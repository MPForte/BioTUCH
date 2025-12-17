# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import utils
from mesh_viewer import MeshViewer

# Add project root to path
from pathlib import Path
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# Import vertex loader directly from the file
sys.path.insert(0, os.path.join(project_root, 'utils'))
import vertex_loader


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.
        fast: Whether to use the new faster implementation (based on the
            Rodrigues formula) instead of the original implementation (which
            first converted to a quaternion and then back to a rotation matrix).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    shape = axis_angle.shape
    device, dtype = axis_angle.device, axis_angle.dtype
    
    # Ensure axis_angle has correct shape
    if len(shape) == 1:
        axis_angle = axis_angle.unsqueeze(0)
        shape = axis_angle.shape
    
    # Calculate angles
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True).unsqueeze(-1)
    
    # Extract components
    rx, ry, rz = axis_angle[..., 0], axis_angle[..., 1], axis_angle[..., 2]
    
    # Create zeros with explicit dimensions
    zeros_shape = tuple(shape[:-1])  # Convert to tuple explicitly
    zeros = torch.zeros(zeros_shape, dtype=dtype, device=device)
    
    # Create cross product matrix with explicit reshaping
    flattened = torch.stack([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=-1)
    
    # Explicitly calculate the new shape
    new_shape = tuple(shape[:-1]) + (3, 3)
    cross_product_matrix = flattened.reshape(new_shape)
    
    # Matrix multiplication
    cross_product_matrix_sqrd = cross_product_matrix @ cross_product_matrix
    
    # Identity matrix with explicit reshaping
    identity = torch.eye(3, dtype=dtype, device=device)
    identity_expanded = identity.expand(tuple(shape[:-1]) + (3, 3))
    
    # Avoid division by zero
    angles_sqrd = angles * angles
    angles_sqrd = torch.where(angles_sqrd == 0, torch.ones_like(angles_sqrd), angles_sqrd)
    
    # Final computation
    result = (
        identity_expanded
        + torch.sinc(angles / torch.pi) * cross_product_matrix
        + ((1 - torch.cos(angles)) / angles_sqrd) * cross_product_matrix_sqrd
    )
    
    return result


@torch.no_grad()
def guess_init(model,
               joints_2d,
               edge_idxs,
               focal_length=5000,
               pose_embedding=None,
               vposer=None,
               use_vposer=True,
               dtype=torch.float32,
               model_type='smpl',
               **kwargs):
    ''' Initializes the camera translation vector

        Parameters
        ----------
        model: nn.Module
            The PyTorch module of the body
        joints_2d: torch.tensor 1xJx2
            The 2D tensor of the joints
        edge_idxs: list of lists
            A list of pairs, each of which represents a limb used to estimate
            the camera translation
        focal_length: float, optional (default = 5000)
            The focal length of the camera
        pose_embedding: torch.tensor 1x32
            The tensor that contains the embedding of V-Poser that is used to
            generate the pose of the model
        dtype: torch.dtype, optional (torch.float32)
            The floating point type used
        vposer: nn.Module, optional (None)
            The PyTorch module that implements the V-Poser decoder
        Returns
        -------
        init_t: torch.tensor 1x3, dtype = torch.float32
            The vector with the estimated camera location

    '''

    body_pose = vposer.decode(
        pose_embedding, output_type='aa').view(1, -1) if use_vposer else None
    if use_vposer and model_type == 'smpl':
        wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                 dtype=body_pose.dtype,
                                 device=body_pose.device)
        body_pose = torch.cat([body_pose, wrist_pose], dim=1)

    output = model(body_pose=body_pose, return_verts=False,
                   return_full_pose=False)
    joints_3d = output.joints
    joints_2d = joints_2d.to(device=joints_3d.device)

    diff3d = []
    diff2d = []
    for edge in edge_idxs:
        diff3d.append(joints_3d[:, edge[0]] - joints_3d[:, edge[1]])
        diff2d.append(joints_2d[:, edge[0]] - joints_2d[:, edge[1]])

    diff3d = torch.stack(diff3d, dim=1)
    diff2d = torch.stack(diff2d, dim=1)

    length_2d = diff2d.pow(2).sum(dim=-1).sqrt()
    length_3d = diff3d.pow(2).sum(dim=-1).sqrt()

    height2d = length_2d.mean(dim=1)
    height3d = length_3d.mean(dim=1)

    est_d = focal_length * (height3d / height2d)

    # just set the z value
    batch_size = joints_3d.shape[0]
    x_coord = torch.zeros([batch_size], device=joints_3d.device,
                          dtype=dtype)
    y_coord = x_coord.clone()
    init_t = torch.stack([x_coord, y_coord, est_d], dim=1)
    return init_t


class FittingMonitor(object):
    def __init__(self, summary_steps=1, visualize=False,
                 maxiters=100, ftol=2e-09, gtol=1e-03,
                 body_color=(1.0, 1.0, 0.9, 1.0),
                 model_type='smpl',
                 interactive=False,
                 **kwargs):
        super(FittingMonitor, self).__init__()

        self.maxiters = maxiters
        self.ftol = ftol
        self.gtol = gtol

        self.visualize = visualize
        self.summary_steps = summary_steps
        self.body_color = body_color
        self.model_type = model_type
        self.interactive = interactive  # Store interactive flag

    def __enter__(self):
        self.steps = 0
        if self.visualize:
            self.mv = MeshViewer(body_color=self.body_color)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if self.visualize:
            self.mv.close_viewer()

    def set_colors(self, vertex_color):
        batch_size = self.colors.shape[0]

        self.colors = np.tile(
            np.array(vertex_color).reshape(1, 3),
            [batch_size, 1])

    # Modify the run_fitting method in FittingMonitor class:

    def run_fitting(self, optimizer, closure, params, body_model,
                    use_vposer=True, pose_embedding=None, vposer=None,
                    **kwargs):
        ''' Helper function for running an optimization process '''
        
        append_wrists = self.model_type == 'smpl' and use_vposer
        prev_loss = None
        
        try:
            for n in range(self.maxiters):
                try:
                    # Check if Trust Region optimizer has converged
                    if hasattr(optimizer, 'has_converged') and optimizer.has_converged:
                        break
                    
                    # Smart memory management - only clear when needed
                    if n % 10 == 0 and torch.cuda.is_available():
                        current_memory = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
                        if current_memory > 0.8:
                            import gc
                            gc.collect()
                            torch.cuda.empty_cache()
        
                    loss = optimizer.step(closure)

                    if torch.isnan(loss).sum() > 0:
                        print('NaN loss value, stopping!')
                        break

                    if torch.isinf(loss).sum() > 0:
                        print('Infinite loss value, stopping!')
                        break

                    # Check ftol convergence
                    if n > 0 and prev_loss is not None and self.ftol > 0:
                        loss_rel_change = utils.rel_change(prev_loss, loss.item())
                        if loss_rel_change <= self.ftol:
                            break

                    prev_loss = loss.item()

                except StopOptimizationError:
                    if self.interactive:
                        print("✓ Contact achieved!")
                    break
                    

        except Exception as e:
            print(f"Unexpected error during optimization: {str(e)}")
            raise

        return prev_loss
    def create_fitting_closure(self,
                       optimizer,
                       body_model,
                       camera=None,
                       gt_joints=None,
                       loss=None,
                       joints_conf=None,
                       joint_weights=None,
                       return_verts=True,
                       return_full_pose=False,
                       use_vposer=False,
                       vposer=None,
                       pose_embedding=None,
                       create_graph=False,
                       pose=None,
                    #    optimized_body_params=None,
                       joint_mask=None,
                       **kwargs):
        faces_tensor = body_model.faces_tensor.view(-1)
        append_wrists = self.model_type == 'smpl' and use_vposer

        def fitting_func(backward=True):
            if backward:
                optimizer.zero_grad()

            # --- Generate body_pose ---
            if use_vposer:
                assert pose_embedding is not None, "pose_embedding required with VPoser"
                body_pose = vposer.decode(pose_embedding, output_type='aa').view(1, -1)

                if append_wrists:
                    wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                            dtype=body_pose.dtype,
                                            device=body_pose.device,
                                            requires_grad=True)
                    body_pose = torch.cat([body_pose, wrist_pose], dim=1)
            else:
                if pose is not None:
                    # Use pose directly to preserve gradient connections
                    body_pose = pose

                elif hasattr(body_model, 'body_pose'):
                    body_pose = body_model.body_pose.clone()
                else:
                    raise RuntimeError("No valid body pose available")

            # CRITICAL FIX: If body_pose is not a leaf tensor, call retain_grad()
            if not body_pose.is_leaf and body_pose.requires_grad and backward:
                body_pose.retain_grad()

            # --- Apply contact_info (e.g. fixed hand params) ---
            if hasattr(loss, 'contact_info'):
                for hand_side in ['right', 'left']:
                    info = loss.contact_info.get(hand_side)
                    if info and info.get('optimized_params') is not None:
                        params = info['optimized_params']
                        indices = loss._get_param_indices(hand_side)

                        if len(params.shape) == 1:
                            params = params.unsqueeze(0)

                        if len(indices) == params.reshape(-1).shape[0]:
                            # Safe assignment
                            # Flatten and clone the pose
                            flat_pose = body_pose.view(-1).clone()
                            flat_params = params.view(-1)

                            # Replace the specified indices
                            flat_pose[indices] = flat_params

                            # Reshape to original pose shape
                            body_pose = flat_pose.view_as(body_pose)
                            body_pose.requires_grad_(True)

                        else:
                            print(f"[Warning] Parameter shape mismatch for {hand_side} hand")
                            print(f"Expected {len(indices)} but got {params.reshape(-1).shape[0]}")

            # --- Forward model pass ---
            body_model_output = body_model(
                return_verts=return_verts,
                body_pose=body_pose,
                return_full_pose=return_full_pose
            )

            # --- Compute total loss ---
            total_loss = loss(
                body_model_output,
                camera=camera,
                gt_joints=gt_joints,
                body_model_faces=faces_tensor,
                joints_conf=joints_conf,
                joint_weights=joint_weights,
                pose_embedding=pose_embedding,
                use_vposer=use_vposer,
                pose=body_pose,
                **kwargs
            )


            # --- Backward pass ---
            if backward:
                total_loss.backward(create_graph=create_graph)

                # IMPROVED: Zero gradients on frozen joints with better error handling
                if joint_mask is not None:
                    if body_pose.grad is not None:
                        with torch.no_grad():
                            # Create a copy of the gradient to avoid in-place modification issues
                            original_grad = body_pose.grad.clone()
                            
                            # Zero gradients for joints that are frozen (mask False)
                            masked_grad = original_grad.clone()
                            masked_grad[:, ~joint_mask] = 0
                            
                            # Replace the gradient
                            body_pose.grad.data.copy_(masked_grad)

            return total_loss

        return fitting_func
class StopOptimizationError(Exception):
    pass

def create_loss(loss_type='smplify', **kwargs):
    if loss_type == 'smplify':
        return SMPLifyLoss(loss_type, **kwargs)
    elif loss_type == 'camera_init':
        return SMPLifyCameraInitLoss(**kwargs)
    elif loss_type == 'biotuch':
        return BioTUCHLoss(loss_type, **kwargs)
    else:
        raise ValueError('Unknown loss type: {}'.format(loss_type))


def L2Loss(pose_curr, pose_ref, size, smooth_weight):
    quat_pose_curr = axis_angle_to_rotation_6d(torch.reshape(pose_curr, size))
    quat_pose_ref = axis_angle_to_rotation_6d(torch.reshape(pose_ref, size))
    loss = (torch.sum((quat_pose_curr - quat_pose_ref) ** 2) * smooth_weight ** 2)
    return loss


def axis_angle_to_rotation_6d(axis_angle: torch.Tensor) -> torch.Tensor:
    # https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )

    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    matrix = o.reshape(quaternions.shape[:-1] + (3, 3))

    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


class SMPLifyLoss(nn.Module):

    def __init__(self,
                 loss_type,
                 rho=100,
                 use_joints_conf=True,
                 dtype=torch.float32,
                 data_weight=1.0,
                 **kwargs):

        super(SMPLifyLoss, self).__init__()

        self.use_joints_conf = use_joints_conf

        self.robustifier = utils.GMoF(rho=rho)
        self.rho = rho

        self.register_buffer('data_weight',
                             torch.tensor(data_weight, dtype=dtype))

        self.loss_type = loss_type


    def reset_loss_weights(self, loss_weight_dict):
        for key in loss_weight_dict:
            if hasattr(self, key):
                weight_tensor = getattr(self, key)
                if 'torch.Tensor' in str(type(loss_weight_dict[key])):
                    weight_tensor = loss_weight_dict[key].clone().detach()
                else:
                    weight_tensor = torch.tensor(loss_weight_dict[key],
                                                 dtype=weight_tensor.dtype,
                                                 device=weight_tensor.device)
                setattr(self, key, weight_tensor)

    def forward(self, body_model_output, camera, gt_joints, joints_conf,
                joint_weights,
                **kwargs):
        projected_joints = camera(body_model_output.joints)
        # Calculate the weights for each joints
        weights = (joint_weights * joints_conf
                   if self.use_joints_conf else
                   joint_weights).unsqueeze(dim=-1)

        # Calculate the distance of the projected joints from
        # the ground truth 2D detections
        joint_diff = self.robustifier(gt_joints - projected_joints)
        joint_loss = (torch.sum(weights ** 2 * joint_diff) *
                      self.data_weight ** 2)
        return joint_loss


class SMPLifyCameraInitLoss(nn.Module):

    def __init__(self, init_joints_idxs, trans_estimation=None,
                 reduction='sum',
                 data_weight=1.0,
                 depth_loss_weight=1e2, dtype=torch.float32,
                 **kwargs):
        super(SMPLifyCameraInitLoss, self).__init__()
        self.dtype = dtype

        if trans_estimation is not None:
            self.register_buffer(
                'trans_estimation',
                utils.to_tensor(trans_estimation, dtype=dtype))
        else:
            self.trans_estimation = trans_estimation

        self.register_buffer('data_weight',
                             torch.tensor(data_weight, dtype=dtype))
        self.register_buffer(
            'init_joints_idxs',
            utils.to_tensor(init_joints_idxs, dtype=torch.long))
        self.register_buffer('depth_loss_weight',
                             torch.tensor(depth_loss_weight, dtype=dtype))

    def reset_loss_weights(self, loss_weight_dict):
        for key in loss_weight_dict:
            if hasattr(self, key):
                weight_tensor = getattr(self, key)
                weight_tensor = torch.tensor(loss_weight_dict[key],
                                             dtype=weight_tensor.dtype,
                                             device=weight_tensor.device)
                setattr(self, key, weight_tensor)

    def forward(self, body_model_output, camera, gt_joints,
                **kwargs):

        projected_joints = camera(body_model_output.joints)

        joint_error = torch.pow(
            torch.index_select(gt_joints, 1, self.init_joints_idxs) -
            torch.index_select(projected_joints, 1, self.init_joints_idxs),
            2)
        joint_loss = torch.sum(joint_error) * self.data_weight ** 2

        depth_loss = 0.0
        if (self.depth_loss_weight.item() > 0 and self.trans_estimation is not
                None):
            depth_loss = self.depth_loss_weight ** 2 * torch.sum((
                camera.translation[:, 2] - self.trans_estimation[:, 2]).pow(2))
        return depth_loss + joint_loss 
    

class BioTUCHLoss(SMPLifyLoss):
    def __init__(self, loss_type, bio_contact, active_hand=None, interactive=False, **kwargs):
        super(BioTUCHLoss, self).__init__(loss_type=loss_type, **kwargs)

        self.joint_indices = {'right': [16, 18, 20], 'left': [15, 17, 19]}
        self.keypoint_indices = {'right': [2, 3, 4], 'left': [5, 6, 7]}

        self.cached_proximity_results = {'right': None, 'left': None}
        self.active_hand = active_hand
        self.interactive = interactive  # Store interactive flag
        self.bio_contact_weight = kwargs.get('bio_contact_weight')
        self.contact_weight = kwargs.get('contact_weight', 1.0)
        self.consistency_weight = kwargs.get('consistency_weight', 0.25)
        self.penetration_weight = kwargs.get('penetration_weight', 1.0)
        self.frame_loss_weight = kwargs.get('frame_loss_weight', 1.0)    
        self.contact_distance_threshold = kwargs.get('contact_distance_threshold', 0.005)
        self.contact_mode = kwargs.get('contact_mode', 'all_axes')
        self.distance_history = {'right': [], 'left': []}
        self.iteration_count = {'right': 0, 'left': 0}

        device = kwargs.get('device', 'cuda')
        self.register_buffer('bio_contact', torch.tensor(bio_contact, dtype=bool))
        self._load_vertex_indices(device)

        self.initial_pairs = {'right': [], 'left': []}
        self.first_pass = {'right': True, 'left': True}
        self.contact_found = {'right': False, 'left': False}
        self.contact_info = {'right': None, 'left': None}
        self.distance_metric = {'right': None, 'left': None}
        self.best_vertices = {'right': None, 'left': None}
        self.iteration_counter = {}

    def _get_top_candidates(self, distances, hand_verts, body_verts, top_k=500):
        flat_distances = distances.view(-1)
        top_k = min(top_k, flat_distances.numel())

        _, top_indices = torch.topk(flat_distances, k=top_k, largest=False)

        hand_indices = top_indices // body_verts.shape[0]
        body_indices_local = top_indices % body_verts.shape[0]
        diffs = hand_verts[hand_indices] - body_verts[body_indices_local]

        return hand_indices, body_indices_local, top_indices, diffs


    def find_multiple_contact_pairs(self, hand_side, hand_verts, body_verts, body_indices, view_direction):
        loader = vertex_loader.vertex_loader
        device = hand_verts.device

        distances = torch.cdist(hand_verts, body_verts)
        hand_indices, body_indices_local, _, diffs = self._get_top_candidates(
            distances, hand_verts, body_verts, top_k=500
        )

        # Use region selection weights directly on body coordinates
        x_weight = self.region_selection_weights['x_weight']
        y_weight = self.region_selection_weights['y_weight'] 
        z_weight = self.region_selection_weights['z_weight']

        weighted_distances = (
            x_weight * torch.abs(diffs[:, 0]) +  # body X differences
            y_weight * torch.abs(diffs[:, 1]) +  # body Y differences
            z_weight * torch.abs(diffs[:, 2])    # body Z differences
        )

        min_val = weighted_distances.min().item()
        threshold = min_val + 0.02
        valid_mask = weighted_distances <= threshold
        if not valid_mask.any():
            return None

        region_vertex_sets = self._define_body_regions(hand_side, loader, device)

        valid_indices = torch.nonzero(valid_mask).squeeze(1)
        contact_pairs = self._group_candidates_by_region(
            valid_indices=valid_indices,
            hand_indices=hand_indices,
            body_indices_local=body_indices_local,
            body_indices=body_indices,
            camera_weighted_distances=weighted_distances,
            flat_distances=distances.view(-1),
            top_indices=None,
            hand_verts=hand_verts,
            body_verts=body_verts,
            region_vertex_sets=region_vertex_sets,
            hand_side=hand_side
        )

        return contact_pairs if contact_pairs and len(contact_pairs) >= 2 else None

    def _calculate_camera_weighted_distances(self, diffs, view_direction):
        # Use region selection weights directly on body coordinates
        x_weight = self.region_selection_weights['x_weight']
        y_weight = self.region_selection_weights['y_weight'] 
        z_weight = self.region_selection_weights['z_weight']
        
        return (
            x_weight * torch.abs(diffs[:, 0]) +  # body X differences
            y_weight * torch.abs(diffs[:, 1]) +  # body Y differences
            z_weight * torch.abs(diffs[:, 2])    # body Z differences
        )

    def _load_vertex_indices(self, device):
        """Load vertex indices for different body parts."""
        loader = vertex_loader.vertex_loader
        
        # Right hand and arm vertices
        self.register_buffer('right_hand', loader.get_right_hand(device))
        self.register_buffer('right_limbs', torch.cat([
            loader.get_right_hand(device),
            loader.get_vertices('rightForeArm', device),
        ]))
        self.register_buffer('right_excluded_vertices', torch.cat([
            self.right_limbs,
            loader.get_vertices('rightArm', device),
            loader.get_lower_body_vertices(device)
        ]))

        # Left hand and arm vertices
        self.register_buffer('left_hand', loader.get_left_hand(device))
        self.register_buffer('left_limbs', torch.cat([
            loader.get_left_hand(device),
            loader.get_vertices('leftForeArm', device),
        ]))
        self.register_buffer('left_excluded_vertices', torch.cat([
            self.left_limbs,
            loader.get_vertices('leftArm', device),
            loader.get_lower_body_vertices(device)
        ]))

    def initialize_adaptive_weights(self, camera, global_orient):
        """Initialize adaptive weights - MUST be called before any other processing."""
        if camera is not None and global_orient is not None:
            # Get camera view direction in body space
            cam_view_body = self._get_camera_view_direction_in_body(camera, global_orient)
            cam_z_weights = torch.abs(cam_view_body)
            base_weight = 1
            
            # Different emphasis for selection vs optimization
            selection_depth_emphasis = 0.25 #0.1  # LOW weight for camera Z during selection #data-driven would be 0.296
            optimization_depth_emphasis = 4.0   # HIGH weight for camera Z during optimization #data-driven would be 3.37
            
            # Calculate selection weights (Z gets LESS weight)
            selection_weights = base_weight * (1 + (selection_depth_emphasis - 1) * cam_z_weights)
            self.region_selection_weights = {
                'x_weight': selection_weights[0].item(),
                'y_weight': selection_weights[1].item(),
                'z_weight': selection_weights[2].item()
            }
            
            # Calculate optimization weights (Z gets MORE weight)  
            opt_weights = base_weight * (1 + (optimization_depth_emphasis - 1) * cam_z_weights)
            self.optimization_weights = {
                'x_weight': opt_weights[0].item(),
                'y_weight': opt_weights[1].item(),
                'z_weight': opt_weights[2].item()
            }
            
            return True
        return False


    # ---------- Contact Detection Methods ----------
    
    def _calculate_proximity_metric(self, hand_side, hand_verts, body_verts, body_indices, diffs_data, view_direction):
        """Calculate proximity metric with camera-dependent weighting - OPTIMIZED VERSION."""
        
        # Unpack the optimized data structure
        diffs, hand_indices, body_indices_local, full_distances = diffs_data

        # Apply region selection weights directly to body coordinates
        x_weight = self.region_selection_weights['x_weight']
        y_weight = self.region_selection_weights['y_weight']
        z_weight = self.region_selection_weights['z_weight']

        weighted_distances = (
            x_weight * torch.abs(diffs[:, 0]) +  # body X differences
            y_weight * torch.abs(diffs[:, 1]) +  # body Y differences
            z_weight * torch.abs(diffs[:, 2])    # body Z differences
        )
                        
        # Find minimum weighted distance among top candidates
        min_val, min_idx = torch.min(weighted_distances, dim=0)
        
        # Map back to original indices
        original_hand_idx = hand_indices[min_idx].item()
        original_body_idx = body_indices_local[min_idx].item()
        
        # Get the actual body vertex index from body_indices
        actual_body_idx = body_indices[original_body_idx].item()
        
        # Get hand vertex index in the original hand vertex array
        hand_vertex = self.right_hand[original_hand_idx].item() if hand_side == 'right' else self.left_hand[original_hand_idx].item()
        
        # Get per-axis differences for the winning candidate
        winning_diffs = diffs[min_idx]
        
        # weighted_dist = x_weight * abs(cam_x_component[min_idx].item()) + y_weight * abs(cam_y_component[min_idx].item()) + z_weight * abs(cam_z_component[min_idx].item())
        weighted_dist = x_weight * abs(winning_diffs[0].item()) + y_weight * abs(winning_diffs[1].item()) + z_weight * abs(winning_diffs[2].item())
        
        # Create result in same format as before
        result = {
            'valid': True,
            'camera_weighted_dist': min_val.item(),  # Camera viewing plane weighting
            'adaptive_weighted_dist': weighted_dist,  # x/y/z axis weighting  
            'hand_idx': original_hand_idx,
            'body_idx': actual_body_idx,
            'distance_type': 'proximity'
        }
        
        return result
        
    
    def _define_body_regions(self, hand_side, loader, device):
        # Torso combines spine, spine1, and spine2
        torso = torch.cat([
            loader.get_vertices('spine', device),
            loader.get_vertices('spine1', device),
            loader.get_vertices('spine2', device)
        ])

        if hand_side == 'right':
            regions = {
                'forearm': loader.get_vertices('rightForeArm', device),
                'upperarm': loader.get_vertices('rightArm', device),
                'shoulder': loader.get_vertices('rightShoulder', device),
                'torso': torso,
                'hand': loader.get_right_hand(device)
            }
        else:
            regions = {
                'forearm': loader.get_vertices('leftForeArm', device),
                'upperarm': loader.get_vertices('leftArm', device),
                'shoulder': loader.get_vertices('leftShoulder', device),
                'torso': torso,
                'hand': loader.get_left_hand(device)
            }

        return regions

    def _compute_camera_plane_consistency_loss(self, hand_vert, initial_hand_pos):
        movement = hand_vert - initial_hand_pos
        
        # Check if optimization weights are available
        if hasattr(self, 'optimization_weights') and self.optimization_weights:
            # Use INVERSE of optimization weights
            x_weight = 1.0 / self.optimization_weights['x_weight'] 
            y_weight = 1.0 / self.optimization_weights['y_weight']
            z_weight = 1.0 / self.optimization_weights['z_weight']
            
            weighted_movement = torch.abs(movement) * torch.tensor([x_weight, y_weight, z_weight], device=movement.device)
            return torch.sum(weighted_movement)
        else:
            # Fallback to simple L2 norm
            return torch.norm(movement)

    def determine_active_hands(self, vertices, camera=None, global_orient=None):
        """
        Simplified approach: Determines which hands to optimize using only the proximity metric.
        """
        # Initialize adaptive weights
        self.initialize_adaptive_weights(camera, global_orient)
        
        with torch.no_grad():
            # Get device from vertices
            device = vertices.device
            
            # Get camera viewing direction (pass device)
            if camera is None or global_orient is None:
                view_direction = torch.tensor([0, 0, -1.0], device=device)
                print(f"Using default camera direction (no camera/orientation)")
            else:
                view_direction = self._get_camera_view_direction_in_body(camera, global_orient)

            # CACHE view direction for later use in _compute_hand_loss
            self._cached_view_direction = view_direction
            
            # Process each hand
            results = {'right': {}, 'left': {}}
            for hand_side in ['right', 'left']:               
                # Get vertices and differences
                vertices_data = self._get_vertices_data(hand_side, vertices)
                if not vertices_data:  # No valid body vertices
                    results[hand_side] = self._get_default_result()
                    continue
                    
                hand_verts, body_verts, body_indices, optimized_diffs = vertices_data

                # Use proximity metric
                proximity_result = self._calculate_proximity_metric(
                    hand_side, hand_verts, body_verts, body_indices, 
                    optimized_diffs, view_direction  # Pass the tuple to the optimized function
                )
                
                # If proximity metric fails, use Euclidean distance as fallback
                if not proximity_result['valid']:
                    print("Proximity metric failed, falling back to Euclidean distance with adaptive weighting")
                    diffs, hand_indices, body_indices_local, full_distances = optimized_diffs
                    
                    # Apply adaptive weights to the differences
                    abs_diffs = torch.abs(diffs)
                    x_weight = self.region_selection_weights['x_weight']
                    y_weight = self.region_selection_weights['y_weight'] 
                    z_weight = self.region_selection_weights['z_weight']
                    
                    adaptive_weighted_distances = (
                        x_weight * abs_diffs[:, 0] +
                        y_weight * abs_diffs[:, 1] +
                        z_weight * abs_diffs[:, 2]
                    )
                    
                    # Find minimum adaptive weighted distance
                    adaptive_min, min_idx = torch.min(adaptive_weighted_distances, dim=0)
                    
                    # Get the euclidean distance for this pair
                    euclidean_dist = full_distances.view(-1)[min_idx].item()
                    
                    # Map back to original indices
                    original_hand_idx = hand_indices[min_idx].item()
                    original_body_idx = body_indices_local[min_idx].item()
                    actual_body_idx = body_indices[original_body_idx].item()
                    
                    proximity_result = {
                        'valid': True,
                        'camera_weighted_dist': euclidean_dist,           # Raw euclidean (no camera weighting)
                        'adaptive_weighted_dist': adaptive_min.item(),    # Properly calculated adaptive weighting
                        'hand_idx': original_hand_idx,
                        'body_idx': actual_body_idx,
                        'distance_type': 'euclidean_adaptive'
                    }
                

                self.cached_proximity_results[hand_side] = {
                    'proximity_result': proximity_result,
                    'hand_verts': hand_verts,
                    'body_verts': body_verts, 
                    'body_indices': body_indices,
                    'optimized_diffs': optimized_diffs
                }
                
                results[hand_side] = proximity_result
                
                # Store best vertices if valid
                if proximity_result['valid']:
                    self._store_best_vertices(hand_side, hand_verts, vertices, 
                                            proximity_result, body_indices)
            
            # Print results and decide which hand to optimize
            self._print_results_summary(results)
            self.active_hand = self._decide_active_hand(results, {'epsilon': 0.001})
        
        return
        

    def _get_vertices_data(self, hand_side, vertices):
        """Get vertices data and calculate differences."""
        # Get hand/body vertices based on hand side
        all_indices = torch.arange(vertices.shape[0], device=vertices.device)
        if hand_side == 'right':
            hand_verts = vertices[self.right_hand]
            excluded = self.right_excluded_vertices
        else:
            hand_verts = vertices[self.left_hand]
            excluded = self.left_excluded_vertices
            
        body_mask = ~torch.isin(all_indices, excluded)
        body_verts = vertices[body_mask]
        body_indices = torch.nonzero(body_mask).squeeze(1)
        
        if body_verts.shape[0] == 0:
            return None
            
        # Pre-compute all differences once
        distances = torch.cdist(hand_verts, body_verts)  

        # Find top candidates (much smaller than full 5.7M combinations)
        top_k = min(100, distances.numel())  # Top 100 closest pairs
        flat_distances = distances.view(-1)
        _, top_indices = torch.topk(flat_distances, top_k, largest=False)
        
        # Convert flat indices back to (hand_idx, body_idx) pairs
        hand_indices = torch.div(top_indices, body_verts.shape[0], rounding_mode='trunc')
        body_indices_local = top_indices % body_verts.shape[0]
        
        # Only compute differences for top candidates
        top_hand_verts = hand_verts[hand_indices]
        top_body_verts = body_verts[body_indices_local]
        diffs = top_hand_verts - top_body_verts  # Much smaller tensor!
        
        return hand_verts, body_verts, body_indices, (diffs, hand_indices, body_indices_local, distances)

    
    def _get_default_result(self):
        return {
            'valid': False,
            'euclidean_dist': float('inf'),      # Raw 3D euclidean distance
            'adaptive_weighted_dist': float('inf'),  # Distance weighted by x/y/z adaptive weights
            'camera_weighted_dist': float('inf'),    # Distance weighted by camera viewing plane
            'hand_idx': -1,
            'body_idx': -1,
            'distance_type': None
        }


    def _store_best_vertices(self, hand_side, hand_verts, vertices, result, body_indices):
        """Store the best vertices for a hand."""
        if result['valid'] and result['hand_idx'] >= 0:
            hand_idx = result['hand_idx']
            body_idx = result['body_idx']
            
            self.best_vertices[hand_side] = {
                'hand_idx': hand_idx,
                'hand_vertex': self.right_hand[hand_idx].item() if hand_side == 'right' else self.left_hand[hand_idx].item(),
                'body_vertex': body_idx,
                'hand_pos': hand_verts[hand_idx].detach().clone(),
                'body_pos': vertices[body_idx].detach().clone()
            }
            self.distance_metric[hand_side] = result['distance_type']

    def _print_results_summary(self, results):
        """Print summary of results for both hands."""
        
        for side in ['right', 'left']:
            result = results[side]
            type_str = result.get('distance_type')
            if type_str and result.get('valid', False):
                
                # Print vertex information if available
                if result.get('hand_idx', -1) >= 0:
                    hand_verts = self.right_hand if side == 'right' else self.left_hand
                    hand_vertex_global = hand_verts[result['hand_idx']].item()
                    body_vertex = result['body_idx']
                    
            else:
                print(f"{side.capitalize()} hand: No valid metric found")

    def _decide_active_hand(self, results, config):
        """Use distances to decide which hand to optimize."""        
        # Get weighted and raw distances
        right_camera_weighted = results['right'].get('camera_weighted_dist', float('inf'))
        left_camera_weighted = results['left'].get('camera_weighted_dist', float('inf'))
        
        right_valid = results['right'].get('valid', False)
        left_valid = results['left'].get('valid', False)
        
        
        # Handle near-contact cases
        if right_camera_weighted <= config['epsilon'] and left_camera_weighted <= config['epsilon']:
            return 'both'
        elif right_camera_weighted <= config['epsilon']:
            return 'right'
        elif left_camera_weighted <= config['epsilon']:
            return 'left'
        
        # Handle valid distance comparison
        if right_valid and left_valid:
            min_dist = min(right_camera_weighted, left_camera_weighted)
            max_dist = max(right_camera_weighted, left_camera_weighted)
            
            if min_dist > 0:
                relative_diff = (max_dist - min_dist) / min_dist
                
                raw_similarity_threshold = config.get('raw_similarity', 0.5)
                
                if relative_diff <= raw_similarity_threshold:
                    return 'both'
                else:
                    closer = 'right' if right_camera_weighted < left_camera_weighted else 'left'
                    if self.interactive:
                        print(f"\nOptimizing {closer} hand only")
                    return closer
            else:
                print("\nWarning: Invalid minimum distance detected")
                closer = 'right' if right_camera_weighted < left_camera_weighted else 'left'
                return closer
        
        # Handle single valid case
        if right_valid:
            if self.interactive:
                print("\nOnly right hand has valid distance measurements")
            return 'right'
        elif left_valid:
            if self.interactive:
                print("\nOnly left hand has valid distance measurements")
            return 'left'
        
        # Fallback
        print("\nWarning: No valid distances found for either hand")
        return 'both'  # Default fallback

    # ---------- Optimization Methods ----------

    def create_active_joint_mask(self, device):
        """Creates joint mask based on current contact state."""
        joint_mask = torch.zeros(63, dtype=torch.bool, device=device)
        
        if self.active_hand == 'both':
            for hand_side in ['right', 'left']:
                if not self.contact_found[hand_side]:
                    for joint_idx in self.joint_indices[hand_side]:
                        joint_range = slice(joint_idx * 3, (joint_idx + 1) * 3)
                        joint_mask[joint_range] = True
        else:
            hand_side = self.active_hand
            if not self.contact_found[hand_side]:
                for joint_idx in self.joint_indices[hand_side]:
                    joint_range = slice(joint_idx * 3, (joint_idx + 1) * 3)
                    joint_mask[joint_range] = True

        return joint_mask

    def _get_param_indices(self, hand_side):
        """Helper to get parameter indices for a hand."""
        params = []
        for idx in self.joint_indices[hand_side]:
            params.extend(range(idx * 3, (idx + 1) * 3))
        return params

    def _get_keypoint_indices(self, hand_side):
        """Helper to get keypoint indices for a hand."""
        return self.keypoint_indices[hand_side]

    # ---------- Contact Loss Methods ----------

    def check_contact_for_hand(self, hand_side, hand_verts, body_verts, body_indices):
        """Checks if contact is found for a specific hand."""
        distances = torch.cdist(hand_verts, body_verts)
        min_distances, min_body_idx = torch.min(distances, dim=1)
        
        # Find overall closest vertices
        closest_idx = torch.argmin(min_distances)
        closest_distance = min_distances[closest_idx].item()
        closest_body_idx = min_body_idx[closest_idx].item()
        
        hand_vertex_idx = (self.right_hand[closest_idx] if hand_side == 'right' 
                         else self.left_hand[closest_idx]).item()
        body_vertex_idx = body_indices[closest_body_idx].item()
        
        # Get actual positions for the closest vertices
        hand_pos = hand_verts[closest_idx]
        body_pos = body_verts[closest_body_idx]
        
        # Calculate per-axis distances
        x_diff = abs(hand_pos[0] - body_pos[0])
        y_diff = abs(hand_pos[1] - body_pos[1])
        z_diff = abs(hand_pos[2] - body_pos[2])

        self.iteration_count[hand_side] += 1
    
        self.distance_history[hand_side].append(closest_distance)
        
        # Determine if contact is found
        contact_found = (x_diff < self.contact_distance_threshold and 
                         y_diff < self.contact_distance_threshold and 
                         z_diff < self.contact_distance_threshold)
        
        if contact_found:
            contact_info = {
                'hand_vertex': hand_vertex_idx,
                'body_vertex': body_vertex_idx,
                'distance': closest_distance,
                'x_distance': x_diff.item(),
                'y_distance': y_diff.item(),
                'z_distance': z_diff.item(),
                'mode': self.contact_mode
            }
            return True, contact_info
        
        return False, None

    def compute_interpenetration_loss(self, hand_side, vertices):
        """
        Compute interpenetration loss using piece-wise barrier energy
        Following the approach from ExPose (Choutas et al.)
        """
        # Get relevant mesh parts
        limbs = self.right_limbs if hand_side == 'right' else self.left_limbs
        excluded_vertices = self.right_excluded_vertices if hand_side == 'right' else self.left_excluded_vertices
        
        arm_hand_verts = vertices[limbs]
        
        # Get body vertices for collision checking
        all_indices = torch.arange(vertices.shape[0], device=vertices.device)
        body_mask = ~torch.isin(all_indices, excluded_vertices)
        body_verts = vertices[body_mask]

        # Compute distances between hand vertices and body vertices
        differences = arm_hand_verts.unsqueeze(1) - body_verts.unsqueeze(0)
        distances = torch.norm(differences, dim=2)

        sigma = 0.002  # contact threshold
        epsilon = 0.01  # smoothing term to avoid singularities

        # Compute barrier function
        penetration_mask = distances < sigma
        if not torch.any(penetration_mask):
            return torch.tensor(0.0, device=vertices.device)

        # Barrier energy computation
        penetration_distances = distances[penetration_mask]
        barrier_energy = torch.exp(-penetration_distances / epsilon).sum()

        barrier_weight = 100.0
        penetration_loss = barrier_weight * barrier_energy

        return penetration_loss
    
    def compute_contact_loss_for_hand(self, hand_side, hand_verts, vertices, initial_pairs):
        """
        FIXED: Works for both single and multi-region cases without convergence issues.
        Uses smooth blending instead of hard switching to avoid discontinuities.
        """
        if not initial_pairs:
            return torch.tensor(0.0, device=vertices.device)

        # SINGLE REGION: Use simple approach (like fitting_simple.py)
        if len(initial_pairs) == 1:
            return self._compute_single_region_loss(hand_side, hand_verts, vertices, initial_pairs[0])
        
        # MULTI-REGION: Use smooth blending approach
        return self._compute_multi_region_blended_loss(hand_side, hand_verts, vertices, initial_pairs)

    def _compute_single_region_loss(self, hand_side, hand_verts, vertices, pair):
        """Single region loss - uses optimization weights"""
        best_hand_vert = hand_verts[pair['hand_idx']]
        best_body_vert = vertices[pair['body_vertex']]
        initial_hand_pos = pair.get('initial_hand_pos', best_hand_vert.detach())
        
        # Compute per-axis differences
        diffs = torch.abs(best_hand_vert - best_body_vert)
        x_diff, y_diff, z_diff = diffs[0], diffs[1], diffs[2]
        
        # Use optimization weights here
        x_weight = self.optimization_weights['x_weight']
        y_weight = self.optimization_weights['y_weight']
        z_weight = self.optimization_weights['z_weight']

        # Compute losses
        contact_component = self.contact_weight * (x_weight * x_diff + y_weight * y_diff + z_weight * z_diff)
        consistency_component = self.consistency_weight * self._compute_camera_plane_consistency_loss(
                                best_hand_vert, initial_hand_pos)
        penetration_component = self.penetration_weight * self.compute_interpenetration_loss(hand_side, vertices)

        return contact_component + consistency_component + penetration_component

            
    def _compute_multi_region_blended_loss(self, hand_side, hand_verts, vertices, initial_pairs):
        """Multi-region loss using smooth blending - uses optimization weights"""
        total_contact_loss = torch.tensor(0.0, device=vertices.device)
        total_weight = torch.tensor(0.0, device=vertices.device)
        
        # Use optimization weights here
        x_weight = self.optimization_weights['x_weight']
        y_weight = self.optimization_weights['y_weight']
        z_weight = self.optimization_weights['z_weight']
        
        for pair in initial_pairs:
            hand_vert = hand_verts[pair['hand_idx']]
            
            # FIX: Handle both 'body_vertex' and 'body_idx' keys for compatibility
            if 'body_vertex' in pair:
                body_vert = vertices[pair['body_vertex']]
            elif 'body_idx' in pair:
                body_vert = vertices[pair['body_idx']]
            else:
                print(f"Warning: No body vertex key found in pair: {list(pair.keys())}")
                continue
            
            # Compute this region's distance
            diffs = torch.abs(hand_vert - body_vert)
            x_diff, y_diff, z_diff = diffs[0], diffs[1], diffs[2]
            weighted_distance = x_weight * x_diff + y_weight * y_diff + z_weight * z_diff
            
            # SMOOTH WEIGHTING: Closer regions get higher weight, but all contribute
            region_weight = torch.exp(-weighted_distance * 50.0)  # Exponential weighting
            region_weight = torch.clamp(region_weight, min=0.01)   # Minimum weight to keep all active
            
            # This region's contribution to the loss
            region_loss = weighted_distance * region_weight
            
            total_contact_loss += region_loss
            total_weight += region_weight
        
        # Normalize by total weight to get average weighted loss
        if total_weight > 0:
            contact_component = self.contact_weight * (total_contact_loss / total_weight)
        else:
            contact_component = torch.tensor(0.0, device=vertices.device)
        
        # Add consistency and penetration losses (using first pair for consistency reference)
        first_pair = initial_pairs[0]
        best_hand_vert = hand_verts[first_pair['hand_idx']]
        initial_hand_pos = first_pair.get('initial_hand_pos', best_hand_vert.detach())
        
        consistency_component = self.consistency_weight * self._compute_camera_plane_consistency_loss(
                                best_hand_vert, initial_hand_pos)
        penetration_component = self.penetration_weight * self.compute_interpenetration_loss(hand_side, vertices)
        
        return contact_component + consistency_component + penetration_component

    def _group_candidates_by_region(
        self, valid_indices, hand_indices, body_indices_local, body_indices,
        camera_weighted_distances, flat_distances, top_indices,
        hand_verts, body_verts, region_vertex_sets, hand_side
    ):
        """FIXED: Use consistent key names for multi-region contact pairs"""
        region_contacts = {}

        for idx in valid_indices:
            hi = hand_indices[idx].item()
            bi_local = body_indices_local[idx].item()
            bi_global = body_indices[bi_local].item()

            dist = camera_weighted_distances[idx].item()

            for region_name, region_verts in region_vertex_sets.items():
                if bi_global in region_verts:
                    if region_name not in region_contacts or dist < region_contacts[region_name]['dist']:
                        region_contacts[region_name] = {
                            'dist': dist,
                            'hand_idx': hi,
                            'body_idx': bi_global,  # Keep this for compatibility
                            'vector': hand_verts[hi] - body_verts[bi_local]
                        }

        # Sort by distance and return list of contact pairs
        sorted_contacts = sorted(region_contacts.items(), key=lambda x: x[1]['dist'])

        contact_pairs = []
        for region, info in sorted_contacts:
            contact_pairs.append({
                'region': region,
                'hand_idx': info['hand_idx'],
                'body_idx': info['body_idx'],      
                'body_vertex': info['body_idx'],   
                'vector': info['vector'],
                'dist': info['dist']
            })

        return contact_pairs

    def _compute_hand_loss(self, hand_side, hand_vertices, vertices):
        """
        UPDATED: Maintains both single and multi-region functionality
        """
        # Initialize pairs if needed
        if self.first_pass[hand_side] or not self.initial_pairs[hand_side]:
            cached = self.cached_proximity_results[hand_side]
            if cached is None:
                return None
                
            hand_verts, body_verts, body_indices = cached['hand_verts'], cached['body_verts'], cached['body_indices']
            
            # Try multi-region detection
            view_direction = getattr(self, '_cached_view_direction', 
                                torch.tensor([0, 0, -1.0], device=hand_vertices.device))
            
            multiple_pairs = None
            try:
                multiple_pairs = self.find_multiple_contact_pairs(hand_side, hand_verts, body_verts, body_indices, view_direction)
            except Exception as e:
                print(f"Multi-region detection failed for {hand_side} hand: {str(e)}")
            
            # Decide approach based on what we found
            if multiple_pairs is not None and len(multiple_pairs) >= 2:
                # Use multi-region with smooth blending
                self.initial_pairs[hand_side] = multiple_pairs
                if self.interactive:
                    print(f"✓ Multi-region activated for {hand_side} hand ({len(multiple_pairs)} regions)")
            else:
                # Use single-region approach
                proximity_result = cached['proximity_result']
                if proximity_result['valid']:
                    pair = {
                        'hand_idx': proximity_result['hand_idx'],
                        'hand_vertex': (self.right_hand[proximity_result['hand_idx']] 
                                    if hand_side == 'right' else 
                                    self.left_hand[proximity_result['hand_idx']]).item(),
                        'body_vertex': proximity_result['body_idx'],
                        'initial_hand_pos': hand_verts[proximity_result['hand_idx']].detach().clone(),
                        'initial_body_pos': vertices[proximity_result['body_idx']].detach().clone(),
                        'metric_type': proximity_result['distance_type'],
                        'region': 'single_contact'
                    }
                    self.initial_pairs[hand_side] = [pair]
                    if self.interactive:
                        print(f"✓ Single-region approach for {hand_side} hand")
                else:
                    print(f"✗ No valid contacts found for {hand_side} hand")
                    self.initial_pairs[hand_side] = []
            
            self.first_pass[hand_side] = False
        
        # Compute loss using the appropriate method
        if self.initial_pairs[hand_side]:
            return self.compute_contact_loss_for_hand(
                hand_side, hand_vertices, vertices,
                self.initial_pairs[hand_side]
            )
        
        return None
    def forward(self, body_model_output, camera, gt_joints, joints_conf,
            body_model_faces, joint_weights,
            use_vposer=False, vposer=None, pose_embedding=None,
            pose=None, opt_idx=0, **kwargs):
        
        # Calculate standard frame loss using parent class
        frame_loss = super(BioTUCHLoss, self).forward(
            body_model_output, camera=camera,
            gt_joints=gt_joints, body_model_faces=body_model_faces,
            joints_conf=joints_conf, joint_weights=joint_weights,
            pose_embedding=pose_embedding, use_vposer=use_vposer,
            **kwargs)
        
        # Calculate bio contact loss if applicable
        biotuch_loss = 0.0
        any_contact_this_iteration = False

        if self.bio_contact and self.bio_contact_weight > 0:
            vertices = body_model_output.vertices.squeeze(0)
            
            # Get current body pose based on what's available
            body_pose = self._get_current_body_pose(use_vposer, vposer, pose_embedding, pose, body_model_output)
                    
            # Initialize active hand if not already set
            if self.active_hand is None:
                # Already have adaptive weights calculated, just determine hands
                self.determine_active_hands(vertices)
                
                    
            # Get hands to process
            active_hands = ['right', 'left'] if self.active_hand == 'both' else [self.active_hand]
            remaining_hands = [hand for hand in active_hands if not self.contact_found[hand]]
            
            # If all hands have found contact, terminate optimization
            if not remaining_hands:
                raise StopOptimizationError()

            # Process each remaining hand
            for hand_side in remaining_hands:
                # Track iteration count for this hand
                self._increment_iteration_counter(hand_side)
                
                # Get vertices for contact check
                hand_vertices, body_vertices, body_indices = self._get_contact_vertices(hand_side, vertices)
                
                # Check for contact
                contact_found, contact_info = self.check_contact_for_hand(
                    hand_side, hand_vertices, body_vertices, body_indices
                )
                
                # If contact found, store information and continue
                if contact_found:
                    self._handle_contact_found(hand_side, contact_info, body_pose)
                    any_contact_this_iteration = True
                    continue
                
                # If no contact found, compute and add loss for this hand
                hand_loss = self._compute_hand_loss(
                    hand_side, hand_vertices, vertices)
                
                if hand_loss is not None:
                    biotuch_loss += self.bio_contact_weight * hand_loss

            # If any hand found contact this iteration, trigger an update
            if any_contact_this_iteration:
                raise StopOptimizationError()
            
        return biotuch_loss + self.frame_loss_weight * frame_loss
    
    def _get_camera_view_direction_in_body(self, camera, global_orient):
        """
        Computes the camera's viewing direction in the body's coordinate system.

        Args:
            camera: Object with .rotation (3x3) matrix (world-from-camera)
            global_orient: (1, 3) axis-angle vector (world-from-body)

        Returns:
            view_direction_body: (3,) tensor
        """
        if global_orient is None:
            raise ValueError("global_orient must be provided")

        if not hasattr(camera, 'rotation') or camera.rotation is None:
            raise ValueError("camera.rotation must be defined")

        try:
            cam_view_cam = torch.tensor([0, 0, -1.0],
                                        device=global_orient.device,
                                        dtype=global_orient.dtype).view(1, 3, 1)

            R_cam = camera.rotation.view(1, 3, 3)              # (1, 3, 3)
            cam_view_world = torch.matmul(R_cam, cam_view_cam)  # (1, 3, 1)

            R_body = axis_angle_to_matrix(global_orient).view(1, 3, 3)
            cam_view_body = torch.matmul(R_body.transpose(-2, -1), cam_view_world)  # (1, 3, 1)

            return cam_view_body.view(3)

        except Exception as e:
            print(f"[WARN] Camera view transform failed: {str(e)}. Defaulting to -Z.")
            return torch.tensor([0, 0, -1.0], device=global_orient.device, dtype=global_orient.dtype)


    
    def _get_current_body_pose(self, use_vposer, vposer, pose_embedding, pose, body_model_output):
        """Get the current body pose from available sources."""
        if use_vposer and pose_embedding is not None and vposer is not None:
            body_pose = vposer.decode(pose_embedding, output_type='aa').view(1, -1)
        elif pose is not None:
            body_pose = pose
        elif hasattr(body_model_output, 'body_pose'):
            body_pose = body_model_output.body_pose
        else:
            raise RuntimeError("No valid body pose source available")

        # Ensure body_pose is properly shaped
        if len(body_pose.shape) == 1:
            body_pose = body_pose.unsqueeze(0)
            
        return body_pose
    
    def _increment_iteration_counter(self, hand_side):
        """Increment the iteration counter for a specific hand."""
        if not hasattr(self, 'iteration_counter'):
            self.iteration_counter = {}
        
        if hand_side not in self.iteration_counter:
            self.iteration_counter[hand_side] = 0
        
        self.iteration_counter[hand_side] += 1
    
    def _get_contact_vertices(self, hand_side, vertices):
        """Get vertices for contact detection for a specific hand."""
        # Get hand vertices
        hand_vertices = vertices[self.right_hand if hand_side == 'right' else self.left_hand]
        excluded_vertices = self.right_excluded_vertices if hand_side == 'right' else self.left_excluded_vertices
        
        # Get body vertices (excluding the hand's own parts)
        all_indices = torch.arange(vertices.shape[0], device=vertices.device)
        body_mask = ~torch.isin(all_indices, excluded_vertices)
        body_vertices = vertices[body_mask]
        body_indices = torch.nonzero(body_mask).squeeze(1)
        
        return hand_vertices, body_vertices, body_indices
    
    def _handle_contact_found(self, hand_side, contact_info, body_pose):
        """Handle the case where contact is found for a hand."""
        self.contact_found[hand_side] = True
        params_to_store = self._get_param_indices(hand_side)

        # Ensure we have valid parameters to store
        if body_pose is not None and params_to_store:
            self.contact_info[hand_side] = {
                'hand_vertex': contact_info['hand_vertex'],
                'body_vertex': contact_info['body_vertex'],
                'distance': contact_info['distance'],
                'x_distance': contact_info.get('x_distance', 0),
                'y_distance': contact_info.get('y_distance', 0),
                'z_distance': contact_info.get('z_distance', 0),
                'mode': contact_info.get('mode', 'unknown'),
                'optimized_params': body_pose[:, params_to_store].detach().clone(),
                'iterations': self.iteration_counter[hand_side]
            }
    
    

    def reset_state(self):
        """
        Reset all state variables to prevent memory accumulation between optimizations.
        Call this method at the end of each frame's optimization.
        """
        # Reset iteration tracking
        self.iteration_counter = {}
        
        # Reset contact pairs and tracking
        self.initial_pairs = {'right': [], 'left': []}
        self.first_pass = {'right': True, 'left': True}
        self.contact_found = {'right': False, 'left': False}
        
        # Clear stored vertices and metrics
        self.best_vertices = {'right': None, 'left': None}
        self.distance_metric = {'right': None, 'left': None}
        
        # Don't clear contact_info as it contains the final parameters
        # But make sure tensors are detached
        for hand_side in ['right', 'left']:
            if self.contact_info.get(hand_side) and 'optimized_params' in self.contact_info[hand_side]:
                params = self.contact_info[hand_side]['optimized_params']
                # Ensure params are detached
                if isinstance(params, torch.Tensor) and params.requires_grad:
                    self.contact_info[hand_side]['optimized_params'] = params.detach().clone()
        
        # Reset active hand selection
        self.active_hand = None

        self.distance_history = {'right': [], 'left': []}
        self.iteration_count = {'right': 0, 'left': 0}
        
        # Force garbage collection
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()