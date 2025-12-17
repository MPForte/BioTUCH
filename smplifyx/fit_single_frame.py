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


import time
try:
    import cPickle as pickle
except ImportError:
    import pickle
import os
import os.path as osp
import numpy as np
import torch
from tqdm import tqdm
import copy
import cv2
import PIL.Image as pil_img
from optimizers import optim_factory
import fitting
try:
    from human_body_prior.tools.model_loader import load_vposer
except ImportError:
    load_vposer = None
    use_vposer = False
import contextlib
import warnings

# Suppress smplx tensor construction warning
warnings.filterwarnings('ignore', 
                       message='To copy construct from a tensor',
                       category=UserWarning,
                       module='smplx')

def clear_cuda_cache():
    """Safely clear CUDA cache without affecting computation results."""
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def fit_single_frame(img,
                     keypoints,
                     body_model,
                     camera,
                     joint_weights,
                     body_pose_prior,
                     jaw_prior,
                     left_hand_prior,
                     right_hand_prior,
                     shape_prior,
                     expr_prior,
                     angle_prior,
                     bio_contact,
                     result_fn='out.pkl',
                     mesh_fn='out.obj',
                     out_img_fn='overlay.png',
                     use_cuda=True,
                     init_joints_idxs=(9, 12, 2, 5),
                     use_face=True,
                     use_hands=True,
                     data_weights=None,
                     depth_loss_weight=1e2,
                     interpenetration=True,
                     focal_length=5000.,
                     side_view_thsh=25.,
                     rho=100,
                     vposer_ckpt='',
                     use_joints_conf=False,
                     interactive=True,
                     visualize=False,
                     save_meshes=True,
                     batch_size=1,
                     dtype=torch.float32,
                     left_shoulder_idx=2,
                     right_shoulder_idx=5,
                     start_opt_stage=0,
                     contact_weight=1.0,         
                     consistency_weight=0.25,     
                     penetration_weight=1.0,     
                     frame_loss_weight=1.0,
                     **kwargs):

    device = torch.device('cuda') if use_cuda and torch.cuda.is_available() else torch.device('cpu')

    if data_weights is None:
        data_weights = [1, ] * 5

    init_pose_path = kwargs.get('init_pose_path')
    with open(init_pose_path, 'rb') as pkl_f:
        init_pose_result = pickle.load(pkl_f)


    # Load the mean betas if provided, otherwise use init_pose_result betas
    beta_path = kwargs.get('beta_path')
    if beta_path and os.path.exists(beta_path):
        with open(beta_path, 'rb') as f:
            mean_betas = pickle.load(f)

        # Replace the betas in init_pose_result with the mean betas
        init_pose_result['betas'] = mean_betas

    allowed_keys = {'betas', 'expression', 'jaw_pose', 'left_hand_pose', 'right_hand_pose', 'body_pose'}

    # Add this right after loading the problematic init_pose_result
    if 'expression' in init_pose_result:
        expr = np.array(init_pose_result['expression'])

    if 'betas' in init_pose_result:
        betas = np.array(init_pose_result['betas'])

    def expand_hand_pose(hand_pc_values, hand_key):
        """Expand 12D PCA hand pose to 45D using saved PCA data."""

        # Load PCA data
        data = np.load("data/hand_pca_data.npz")

        if "left" in hand_key:
            hand_pca = data["left_hand_pca"]  # (12, 45)
        else:
            hand_pca = data["right_hand_pca"]  # (12, 45)

        hand_pca = hand_pca.T 

        # Ensure correct shape for input
        hand_pc_values = np.asarray(hand_pc_values)

        # More robust shape handling
        if len(hand_pc_values.shape) == 2:
            if hand_pc_values.shape[1] == 12:  # Shape is (N, 12)
                if hand_pc_values.shape[0] == 1:
                    hand_pc_values = hand_pc_values.flatten()
                else:
                    # Process only the first set if multiple are provided
                    print(f"Warning: Multiple hand poses provided for {hand_key}, using only the first one")
                    hand_pc_values = hand_pc_values[0]
            elif hand_pc_values.shape[0] == 12 and hand_pc_values.shape[1] == 1:
                # Shape is (12, 1)
                hand_pc_values = hand_pc_values.flatten()
        elif len(hand_pc_values.shape) != 1 or hand_pc_values.shape[0] != 12:
            raise ValueError(f"Unexpected hand_pc_values shape: {hand_pc_values.shape}. Expected (12,) or (1, 12)")

        full_hand_pose = np.dot(hand_pca, hand_pc_values)

        # Ensure final shape is (1, 45)
        full_hand_pose = full_hand_pose.reshape(1, 45)

        return full_hand_pose

    # Modify init_pose_result in-place
    for k in ['left_hand_pose', 'right_hand_pose']:
        if k in init_pose_result:
            hand_pose = np.array(init_pose_result[k])
            # Check array dimensions properly
            if len(hand_pose.shape) >= 2 and hand_pose.shape[1] == 12:
                init_pose_result[k] = expand_hand_pose(init_pose_result[k], k)
            elif len(hand_pose.shape) == 1 and hand_pose.shape[0] == 12:
                # Reshape 1D array to 2D before expansion
                init_pose_result[k] = expand_hand_pose(hand_pose.reshape(1, 12), k)

    # Modify expression if needed
    if 'expression' in init_pose_result:
        expr = np.array(init_pose_result['expression'])
        # Check array dimensions properly
        if len(expr.shape) >= 2 and expr.shape[1] == 50:
            # 2D array case (batch, 50)
            init_pose_result['expression'] = expr[:, :10]
        elif len(expr.shape) == 1 and expr.shape[0] == 50:
            # 1D array case with 50 elements
            init_pose_result['expression'] = expr[:10].reshape(1, 10)
        
        # CRITICAL FIX: Ensure expression always has batch dimension
        expr_final = np.array(init_pose_result['expression'])
        if len(expr_final.shape) == 1:
            init_pose_result['expression'] = expr_final.reshape(1, -1)

    # Create prev_params with consistent batch dimensions
    prev_params = {}
    for k, v in init_pose_result.items():
        if k in allowed_keys:
            tensor = torch.tensor(v, dtype=dtype, device=device)
            
            # Ensure all parameters have batch dimension of 1
            if len(tensor.shape) == 1:
                tensor = tensor.unsqueeze(0)  # Add batch dimension
            elif tensor.shape[0] != 1:
                tensor = tensor[:1]  # Take only first batch
                
            prev_params[k] = tensor

    # If camera_translation_path exists, then add translation and global orientation
    transl_globalorient_path = kwargs.get('transl_globalorient_path', None)
    if transl_globalorient_path and os.path.exists(transl_globalorient_path):
        with open(transl_globalorient_path, 'rb') as pkl_f:
            data = pickle.load(pkl_f)
            
            # Load body model parameters with consistent batch dimensions
            for param_name in ['transl', 'global_orient']:
                if param_name in data:
                    param_tensor = torch.tensor(data[param_name], dtype=dtype, device=device)
                    # Ensure batch dimension
                    if len(param_tensor.shape) == 1:
                        param_tensor = param_tensor.unsqueeze(0)
                    elif param_tensor.shape[0] != 1:
                        param_tensor = param_tensor[:1]
                    prev_params[param_name] = param_tensor

            # Load camera parameters safely
            with torch.no_grad():
                if 'camera_state' in data:
                    camera_state = data['camera_state']
                    camera.translation[:] = torch.tensor(camera_state['translation'], dtype=dtype, device=device)
                    camera.center[:] = torch.tensor(camera_state['center'], dtype=dtype, device=device)

                    # Handle other parameters as needed
                    if hasattr(camera, 'rotation') and 'rotation' in camera_state and camera_state['rotation'] is not None:
                        camera.rotation[:] = torch.tensor(camera_state['rotation'], dtype=dtype, device=device)
                else:
                    # Fallback for older pickles
                    print(f"Available keys in data: {list(data.keys())}")
                    if 'camera_translation' in data:
                        camera.translation[:] = torch.tensor(data['camera_translation'], dtype=dtype, device=device)
                    if 'camera_center' in data:
                        camera.center[:] = torch.tensor(data['camera_center'], dtype=dtype, device=device)
            
            # # Re-enable gradients
            # camera.translation.requires_grad = True

    use_vposer = kwargs.get('use_vposer', True)
    vposer, pose_embedding = [None, ] * 2
    if use_vposer:
        pose_embedding = torch.zeros([batch_size, 32],
                                     dtype=dtype, device=device,
                                     requires_grad=True)

        vposer_ckpt = osp.expandvars(vposer_ckpt)
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            vposer, _ = load_vposer(vposer_ckpt, vp_model='snapshot')
        vposer = vposer.to(device=device)
        vposer.eval()

    keypoint_data = torch.tensor(keypoints, dtype=dtype, device=device)
 
    if use_joints_conf:
        joints_conf = keypoint_data[:, :, 2].reshape(len(keypoints), -1)
    gt_joints = keypoint_data[:, :, :2]

    # Transfer the data to the correct device
    gt_joints = gt_joints.to(device=device, dtype=dtype)
    if use_joints_conf:
        joints_conf = joints_conf.to(device=device, dtype=dtype)

    opt_weights_dict = {'data_weight': data_weights}
    
    keys = opt_weights_dict.keys()
    opt_weights = [dict(zip(keys, vals)) for vals in
                   zip(*(opt_weights_dict[k] for k in keys
                         if opt_weights_dict[k] is not None))]
    for weight_list in opt_weights:
        for key in weight_list:
            weight_list[key] = torch.tensor(weight_list[key],
                                            device=device,
                                            dtype=dtype)


    if init_pose_path and os.path.exists(init_pose_path) and transl_globalorient_path and os.path.exists(transl_globalorient_path):
        # Initialize results list with a properly structured entry
        body_model.reset_params(**prev_params)
        
        # Create model output for visualization
        with torch.no_grad():
            model_output = body_model(return_verts=True)
        
        # Create a result dictionary
        result = {}
        for key, val in body_model.named_parameters():
            result[key] = val.detach().cpu().numpy()
        
        # Add entry to results list
        results = [{'loss': 0.0, 'result': result}]
    else:
        # Check if all joints are visible, if not remove not visible joints
        init_joints_idxs_tmp = []
        for joint in init_joints_idxs:
            if torch.all(keypoint_data[:, joint, :].cpu() > 0):
                init_joints_idxs_tmp.append(joint)
        
        # The indices of the joints used for the initialization of the camera
        init_joints_idxs = torch.tensor(init_joints_idxs_tmp, device=device)

        edge_indices = kwargs.get('body_tri_idxs')
        init_t = fitting.guess_init(body_model, gt_joints, edge_indices,
                                    use_vposer=use_vposer, vposer=vposer,
                                    pose_embedding=pose_embedding,
                                    model_type=kwargs.get('model_type', 'smpl'),
                                    focal_length=focal_length, dtype=dtype)

        camera_loss = fitting.create_loss('camera_init',
                                        trans_estimation=init_t,
                                        init_joints_idxs=init_joints_idxs,
                                        depth_loss_weight=depth_loss_weight,
                                        dtype=dtype).to(device=device)
        camera_loss.trans_estimation[:] = init_t

        initial_loss = fitting.create_loss('smplify',  
                                joint_weights=joint_weights,
                                rho=rho,
                                use_joints_conf=use_joints_conf,
                                use_face=use_face, use_hands=use_hands,
                                vposer=vposer,
                                pose_embedding=pose_embedding,
                                body_pose_prior=body_pose_prior,
                                shape_prior=shape_prior,
                                angle_prior=angle_prior,
                                expr_prior=expr_prior,
                                left_hand_prior=left_hand_prior,
                                right_hand_prior=right_hand_prior,
                                jaw_prior=jaw_prior,
                                interpenetration=interpenetration,
                                bio_contact=bio_contact,
                                contact_weight=contact_weight,           
                                consistency_weight=consistency_weight,   
                                penetration_weight=penetration_weight,   
                                frame_loss_weight=frame_loss_weight,     
                                dtype=dtype)
        initial_loss = initial_loss.to(device=device)

        loss = fitting.create_loss(joint_weights=joint_weights,
                                rho=rho,
                                use_joints_conf=use_joints_conf,
                                use_face=use_face, use_hands=use_hands,
                                vposer=vposer,
                                pose_embedding=pose_embedding,
                                body_pose_prior=body_pose_prior,
                                shape_prior=shape_prior,
                                angle_prior=angle_prior,
                                expr_prior=expr_prior,
                                left_hand_prior=left_hand_prior,
                                right_hand_prior=right_hand_prior,
                                jaw_prior=jaw_prior,
                                interpenetration=interpenetration,
                                bio_contact=bio_contact,
                                dtype=dtype,
                                contact_weight=contact_weight,           
                                consistency_weight=consistency_weight,   
                                penetration_weight=penetration_weight,   
                                frame_loss_weight=frame_loss_weight,  
                                **kwargs)
        loss = loss.to(device=device)

        with fitting.FittingMonitor(
                batch_size=batch_size, visualize=visualize, interactive=interactive, **kwargs) as monitor:

            H, W, _ = torch.tensor(img, dtype=dtype).shape

            data_weight = 1000 / H

            # Step 1: Initialization
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=UserWarning)
                body_model.reset_params(**prev_params)

            # The closure passed to the optimizer
            # camera_loss.reset_loss_weights({'data_weight': data_weight})
            if use_vposer:
                with torch.no_grad():
                    pose_embedding.fill_(0)

            # If the distance between the 2D shoulders is smaller than a
            # predefined threshold then try 2 fits, the initial one and a 180
            # degree rotation
            shoulder_dist = torch.dist(gt_joints[:, left_shoulder_idx],
                                        gt_joints[:, right_shoulder_idx])
            try_both_orient = shoulder_dist.item() < side_view_thsh

            # Update the value of the translation of the camera as well as
            # the image center.
            with torch.no_grad():
                camera.translation[:] = init_t.view_as(camera.translation)
                camera.center[:] = torch.tensor([W, H], dtype=dtype) * 0.5

            # Re-enable gradient calculation for the camera translation
            camera.translation.requires_grad = True

            camera_opt_params = [camera.translation, body_model.global_orient]

            camera_optimizer, camera_create_graph = optim_factory.create_optimizer(
                camera_opt_params,
                **kwargs)

            # The closure passed to the optimizer
            fit_camera = monitor.create_fitting_closure(
                camera_optimizer, body_model, camera, gt_joints,
                camera_loss, create_graph=camera_create_graph,
                use_vposer=use_vposer, vposer=vposer,
                pose_embedding=pose_embedding,
                return_full_pose=False, return_verts=False)

            # Step 1: Optimize over the torso joints the camera translation
            # Initialize the computational graph by feeding the initial translation
            # of the camera and the initial pose of the body model.
            camera_init_start = time.time()
            cam_init_loss_val = monitor.run_fitting(camera_optimizer,
                                            fit_camera,
                                            camera_opt_params, body_model,
                                            use_vposer=use_vposer,
                                            pose_embedding=pose_embedding,
                                            vposer=vposer)

            # If the 2D detections/positions of the shoulder joints are too
            # close the rotate the body by 180 degrees and also fit to that
            # orientation
            if try_both_orient:
                body_orient = body_model.global_orient.detach().cpu().numpy()
                flipped_orient = cv2.Rodrigues(body_orient)[0].dot(
                    cv2.Rodrigues(np.array([0., np.pi, 0]))[0])
                flipped_orient = cv2.Rodrigues(flipped_orient)[0].ravel()

                flipped_orient = torch.tensor(flipped_orient, dtype=dtype, device=device).unsqueeze(dim=0)
                orientations = [body_orient, flipped_orient]
            else:
                orientations = [body_model.global_orient.detach().cpu().numpy()]

            # if interactive:
                # if use_cuda and torch.cuda.is_available():
                #     torch.cuda.synchronize()
            tqdm.write('Camera initialization done after {:.4f}'.format(
                time.time() - camera_init_start))
            tqdm.write('Camera initialization final loss {:.4f}'.format(
                cam_init_loss_val))


            # store here the final error for both orientations,
            # and pick the orientation resulting in the lowest error
            results = []

            # Step 2: Optimize the full model
            final_loss_val = 0
            for or_idx, orient in enumerate(tqdm(orientations, desc='Orientation') if interactive else orientations):
                opt_start = time.time()

                new_params = dict(
                    global_orient=orient,
                    transl=body_model.transl,
                    body_pose=init_pose_result.get('body_pose'),
                    betas=init_pose_result.get('betas'),
                    expression=init_pose_result.get('expression'),
                    jaw_pose=init_pose_result.get('jaw_pose'),
                    right_hand_pose=init_pose_result.get('right_hand_pose'),
                    left_hand_pose=init_pose_result.get('left_hand_pose'),
                )

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=UserWarning)
                    body_model.reset_params(**new_params)

                    for name, param in body_model.named_parameters():
                        if name != 'transl':
                            param.requires_grad = False

                if use_vposer:
                    with torch.no_grad():
                        pose_embedding.fill_(0)

                if interactive:
                    opt_idx, curr_weights = 0, list(tqdm(opt_weights[:1], desc='Stage'))[0]
                else:
                    opt_idx, curr_weights = 0, opt_weights[0]

                # final_params = list(filter(lambda x: x.requires_grad, body_params))
                final_params = [body_model.transl]

                if use_vposer:
                    final_params.append(pose_embedding)

                body_optimizer, body_create_graph = optim_factory.create_optimizer(
                    final_params,
                    **kwargs)
                body_optimizer.zero_grad()

                curr_weights['data_weight'] = data_weight

                initial_loss.reset_loss_weights(curr_weights)
                loss.reset_loss_weights(curr_weights)

                closure = monitor.create_fitting_closure(
                    body_optimizer, body_model,
                    camera=camera, gt_joints=gt_joints,
                    joints_conf=joints_conf,
                    joint_weights=joint_weights,
                    loss=initial_loss, create_graph=body_create_graph,
                    use_vposer=use_vposer, vposer=vposer,
                    pose_embedding=pose_embedding,
                    opt_idx=opt_idx + start_opt_stage,
                    return_verts=True, return_full_pose=True)

                if interactive:
                    # if use_cuda and torch.cuda.is_available():
                        # torch.cuda.synchronize()
                    stage_start = time.time()

                final_loss_val = monitor.run_fitting(
                    body_optimizer,
                    closure, final_params,
                    body_model,
                    pose_embedding=pose_embedding, vposer=vposer,
                    use_vposer=use_vposer)

                if interactive:
                    # Stage timing
                    # if use_cuda and torch.cuda.is_available():
                    #     torch.cuda.synchronize()
                    elapsed_stage = time.time() - stage_start
                    tqdm.write('Stage {:03d} done after {:.4f} seconds'.format(opt_idx, elapsed_stage))
                    
                    # Orientation timing  
                    # if use_cuda and torch.cuda.is_available():
                    #     torch.cuda.synchronize()
                    elapsed_orientation = time.time() - opt_start
                    tqdm.write(
                        'Body fitting Orientation {} done after {:.4f} seconds'.format(
                            or_idx, elapsed_orientation))
                    tqdm.write('Body final loss val = {:.5f}'.format(
                        final_loss_val))

                # Get the result of the fitting process
                # Store in it the errors list in order to compare multiple
                # orientations, if they exist
                result = {'camera_' + str(key): val.detach().cpu().numpy()
                        for key, val in camera.named_parameters()}
                result.update({key: val.detach().cpu().numpy()
                            for key, val in body_model.named_parameters()})
                if use_vposer:
                    result['body_pose'] = pose_embedding.detach().cpu().numpy()

                results.append({'loss': final_loss_val,
                                'result': result})
                
            print(camera.translation.detach().cpu().numpy())
            # import ipdb; ipdb.set_trace()
            # After optimization, save both body model parameters and the entire camera
            transl_globalorient_result = {
                # Body model parameters
                'transl': body_model.transl.detach().cpu().numpy(),
                'global_orient': body_model.global_orient.detach().cpu().numpy(),
                
                # Save the entire camera state
                'camera_state': {
                    # Camera parameters
                    'translation': camera.translation.detach().cpu().numpy(),
                    'center': camera.center.detach().cpu().numpy(),
                    
                    # Other camera attributes that might exist
                    'rotation': camera.rotation.detach().cpu().numpy() if hasattr(camera, 'rotation') else None,
                    'focal_length': focal_length,  # Store the scalar focal length
                    
                    # Camera type/class information for reconstruction
                    'camera_type': type(camera).__name__,
                    
                    # Any other camera-specific parameters
                    'additional_params': {
                        name: param.detach().cpu().numpy() 
                        for name, param in camera.named_parameters() 
                        if name not in ['translation', 'center', 'rotation']
                    }
                }
            }
            with open(transl_globalorient_path, 'wb') as f:
                pickle.dump(transl_globalorient_result, f, protocol=2)

    # Additional optimization step for specific joints
    # Only perform for frames with detected contact
    model_output = body_model(return_verts=True)
    if bio_contact == 0:
        if interactive:
            print("No contact detected - skipping contact optimization")
    else:
        # Force CUDA operations to complete and check memory
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
            if current_memory > 0.7:  # Clear if memory > 70%
                clear_cuda_cache()
                print(f"Cleared cache before contact optimization (memory: {current_memory:.1%})")
        
        if transl_globalorient_path:
            with fitting.FittingMonitor(
                batch_size=batch_size, visualize=visualize, interactive=interactive, **kwargs) as monitor:
                loss = fitting.create_loss(joint_weights=joint_weights,
                                        rho=rho,
                                        use_joints_conf=use_joints_conf,
                                        use_face=use_face, use_hands=use_hands,
                                        vposer=vposer,
                                        pose_embedding=pose_embedding,
                                        body_pose_prior=body_pose_prior,
                                        shape_prior=shape_prior,
                                        angle_prior=angle_prior,
                                        expr_prior=expr_prior,
                                        left_hand_prior=left_hand_prior,
                                        right_hand_prior=right_hand_prior,
                                        jaw_prior=jaw_prior,
                                        interpenetration=interpenetration,
                                        bio_contact=bio_contact,
                                        interactive=interactive,
                                        dtype=dtype,
                                        contact_weight=contact_weight,           
                                        consistency_weight=consistency_weight,   
                                        penetration_weight=penetration_weight,   
                                        frame_loss_weight=frame_loss_weight,  
                                        **kwargs)
                loss = loss.to(device=device)

        # Determine which hands are involved in the contact
        if interactive:
            tqdm.write('Determining which hand to optimize...')

        # First determine which hands to optimize
        start_time = time.time()
        model_output = body_model(return_verts=True)
        vertices = model_output.vertices.squeeze(0)
        loss.determine_active_hands(vertices, camera=camera, global_orient=body_model.global_orient)

        # Initialize pose with proper gradient tracking
        if use_vposer:
            body_pose = vposer.decode(pose_embedding, output_type='aa').view(1, -1)
            pose = body_pose.clone().detach().requires_grad_(True)
        else:
            if 'body_pose' in prev_params:
                pose = prev_params['body_pose'].clone().detach().requires_grad_(True)
            else:
                # Fallback with batch dimension fix
                init_body_pose = torch.tensor(init_pose_result.get('body_pose'), dtype=dtype, device=device)
                if len(init_body_pose.shape) == 1:
                    init_body_pose = init_body_pose.unsqueeze(0)
                pose = init_body_pose.clone().detach().requires_grad_(True)

        assert pose.requires_grad and pose.is_leaf, "Pose must be a leaf tensor with requires_grad=True"

        last_valid_pose = pose.clone().detach()
        memory_threshold = 0.9

        MAX_ITERATIONS = 3
        iteration = 0
        joint_weights_tensor = torch.zeros(1, 118, device=device)
        while iteration < MAX_ITERATIONS:            
            if not pose.is_leaf:
                
                # Recreate pose as a leaf tensor
                pose_data = pose.detach().clone()
                pose = pose_data.requires_grad_(True)
                
                assert pose.is_leaf and pose.requires_grad, "Failed to recreate pose as leaf tensor"
            
            if not pose.requires_grad:
                pose.requires_grad_(True)
            
            # Check memory usage first
            if torch.cuda.is_available():
                current_memory = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
                if current_memory > memory_threshold:
                    print(f"\n*** HIGH MEMORY USAGE DETECTED ({current_memory:.1%}) ***")
                    print("Using last valid pose and stopping further iterations")
                    
                    if last_valid_pose is not None:
                        print("Applying last valid pose")
                        with torch.no_grad():
                            pose.copy_(last_valid_pose)
                        # Re-enable gradients after copying
                        pose.requires_grad_(True)
                        
                        # Create final model output with last valid pose
                        model_output = body_model(
                            return_verts=True,
                            body_pose=pose,
                            betas=body_model.betas,
                            global_orient=body_model.global_orient,
                            transl=body_model.transl,
                            expression=body_model.expression,
                            jaw_pose=body_model.jaw_pose,
                            left_hand_pose=body_model.left_hand_pose,
                            right_hand_pose=body_model.right_hand_pose
                        )
                    break
            clear_cuda_cache()
           
            # Get updated joint mask based on current contact state
            joint_mask = loss.create_active_joint_mask(device)
            if not torch.any(joint_mask):
                if interactive:
                    print("\n✓ All active hands have found contact!")
                break

            # Apply any stored contact parameters
            if hasattr(loss, 'contact_info'):
                for hand_side in ['right', 'left']:
                    if (loss.contact_info.get(hand_side) and 
                        loss.contact_info[hand_side].get('optimized_params') is not None):
                        
                        params = loss.contact_info[hand_side]['optimized_params']
                        indices = loss._get_param_indices(hand_side)
                        
                        # Ensure dimensions match
                        if len(params.shape) == 1:
                            params = params.unsqueeze(0)
                        if len(indices) == len(params.reshape(-1)):
                            print(f"Before applying params: pose.is_leaf={pose.is_leaf}, requires_grad={pose.requires_grad}")
                            
                            # CRITICAL: Use .data assignment to preserve leaf status
                            with torch.no_grad():
                                pose.data[:, indices] = params.data
                            
                            print(f"After applying params: pose.is_leaf={pose.is_leaf}, requires_grad={pose.requires_grad}")
                            print(f"- Applied {hand_side} arm parameters")
                        else:
                            print(f"Warning: Parameter shape mismatch for {hand_side} hand")

            # Setup weights for active joints only
            keypoint_indices = []
            for hand_side in ['right', 'left']:
                if not loss.contact_found.get(hand_side, True):
                    keypoint_indices.extend(loss._get_keypoint_indices(hand_side))

            joint_weights_tensor.zero_()
            if keypoint_indices:
                joint_weights_tensor[0, keypoint_indices] = 1.0

            # Create optimizer for remaining active joints
            joint_optimizer, joint_create_graph = optim_factory.create_optimizer(
                [pose],  # pose should be a leaf tensor here
                **kwargs)
            
            try:
                # Create closure for optimization
                joint_closure = monitor.create_fitting_closure(
                    joint_optimizer, body_model,
                    camera=camera, gt_joints=gt_joints,
                    joints_conf=joints_conf,
                    joint_weights=joint_weights,
                    loss=loss, create_graph=joint_create_graph,
                    pose=pose,
                    joint_mask=joint_mask,
                    use_vposer=use_vposer, vposer=vposer,
                    pose_embedding=pose_embedding,
                    return_verts=True, return_full_pose=True)

                # Run optimization step
                if interactive:
                    start_time = time.time()

                specific_joint_loss = monitor.run_fitting(
                    joint_optimizer,
                    joint_closure,
                    [pose],
                    body_model,
                    pose_embedding=pose_embedding,
                    vposer=vposer,
                    use_vposer=use_vposer)

                if interactive:
                    elapsed = time.time() - start_time
                    tqdm.write(f'Iteration {iteration}: Joint optimization step done after {elapsed:.4f} seconds')
                    if specific_joint_loss is not None:
                        tqdm.write(f'Joint loss val = {specific_joint_loss:.5f}')

                # Store this as last valid pose after successful optimization
                last_valid_pose = pose.detach().clone()
                
                # Create model output with updated pose
                model_output = body_model(
                    return_verts=True,
                    body_pose=pose,
                    betas=body_model.betas,
                    global_orient=body_model.global_orient,
                    transl=body_model.transl,
                    expression=body_model.expression,
                    jaw_pose=body_model.jaw_pose,
                    left_hand_pose=body_model.left_hand_pose,
                    right_hand_pose=body_model.right_hand_pose
                )
                
            except (fitting.StopOptimizationError, RuntimeError) as e:
                # Also catch RuntimeError for memory issues
                if isinstance(e, RuntimeError) and "CUDA out of memory" in str(e):
                    print(f"\n*** CUDA OUT OF MEMORY DURING ITERATION {iteration} ***")
                    print("Using last valid pose and stopping further iterations")
                    
                    if last_valid_pose is not None:
                        print("Applying last valid pose")
                        with torch.no_grad():
                            pose.copy_(last_valid_pose)
                        
                        # Create final model output with last valid pose
                        model_output = body_model(
                            return_verts=True,
                            body_pose=pose,
                            betas=body_model.betas,
                            global_orient=body_model.global_orient,
                            transl=body_model.transl,
                            expression=body_model.expression,
                            jaw_pose=body_model.jaw_pose,
                            left_hand_pose=body_model.left_hand_pose,
                            right_hand_pose=body_model.right_hand_pose
                        )
                    break
                elif isinstance(e, fitting.StopOptimizationError):
                    if interactive:
                        print("Contact found - updating parameters and continuing optimization")
                    continue  # Skip to next iteration without incrementing counter
                else:
                    # Other runtime errors - log and continue
                    print(f"Error during optimization: {str(e)}")
                    if last_valid_pose is not None:
                        with torch.no_grad():
                            pose.copy_(last_valid_pose)
                        model_output = body_model(
                            return_verts=True,
                            body_pose=pose,
                            betas=body_model.betas,
                            global_orient=body_model.global_orient,
                            transl=body_model.transl,
                            expression=body_model.expression,
                            jaw_pose=body_model.jaw_pose,
                            left_hand_pose=body_model.left_hand_pose,
                            right_hand_pose=body_model.right_hand_pose
                        )
                    break
                    
            # Always clear after each iteration
            del joint_optimizer, joint_closure #optimized_body_params, 
            # clear_cuda_cache()
            
            iteration += 1

        # NEW: Add fine-tuning for near-contact situations
        if iteration >= MAX_ITERATIONS:  # After max iterations reached
            if interactive:
                print("\n🔍 Checking for near-contact situations...")
            
            # Memory check before fine-tuning
            if torch.cuda.is_available():
                current_memory = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
                if current_memory > memory_threshold:
                    print(f"\n*** HIGH MEMORY USAGE DETECTED ({current_memory:.1%}) ***")
                    print("Skipping fine-tuning due to memory constraints")
                    
                    if last_valid_pose is not None:
                        print("Using last valid pose")
                        with torch.no_grad():
                            pose.copy_(last_valid_pose)
                        pose.requires_grad_(True)
                        
                        model_output = body_model(
                            return_verts=True,
                            body_pose=pose,
                            betas=body_model.betas,
                            global_orient=body_model.global_orient,
                            transl=body_model.transl,
                            expression=body_model.expression,
                            jaw_pose=body_model.jaw_pose,
                            left_hand_pose=body_model.left_hand_pose,
                            right_hand_pose=body_model.right_hand_pose
                        )
                else:
                    # Proceed with fine-tuning if memory is OK                    
                    # Only check hands that were originally selected as active
                    active_hands = []
                    if loss.active_hand == 'both':
                        active_hands = ['right', 'left']
                    elif loss.active_hand in ['right', 'left']:
                        active_hands = [loss.active_hand]

                    if interactive:
                        print(f"Checking contact status for originally active hands: {active_hands}")

                    # Check if any active hand still needs contact
                    hands_needing_contact = []
                    for hand_side in active_hands:
                        if not loss.contact_found.get(hand_side, False):  # Hand hasn't found contact yet
                            hands_needing_contact.append(hand_side)
                            if interactive:
                                print(f"{hand_side.capitalize()} hand: Contact not yet achieved")

                    # If any active hand hasn't found contact, use fine-tuning
                    if hands_needing_contact:
                        if interactive:
                            print(f"\n🔧 Fine-tuning needed for: {hands_needing_contact}")
                        # Use simple gradient descent for fine-tuning
                        simple_optimizer = torch.optim.Adam([pose], lr=0.01)  # Much smaller learning rate
                        
                        for fine_tune_iter in range(50):  # Limited iterations
                            simple_optimizer.zero_grad()
                            
                            # Forward pass
                            model_output_fine = body_model(return_verts=True, body_pose=pose)
                            
                            # Compute loss (handle contact found)
                            try:
                                total_loss_fine = loss(
                                    model_output_fine,
                                    camera=camera,
                                    gt_joints=gt_joints,
                                    body_model_faces=body_model.faces_tensor.view(-1),
                                    joints_conf=joints_conf,
                                    joint_weights=joint_weights,
                                    use_vposer=use_vposer, vposer=vposer,
                                    pose_embedding=pose_embedding,
                                    pose=pose
                                )
                                
                                # Backward pass
                                total_loss_fine.backward()
                                
                            except fitting.StopOptimizationError:
                                if interactive:
                                    print("✓ Fine-tuning achieved contact!")
                                # Update model output with successful pose
                                model_output = model_output_fine
                                break
                            
                            # Apply gradient mask
                            if pose.grad is not None:
                                joint_mask = loss.create_active_joint_mask(device)
                                with torch.no_grad():
                                    pose.grad[:, ~joint_mask] = 0
                            
                            # Gradient descent step
                            simple_optimizer.step()
                            
                            # Check for contact every 10 iterations  
                            if fine_tune_iter % 10 == 0 or fine_tune_iter == 49:
                                vertices_fine = model_output_fine.vertices.squeeze(0)
                                
                                for hand_side in active_hands:
                                    if not loss.contact_found.get(hand_side, True):
                                        hand_verts, body_verts, body_idxs = loss._get_contact_vertices(hand_side, vertices_fine)
                                        contact_found, contact_info = loss.check_contact_for_hand(hand_side, hand_verts, body_verts, body_idxs)
                                        
                                        if contact_found:
                                            if interactive:
                                                print(f"  ✓ {hand_side} hand contact found (iteration {fine_tune_iter})")
                                            loss.contact_found[hand_side] = True
                                            loss.contact_info[hand_side] = contact_info
                                
                                # Update joint mask to exclude hands that found contact (same as trust region)
                                joint_mask = loss.create_active_joint_mask(device)
                                if not torch.any(joint_mask):
                                    if interactive:
                                        print("  ✓ Fine-tuning complete - all contacts achieved!")
                                    break
                                elif interactive and fine_tune_iter % 10 == 0:
                                    active_joint_count = torch.sum(joint_mask).item()
                                    print(f"  Fine-tuning iteration {fine_tune_iter}, active joints: {active_joint_count}")

                            # Apply the updated joint mask to gradients
                            if pose.grad is not None:
                                joint_mask = loss.create_active_joint_mask(device)  # Get current mask
                                with torch.no_grad():
                                    pose.grad[:, ~joint_mask] = 0  # Freeze joints for hands that found contact
                        # Update model output with fine-tuned pose
                        model_output = model_output_fine
                        if interactive:
                            print(f"✓ Fine-tuning completed")
                    else:
                        if interactive:
                            print("No near-contact situations detected - keeping current result")
            else:
                print("CUDA not available - skipping fine-tuning")

        if hasattr(loss, 'reset_state'):
            loss.reset_state()

    with open(result_fn, 'wb') as result_file:
        if len(results) > 1:
            min_idx = (0 if results[0]['loss'] < results[1]['loss']
                        else 1)
        else:
            min_idx = 0

        if use_vposer:
            body_pose = vposer.decode(
                pose_embedding,
                output_type='aa').view(1, -1) if use_vposer else None

            model_type = kwargs.get('model_type', 'smpl')
            append_wrists = model_type == 'smpl' and use_vposer
            if append_wrists:
                wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                        dtype=body_pose.dtype,
                                        device=body_pose.device)
                body_pose = torch.cat([body_pose, wrist_pose], dim=1)

        # Store all parameters from model_output in the results dictionary
        results[min_idx]['result'].update({
            'betas': model_output.betas.detach().cpu().numpy(),
            'body_pose': model_output.body_pose.detach().cpu().numpy(),
            'left_hand_pose': model_output.left_hand_pose.detach().cpu().numpy(),
            'right_hand_pose': model_output.right_hand_pose.detach().cpu().numpy(),
            'jaw_pose': model_output.jaw_pose.detach().cpu().numpy(),
            'expression': model_output.expression.detach().cpu().numpy(),
            'transl': body_model.transl.detach().cpu().numpy(),  
            'global_orient': model_output.global_orient.detach().cpu().numpy(),
            'vertices': model_output.vertices.detach().cpu().numpy()
        })

        # Save the results
        pickle.dump(results[min_idx]['result'], result_file, protocol=2)

    if save_meshes or visualize:
        import trimesh
        vertices = model_output.vertices.detach().cpu().numpy().squeeze()
        # Create vertex colors array (default to light pink)
        vertex_colors = np.ones((len(vertices), 4)) * [0.9, 0.5, 0.9, 1.0]

        # Create mesh with vertex colors
        out_mesh = trimesh.Trimesh(
            vertices, 
            body_model.faces, 
            vertex_colors=vertex_colors,
            process=False)
        
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        out_mesh.apply_transform(rot)

        os.environ['PYOPENGL_PLATFORM'] = 'egl'
        if 'GPU_DEVICE_ORDINAL' in os.environ:
            os.environ['EGL_DEVICE_ID'] = os.environ['GPU_DEVICE_ORDINAL'].split(',')[0]
        import pyrender

        # Create the basic material (will be colored by vertex colors)
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(1.0, 1.0, 1.0, 0.7))  # White base color to let vertex colors show

        # Create a mesh that preserves vertex colors
        mesh = pyrender.Mesh.from_trimesh(
                out_mesh,
                material=material,
                smooth=False)  # Set smooth=False to preserve vertex colors

        out_mesh.export(mesh_fn)                
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0])
        scene.add(mesh, 'mesh')

        # Rest of the visualization code remains the same
        height, width = img.shape[:2]
        camera_center = camera.center.detach().cpu().numpy().squeeze()
        camera_transl = camera.translation.detach().cpu().numpy().squeeze()
        camera_transl[0] *= -1.0

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_transl

        camera = pyrender.camera.IntrinsicsCamera(
            fx=focal_length, fy=focal_length,
            cx=camera_center[0], cy=camera_center[1])
        scene.add(camera, pose=camera_pose)

        light_node = pyrender.DirectionalLight(color=np.ones(3), intensity=2.5)
        scene.add(light_node, pose=camera_pose)

        r = pyrender.OffscreenRenderer(viewport_width=width,
                                        viewport_height=height,
                                        point_size=1.0)
        color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
        img_uint8 = (img * 255).astype(np.uint8)
        img_rgba = pil_img.fromarray(img_uint8).convert('RGBA')

        # Convert rendered mesh to Image
        mesh_rgba = pil_img.fromarray(color.astype(np.uint8), 'RGBA')

        # Overlay mesh on original image
        output_img = pil_img.alpha_composite(img_rgba, mesh_rgba)

        if height > 1080:
            output_img = output_img.resize((int(width/2), int(height/2)), pil_img.ANTIALIAS)

        output_img.save(out_img_fn)

    return copy.deepcopy(out_mesh)