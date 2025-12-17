#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simplified BioTUCH fitting script that handles missing keypoint files.
This script takes an initialization body and applies mean betas and
translation/global orientation to create output files without requiring keypoints.
It will always render and save images with the mesh overlaid on original input images.
"""

import os
import argparse
import pickle
import numpy as np
import torch
import smplx
from pathlib import Path
from tqdm import tqdm
import yaml
import cv2
import PIL.Image as pil_img
import types
import re
import warnings

# Suppress smplx tensor construction warning
warnings.filterwarnings('ignore', 
                       message='To copy construct from a tensor',
                       category=UserWarning,
                       module='smplx')


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
    
    if len(hand_pc_values.shape) == 2 and hand_pc_values.shape[0] == 1:
        hand_pc_values = hand_pc_values.flatten() 
    elif len(hand_pc_values.shape) != 1:
        raise ValueError(f"Unexpected hand_pc_values shape: {hand_pc_values.shape}. Expected (12,)")
    
    full_hand_pose = np.dot(hand_pca, hand_pc_values)
    
    # Ensure final shape is (1, 45)
    full_hand_pose = full_hand_pose.reshape(1, 45)
    
    return full_hand_pose


def create_output_mesh(model_output, body_model, mesh_fn, out_img_fn, camera, img=None, focal_length=5000., device='cuda'):
    """Create and save mesh from model output, renders and saves visualization exactly as in fit_single_frame.py"""
    import trimesh
    
    # Get vertices from model output
    vertices = model_output.vertices.detach().cpu().numpy().squeeze()
    
    # Create mesh with pink vertex colors (exactly as in fit_single_frame.py)
    vertex_colors = np.ones((len(vertices), 4)) * [0.9, 0.5, 0.9, 1.0]
    
    # Create mesh with vertex colors
    out_mesh = trimesh.Trimesh(
        vertices, 
        body_model.faces, 
        vertex_colors=vertex_colors,
        process=False)
    
    # Apply rotation to match the standard orientation
    rot = trimesh.transformations.rotation_matrix(
        np.radians(180), [1, 0, 0])
    out_mesh.apply_transform(rot)
    
    # Save the mesh
    out_mesh.export(mesh_fn)
    
    # If no image is provided, create a default image
    if img is None:
        # Default image size (adjust as needed)
        img = np.ones((1080, 1920, 3), dtype=np.uint8) * 240  # Light gray background
    
    # Setup pyrender - MUST match fit_single_frame.py exactly
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    if 'GPU_DEVICE_ORDINAL' in os.environ:
        os.environ['EGL_DEVICE_ID'] = os.environ['GPU_DEVICE_ORDINAL'].split(',')[0]
    import pyrender
    
    # Create a scene with transparent background
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0])
    
    # Create the material - EXACTLY match fit_single_frame.py
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',
        baseColorFactor=(1.0, 1.0, 1.0, 0.7))  # Semi-transparent white
    
    # Create mesh from trimesh object
    mesh = pyrender.Mesh.from_trimesh(
        out_mesh,
        material=material,
        smooth=False)  # Important: smooth=False to preserve vertex colors
    
    # Add mesh to scene
    scene.add(mesh, 'mesh')
    
    # Set up camera - EXACTLY as in fit_single_frame.py
    height, width = img.shape[:2]
    camera_center = camera.center.detach().cpu().numpy().squeeze()
    camera_transl = camera.translation.detach().cpu().numpy().squeeze()
    camera_transl[0] *= -1.0  # This flip is important!
    
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = camera_transl
    
    # Create camera
    camera_node = pyrender.camera.IntrinsicsCamera(
        fx=focal_length, fy=focal_length,
        cx=camera_center[0], cy=camera_center[1])
    scene.add(camera_node, pose=camera_pose)
    
    # Add directional light
    light_node = pyrender.DirectionalLight(color=np.ones(3), intensity=2.5)
    scene.add(light_node, pose=camera_pose)
    
    # Render the scene
    r = pyrender.OffscreenRenderer(viewport_width=width,
                                  viewport_height=height,
                                  point_size=1.0)
    color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    
    # Convert image to RGBA
    img_uint8 = (img * 255).astype(np.uint8) if img.dtype == np.float32 else img
    img_rgba = pil_img.fromarray(img_uint8).convert('RGBA')
    
    # Convert rendered mesh to Image
    mesh_rgba = pil_img.fromarray(color.astype(np.uint8))
    
    # Overlay mesh on original image
    output_img = pil_img.alpha_composite(img_rgba, mesh_rgba)
    
    # Resize if necessary - as in fit_single_frame.py
    if height > 1080:
        from PIL import Image
        output_img = output_img.resize(
            (int(width/2), int(height/2)), 
            Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.ANTIALIAS
        )
    
    # Save the final image
    output_img.save(out_img_fn)
    
    r.delete()  # Clean up renderer
    
    return


def process_frame(init_pose_path, transl_globalorient_path, beta_path, result_fn, mesh_fn, out_img_fn, 
                 body_model, config, img_path=None, input_img_folder=None, frame_num=None, focal_length=5000.):
    """Process a single frame using available initialization files and always render an image"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # Load initialization parameters
    with open(init_pose_path, 'rb') as f:
        init_pose_result = pickle.load(f)
    
    # Load translation and global orientation
    with open(transl_globalorient_path, 'rb') as f:
        transl_orient_data = pickle.load(f)
    
    # Load mean betas if available
    if beta_path and os.path.exists(beta_path):
        with open(beta_path, 'rb') as f:
            mean_betas = pickle.load(f)
    else:
        mean_betas = init_pose_result.get('betas')
    
    # Fix betas to ensure they only have 10 components
    if mean_betas is not None and hasattr(mean_betas, 'shape'):
        if len(mean_betas.shape) == 1:
            # If it's a 1D array, take first 10 and reshape
            mean_betas = mean_betas[:10].reshape(1, 10)
        elif mean_betas.shape[-1] > 10:
            # If last dimension > 10, take only first 10
            mean_betas = mean_betas[..., :10]
    
    # Handle expression params - limit to 10 components
    if 'expression' in init_pose_result and hasattr(init_pose_result['expression'], 'shape'):
        expr = init_pose_result['expression']
        if expr.shape[-1] > 10:
            init_pose_result['expression'] = expr[..., :10]
    
    # Use the same hand pose expansion as in fit_single_frame.py
    # Modify hand poses if they have 12 components (PCA)
    for k in ['left_hand_pose', 'right_hand_pose']:
        if k in init_pose_result:
            hand_pose = init_pose_result[k]
            if hasattr(hand_pose, 'shape') and len(hand_pose.shape) > 0:
                if hand_pose.shape[-1] == 12:
                    try:
                        init_pose_result[k] = expand_hand_pose(hand_pose, k)
                    except Exception as e:
                        print(f"Error expanding {k}: {e}")
                        init_pose_result[k] = np.zeros((1, 45))
                elif hand_pose.shape[-1] != 45:
                    print(f"Warning: {k} has shape {hand_pose.shape}, using zeros")
                    init_pose_result[k] = np.zeros((1, 45))

    # Prepare parameters with proper shapes
    params = {
        'betas': torch.tensor(mean_betas, dtype=torch.float32, device=device),
        'body_pose': torch.tensor(init_pose_result.get('body_pose', np.zeros((1, 63))), dtype=torch.float32, device=device),
        'global_orient': torch.tensor(transl_orient_data.get('global_orient'), dtype=torch.float32, device=device),
        'transl': torch.tensor(transl_orient_data.get('transl'), dtype=torch.float32, device=device),
        'left_hand_pose': torch.tensor(init_pose_result.get('left_hand_pose', np.zeros((1, 45))), dtype=torch.float32, device=device),
        'right_hand_pose': torch.tensor(init_pose_result.get('right_hand_pose', np.zeros((1, 45))), dtype=torch.float32, device=device),
        'jaw_pose': torch.tensor(init_pose_result.get('jaw_pose', np.zeros((1, 3))), dtype=torch.float32, device=device),
        'expression': torch.tensor(init_pose_result.get('expression', np.zeros((1, 10))), dtype=torch.float32, device=device)
    }
    
    # Reset model parameters
    body_model.reset_params(**params)
    
    # Forward pass to get vertices
    with torch.no_grad():
        model_output = body_model(return_verts=True)
    
    # Create result dictionary exactly as in fit_single_frame.py
    result = {
        'betas': model_output.betas.detach().cpu().numpy(),
        'body_pose': model_output.body_pose.detach().cpu().numpy(),
        'left_hand_pose': model_output.left_hand_pose.detach().cpu().numpy(),
        'right_hand_pose': model_output.right_hand_pose.detach().cpu().numpy(),
        'jaw_pose': model_output.jaw_pose.detach().cpu().numpy(),
        'expression': model_output.expression.detach().cpu().numpy(),
        'transl': body_model.transl.detach().cpu().numpy(),
        'global_orient': model_output.global_orient.detach().cpu().numpy(),
        'vertices': model_output.vertices.detach().cpu().numpy()
    }
    
    # Save result using protocol 2 as in fit_single_frame.py
    # results_dir already exists from contact frame processing
    with open(result_fn, 'wb') as f:
        pickle.dump(result, f, protocol=2)
    
    # Load image using biotuch.py's approach (input_img_folder always provided)
    # Glob all images and filter by frame number
    input_img_folder = Path(input_img_folder)
    
    # Same approach as biotuch.py: extract frame numbers and match
    def get_frame_number(filepath):
        """Extract frame number from filename (same as biotuch.py)"""
        match = re.search(r'\d+', Path(filepath).stem)
        return int(match.group()) if match else None
    
    # Find all image files
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
        image_files.extend(input_img_folder.glob(ext))
    
    # Find the file with matching frame number
    img = None
    for img_file in image_files:
        if get_frame_number(img_file) == frame_num:
            try:
                img = cv2.imread(str(img_file))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    break
            except Exception as e:
                print(f"Error loading image {img_file}: {e}")
    
    # Fallback to default if image not found
    if img is None:
        print(f"Could not find image for frame {frame_num} in {input_img_folder}")
        img = np.ones((1080, 1920, 3), dtype=np.uint8) * 240  # Light gray background
        print("Using default image")
    
    # Create camera object exactly as in fit_single_frame.py
    camera = types.SimpleNamespace()
    
    # Set camera basic properties
    camera.translation = torch.tensor(transl_orient_data.get('transl', np.zeros(3)), dtype=torch.float32, device=device)
    H, W = img.shape[:2]
    camera.center = torch.tensor([W/2, H/2], dtype=torch.float32, device=device)
    camera.rotation = torch.eye(3, dtype=torch.float32, device=device)
    
    # If camera_state is in transl_orient_data, use those values
    if 'camera_state' in transl_orient_data:
        camera_state = transl_orient_data['camera_state']
        if 'translation' in camera_state:
            camera.translation = torch.tensor(camera_state['translation'], dtype=torch.float32, device=device)
        if 'center' in camera_state:
            camera.center = torch.tensor(camera_state['center'], dtype=torch.float32, device=device)
        if 'rotation' in camera_state and camera_state['rotation'] is not None:
            camera.rotation = torch.tensor(camera_state['rotation'], dtype=torch.float32, device=device)
        if 'focal_length' in camera_state and camera_state['focal_length'] is not None:
            focal_length = camera_state['focal_length']
    
    # Create and save mesh with visualization - exactly as in fit_single_frame.py
    create_output_mesh(
        model_output, 
        body_model, 
        mesh_fn, 
        out_img_fn, 
        camera, 
        img, 
        focal_length, 
        device
    )
        
    return result

def build_frame_mapping(init_folder):
    """Build a mapping from frame numbers to file paths (same logic as biotuch.py)"""
    init_folder = Path(init_folder)
    frame_mapping = {}
    
    for file_path in init_folder.glob("*.pkl"):
        # Extract first number from filename (same as biotuch.py)
        match = re.search(r'(\d+)', file_path.name)
        if match:
            frame_num = int(match.group(1))
            frame_mapping[frame_num] = file_path
    
    return frame_mapping


def process_all_frames(frames, output_dir, init_body_folder, beta_path, transl_globalorient_path, config_path, model_path, img_folder=None):
    """Process all frames in the list"""    
    output_dir = Path(output_dir)
    results_dir = output_dir / "results"
    meshes_dir = output_dir / "meshes"
    images_dir = output_dir / "images"
    
    # Directories already created by contact frame processing in biotuch.py

    # Create body model once and reuse
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: Could not read config file: {e}. Using default config.")
        config = {
            'model_type': 'smplx',
            'gender': 'neutral',
            'use_pca': False,
            'flat_hand_mean': False,
            'num_pca_comps': 12
        }

    # Create shared body model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    body_model = smplx.create(
        model_path=model_path,
        model_type=config.get('model_type', 'smplx'),
        gender=config.get('gender', 'neutral'),
        use_face_contour=True,
        batch_size=1,
        use_pca=config.get('use_pca', False),
        flat_hand_mean=config.get('flat_hand_mean', False),
        num_pca_comps=config.get('num_pca_comps', 12)
    )
    body_model = body_model.to(device=device)
    
    # Build frame mapping once
    print("Building frame mapping...")
    frame_mapping = build_frame_mapping(init_body_folder)
    print(f"Found {len(frame_mapping)} initialization files")
    
    processed_frames = []
    failed_frames = []
    
    for frame in tqdm(frames, desc="Processing frames"):
        result_fn = results_dir / f"{frame:05d}.pkl"
        mesh_fn = meshes_dir / f"{frame:05d}.obj"
        out_img_fn = images_dir / f"{frame:05d}.png"  # Always define output image path
        
        # Skip if result already exists
        if result_fn.exists() and mesh_fn.exists() and out_img_fn.exists():
            print(f"Skipping frame {frame} (output files already exist)")
            processed_frames.append(frame)
            continue
        
        # Find initialization file using mapping
        init_file = frame_mapping.get(frame)
        
        if init_file is None:
            print(f"Warning: No initialization file found for frame {frame}")
            failed_frames.append(frame)
            continue
        
        try:
            process_frame(
                init_pose_path=init_file,
                transl_globalorient_path=transl_globalorient_path,
                beta_path=beta_path,
                result_fn=result_fn,
                mesh_fn=mesh_fn,
                out_img_fn=out_img_fn,
                body_model=body_model,        
                config=config,                
                input_img_folder=img_folder,
                frame_num=frame
            )
            processed_frames.append(frame)
        except Exception as e:
            print(f"Error processing frame {frame}: {str(e)}")
            import traceback
            traceback.print_exc()
            failed_frames.append(frame)
    
    print(f"\nProcessed {len(processed_frames)} frames successfully")
    if failed_frames:
        print(f"Failed to process {len(failed_frames)} frames: {failed_frames}")
    
    return processed_frames, failed_frames
def main():
    parser = argparse.ArgumentParser(description='Process frames without keypoints using initialization files')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--init_body_folder', required=True, help='Folder containing initialization bodies')
    parser.add_argument('--beta_path', required=True, help='Path to precomputed mean betas file')
    parser.add_argument('--transl_globalorient_path', required=True, help='Path to translation and global orientation file')
    parser.add_argument('--frames', required=True, help='Comma-separated list of frames to process (e.g., "150,151,152")')
    parser.add_argument('--model_path', required=True, help='Path to SMPLX model folder')
    parser.add_argument('--config_path', required=True, help='Path to config YAML file')
    parser.add_argument('--img_folder', required=True, help='Folder containing input images')
    parser.add_argument('--focal_length', type=float, default=5000.0, help='Camera focal length (default: 5000.0)')
    
    args = parser.parse_args()
    
    # Collect frames to process (always provided by biotuch.py)
    frames_to_process = [int(f) for f in args.frames.split(',')]
    frames_to_process = sorted(list(set(frames_to_process)))
    print(f"Processing {len(frames_to_process)} frames: {frames_to_process[:5]}...")
    
    # Model path always provided by biotuch.py
    model_path = args.model_path

    # Process all frames
    processed_frames, failed_frames = process_all_frames(
        frames=frames_to_process,
        output_dir=args.output_dir,
        init_body_folder=args.init_body_folder,
        beta_path=args.beta_path,
        transl_globalorient_path=args.transl_globalorient_path,
        config_path=args.config_path,
        model_path=model_path,  
        img_folder=args.img_folder
    )
    
    # Write report
    with open(os.path.join(args.output_dir, "processing_report.txt"), "w") as f:
        f.write(f"Total frames requested: {len(frames_to_process)}\n")
        f.write(f"Successfully processed: {len(processed_frames)}\n")
        f.write(f"Failed to process: {len(failed_frames)}\n")
        if failed_frames:
            f.write("\nFailed frames:\n")
            f.write(", ".join(map(str, failed_frames)))

if __name__ == "__main__":
    main()