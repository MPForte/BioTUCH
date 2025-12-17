import os
from subprocess import run
from pathlib import Path
import numpy as np
from tqdm import tqdm
import re
from .dwpose_infer import process_images
import pickle


def get_all_available_frames(image_dir_path):
    """Get all available frame numbers from the image directory"""
    image_dir_path = Path(image_dir_path)
    all_frames = []
    
    # Get all image files
    image_files = list(image_dir_path.glob('*.[jJ][pP][gG]')) + \
                  list(image_dir_path.glob('*.[pP][nN][gG]')) + \
                  list(image_dir_path.glob('*.[jJ][pP][eE][gG]'))
    
    for image_file in image_files:
        # Extract the first number found in filename
        match = re.search(r'\d+', image_file.stem)
        if match:
            all_frames.append(int(match.group()))
    
    if not all_frames:
        raise ValueError(f"No numbered frames found in {image_dir_path}")
    
    print(f"Found {len(all_frames)} frames (range: {min(all_frames)}-{max(all_frames)})")
    
    return sorted(set(all_frames))


def compute_betas(*, init_folder, beta_path):
    """
    Compute mean betas from all initialization files in the folder.
    Only uses the first 10 beta values.
    
    Args:
        init_folder: Path to folder containing initialization files
        beta_path: Path where the computed mean betas will be saved
    """
    print(f"Computing mean betas from {init_folder}")
    init_folder = Path(init_folder)
    
    # Find all pkl files
    betas = []
    pkl_files = list(init_folder.glob("*.pkl"))
    
    if not pkl_files:
        raise ValueError(f"No pkl files found in {init_folder}")
    
    # Load betas from each file
    for pkl_file in tqdm(pkl_files, desc="Loading betas"):
        try:
            with pkl_file.open("rb") as file:
                data = pickle.load(file)
                
            # Check if betas exists in the data
            if 'betas' in data:
                # Extract the first dimension if it exists
                if isinstance(data['betas'], np.ndarray) and data['betas'].ndim > 1:
                    beta_value = data['betas'][0]
                else:
                    beta_value = data['betas']
                
                # Take only the first 10 values
                betas.append(beta_value[:10])
            else:
                print(f"Warning: No 'betas' field found in {pkl_file}")
        except Exception as e:
            print(f"Error loading {pkl_file}: {str(e)}")
    
    if not betas:
        raise ValueError("No valid betas found in any initialization files")
    
    # Compute mean betas
    mean_betas = np.mean(betas, axis=0)
    print(f"Computed mean betas with shape {mean_betas.shape} from {len(betas)} files")
    
    # Reshape to match expected format (batch dimension)
    mean_betas_reshaped = np.reshape(mean_betas, (1, -1))
    print(f"Reshaped to {mean_betas_reshaped.shape} to match expected format")
    
    # Save mean betas
    with open(beta_path, 'wb') as file:
        pickle.dump(mean_betas_reshaped, file, pickle.HIGHEST_PROTOCOL)
    
    print(f"Saved mean betas to {beta_path}")
    return mean_betas_reshaped


def copy_frames(*, image_dir_path, output_folder, contact_frames):
    """
    Copy or symlink image files from the input directory to the output folder,
    filtering to include only the frames in contact_frames.
    
    Args:
        image_dir_path: Path to directory containing source images
        output_folder: Path to output directory for symlinks
        contact_frames: List of frame numbers to include
    """
    def get_frame_number(filepath):
        """Extract frame number from filename"""
        match = re.search(r'\d+', Path(filepath).stem)
        return int(match.group()) if match else 0
    
    # Gather all image files
    image_dir = Path(image_dir_path)
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
        image_files.extend(image_dir.glob(ext))
    
    if not image_files:
        raise ValueError(f"No image files found in {image_dir}")
    
    # Filter and sort by frame number
    contact_files = [f for f in image_files if get_frame_number(f) in contact_frames]
    contact_files.sort(key=get_frame_number)
    
    print(f"Copying {len(contact_files)} frames from {len(image_files)} total frames")
    
    # Copy/symlink files
    for src_file in tqdm(contact_files, desc="Copying frames"):
        frame_num = get_frame_number(src_file)
        dst_file = output_folder / f"{frame_num:05d}{src_file.suffix}"
        
        if dst_file.exists():
            dst_file.unlink()
        
        try:
            os.symlink(src_file.resolve(), dst_file)
        except Exception as e:
            print(f"Error creating symlink for frame {frame_num}: {e}")


def run_dwpose(*, output_folder, **kwargs):
    """
    Wrapper function for DWPose to match the interface of other keypoint detection utilities
    
    Args:
        output_folder (str or Path): Base output folder
        **kwargs: Additional arguments to pass to process_images
    """
    images_folder = Path(output_folder) / "images"
    keypoints_folder = Path(output_folder) / "keypoints"
    
    # Ensure keypoints folder exists
    keypoints_folder.mkdir(parents=True, exist_ok=True)
    
    # Call process_images with images and keypoints folders
    process_images(
        input_folder=str(images_folder), 
        output_folder=str(keypoints_folder)
    )


def process_frames_without_contact(output_dir, init_folder, frames, beta_path, transl_globalorient_path, config_path, img_folder):
    """
    Process frames without contact using subprocess to call the script.
    This creates the result pkl files and meshes for frames without keypoints.
    
    Args:
        output_dir: Output directory for results
        init_folder: Folder containing initialization files
        frames: List of frame numbers to process
        beta_path: Path to mean betas file
        transl_globalorient_path: Path to translation/orientation file
        config_path: Path to configuration file
        img_folder: Folder containing input images
        
    Returns:
        List of processed frames
    """
    from pathlib import Path
    
    frames_str = ",".join(map(str, frames))
    
    # Get model path (BioTUCH/data/models)
    base_dir = Path(__file__).parent.parent  # Go up from utils/ to BioTUCH/
    model_path = base_dir / "data" / "models"
    
    # Build the command
    cmd = [
        "python", "smplifyx/process_frames_without_contact.py",
        "--output_dir", str(output_dir),
        "--init_body_folder", str(init_folder),
        "--beta_path", str(beta_path),
        "--transl_globalorient_path", str(transl_globalorient_path),
        "--frames", frames_str,
        "--config_path", config_path,
        "--img_folder", str(img_folder),
        "--model_path", str(model_path)  # Add model path
    ]

    # Run the command
    run(cmd, check=True)
    
    return frames  # Assume all frames were processed successfully


def create_video(input_folder, output_folder, fps=30):
    """
    Create output video.
    
    Args:
        input_folder (str or Path): Path to input folder containing frames
        output_folder (str or Path): Directory to save the output video
        fps (int, optional): Frames per second for the output video. Default is 30.
    
    Returns:
        Result of the ffmpeg command
    """
    print("\nCreating input frames video...")
    
    frames_dir = Path(input_folder)
    video_path = Path(output_folder) / "output.mp4"
    
    if not frames_dir.exists():
        raise ValueError(f"Frames directory not found: {frames_dir}")
    
    # Find image extension
    for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
        if list(frames_dir.glob(f"*{ext}")):
            input_pattern = str(frames_dir / f"*{ext}")
            break
    else:
        raise ValueError(f"No image frames found in {frames_dir}")
    
    print(f"Creating video: {video_path}")
    
    return run(
        ["ffmpeg", "-r", str(fps), "-pattern_type", "glob", 
         "-i", input_pattern, "-y", str(video_path), "-nostdin"], 
        check=True
    )