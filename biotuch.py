import argparse
import json
import shutil
from pathlib import Path
from subprocess import run
import tqdm.auto as tqdm
import re

from utils import (
    get_all_available_frames,
    compute_betas,
    copy_frames,
    run_dwpose,
    process_frames_without_contact,
    create_video
)


def load_contact_frames(contact_json_path):
    """
    Load contact frames from a JSON file.
    
    Args:
        contact_json_path: Path to contact.json file
        
    Returns:
        List of contact frame numbers
    """
    with open(contact_json_path, 'r') as f:
        contact_frames = json.load(f)
    
    if not isinstance(contact_frames, list):
        raise ValueError(f"Expected contact.json to contain a list, got {type(contact_frames)}")
    
    print(f"Loaded {len(contact_frames)} contact frames from {contact_json_path}")
    
    return contact_frames


def call_biotuch(
    *,
    output_folder,
    data_folder,
    config_file,
    beta_path,
    transl_globalorient_path,
    init_body_folder,
    bio_contact,
    interactive
):
    cmd = [
        "python",
        "smplifyx/main.py",
        "--config",
        config_file,
        "--output_folder",
        output_folder,
        "--data_folder",
        data_folder,
        "--beta_path",
        beta_path,
        "--transl_globalorient_path",
        transl_globalorient_path,
        "--init_body_folder",
        init_body_folder,
        "--bio_contact",
        str(bio_contact),
        "--interactive",
        "true" if interactive else "false"
    ]

    return run(cmd, check=True)
    
def run_videobio(
    tmp_folder,
    output_folder,
    init_folder,
    contact_frames_set,
    beta_path,
    transl_globalorient_path,
    interactive=False,
    ):
    # Get all image files with keypoints
    def get_frame_number(filepath):
        match = re.search(r'(\d+)', filepath.name)
        return int(match.group(1)) if match else None

    keypoints_folder = tmp_folder.joinpath("keypoints")
    images_folder = tmp_folder.joinpath("images")
    
    # Get list of frames that have both image and keypoints
    image_files = sorted(images_folder.glob("*"), key=get_frame_number)
    frames_with_keypoints = []
    
    for image_file in image_files:
        frame_num = get_frame_number(image_file)
        if frame_num is not None:
            keypoint_file = keypoints_folder / f"{image_file.stem}_keypoints.json"
            if keypoint_file.exists():
                frames_with_keypoints.append({
                    'frame_num': frame_num,
                    'img_path': image_file,
                    'keypoint_path': keypoint_file
                })
    
    print(f"Processing {len(frames_with_keypoints)} frames with keypoints...")
    
    # Process each frame independently
    for frame_data in tqdm.tqdm(frames_with_keypoints, desc="Optimizing frames"):
        frame_num = frame_data['frame_num']
        image_file = frame_data['img_path']
        keypoint_file = frame_data['keypoint_path']
        
        # Setup per-frame data folder (will be cleaned after processing)
        tmp_data_path = tmp_folder / "tmp" / "data"
        shutil.rmtree(tmp_data_path, ignore_errors=True)
        tmp_data_path.mkdir(parents=True)
        
        # Create images subfolder with symlink to single image
        images_path = tmp_data_path / "images"
        images_path.mkdir()
        images_path.joinpath(image_file.name).symlink_to(image_file.resolve())
        
        # Create keypoints subfolder with symlink to single keypoint
        keypoints_path = tmp_data_path / "keypoints"
        keypoints_path.mkdir()
        keypoints_path.joinpath(keypoint_file.name).symlink_to(keypoint_file.resolve())
        
        # Get bio_contact for this frame
        bio_contact = 1 if frame_num in contact_frames_set else 0
        
        # Call main.py for this single frame
        try:
            call_biotuch(
                output_folder=str(output_folder),
                data_folder=str(tmp_data_path),
                config_file=args.cfg_file,
                beta_path=str(beta_path),
                transl_globalorient_path=str(transl_globalorient_path),
                init_body_folder=str(init_folder),
                bio_contact=bio_contact,
                interactive=interactive
            )
        except Exception as e:
            print(f"\nError processing frame {frame_num}: {str(e)}")
            with open(output_folder / "FAILED.txt", "w") as f:
                f.write(f"Processing failed at frame {frame_num}\nError: {str(e)}")
            raise
        
        # Cleanup tmp/data after processing this frame
        shutil.rmtree(tmp_data_path, ignore_errors=True)

def main(args):
    # Setup paths based on new structure
    # BioTUCH/
    #   ├── biotuch.py
    #   ├── data/
    #   │   ├── annotator/  (DWpose)
    #   │   └── demo/
    #   │       ├── input/
    #   │       │   ├── frames/
    #   │       │   ├── contact.json
    #   │       │   └── initialization/
    #   │       └── output/
    
    base_dir = Path(__file__).parent
    demo_dir = base_dir / "data" / "demo"
    
    # Use provided input folder or default to data/demo/input
    if args.input_folder:
        input_dir = Path(args.input_folder)
    else:
        input_dir = demo_dir / "input"
    
    # Load paths from input directory
    init_dir = input_dir / "initialization"
    contact_json_path = input_dir / "contact.json"
    frames_dir = input_dir / "frames"
    
    # Use provided output folder or default to data/demo/output
    if args.output_folder:
        output_base = Path(args.output_folder)
    else:
        output_base = demo_dir / "output"
    
    output_base.mkdir(exist_ok=True, parents=True)
    
    # Print the folders being used
    print(f"Input folder: {input_dir}")
    print(f"Output folder: {output_base}")
    
    # Verify required files exist
    if not contact_json_path.exists():
        raise FileNotFoundError(f"contact.json not found at {contact_json_path}")
    
    if not frames_dir.exists():
        raise FileNotFoundError(f"frames directory not found at {frames_dir}")
    
    if not init_dir.exists():
        raise FileNotFoundError(f"initialization directory not found at {init_dir}")
    
    # Load contact frames from JSON
    contact_frames = load_contact_frames(contact_json_path)
    
    # Create specific output folder with timestamp or name
    output_folder = output_base
    output_folder.mkdir(exist_ok=True, parents=True)
    
    # Create temporary folder inside output folder
    tmp_folder = output_folder / ".tmp"
    tmp_folder.mkdir(exist_ok=True, parents=True)
    
    images_folder = tmp_folder.joinpath("images")
    images_folder.mkdir(exist_ok=True, parents=True)
    
    # Paths for beta and transformation data
    beta_path = tmp_folder / "mean_betas.npy"
    transl_globalorient_path = tmp_folder / "transl_globalorient.json"
    
    # Get all available frames from the frames directory
    all_available_frames = get_all_available_frames(frames_dir)
    all_available_frames.sort()
    
    print(f"Total frames in directory: {len(all_available_frames)}")
    print(f"Frame range: {min(all_available_frames)} to {max(all_available_frames)}")
    
    # Determine starting frame - simply use the first available frame
    if not all_available_frames:
        raise ValueError("No frames found in the frames directory!")
    
    # The first frame to analyze is simply the first available frame
    first_frame = min(all_available_frames)
    print(f"Starting from first available frame: {first_frame}")
    
    # Filter to include only contact frames that exist in the image directory
    available_contact_frames = [frame for frame in contact_frames if frame in all_available_frames]
    available_contact_frames.sort()
    
    print(f"All available frames: {len(all_available_frames)} (min: {min(all_available_frames)}, max: {max(all_available_frames)})")
    print(f"Contact frames available in image directory: {len(available_contact_frames)}")
    
    if not available_contact_frames:
        print("WARNING: No contact frames found in the image directory!")
        
    # Run preprocessing steps
    if not args.skip_preprocessing:
        # Compute mean betas from initialization files (initialization folder)
        compute_betas(init_folder=init_dir, beta_path=beta_path) 
        
        # First, clear the images folder to ensure only our specific frames will be there
        for file in images_folder.glob("*"):
            file.unlink()
        
        # Create list of frames for DWPose (first frame + all contact frames)
        dwpose_frames = [first_frame] + available_contact_frames
        dwpose_frames = list(set(dwpose_frames))  # Remove duplicates in case first frame is also a contact frame
        dwpose_frames.sort()

        print(f"Total frames for DWPose processing: {len(dwpose_frames)}")
        print(f"  - First frame: {first_frame}")
        print(f"  - Contact frames: {len(available_contact_frames)}")
                
        # Copy ONLY the frames we want to process with DWPose
        copy_frames(
            image_dir_path=frames_dir,
            output_folder=images_folder,
            contact_frames=dwpose_frames  # First frame + contact frames
        )
        
        # Run DWPose on the specific frames we copied
        print(f"Extracting 2D keypoints with DWPose for {len(dwpose_frames)} frames...")
        keypoints_folder = tmp_folder.joinpath("keypoints")
        keypoints_folder.mkdir(exist_ok=True, parents=True)
        run_dwpose(
            output_folder=tmp_folder,
            confidence=0.5,
            static_image_mode=False,
            keypoint_folder=keypoints_folder
        )
    # Pass contact.json path directly to main.py
    contact_json_path = contact_json_path
    
    # All frames to process (for non-contact frame processing later)
    frames_to_process = all_available_frames
    
    # Convert contact frames to set for fast lookup
    contact_frames_set = set(available_contact_frames)
    
    # Run main optimization with mean betas for frames with keypoints
    print(f"Running VideoBio optimization for frames with keypoints (starting from frame {first_frame})...")
    run_videobio(
        tmp_folder=tmp_folder,
        output_folder=output_folder,
        init_folder=init_dir,
        contact_frames_set=contact_frames_set,
        beta_path=beta_path,
        transl_globalorient_path=transl_globalorient_path,
        interactive=args.interactive
    )

    # After run_videobio completes
    # Process frames without contact
    print("Processing frames without contact...")
    non_contact_frames = [frame for frame in frames_to_process 
                     if frame not in available_contact_frames and frame != first_frame]
    process_frames_without_contact(
        output_dir=output_folder,
        init_folder=init_dir,
        frames=non_contact_frames,
        beta_path=beta_path,
        transl_globalorient_path=transl_globalorient_path,
        config_path=args.cfg_file,
        img_folder=frames_dir
    )

    # Create visualization video from processed output images
    print("\nCreating video from processed images...")
    output_images_folder = output_folder / "images"
    if output_images_folder.exists():
        create_video(
            input_folder=output_images_folder,
            output_folder=output_folder
        )

    # Clean up temporary folder
    print("\nCleaning up temporary files...")
    if tmp_folder.exists():
        shutil.rmtree(tmp_folder)
        print(f"Removed temporary folder: {tmp_folder}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", required=False, type=str, default=None, help="Input folder (default: data/demo/input)")
    parser.add_argument("--output_folder", required=False, type=str, default=None, help="Output folder (default: data/demo/output)")
    parser.add_argument("--skip_preprocessing", required=False, action="store_true", help="Skip preprocessing of data")
    parser.add_argument("--cfg_file", required=True, help="Path to the configuration file")
    parser.add_argument("--interactive", action='store_true', help="Enable verbose debug output")
    
    args = parser.parse_args()
    
    main(args)