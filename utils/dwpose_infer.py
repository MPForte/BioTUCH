import cv2
import os
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
import json
import numpy as np
from tqdm import tqdm

import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from data.annotator.dwpose import DWposeDetector

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def select_best_person(keypoints, scores, visibility_threshold=0.4):
    """
    Select the person with the most visible keypoints
    Args:
        keypoints: list of keypoints for each person
        scores: confidence scores for each keypoint
        visibility_threshold: minimum confidence score to consider a keypoint visible
    Returns:
        keypoints and scores for the person with most visible keypoints
    """
    if len(keypoints) == 0:
        return None, None
    
    # Count visible keypoints for each person
    visible_counts = []
    for person_scores in scores:
        # Count keypoints with confidence above threshold
        visible_count = np.sum(person_scores > visibility_threshold)
        visible_counts.append(visible_count)
    
    # Get the person with most visible keypoints
    best_idx = np.argmax(visible_counts)
    
    return keypoints[best_idx], scores[best_idx]

def convert_to_openpose_format(keypoints, scores):
    """
    Convert DWpose keypoints to OpenPose format
    Returns flat arrays of [x, y, confidence] triplets
    """
    # Initialize arrays for each part
    pose_keypoints_2d = []
    face_keypoints_2d = []
    hand_left_keypoints_2d = []
    hand_right_keypoints_2d = []

    # Extract body parts like in the template
    body = keypoints[:18].copy()  # First 18 are body keypoints
    foot = keypoints[18:24]      # 18-24 are foot keypoints
    faces = keypoints[24:92]     # 24-92 are face keypoints
    hands = np.vstack([keypoints[92:113], keypoints[113:]])  # Hand keypoints

    # Calculate mid-point between hips for midhip (keypoint 8)
    left_hip = body[11]   # Left hip in DWpose
    right_hip = body[8]   # Right hip in DWpose
    midhip_point = [(left_hip[0] + right_hip[0])/2, 
                    (left_hip[1] + right_hip[1])/2]
    midhip_conf = min(scores[11], scores[8])

    # Map DWpose body keypoints to OpenPose format
    openpose_mapping = {
        0: 0,   # Nose
        1: 1,   # Neck
        2: 2,   # Right Shoulder
        3: 3,   # Right Elbow
        4: 4,   # Right Wrist
        5: 5,   # Left Shoulder
        6: 6,   # Left Elbow
        7: 7,   # Left Wrist
        8: None,  # Midhip (calculated)
        9: 8,   # Right Hip
        10: 9,  # Right Knee
        11: 10, # Right Ankle
        12: 11, # Left Hip
        13: 12, # Left Knee
        14: 13, # Left Ankle
        15: 14, # Right Eye
        16: 15, # Left Eye
        17: 16, # Right Ear
        18: 17  # Left Ear
    }

    # Process body keypoints
    for i in range(19):  # OpenPose format body points
        if i == 8:  # Midhip point
            x, y = midhip_point
            conf = midhip_conf
        elif i in openpose_mapping and openpose_mapping[i] is not None:
            idx = openpose_mapping[i]
            if idx < len(body):
                x, y = body[idx]
                conf = float(scores[idx]) if idx < len(scores) else 0.0
            else:
                x, y, conf = 0.0, 0.0, 0.0
        else:
            x, y, conf = 0.0, 0.0, 0.0
        pose_keypoints_2d.extend([float(x), float(y), conf])

    # Add foot keypoints (19-24 in OpenPose format)
    for i in range(6):  # 6 foot keypoints
        if i < len(foot):
            x, y = foot[i]
            conf = float(scores[18 + i]) if 18 + i < len(scores) else 0.0
            pose_keypoints_2d.extend([float(x), float(y), conf])
        else:
            pose_keypoints_2d.extend([0.0, 0.0, 0.0])

    # Process face keypoints
    for i in range(len(faces)):
        x, y = faces[i]
        conf = float(scores[24 + i]) if 24 + i < len(scores) else 0.0
        face_keypoints_2d.extend([float(x), float(y), conf])

    # Process hand keypoints
    # Left hand
    for i in range(21):  # First 21 points are left hand
        if i < len(hands) // 2:
            x, y = hands[i]
            conf = float(scores[92 + i]) if 92 + i < len(scores) else 0.0
            hand_left_keypoints_2d.extend([float(x), float(y), conf])
        else:
            hand_left_keypoints_2d.extend([0.0, 0.0, 0.0])

    # Right hand
    for i in range(21):  # Next 21 points are right hand
        idx = i + 21
        if idx < len(hands):
            x, y = hands[idx]
            conf = float(scores[113 + i]) if 113 + i < len(scores) else 0.0
            hand_right_keypoints_2d.extend([float(x), float(y), conf])
        else:
            hand_right_keypoints_2d.extend([0.0, 0.0, 0.0])

    return {
        'pose_keypoints_2d': pose_keypoints_2d,
        'face_keypoints_2d': face_keypoints_2d,
        'hand_left_keypoints_2d': hand_left_keypoints_2d,
        'hand_right_keypoints_2d': hand_right_keypoints_2d
    }

def process_images(input_folder, output_folder):
    # Create output folders if they don't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize pose detector
    pose = DWposeDetector()
    
    # Supported image extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    
    # Process each image in the input folder
    for filename in tqdm(os.listdir(input_folder)):
        if filename.lower().endswith(tuple(valid_extensions)):
            base_name, _ = os.path.splitext(filename)
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"{base_name}_dwpose.jpg")
            vis_path = os.path.join(output_folder, f"{base_name}_dwpose2op.jpg")
            keypoints_path = os.path.join(output_folder, f"{base_name}_keypoints.json")
         
            try:
                # Read and process image
                oriImg = cv2.imread(input_path)  # B,G,R order
                
                if oriImg is None:
                    print(f"Error reading image {filename}")
                    continue
                
                # Process image with DWpose
                processed_img = pose(oriImg)
                keypoints, scores = pose.pose_estimation(oriImg)
                
                # Select the person with most visible keypoints
                best_keypoints, best_scores = select_best_person(keypoints, scores, visibility_threshold=0.5)
                
                if best_keypoints is None:
                    print(f"No person detected in {filename}")
                    continue
                
                # Convert to OpenPose format for the best person
                person_data = convert_to_openpose_format(best_keypoints, best_scores)
                person_data['person_id'] = [-1]
                person_data['pose_keypoints_3d'] = []
                person_data['face_keypoints_3d'] = []
                person_data['hand_left_keypoints_3d'] = []
                person_data['hand_right_keypoints_3d'] = []
                
                # Create final OpenPose format JSON
                output_dict = {
                    'version': 1.3,
                    'people': [person_data]
                }
                
                # Uncomment only during debugging. This is for visualizing the keypoints
                # # Create visualization using the OpenPose format data
                # vis_img = draw_keypoints_on_image(oriImg, person_data, threshold=0.5)
                # cv2.imwrite(vis_path, vis_img)
                
                # # Save DWpose visualization
                # plt.imsave(output_path, processed_img)
                
                # Save keypoints
                with open(keypoints_path, 'w') as f:
                    json.dump(output_dict, f)
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                import traceback
                print(traceback.format_exc())

def draw_keypoints_on_image(image, person_data, threshold=0.5):
    """
    Draw keypoints and connections on the image with keypoint index labels
    Args:
        image: original image
        person_data: person data in OpenPose format
        threshold: minimum confidence score to draw a keypoint
    """
    # Create a copy of the image
    vis_img = image.copy()
    H, W = image.shape[:2]

    # Colors for different parts
    colors = {
        'body': (0, 255, 0),    # Green
        'face': (255, 0, 0),    # Blue
        'hand': (0, 0, 255),    # Red
        'foot': (0, 255, 255),  # Cyan
        'text': (0, 0, 0)       # Black for labels
    }

    # Extract keypoints and confidences from person_data
    pose_data = np.array(person_data['pose_keypoints_2d']).reshape(-1, 3)
    keypoints = pose_data[:, :2]
    scores = pose_data[:, 2]

    # Body keypoint connections including feet
    body_connections = [
        # Body
        [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],  # Arms
        [8, 9], [9, 10], [10, 11],     # Right leg
        [8, 12], [12, 13], [13, 14],   # Left leg
        [1, 8],                        # Neck to midhip
        # Face
        [1, 0],     # Neck to nose
        [0, 15], [15, 17],  # Right face (nose -> right eye -> right ear)
        [0, 16], [16, 18],  # Left face (nose -> left eye -> left ear)
        # Feet
        [11, 22], [11, 24],  # Left ankle to left foot points
        [14, 19], [14, 21],  # Right ankle to right foot points
        [22, 23],  # Left foot connection
        [19, 20]   # Right foot connection
    ]

    # Draw body connections
    for connection in body_connections:
        if scores[connection[0]] > threshold and scores[connection[1]] > threshold:
            x1, y1 = keypoints[connection[0]]
            x2, y2 = keypoints[connection[1]]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # Use foot color for foot connections
            color = colors['foot'] if min(connection) >= 19 else colors['body']
            cv2.line(vis_img, (x1, y1), (x2, y2), color, 2)

    # Draw body keypoints and their indices (including feet)
    max_keypoint = 25  # Including feet points (0-24)
    for i in range(max_keypoint):
        if scores[i] > threshold:
            x, y = keypoints[i]
            x, y = int(x), int(y)
            # Choose color based on keypoint type
            color = colors['foot'] if i >= 19 else colors['body']
            cv2.circle(vis_img, (x, y), 4, color, -1)
            cv2.putText(vis_img, str(i), (x + 5, y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['text'], 2)

    # Draw face keypoints
    face_data = np.array(person_data['face_keypoints_2d']).reshape(-1, 3)
    for x, y, conf in face_data:
        if conf > threshold:
            pos = (int(x), int(y))
            cv2.circle(vis_img, pos, 1, colors['face'], -1)

    # Hand keypoint connections
    hand_connections = [
        [0, 1], [1, 2], [2, 3], [3, 4],           # Thumb
        [0, 5], [5, 6], [6, 7], [7, 8],           # Index finger
        [0, 9], [9, 10], [10, 11], [11, 12],      # Middle finger
        [0, 13], [13, 14], [14, 15], [15, 16],    # Ring finger
        [0, 17], [17, 18], [18, 19], [19, 20]     # Pinky
    ]

    # Draw hands with connections and numbering
    hand_data = np.array(person_data['hand_left_keypoints_2d']).reshape(-1, 3)
    for connection in hand_connections:
        if hand_data[connection[0], 2] > threshold and hand_data[connection[1], 2] > threshold:
            x1, y1 = hand_data[connection[0], :2]
            x2, y2 = hand_data[connection[1], :2]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.line(vis_img, (x1, y1), (x2, y2), colors['hand'], 1)

    for i, (x, y, conf) in enumerate(hand_data):
        if conf > threshold:
            pos = (int(x), int(y))
            cv2.circle(vis_img, pos, 2, colors['hand'], -1)
            cv2.putText(vis_img, str(i), (pos[0] + 3, pos[1] + 3), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, colors['text'], 1)

    hand_data = np.array(person_data['hand_right_keypoints_2d']).reshape(-1, 3)
    for connection in hand_connections:
        if hand_data[connection[0], 2] > threshold and hand_data[connection[1], 2] > threshold:
            x1, y1 = hand_data[connection[0], :2]
            x2, y2 = hand_data[connection[1], :2]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.line(vis_img, (x1, y1), (x2, y2), colors['hand'], 1)

    for i, (x, y, conf) in enumerate(hand_data):
        if conf > threshold:
            pos = (int(x), int(y))
            cv2.circle(vis_img, pos, 2, colors['hand'], -1)
            cv2.putText(vis_img, str(i), (pos[0] + 3, pos[1] + 3), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, colors['text'], 1)

    return vis_img
def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Process images with DWpose detector and extract keypoints')
    
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Input directory containing images to process'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for processed images and keypoints'
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Process all images
    print(args.input_dir, args.output_dir)
    process_images(args.input_dir, args.output_dir)