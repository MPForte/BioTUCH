import json
from functools import lru_cache
from pathlib import Path
import torch
import os

class SMPLXVertexLoader:
    """
    Utility class to efficiently load and cache SMPL-X vertex indices for different body parts
    """
    _instance = None

    def __new__(cls, segmentation_file=None):
        if cls._instance is None:
            cls._instance = super(SMPLXVertexLoader, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self, segmentation_file=None):
        if self.initialized:
            return
            
        if segmentation_file is None:
            # Default to looking in the same directory as this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            segmentation_file = os.path.join('.', 'data', 'smplx_vert_segmentation.json')
        
        self.segmentation_file = Path(segmentation_file)
        self._load_segmentation()
        self.initialized = True
    
    @lru_cache(maxsize=None)
    def _load_segmentation(self):
        """Load and cache the vertex segmentation from JSON file"""
        if not self.segmentation_file.exists():
            raise FileNotFoundError(f"Vertex segmentation file not found: {self.segmentation_file}")
            
        with open(self.segmentation_file, 'r') as f:
            self._segmentation = json.load(f)

        # Define upper and lower body part mappings
        self._lower_body_parts = [
            'rightUpLeg', 'leftUpLeg',  # Thighs
            'rightLeg', 'leftLeg',      # Shins  
            'rightFoot', 'leftFoot',    # Feet
            'rightToeBase', 'leftToeBase',  # Toes
            'hips'  # Hip vertices
        ]
        
        self._upper_body_parts = [
            'spine', 'spine1', 'spine2',  # Spine segments
            'leftShoulder', 'rightShoulder',
            'leftArm', 'rightArm',
            'leftForeArm', 'rightForeArm',
            'leftHand', 'rightHand',
            'neck', 'head'
        ]
    
    def get_vertices(self, part_name: str, device=None):
        """
        Get vertex indices for a specific body part
        
        Args:
            part_name (str): Name of the body part (e.g. 'leftHand', 'rightForeArm')
            device: Optional torch device to place the tensor on
            
        Returns:
            torch.Tensor: Tensor of vertex indices
        """
        if part_name not in self._segmentation:
            raise ValueError(f"Unknown body part: {part_name}")
            
        vertices = torch.tensor(self._segmentation[part_name], dtype=torch.long)
        if device is not None:
            vertices = vertices.to(device)
            
        return vertices
    
    def get_left_hand(self, device=None):
        """Get left hand vertices including all fingers"""
        left_hand = self._segmentation['leftHand']
        left_hand_index = self._segmentation.get('leftHandIndex1', [])
        # Combine and remove duplicates
        full_left_hand = sorted(set(left_hand + left_hand_index))
        # Convert to tensor
        vertices = torch.tensor(full_left_hand, dtype=torch.long)
        if device is not None:
            vertices = vertices.to(device)
        return vertices

    def get_right_hand(self, device=None):
        """Get right hand vertices including all fingers"""
        right_hand = self._segmentation['rightHand']
        right_hand_index = self._segmentation.get('rightHandIndex1', [])
        # Combine and remove duplicates
        full_right_hand = sorted(set(right_hand + right_hand_index))
        # Convert to tensor
        vertices = torch.tensor(full_right_hand, dtype=torch.long)
        if device is not None:
            vertices = vertices.to(device)
        return vertices

    def get_lower_body_vertices(self, device=None):
        """Get all lower body vertices including legs, feet, and hips"""
        # Collect vertices from all lower body parts
        lower_body_vertices = set()
        for part in self._lower_body_parts:
            if part in self._segmentation:
                lower_body_vertices.update(self._segmentation[part])
        
        # Convert to sorted list and then tensor
        vertices = torch.tensor(sorted(lower_body_vertices), dtype=torch.long)
        if device is not None:
            vertices = vertices.to(device)
        return vertices

    def get_upper_body_vertices(self, device=None):
        """Get all upper body vertices including torso, arms, and head"""
        # Collect vertices from all upper body parts
        upper_body_vertices = set()
        for part in self._upper_body_parts:
            if part in self._segmentation:
                upper_body_vertices.update(self._segmentation[part])
        
        # Convert to sorted list and then tensor
        vertices = torch.tensor(sorted(upper_body_vertices), dtype=torch.long)
        if device is not None:
            vertices = vertices.to(device)
        return vertices

    def get_hips_vertices(self, device=None):
        """Get hip vertices"""
        if 'hips' not in self._segmentation:
            raise ValueError("Hip vertices not found in segmentation")
            
        vertices = torch.tensor(self._segmentation['hips'], dtype=torch.long)
        if device is not None:
            vertices = vertices.to(device)
        return vertices

# Create singleton instance
vertex_loader = SMPLXVertexLoader()