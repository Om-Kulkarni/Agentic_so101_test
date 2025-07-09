"""
LLM Planner using Gemini and SmolVLA for the SO101 kitting task.
Gemini serves as the high-level task planner to decide which cube to pick up,
while SmolVLA acts as the low-level controller to execute the picking action.
"""

import os
from pathlib import Path
from enum import Enum
import logging
import base64
from typing import Optional, Dict, Union, Any
import cv2
import torch
import numpy as np
import requests  # For direct API calls to Gemini
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig 
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Camera configurations
TOP_CAMERA_PORT = 2  # /dev/video2 - Top-down view
WRIST_CAMERA_PORT = 4  # /dev/video4 - Gripper-mounted camera

# Available tasks for SmolVLA
class TaskPrompt(Enum):
    PICK_RED = "Pick up red cube"
    PICK_BLUE = "Pick up blue cube"

class LLMPlanner:
    """
    A planner that uses Gemini as the high-level task selector and 
    SmolVLA as the low-level action controller.
    """
    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        """
        Initialize the LLM planner with Gemini and SmolVLA.
        
        Args:
            model_path: Path to the finetuned SmolVLA model weights
        """
        self._init_gemini()
        self._init_smolvla(model_path)
        self._init_cameras()

    def _init_gemini(self):
        """Initialize Gemini API configuration"""
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
            
        # Set up API configuration
        self.gemini_endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-vision:generateContent"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Prompt template for cube selection
        self.gemini_prompt = (
            "The image shows a robotic workspace with red and blue cubes. "
            "Based on their positions, which cube should be picked up next? "
            "Only respond with either 'PICK_RED' or 'PICK_BLUE'. "
            "Consider factors like: reachability, current gripper position, and task efficiency."
        )

    def _init_smolvla(self, model_path: Optional[Union[str, Path]] = None):
        """Initialize SmolVLA for low-level control"""
        # Handle model path
        if model_path is None:
            model_dir = Path(__file__).parent.parent / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / "smolvla_finetuned.safetensors"
        elif isinstance(model_path, str):
            model_path = Path(model_path)
            
        # Verify model exists
        if not model_path.exists():
            raise FileNotFoundError(f"SmolVLA model not found at {model_path}")
            
        # Configure SmolVLA
        config = SmolVLAConfig(
            n_obs_steps=1,
            chunk_size=50,
            n_action_steps=50,
            resize_imgs_with_padding=(512, 512),
            vlm_model_name="HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
        )
        
        # Initialize policy and load weights
        self.policy = SmolVLAPolicy(config)
        self.policy.from_pretrained(model_path)
        self.policy.eval()  # Set to evaluation mode

        logger.info("SmolVLA model initialized successfully")

    def _init_cameras(self):
        """Initialize the camera connections"""
        self.top_camera = cv2.VideoCapture(TOP_CAMERA_PORT)
        self.wrist_camera = cv2.VideoCapture(WRIST_CAMERA_PORT)
        
        if not self.top_camera.isOpened() or not self.wrist_camera.isOpened():
            raise RuntimeError("Failed to open one or both cameras")

    def get_camera_images(self):
        """Capture current images from both cameras"""
        _, top_frame = self.top_camera.read()
        _, wrist_frame = self.wrist_camera.read()
        
        if top_frame is None or wrist_frame is None:
            raise RuntimeError("Failed to capture images from cameras")
            
        return {
            "top": top_frame,
            "wrist": wrist_frame
        }

    def select_next_cube(self, image: np.ndarray) -> TaskPrompt:
        """Use Gemini to decide which cube to pick up next"""
        # Convert image for Gemini API
        _, encoded_img = cv2.imencode(".jpg", image)
        image_bytes = base64.b64encode(encoded_img.tobytes()).decode('utf-8')
        
        # Create request payload
        payload = {
            "contents": [{
                "parts": [
                    {"text": self.gemini_prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_bytes
                        }
                    }
                ]
            }]
        }
        
        # Make request to Gemini API
        try:
            response = requests.post(
                self.gemini_endpoint,
                headers=self.headers,
                json=payload
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract response text
            candidates = result.get("candidates", [])
            if candidates and candidates[0].get("content"):
                task_name = candidates[0]["content"].get("parts", [{}])[0].get("text", "")
            else:
                logger.warning("No valid response from Gemini API")
                task_name = "PICK_RED"
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Gemini API request failed: {e}")
            task_name = "PICK_RED"  # Default fallback
        
        try:
            return TaskPrompt[task_name]
        except KeyError:
            logger.warning(f"Invalid task name from Gemini: {task_name}, defaulting to PICK_RED")
            return TaskPrompt.PICK_RED

    def execute_action(self, task_prompt: TaskPrompt, 
                      images: Dict[str, np.ndarray], 
                      robot_state: np.ndarray) -> np.ndarray:
        """Execute the selected task using SmolVLA"""
        # Prepare batch for SmolVLA
        batch = {
            "observation.images.camera_1": torch.from_numpy(images["top"]).permute(2, 0, 1).float() / 255.0,
            "observation.images.camera_2": torch.from_numpy(images["wrist"]).permute(2, 0, 1).float() / 255.0,
            "observation.state": torch.from_numpy(robot_state).float(),
            "task": task_prompt.value
        }
        
        # Get action from policy
        with torch.no_grad():
            action = self.policy.select_action(batch)
        
        return action.cpu().numpy()

    def cleanup(self):
        """Release camera resources"""
        self.top_camera.release()
        self.wrist_camera.release()
        
    def __del__(self):
        """Ensure cleanup on deletion"""
        self.cleanup()
