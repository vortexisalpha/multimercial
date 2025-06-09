"""
Working Video Advertisement Processor

This module provides actual video processing capabilities for the pipeline.
"""

import cv2
import numpy as np
import os
import time
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class VideoAdProcessor:
    """Real video advertisement processor using OpenCV."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the video processor."""
        self.config = config or {
            "default_opacity": 0.85,
            "default_scale": 0.3,
            "default_position": "bottom_right",
            "quality": "high"
        }
        logger.info("VideoAdProcessor initialized")
    
    def resize_advertisement(self, ad_image: np.ndarray, target_width: int, target_height: int, scale_factor: float = 0.3) -> np.ndarray:
        """Resize advertisement to fit video dimensions."""
        ad_height, ad_width = ad_image.shape[:2]
        
        # Calculate new size
        new_width = int(target_width * scale_factor)
        new_height = int((new_width / ad_width) * ad_height)
        
        # Ensure ad doesn't exceed reasonable bounds
        if new_height > target_height * 0.4:
            new_height = int(target_height * 0.4)
            new_width = int((new_height / ad_height) * ad_width)
        
        return cv2.resize(ad_image, (new_width, new_height))
    
    def get_placement_position(self, frame_width: int, frame_height: int, ad_width: int, ad_height: int, strategy: str = "bottom_right") -> Tuple[int, int]:
        """Calculate advertisement placement position."""
        margin = 20
        
        positions = {
            "top_left": (margin, margin),
            "top_right": (frame_width - ad_width - margin, margin),
            "bottom_left": (margin, frame_height - ad_height - margin),
            "bottom_right": (frame_width - ad_width - margin, frame_height - ad_height - margin),
            "center": ((frame_width - ad_width) // 2, (frame_height - ad_height) // 2)
        }
        
        return positions.get(strategy, positions["bottom_right"])
    
    def apply_advertisement_overlay(self, frame: np.ndarray, ad_image: np.ndarray, position: Tuple[int, int], opacity: float = 0.85) -> np.ndarray:
        """Apply advertisement overlay to a frame."""
        x, y = position
        ad_height, ad_width = ad_image.shape[:2]
        frame_height, frame_width = frame.shape[:2]
        
        # Boundary checks
        if x + ad_width > frame_width:
            ad_width = frame_width - x
            ad_image = ad_image[:, :ad_width]
        if y + ad_height > frame_height:
            ad_height = frame_height - y
            ad_image = ad_image[:ad_height, :]
        
        if ad_width <= 0 or ad_height <= 0:
            return frame
        
        # Create overlay
        overlay = frame.copy()
        overlay[y:y+ad_height, x:x+ad_width] = ad_image
        
        # Blend
        result = cv2.addWeighted(frame, 1-opacity, overlay, opacity, 0)
        return result
    
    def add_advertisement_styling(self, ad_image: np.ndarray) -> np.ndarray:
        """Add border and styling to advertisement."""
        # Add white border
        border_size = 3
        ad_with_border = cv2.copyMakeBorder(
            ad_image, border_size, border_size, border_size, border_size,
            cv2.BORDER_CONSTANT, value=[255, 255, 255]
        )
        
        # Add shadow
        shadow_size = 2
        ad_with_shadow = cv2.copyMakeBorder(
            ad_with_border, shadow_size, shadow_size, shadow_size, shadow_size,
            cv2.BORDER_CONSTANT, value=[128, 128, 128]
        )
        
        return ad_with_shadow
    
    def process_video(self, video_path: str, ad_image_path: str, output_path: str, placement_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process video with advertisement placement.
        
        Args:
            video_path: Path to input video
            ad_image_path: Path to advertisement image
            output_path: Path for output video
            placement_config: Placement configuration
        
        Returns:
            Processing result dictionary
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting video processing: {video_path}")
            
            # Load advertisement
            ad_image = cv2.imread(ad_image_path)
            if ad_image is None:
                raise ValueError(f"Could not load advertisement: {ad_image_path}")
            
            # Style the advertisement
            ad_image = self.add_advertisement_styling(ad_image)
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"Video properties: {frame_width}x{frame_height}, {fps}fps, {total_frames} frames")
            
            # Prepare advertisement
            scale_factor = placement_config.get("scale_factor", 0.3)
            ad_resized = self.resize_advertisement(ad_image, frame_width, frame_height, scale_factor)
            
            # Get placement position
            strategy = placement_config.get("strategy", "bottom_right")
            ad_height, ad_width = ad_resized.shape[:2]
            position = self.get_placement_position(frame_width, frame_height, ad_width, ad_height, strategy)
            
            # Setup output
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            
            # Calculate timing
            start_frame = int(placement_config.get("start_time", 2.0) * fps)
            duration = placement_config.get("duration", 10.0)
            end_frame = start_frame + int(duration * fps)
            opacity = placement_config.get("opacity", 0.85)
            
            logger.info(f"Advertisement will show from frame {start_frame} to {end_frame}")
            
            # Process frames
            frame_count = 0
            frames_with_ad = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Apply advertisement if in time range
                if start_frame <= frame_count <= end_frame:
                    frame = self.apply_advertisement_overlay(frame, ad_resized, position, opacity)
                    frames_with_ad += 1
                
                out.write(frame)
                frame_count += 1
                
                # Progress logging
                if frame_count % (fps * 5) == 0:  # Every 5 seconds
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"Processing progress: {progress:.1f}%")
            
            # Cleanup
            cap.release()
            out.release()
            
            processing_time = time.time() - start_time
            
            result = {
                "status": "completed",
                "input_video": video_path,
                "advertisement": ad_image_path,
                "output_video": output_path,
                "processing_time": processing_time,
                "total_frames": frame_count,
                "frames_with_advertisement": frames_with_ad,
                "video_properties": {
                    "width": frame_width,
                    "height": frame_height,
                    "fps": fps,
                    "duration_seconds": frame_count / fps
                },
                "placement_config": placement_config
            }
            
            logger.info(f"Video processing completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            raise
    
    def validate_inputs(self, video_path: str, ad_image_path: str) -> bool:
        """Validate input files exist and are readable."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        if not os.path.exists(ad_image_path):
            raise FileNotFoundError(f"Advertisement image not found: {ad_image_path}")
        
        # Test if files can be opened
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            raise ValueError(f"Cannot open video file: {video_path}")
        cap.release()
        
        ad_test = cv2.imread(ad_image_path)
        if ad_test is None:
            raise ValueError(f"Cannot load advertisement image: {ad_image_path}")
        
        return True 