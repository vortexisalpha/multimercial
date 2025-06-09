#!/usr/bin/env python3
"""
Video Advertisement Placement Demo Script

This script demonstrates the video advertisement placement functionality
by actually processing a video and overlaying an advertisement image.
"""

import cv2
import numpy as np
import os
import sys
import time
from pathlib import Path
import json

# Add src to path for imports
sys.path.insert(0, 'src')

def resize_ad_to_fit(ad_image, video_width, video_height, scale_factor=0.3):
    """Resize advertisement to fit nicely in the video."""
    ad_height, ad_width = ad_image.shape[:2]
    
    # Calculate new size (30% of video width by default)
    new_width = int(video_width * scale_factor)
    new_height = int((new_width / ad_width) * ad_height)
    
    # Ensure ad doesn't exceed video dimensions
    if new_height > video_height * 0.4:
        new_height = int(video_height * 0.4)
        new_width = int((new_height / ad_height) * ad_width)
    
    return cv2.resize(ad_image, (new_width, new_height))

def find_ad_placement_position(frame_width, frame_height, ad_width, ad_height, strategy="bottom_right"):
    """Find the best position to place the advertisement."""
    
    positions = {
        "top_left": (20, 20),
        "top_right": (frame_width - ad_width - 20, 20),
        "bottom_left": (20, frame_height - ad_height - 20),
        "bottom_right": (frame_width - ad_width - 20, frame_height - ad_height - 20),
        "center": ((frame_width - ad_width) // 2, (frame_height - ad_height) // 2)
    }
    
    return positions.get(strategy, positions["bottom_right"])

def create_transparent_overlay(frame, ad_image, position, opacity=0.8):
    """Create a semi-transparent overlay of the advertisement."""
    x, y = position
    ad_height, ad_width = ad_image.shape[:2]
    
    # Ensure we don't go out of bounds
    frame_height, frame_width = frame.shape[:2]
    if x + ad_width > frame_width:
        ad_width = frame_width - x
        ad_image = ad_image[:, :ad_width]
    if y + ad_height > frame_height:
        ad_height = frame_height - y
        ad_image = ad_image[:ad_height, :]
    
    # Create overlay
    overlay = frame.copy()
    overlay[y:y+ad_height, x:x+ad_width] = ad_image
    
    # Blend with original frame
    result = cv2.addWeighted(frame, 1-opacity, overlay, opacity, 0)
    return result

def add_border_and_shadow(ad_image):
    """Add a nice border and shadow effect to the advertisement."""
    # Add a white border
    border_size = 3
    ad_with_border = cv2.copyMakeBorder(
        ad_image, border_size, border_size, border_size, border_size,
        cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )
    
    # Add a subtle shadow (optional - just darken the border)
    shadow_color = [128, 128, 128]
    ad_with_shadow = cv2.copyMakeBorder(
        ad_with_border, 2, 2, 2, 2,
        cv2.BORDER_CONSTANT, value=shadow_color
    )
    
    return ad_with_shadow

def process_video_with_advertisement(video_path, ad_image_path, output_path, placement_config=None):
    """
    Process a video and overlay an advertisement.
    
    Args:
        video_path: Path to input video
        ad_image_path: Path to advertisement image
        output_path: Path for output video
        placement_config: Configuration for ad placement
    """
    print(f"🎬 Starting video processing...")
    print(f"📹 Input video: {video_path}")
    print(f"📷 Advertisement: {ad_image_path}")
    print(f"💾 Output: {output_path}")
    
    # Default configuration
    if placement_config is None:
        placement_config = {
            "strategy": "bottom_right",
            "opacity": 0.85,
            "scale_factor": 0.25,
            "start_time": 2.0,  # Start showing ad after 2 seconds
            "duration": 10.0    # Show ad for 10 seconds
        }
    
    # Load advertisement image
    print("📸 Loading advertisement image...")
    ad_image = cv2.imread(ad_image_path)
    if ad_image is None:
        raise ValueError(f"Could not load advertisement image: {ad_image_path}")
    
    # Add border and styling to ad
    ad_image = add_border_and_shadow(ad_image)
    
    # Open video
    print("🎥 Loading video...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📊 Video info: {frame_width}x{frame_height} @ {fps}fps, {total_frames} frames")
    
    # Resize advertisement to fit video
    ad_resized = resize_ad_to_fit(ad_image, frame_width, frame_height, placement_config["scale_factor"])
    ad_position = find_ad_placement_position(frame_width, frame_height, ad_resized.shape[1], ad_resized.shape[0], placement_config["strategy"])
    
    print(f"📐 Advertisement resized to: {ad_resized.shape[1]}x{ad_resized.shape[0]}")
    print(f"📍 Placement position: {ad_position}")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Calculate frame ranges for ad display
    start_frame = int(placement_config["start_time"] * fps)
    end_frame = start_frame + int(placement_config["duration"] * fps)
    
    print(f"⏰ Advertisement will show from frame {start_frame} to {end_frame}")
    
    # Process frames
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Add advertisement overlay if in the specified time range
        if start_frame <= frame_count <= end_frame:
            frame = create_transparent_overlay(
                frame, ad_resized, ad_position, placement_config["opacity"]
            )
        
        # Write frame
        out.write(frame)
        frame_count += 1
        
        # Progress update
        if frame_count % (fps * 2) == 0:  # Every 2 seconds
            progress = (frame_count / total_frames) * 100
            elapsed = time.time() - start_time
            eta = (elapsed / frame_count) * (total_frames - frame_count)
            print(f"⏳ Progress: {progress:.1f}% | ETA: {eta:.1f}s")
    
    # Cleanup
    cap.release()
    out.release()
    
    processing_time = time.time() - start_time
    print(f"✅ Video processing complete!")
    print(f"⏱️  Total processing time: {processing_time:.2f} seconds")
    print(f"💾 Output saved to: {output_path}")
    
    return {
        "status": "completed",
        "input_video": video_path,
        "advertisement": ad_image_path,
        "output_video": output_path,
        "processing_time": processing_time,
        "frames_processed": frame_count,
        "placement_config": placement_config
    }

def main():
    """Main function to demonstrate video ad placement."""
    print("🚀 Video Advertisement Placement Demo")
    print("=" * 50)
    
    # Paths
    video_path = "/Users/ffmbp/Desktop/multimercial/logs/vid.mp4"
    ad_image_path = "./uploads/advertisements/images.png"
    output_path = "./outputs/video_with_ad.mp4"
    
    # Create output directory
    os.makedirs("./outputs", exist_ok=True)
    
    # Check if files exist
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return
    
    if not os.path.exists(ad_image_path):
        print(f"❌ Advertisement image not found: {ad_image_path}")
        return
    
    # Custom placement configuration
    placement_config = {
        "strategy": "bottom_right",  # Options: top_left, top_right, bottom_left, bottom_right, center
        "opacity": 0.85,            # 0.0 (transparent) to 1.0 (opaque)
        "scale_factor": 0.3,        # Size relative to video width
        "start_time": 3.0,          # Start showing ad after 3 seconds
        "duration": 15.0            # Show ad for 15 seconds
    }
    
    try:
        # Process the video
        result = process_video_with_advertisement(
            video_path, ad_image_path, output_path, placement_config
        )
        
        # Save processing report
        report_path = "./outputs/processing_report.json"
        with open(report_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n📊 Processing report saved to: {report_path}")
        print("\n🎉 Demo completed successfully!")
        print(f"🎬 Watch the result: {output_path}")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 