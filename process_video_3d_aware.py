#!/usr/bin/env python3
"""
3D Scene-Aware Video Advertisement Placement Demo

This script demonstrates the difference between simple 2D overlay 
and 3D scene-aware advertisement placement that respects depth and occlusion.
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

from video_ad_placement.pipeline.depth_aware_processor import DepthAwareProcessor

def compare_placement_methods(video_path, ad_image_path, output_dir):
    """Compare 2D overlay vs 3D scene-aware placement."""
    
    print("🎬 3D Scene-Aware Advertisement Placement Demo")
    print("=" * 60)
    print()
    print("This demo compares:")
    print("  1. 🔲 Simple 2D overlay (current method)")
    print("  2. 🧊 3D scene-aware placement (new method)")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Configuration for both methods
    placement_config = {
        "strategy": "bottom_right",
        "opacity": 0.85,
        "scale_factor": 0.25,
        "start_time": 3.0,
        "duration": 15.0,
        "placement_depth": 0.8  # Place ad in background layer
    }
    
    print(f"📹 Input video: {video_path}")
    print(f"📷 Advertisement: {ad_image_path}")
    print(f"📊 Configuration: {placement_config}")
    print()
    
    # Method 1: Simple 2D overlay (existing method)
    print("🔲 1. Processing with simple 2D overlay...")
    output_2d = os.path.join(output_dir, "video_with_ad_2d_overlay.mp4")
    result_2d = process_with_2d_overlay(video_path, ad_image_path, output_2d, placement_config)
    
    print(f"✅ 2D overlay completed in {result_2d['processing_time']:.2f}s")
    print(f"💾 Output: {output_2d}")
    
    # Method 2: 3D scene-aware placement
    print("\n🧊 2. Processing with 3D scene-aware placement...")
    output_3d = os.path.join(output_dir, "video_with_ad_3d_aware.mp4")
    
    try:
        depth_processor = DepthAwareProcessor({
            "placement_depth": 0.8,  # Place in background
            "occlusion_threshold": 0.3
        })
        
        result_3d = depth_processor.process_video_with_depth_awareness(
            video_path, ad_image_path, output_3d, placement_config
        )
        
        print(f"✅ 3D placement completed in {result_3d['processing_time']:.2f}s")
        print(f"💾 Output: {output_3d}")
        print(f"🧊 Depth processing: {'Enabled' if result_3d['depth_processing'] else 'Fallback mode'}")
        
    except Exception as e:
        print(f"❌ 3D processing failed: {e}")
        result_3d = None
    
    # Create comparison report
    comparison_report = {
        "demo_type": "2D_vs_3D_placement_comparison",
        "input_video": video_path,
        "advertisement": ad_image_path,
        "placement_config": placement_config,
        "methods": {
            "2d_overlay": {
                "description": "Simple 2D image overlay - flat compositing",
                "output": output_2d,
                "processing_time": result_2d['processing_time'],
                "method": "cv2.addWeighted() - no depth awareness"
            }
        }
    }
    
    if result_3d:
        comparison_report["methods"]["3d_aware"] = {
            "description": "3D scene-aware placement with depth and occlusion",
            "output": output_3d,
            "processing_time": result_3d['processing_time'],
            "method": "depth estimation + occlusion masking",
            "depth_processing": result_3d['depth_processing']
        }
    
    # Save comparison report
    report_path = os.path.join(output_dir, "placement_comparison_report.json")
    with open(report_path, 'w') as f:
        json.dump(comparison_report, f, indent=2)
    
    # Display results
    print("\n📊 COMPARISON RESULTS")
    print("=" * 40)
    print(f"📄 Report saved: {report_path}")
    print()
    print("🔍 Key Differences:")
    print("  🔲 2D Overlay:")
    print("    • Advertisement appears as flat layer on top")
    print("    • Always visible regardless of scene depth")
    print("    • Will appear in front of ALL objects (like tables)")
    print("    • Fast processing, simple implementation")
    print()
    if result_3d:
        print("  🧊 3D Scene-Aware:")
        print("    • Advertisement respects scene depth")
        print("    • Can appear behind objects when appropriate")
        print("    • Occlusion handling for realistic placement")
        print("    • Depth-based visibility calculations")
    
    print(f"\n🎬 Compare the videos:")
    print(f"  2D: {output_2d}")
    if result_3d:
        print(f"  3D: {output_3d}")
    
    return comparison_report

def process_with_2d_overlay(video_path, ad_image_path, output_path, config):
    """Process video with simple 2D overlay (existing method)."""
    start_time = time.time()
    
    # Load advertisement
    ad_image = cv2.imread(ad_image_path)
    if ad_image is None:
        raise ValueError(f"Could not load advertisement: {ad_image_path}")
    
    # Add simple styling
    ad_image = add_border_and_shadow(ad_image)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Resize advertisement
    scale_factor = config.get("scale_factor", 0.25)
    new_width = int(frame_width * scale_factor)
    new_height = int((new_width / ad_image.shape[1]) * ad_image.shape[0])
    ad_resized = cv2.resize(ad_image, (new_width, new_height))
    
    # Calculate position
    strategy = config.get("strategy", "bottom_right")
    if strategy == "bottom_right":
        x = frame_width - new_width - 20
        y = frame_height - new_height - 20
    elif strategy == "bottom_left":
        x = 20
        y = frame_height - new_height - 20
    else:  # Default
        x = frame_width - new_width - 20
        y = frame_height - new_height - 20
    
    # Setup output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Calculate timing
    start_frame = int(config.get("start_time", 2.0) * fps)
    duration = config.get("duration", 10.0)
    end_frame = start_frame + int(duration * fps)
    opacity = config.get("opacity", 0.85)
    
    # Process frames
    frame_count = 0
    frames_with_ad = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply simple 2D overlay if in time range
        if start_frame <= frame_count <= end_frame:
            # Simple overlay - this is the "flat" method
            overlay = frame.copy()
            overlay[y:y+new_height, x:x+new_width] = ad_resized
            frame = cv2.addWeighted(frame, 1-opacity, overlay, opacity, 0)
            frames_with_ad += 1
        
        out.write(frame)
        frame_count += 1
    
    cap.release()
    out.release()
    
    processing_time = time.time() - start_time
    
    return {
        "status": "completed",
        "method": "2d_overlay",
        "processing_time": processing_time,
        "total_frames": frame_count,
        "frames_with_advertisement": frames_with_ad
    }

def add_border_and_shadow(ad_image):
    """Add styling to advertisement."""
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

def main():
    """Main demo function."""
    print("🚀 Starting 3D Scene-Aware Advertisement Placement Demo")
    print()
    
    # Paths
    video_path = "/Users/ffmbp/Desktop/multimercial/logs/vid.mp4"
    ad_image_path = "./uploads/advertisements/images.png"
    output_dir = "./outputs/3d_comparison"
    
    # Check prerequisites
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return 1
    
    if not os.path.exists(ad_image_path):
        print(f"❌ Advertisement image not found: {ad_image_path}")
        return 1
    
    try:
        # Run comparison
        report = compare_placement_methods(video_path, ad_image_path, output_dir)
        
        print("\n🎉 Demo completed successfully!")
        print("\n💡 Understanding the Difference:")
        print("   The 2D method you saw before places the ad as a flat overlay")
        print("   The 3D method considers scene depth and occlusion")
        print("   Watch both videos to see how they handle objects in the scene")
        
        return 0
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 