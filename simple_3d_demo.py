#!/usr/bin/env python3
"""
Simple 3D Scene-Aware Advertisement Placement Demo

This script demonstrates the difference between simple 2D overlay 
and depth-aware advertisement placement without complex dependencies.
"""

import cv2
import numpy as np
import os
import time
import json

def estimate_simple_depth(frame):
    """Simple depth estimation using gradient and blur."""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate gradients (edges indicate closer objects)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize gradient
    gradient_norm = gradient_magnitude / gradient_magnitude.max()
    
    # Simple depth heuristic: areas with high gradients are closer
    # Invert so that 0 = close, 1 = far
    depth_estimate = 1.0 - gradient_norm
    
    # Smooth the depth map
    depth_smoothed = cv2.GaussianBlur(depth_estimate, (15, 15), 0)
    
    return depth_smoothed

def detect_foreground_objects(frame):
    """Simple foreground object detection using edge detection."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Find strong edges (likely object boundaries)
    edges = cv2.Canny(gray, 100, 200)
    
    # Dilate to connect nearby edges
    kernel = np.ones((5, 5), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter significant objects
    objects = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 2000:  # Significant objects only
            x, y, w, h = cv2.boundingRect(contour)
            objects.append((x, y, x+w, y+h))
    
    return objects

def apply_depth_aware_placement(frame, ad_image, position, depth_map, placement_depth=0.7):
    """Apply advertisement with depth awareness."""
    x, y = position
    ad_h, ad_w = ad_image.shape[:2]
    frame_h, frame_w = frame.shape[:2]
    
    # Ensure advertisement fits in frame
    if x + ad_w > frame_w:
        ad_w = frame_w - x
        ad_image = ad_image[:, :ad_w]
    if y + ad_h > frame_h:
        ad_h = frame_h - y
        ad_image = ad_image[:ad_h, :]
    
    if ad_w <= 0 or ad_h <= 0:
        return frame
    
    result = frame.copy()
    
    # Get depth values for the placement region
    placement_region_depth = depth_map[y:y+ad_h, x:x+ad_w]
    
    # Create occlusion mask
    # Objects with depth < placement_depth will occlude the ad
    occlusion_threshold = placement_depth - 0.2  # Some tolerance
    visible_mask = placement_region_depth >= occlusion_threshold
    
    # Smooth the mask to avoid harsh edges
    visible_mask_float = visible_mask.astype(np.float32)
    visible_mask_smooth = cv2.GaussianBlur(visible_mask_float, (7, 7), 0)
    
    # Apply advertisement with occlusion
    for i in range(ad_h):
        for j in range(ad_w):
            visibility = visible_mask_smooth[i, j]
            
            if visibility > 0.1:  # Pixel is at least partially visible
                # Blend based on visibility and base opacity
                alpha = 0.85 * visibility
                
                result[y+i, x+j] = (
                    alpha * ad_image[i, j] + 
                    (1 - alpha) * frame[y+i, x+j]
                )
    
    return result

def create_visualization_overlay(frame, depth_map, objects):
    """Create a visualization showing depth and detected objects."""
    # Create depth visualization
    depth_colored = cv2.applyColorMap((depth_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Blend with original frame
    depth_overlay = cv2.addWeighted(frame, 0.7, depth_colored, 0.3, 0)
    
    # Draw detected objects
    for obj in objects:
        x1, y1, x2, y2 = obj
        cv2.rectangle(depth_overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(depth_overlay, "Object", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return depth_overlay

def process_video_comparison(video_path, ad_image_path, output_dir):
    """Process video with both 2D and depth-aware methods."""
    
    print("🎬 Simple 3D Scene-Aware Advertisement Demo")
    print("=" * 50)
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load advertisement
    ad_image = cv2.imread(ad_image_path)
    if ad_image is None:
        raise ValueError(f"Could not load advertisement: {ad_image_path}")
    
    # Add styling
    ad_image = add_simple_styling(ad_image)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video: {frame_width}x{frame_height}, {fps}fps, {total_frames} frames")
    
    # Resize advertisement
    scale_factor = 0.25
    new_width = int(frame_width * scale_factor)
    new_height = int((new_width / ad_image.shape[1]) * ad_image.shape[0])
    ad_resized = cv2.resize(ad_image, (new_width, new_height))
    
    # Position (bottom-right)
    ad_x = frame_width - new_width - 30
    ad_y = frame_height - new_height - 30
    position = (ad_x, ad_y)
    
    # Setup outputs
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # 2D overlay output
    out_2d = cv2.VideoWriter(
        os.path.join(output_dir, "simple_2d_overlay.mp4"), 
        fourcc, fps, (frame_width, frame_height)
    )
    
    # 3D aware output
    out_3d = cv2.VideoWriter(
        os.path.join(output_dir, "simple_3d_aware.mp4"), 
        fourcc, fps, (frame_width, frame_height)
    )
    
    # Depth visualization output
    out_depth = cv2.VideoWriter(
        os.path.join(output_dir, "depth_visualization.mp4"), 
        fourcc, fps, (frame_width, frame_height)
    )
    
    # Processing parameters
    start_frame = int(3.0 * fps)  # Start at 3 seconds
    end_frame = start_frame + int(15.0 * fps)  # Show for 15 seconds
    placement_depth = 0.8  # Place ad in background layer
    
    print(f"🎯 Advertisement placement: {position}")
    print(f"⏱️  Timing: frames {start_frame} to {end_frame}")
    print(f"🧊 Target depth: {placement_depth} (0=foreground, 1=background)")
    print()
    
    # Process frames
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply advertisements if in time range
        if start_frame <= frame_count <= end_frame:
            
            # Method 1: Simple 2D overlay
            frame_2d = frame.copy()
            overlay = frame_2d.copy()
            overlay[ad_y:ad_y+new_height, ad_x:ad_x+new_width] = ad_resized
            frame_2d = cv2.addWeighted(frame_2d, 0.15, overlay, 0.85, 0)  # Simple blend
            
            # Method 2: Depth-aware placement
            depth_map = estimate_simple_depth(frame)
            objects = detect_foreground_objects(frame)
            frame_3d = apply_depth_aware_placement(frame, ad_resized, position, depth_map, placement_depth)
            
            # Create depth visualization
            frame_depth_viz = create_visualization_overlay(frame, depth_map, objects)
            
            # Add labels to distinguish the methods
            cv2.putText(frame_2d, "2D Overlay (Flat)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame_3d, "3D Depth-Aware", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame_depth_viz, "Depth Analysis", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
        else:
            frame_2d = frame.copy()
            frame_3d = frame.copy()
            frame_depth_viz = frame.copy()
        
        # Write frames
        out_2d.write(frame_2d)
        out_3d.write(frame_3d)
        out_depth.write(frame_depth_viz)
        
        frame_count += 1
        
        # Progress update
        if frame_count % (fps * 5) == 0:
            progress = (frame_count / total_frames) * 100
            print(f"⏳ Processing: {progress:.1f}%")
    
    # Cleanup
    cap.release()
    out_2d.release()
    out_3d.release()
    out_depth.release()
    
    processing_time = time.time() - start_time
    
    # Create report
    report = {
        "demo_type": "simple_3d_comparison",
        "processing_time": processing_time,
        "total_frames": frame_count,
        "outputs": {
            "2d_overlay": os.path.join(output_dir, "simple_2d_overlay.mp4"),
            "3d_aware": os.path.join(output_dir, "simple_3d_aware.mp4"),
            "depth_viz": os.path.join(output_dir, "depth_visualization.mp4")
        },
        "explanation": {
            "2d_method": "Simple overlay - ad always appears on top",
            "3d_method": "Depth-aware - ad can be occluded by foreground objects",
            "key_difference": "The 3D method respects scene depth and occlusion"
        }
    }
    
    report_path = os.path.join(output_dir, "comparison_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    return report

def add_simple_styling(ad_image):
    """Add border and shadow to advertisement."""
    # White border
    ad_with_border = cv2.copyMakeBorder(ad_image, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    # Shadow
    ad_with_shadow = cv2.copyMakeBorder(ad_with_border, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[128, 128, 128])
    return ad_with_shadow

def main():
    """Main demo function."""
    print("🚀 Starting Simple 3D Advertisement Placement Demo")
    print()
    
    # Paths
    video_path = "/Users/ffmbp/Desktop/multimercial/logs/videoplayback.mp4"
    ad_image_path = "./uploads/advertisements/images.png"
    output_dir = "./outputs/simple_3d_demo"
    
    # Debug: Show exactly which video file we're using
    print(f"🎯 Using video file: {video_path}")
    
    # Check files
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return 1
    
    if not os.path.exists(ad_image_path):
        print(f"❌ Advertisement not found: {ad_image_path}")
        return 1
    
    try:
        report = process_video_comparison(video_path, ad_image_path, output_dir)
        
        print(f"\n✅ Processing completed in {report['processing_time']:.2f}s")
        print(f"📁 Outputs saved to: {output_dir}")
        print()
        print("🎬 Generated Videos:")
        print(f"  📺 2D Overlay: {report['outputs']['2d_overlay']}")
        print(f"  🧊 3D Aware:   {report['outputs']['3d_aware']}")
        print(f"  🔍 Depth Viz:  {report['outputs']['depth_viz']}")
        print()
        print("🔍 Key Differences:")
        print("  📺 2D Overlay: Advertisement always appears as flat layer on top")
        print("  🧊 3D Aware:   Advertisement respects depth - can be hidden behind objects")
        print("  🔍 Depth Viz:  Shows depth estimation and object detection")
        print()
        print("💡 Watch the videos to see how the 3D method handles occlusion!")
        print("   The advertisement should appear behind foreground objects in the 3D version.")
        
        return 0
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main()) 