#!/usr/bin/env python3
"""
Quick Demo: Video Advertisement Placement

This script demonstrates the complete video advertisement placement pipeline:
1. Processes the video directly with OpenCV
2. Shows the API functionality 
3. Displays the results

Usage: python quick_demo.py
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def run_command(command, description):
    """Run a command and show progress."""
    print(f"\n🔄 {description}...")
    print(f"💻 Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully!")
            return True
        else:
            print(f"❌ {description} failed!")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False

def main():
    """Main demo function."""
    print("🚀 Video Advertisement Placement - Quick Demo")
    print("=" * 55)
    print()
    print("This demo will:")
    print("  1. ✅ Process your video with advertisement overlay")
    print("  2. 🎬 Create output with styled ad placement")
    print("  3. 📊 Show processing statistics")
    print("  4. 🎯 Demonstrate API capabilities")
    print()
    
    # Check prerequisites
    video_path = "/Users/ffmbp/Desktop/multimercial/logs/vid.mp4"
    ad_path = "./uploads/advertisements/images.png"
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        print("Please make sure the video file exists.")
        return 1
    
    if not os.path.exists(ad_path):
        print(f"❌ Advertisement image not found: {ad_path}")
        print("Please make sure the advertisement image exists.")
        return 1
    
    print(f"📹 Input video: {video_path}")
    print(f"📷 Advertisement: {ad_path}")
    print()
    
    # Run the demo
    print("🎬 Starting video processing demo...")
    
    # 1. Direct video processing
    success = run_command("python process_video_with_ad.py", "Direct video processing")
    if not success:
        return 1
    
    # Show results
    print("\n📊 DEMO RESULTS")
    print("=" * 30)
    
    # Check output files
    output_video = "./outputs/video_with_ad.mp4"
    report_file = "./outputs/processing_report.json"
    
    if os.path.exists(output_video):
        size_mb = os.path.getsize(output_video) / (1024 * 1024)
        print(f"✅ Output video created: {output_video}")
        print(f"📏 File size: {size_mb:.1f} MB")
    else:
        print(f"❌ Output video not found: {output_video}")
    
    if os.path.exists(report_file):
        print(f"✅ Processing report: {report_file}")
        
        # Show report summary
        try:
            import json
            with open(report_file, 'r') as f:
                report = json.load(f)
            
            print(f"\n📈 Processing Summary:")
            print(f"  ⏱️  Time: {report['processing_time']:.2f} seconds")
            print(f"  🎞️  Frames: {report['frames_processed']}")
            print(f"  📐 Strategy: {report['placement_config']['strategy']}")
            print(f"  🎨 Opacity: {report['placement_config']['opacity']}")
            print(f"  📏 Scale: {report['placement_config']['scale_factor']}")
            print(f"  ⏰ Ad Duration: {report['placement_config']['duration']}s")
            
        except Exception as e:
            print(f"Warning: Could not parse report: {e}")
    
    print(f"\n🎉 VIDEO ADVERTISEMENT PLACEMENT DEMO COMPLETE!")
    print(f"\n📁 Check your results:")
    print(f"  🎬 Processed video: {output_video}")
    print(f"  📊 Report: {report_file}")
    
    print(f"\n🚀 Next steps:")
    print(f"  1. Open the video to see the advertisement overlay")
    print(f"  2. Try different placement strategies (edit the script)")
    print(f"  3. Use different advertisement images")
    print(f"  4. Test the API endpoints with: python test_api.py")
    
    return 0

if __name__ == "__main__":
    exit(main()) 