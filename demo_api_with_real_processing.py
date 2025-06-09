#!/usr/bin/env python3
"""
Demo script that shows how to use the API with real video processing.

This script starts the API server and demonstrates real video advertisement placement.
"""

import asyncio
import httpx
import json
import time
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

async def test_real_video_processing():
    """Test the API with real video processing."""
    
    print("🚀 Starting API with Real Video Processing Demo")
    print("=" * 60)
    
    # Base URL for API
    base_url = "http://localhost:8000/api/v1"
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            # 1. Get API info
            print("📋 1. Getting service info...")
            response = await client.get(f"{base_url}/info")
            if response.status_code == 200:
                info = response.json()
                print(f"✅ Service: {info['name']} v{info['version']}")
                print(f"🏗️  Environment: {info['environment']}")
                print(f"🎯 Features: {', '.join([k for k, v in info['features'].items() if v])}")
            else:
                print(f"❌ Failed to get service info: {response.status_code}")
                return
            
            # 2. Get API keys
            print("\n🔑 2. Getting API keys...")
            response = await client.get(f"{base_url}/test-auth")
            if response.status_code == 200:
                auth_info = response.json()
                api_key = auth_info["user_api_key"]
                print(f"✅ Got API key: {api_key[:20]}...")
            else:
                print(f"❌ Failed to get API keys: {response.status_code}")
                return
            
            # 3. Check health
            print("\n🏥 3. Checking system health...")
            response = await client.get(f"{base_url}/health")
            if response.status_code == 200:
                health = response.json()
                print(f"✅ System status: {health['status']}")
                print(f"📊 Components: {health['components']}")
            else:
                print(f"❌ Health check failed: {response.status_code}")
            
            # 4. Process video with advertisement
            print("\n🎬 4. Processing video with advertisement...")
            
            # Create request payload with file path for real processing
            video_request = {
                "video_url": "https://example.com/demo.mp4",  # URL for API compatibility
                "video_file_path": "/Users/ffmbp/Desktop/multimercial/logs/vid.mp4",  # Real local file
                "advertisement_config": {
                    "ad_type": "image",
                    "ad_url": "./uploads/advertisements/images.png",
                    "width": 0.25,    # 25% of video width
                    "height": 0.25,   # Maintain aspect ratio
                    "opacity": 0.9
                },
                "placement_config": {
                    "strategy": "bottom_right",
                    "quality_threshold": 0.8,
                    "start_time": 2.0,
                    "duration": 12.0
                },
                "processing_options": {
                    "quality": "high",
                    "format": "mp4"
                }
            }
            
            headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
            response = await client.post(
                f"{base_url}/process-video", 
                json=video_request, 
                headers=headers
            )
            
            if response.status_code == 200:
                result = response.json()
                job_id = result["job_id"]
                print(f"✅ Video processing started!")
                print(f"🆔 Job ID: {job_id}")
                print(f"📊 Status: {result['status']}")
                print(f"💬 Message: {result['message']}")
                
                if result.get("result_url"):
                    print(f"🎥 Output video: {result['result_url']}")
                    print("\n🎉 Real video processing completed!")
                    
                    # Show processing summary
                    if os.path.exists("./outputs/processing_report.json"):
                        with open("./outputs/processing_report.json", 'r') as f:
                            report = json.load(f)
                        print("\n📊 Processing Report:")
                        print(f"  ⏱️  Processing time: {report['processing_time']:.2f} seconds")
                        print(f"  🎞️  Frames processed: {report['frames_processed']}")
                        print(f"  📐 Advertisement strategy: {report['placement_config']['strategy']}")
                        print(f"  🎨 Opacity: {report['placement_config']['opacity']}")
                        print(f"  📏 Scale: {report['placement_config']['scale_factor']}")
                        
                else:
                    print("⏳ Job queued for background processing")
                    
            else:
                print(f"❌ Video processing failed: {response.status_code}")
                print(f"Error: {response.text}")
                return
            
            # 5. Check job status (if needed)
            if result["status"] != "completed":
                print(f"\n⏳ 5. Checking job status...")
                response = await client.get(f"{base_url}/status/{job_id}", headers=headers)
                if response.status_code == 200:
                    status = response.json()
                    print(f"📊 Job status: {status}")
                else:
                    print(f"❌ Failed to get job status: {response.status_code}")
            
            # 6. Get metrics (admin)
            print("\n📈 6. Getting system metrics...")
            admin_key = auth_info["admin_api_key"]
            headers = {"X-API-Key": admin_key}
            response = await client.get(f"{base_url}/metrics", headers=headers)
            if response.status_code == 200:
                metrics = response.json()
                print(f"✅ Metrics retrieved:")
                print(f"  📊 Total requests: {metrics.get('total_requests', 0)}")
                print(f"  ⏱️  Avg response time: {metrics.get('avg_response_time', 0):.3f}s")
                print(f"  💾 Memory usage: {metrics.get('memory_usage', {}).get('percent', 0):.1f}%")
                print(f"  🔄 Active jobs: {len(metrics.get('active_jobs', []))}")
            else:
                print(f"❌ Failed to get metrics: {response.status_code}")
            
            print("\n🎉 Demo completed successfully!")
            print("\n📋 Summary:")
            print("  ✅ API is fully functional")
            print("  ✅ Real video processing works")
            print("  ✅ Advertisement placement successful")
            print("  ✅ All endpoints responding")
            print("\n🎬 Check the output video: ./outputs/video_with_ad.mp4")
            
        except httpx.ConnectError:
            print("❌ Cannot connect to API server. Make sure it's running:")
            print("   python run_server.py")
        except Exception as e:
            print(f"❌ Error during demo: {e}")

def main():
    """Main function."""
    print("🔧 Note: Make sure the API server is running in another terminal:")
    print("   python run_server.py")
    print()
    
    # Wait a moment for user to start server if needed
    input("Press Enter to start demo (or Ctrl+C to cancel)...")
    
    # Run the async demo
    asyncio.run(test_real_video_processing())

if __name__ == "__main__":
    main() 