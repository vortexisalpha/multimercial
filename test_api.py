#!/usr/bin/env python3
"""
API Test Client

Simple script to test the Video Ad Placement API endpoints.
"""

import asyncio
import json
import time
from typing import Dict, Any

import httpx

BASE_URL = "http://localhost:8000"
API_PREFIX = "/api/v1"


class APITestClient:
    """Simple API test client."""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.api_keys = {}
        
    async def test_api(self):
        """Run comprehensive API tests."""
        print("🚀 Starting API tests...")
        
        async with httpx.AsyncClient() as client:
            # Test 1: API Info (no auth required)
            print("\n1. Testing API info endpoint...")
            await self._test_api_info(client)
            
            # Test 2: Health check (no auth required)
            print("\n2. Testing health check endpoint...")
            await self._test_health_check(client)
            
            # Test 3: Get test API keys
            print("\n3. Getting test API keys...")
            await self._get_test_api_keys(client)
            
            # Test 4: Test authentication
            print("\n4. Testing authentication...")
            await self._test_authentication(client)
            
            # Test 5: Test rate limiting
            print("\n5. Testing rate limiting...")
            await self._test_rate_limiting(client)
            
            # Test 6: Test video processing endpoint
            print("\n6. Testing video processing endpoint...")
            await self._test_video_processing(client)
            
            # Test 7: Test metrics endpoint (admin only)
            print("\n7. Testing metrics endpoint...")
            await self._test_metrics(client)
            
            print("\n✅ All tests completed!")
    
    async def _test_api_info(self, client: httpx.AsyncClient):
        """Test API info endpoint."""
        try:
            response = await client.get(f"{self.base_url}{API_PREFIX}/info")
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ API Info: {data['name']} v{data['version']}")
                print(f"      Environment: {data['environment']}")
                print(f"      Features: {list(data['features'].keys())}")
            else:
                print(f"   ❌ API Info failed: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ API Info error: {e}")
    
    async def _test_health_check(self, client: httpx.AsyncClient):
        """Test health check endpoint."""
        try:
            response = await client.get(f"{self.base_url}{API_PREFIX}/health")
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Health: {data['status']}")
                
                for component, status in data['components'].items():
                    component_status = status.get('status', 'unknown')
                    print(f"      {component}: {component_status}")
                    
            else:
                print(f"   ❌ Health check failed: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Health check error: {e}")
    
    async def _get_test_api_keys(self, client: httpx.AsyncClient):
        """Get test API keys."""
        try:
            response = await client.get(f"{self.base_url}{API_PREFIX}/test-auth")
            
            if response.status_code == 200:
                data = response.json()
                self.api_keys = {
                    'admin': data.get('admin_api_key'),
                    'user': data.get('user_api_key')
                }
                print(f"   ✅ Got API keys:")
                print(f"      Admin: {self.api_keys['admin'][:16]}...")
                print(f"      User:  {self.api_keys['user'][:16]}...")
            else:
                print(f"   ❌ Failed to get API keys: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ API keys error: {e}")
    
    async def _test_authentication(self, client: httpx.AsyncClient):
        """Test authentication."""
        try:
            # Test without API key (should fail)
            response = await client.get(f"{self.base_url}{API_PREFIX}/metrics")
            if response.status_code == 401:
                print("   ✅ Authentication required for protected endpoints")
            else:
                print(f"   ❌ Expected 401, got {response.status_code}")
            
            # Test with invalid API key (should fail)
            headers = {"X-API-Key": "invalid_key"}
            response = await client.get(f"{self.base_url}{API_PREFIX}/metrics", headers=headers)
            if response.status_code == 401:
                print("   ✅ Invalid API key rejected")
            else:
                print(f"   ❌ Invalid key should be rejected, got {response.status_code}")
            
            # Test with valid API key (should work)
            if self.api_keys.get('admin'):
                headers = {"X-API-Key": self.api_keys['admin']}
                response = await client.get(f"{self.base_url}{API_PREFIX}/metrics", headers=headers)
                if response.status_code == 200:
                    print("   ✅ Valid API key accepted")
                else:
                    print(f"   ❌ Valid key failed: {response.status_code}")
            
        except Exception as e:
            print(f"   ❌ Authentication test error: {e}")
    
    async def _test_rate_limiting(self, client: httpx.AsyncClient):
        """Test rate limiting."""
        try:
            # Make multiple rapid requests to test rate limiting
            responses = []
            for i in range(5):
                response = await client.get(f"{self.base_url}{API_PREFIX}/info")
                responses.append(response.status_code)
            
            if all(status == 200 for status in responses):
                print("   ✅ Rate limiting allows normal requests")
            else:
                print(f"   ❌ Some requests failed: {responses}")
                
        except Exception as e:
            print(f"   ❌ Rate limiting test error: {e}")
    
    async def _test_video_processing(self, client: httpx.AsyncClient):
        """Test video processing endpoint."""
        try:
            if not self.api_keys.get('user'):
                print("   ⚠️  No user API key available")
                return
                
            headers = {"X-API-Key": self.api_keys['user']}
            
            # Test video processing request
            test_request = {
                "video_url": "https://example.com/test-video.mp4",
                "advertisement_config": {
                    "ad_type": "image",
                    "ad_url": "https://example.com/test-ad.jpg",
                    "width": 1.0,
                    "height": 0.6
                },
                "placement_config": {
                    "strategy": "automatic",
                    "quality_threshold": 0.7
                },
                "processing_options": {
                    "quality_level": "medium",
                    "use_gpu": False
                }
            }
            
            response = await client.post(
                f"{self.base_url}{API_PREFIX}/process-video",
                json=test_request,
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                job_id = data.get('job_id')
                print(f"   ✅ Video processing started: {job_id}")
                
                # Test job status
                await asyncio.sleep(1)  # Wait a bit
                status_response = await client.get(
                    f"{self.base_url}{API_PREFIX}/status/{job_id}",
                    headers=headers
                )
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    print(f"      Job status: {status_data.get('status')}")
                    print(f"      Progress: {status_data.get('progress', 0):.1%}")
                else:
                    print(f"   ❌ Status check failed: {status_response.status_code}")
                    
            else:
                print(f"   ❌ Video processing failed: {response.status_code}")
                if response.status_code == 422:
                    print(f"      Validation error: {response.json()}")
                    
        except Exception as e:
            print(f"   ❌ Video processing test error: {e}")
    
    async def _test_metrics(self, client: httpx.AsyncClient):
        """Test metrics endpoint."""
        try:
            if not self.api_keys.get('admin'):
                print("   ⚠️  No admin API key available")
                return
                
            headers = {"X-API-Key": self.api_keys['admin']}
            
            response = await client.get(f"{self.base_url}{API_PREFIX}/metrics", headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Metrics retrieved:")
                print(f"      Total requests: {data.get('total_requests', 0)}")
                print(f"      Requests/min: {data.get('requests_per_minute', 0):.1f}")
                print(f"      Error rate: {data.get('error_rate', 0):.1%}")
            else:
                print(f"   ❌ Metrics failed: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Metrics test error: {e}")


async def main():
    """Main test function."""
    print("Video Ad Placement API Test Suite")
    print("=" * 50)
    
    # Wait for server to be ready
    print("Waiting for server to be ready...")
    
    max_attempts = 30
    for attempt in range(max_attempts):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{BASE_URL}{API_PREFIX}/health", timeout=5.0)
                if response.status_code == 200:
                    print("✅ Server is ready!")
                    break
        except Exception:
            pass
        
        if attempt < max_attempts - 1:
            await asyncio.sleep(1)
        else:
            print("❌ Server not responding after 30 seconds")
            return
    
    # Run tests
    test_client = APITestClient()
    await test_client.test_api()


if __name__ == "__main__":
    asyncio.run(main()) 