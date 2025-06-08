#!/usr/bin/env python3
"""
Health Check Script for Video Advertisement Placement System

Performs comprehensive health checks including:
- API endpoint availability
- GPU availability and memory
- Model loading status
- Database connections
- Cache connections
- File system access
"""

import sys
import os
import time
import requests
import json
import subprocess
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, '/app/src')

def check_gpu_availability():
    """Check if GPU is available and accessible."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device)
            memory_allocated = torch.cuda.memory_allocated(current_device)
            memory_total = torch.cuda.get_device_properties(current_device).total_memory
            
            # Check if GPU has sufficient memory (at least 1GB free)
            memory_free = memory_total - memory_allocated
            if memory_free < 1024**3:  # 1GB
                return False, f"Insufficient GPU memory: {memory_free / 1024**3:.1f}GB free"
            
            return True, f"GPU available: {gpu_name} ({gpu_count} devices)"
        else:
            return False, "CUDA not available"
    except Exception as e:
        return False, f"GPU check failed: {str(e)}"

def check_api_endpoint():
    """Check if the main API endpoint is responding."""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            return True, "API endpoint healthy"
        else:
            return False, f"API returned status {response.status_code}"
    except requests.exceptions.RequestException as e:
        return False, f"API endpoint unreachable: {str(e)}"

def check_model_cache():
    """Check if model cache directory is accessible and has models."""
    try:
        model_cache_dir = Path(os.environ.get('MODEL_CACHE_DIR', '/models'))
        
        if not model_cache_dir.exists():
            return False, f"Model cache directory does not exist: {model_cache_dir}"
        
        if not model_cache_dir.is_dir():
            return False, f"Model cache path is not a directory: {model_cache_dir}"
        
        # Check if directory is writable
        test_file = model_cache_dir / '.health_check'
        try:
            test_file.touch()
            test_file.unlink()
        except PermissionError:
            return False, f"Model cache directory not writable: {model_cache_dir}"
        
        # Check if there are any model files
        model_files = list(model_cache_dir.glob('**/*.pth')) + list(model_cache_dir.glob('**/*.pt'))
        if len(model_files) == 0:
            return False, f"No model files found in cache directory: {model_cache_dir}"
        
        return True, f"Model cache healthy: {len(model_files)} models available"
    except Exception as e:
        return False, f"Model cache check failed: {str(e)}"

def check_database_connection():
    """Check database connection if configured."""
    try:
        db_url = os.environ.get('DATABASE_URL')
        if not db_url:
            return True, "Database not configured (optional)"
        
        import psycopg2
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        cursor.close()
        conn.close()
        
        return True, "Database connection healthy"
    except Exception as e:
        return False, f"Database connection failed: {str(e)}"

def check_redis_connection():
    """Check Redis connection if configured."""
    try:
        redis_url = os.environ.get('REDIS_URL', os.environ.get('CELERY_BROKER_URL'))
        if not redis_url:
            return True, "Redis not configured (optional)"
        
        import redis
        r = redis.from_url(redis_url)
        r.ping()
        
        return True, "Redis connection healthy"
    except Exception as e:
        return False, f"Redis connection failed: {str(e)}"

def check_file_system_access():
    """Check file system access for required directories."""
    try:
        directories = [
            os.environ.get('DATA_DIR', '/data'),
            os.environ.get('CACHE_DIR', '/cache'),
            os.environ.get('TMP_DIR', '/tmp/video_processing'),
            '/app/logs'
        ]
        
        for directory in directories:
            dir_path = Path(directory)
            
            # Check if directory exists and is accessible
            if not dir_path.exists():
                return False, f"Directory does not exist: {directory}"
            
            if not dir_path.is_dir():
                return False, f"Path is not a directory: {directory}"
            
            # Check if directory is writable
            test_file = dir_path / '.health_check'
            try:
                test_file.touch()
                test_file.unlink()
            except PermissionError:
                return False, f"Directory not writable: {directory}"
        
        return True, "File system access healthy"
    except Exception as e:
        return False, f"File system check failed: {str(e)}"

def check_ffmpeg_availability():
    """Check if FFmpeg is available and working."""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            # Check for GPU acceleration support
            if 'cuda' in result.stdout.lower():
                return True, "FFmpeg with CUDA support available"
            else:
                return True, "FFmpeg available (no CUDA support detected)"
        else:
            return False, "FFmpeg not working properly"
    except subprocess.TimeoutExpired:
        return False, "FFmpeg check timed out"
    except FileNotFoundError:
        return False, "FFmpeg not found"
    except Exception as e:
        return False, f"FFmpeg check failed: {str(e)}"

def check_python_dependencies():
    """Check if critical Python dependencies are importable."""
    critical_deps = [
        'torch',
        'torchvision',
        'cv2',
        'numpy',
        'PIL',
        'open3d',
        'sklearn',
        'scipy'
    ]
    
    try:
        for dep in critical_deps:
            __import__(dep)
        
        return True, f"All critical dependencies available ({len(critical_deps)} checked)"
    except ImportError as e:
        return False, f"Missing critical dependency: {str(e)}"
    except Exception as e:
        return False, f"Dependency check failed: {str(e)}"

def run_health_checks():
    """Run all health checks and return overall status."""
    checks = [
        ("GPU Availability", check_gpu_availability),
        ("API Endpoint", check_api_endpoint),
        ("Model Cache", check_model_cache),
        ("Database Connection", check_database_connection),
        ("Redis Connection", check_redis_connection),
        ("File System Access", check_file_system_access),
        ("FFmpeg Availability", check_ffmpeg_availability),
        ("Python Dependencies", check_python_dependencies),
    ]
    
    results = {}
    all_healthy = True
    
    for check_name, check_function in checks:
        try:
            is_healthy, message = check_function()
            results[check_name] = {
                "healthy": is_healthy,
                "message": message
            }
            if not is_healthy:
                all_healthy = False
        except Exception as e:
            results[check_name] = {
                "healthy": False,
                "message": f"Check failed with exception: {str(e)}"
            }
            all_healthy = False
    
    return all_healthy, results

def main():
    """Main health check function."""
    print("Running health checks for Video Advertisement Placement System...")
    
    start_time = time.time()
    overall_healthy, results = run_health_checks()
    end_time = time.time()
    
    # Print results
    print(f"\nHealth Check Results (completed in {end_time - start_time:.2f}s):")
    print("=" * 60)
    
    for check_name, result in results.items():
        status = "✓" if result["healthy"] else "✗"
        print(f"{status} {check_name}: {result['message']}")
    
    print("=" * 60)
    
    if overall_healthy:
        print("✓ Overall Status: HEALTHY")
        sys.exit(0)
    else:
        print("✗ Overall Status: UNHEALTHY")
        sys.exit(1)

if __name__ == "__main__":
    main() 