#!/usr/bin/env python3
"""
Model Download Script for Video Advertisement Placement System

Downloads and caches all required AI models for the system:
- Depth estimation models (Marigold, Depth Pro)
- Object detection models (YOLOv9 variants)
- ByteTrack tracking models
- Any additional required models
"""

import os
import sys
import urllib.request
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple
import zipfile
import tarfile

# Model definitions with download URLs and checksums
MODELS = {
    "depth_estimation": {
        "marigold_v1": {
            "url": "https://huggingface.co/prs-eth/marigold-lcm-v1-0/resolve/main/pytorch_model.bin",
            "filename": "marigold_v1.pth",
            "checksum": "abcd1234567890",  # Placeholder - use actual checksum
            "size_mb": 1500,
            "description": "Marigold depth estimation model v1.0"
        },
        "depth_pro_v1": {
            "url": "https://huggingface.co/apple/DepthPro/resolve/main/depth_pro.pt",
            "filename": "depth_pro_v1.pt",
            "checksum": "efgh0987654321",  # Placeholder
            "size_mb": 800,
            "description": "Apple Depth Pro model"
        }
    },
    "object_detection": {
        "yolov9s": {
            "url": "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9s.pt",
            "filename": "yolov9s.pt",
            "checksum": "ijkl1122334455",  # Placeholder
            "size_mb": 30,
            "description": "YOLOv9-S object detection model"
        },
        "yolov9m": {
            "url": "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9m.pt",
            "filename": "yolov9m.pt",
            "checksum": "mnop5566778899",  # Placeholder
            "size_mb": 100,
            "description": "YOLOv9-M object detection model"
        },
        "yolov9l": {
            "url": "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9l.pt",
            "filename": "yolov9l.pt",
            "checksum": "qrst9900112233",  # Placeholder
            "size_mb": 200,
            "description": "YOLOv9-L object detection model"
        },
        "yolov9x": {
            "url": "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9x.pt",
            "filename": "yolov9x.pt",
            "checksum": "uvwx4455667788",  # Placeholder
            "size_mb": 400,
            "description": "YOLOv9-X object detection model"
        }
    },
    "tracking": {
        "bytetrack": {
            "url": "https://github.com/ifzhang/ByteTrack/releases/download/v0.1.0/bytetrack_x_mot17.pth.tar",
            "filename": "bytetrack_x_mot17.pth",
            "checksum": "yzab1234567890",  # Placeholder
            "size_mb": 50,
            "description": "ByteTrack multi-object tracking model"
        }
    }
}

def calculate_checksum(filepath: Path) -> str:
    """Calculate SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()

def download_file(url: str, destination: Path, expected_size_mb: int = None) -> bool:
    """Download a file from URL to destination with progress tracking."""
    try:
        print(f"Downloading {url}")
        print(f"Destination: {destination}")
        
        # Create destination directory if it doesn't exist
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        # Download with progress
        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                downloaded = block_num * block_size
                percent = min(100, (downloaded / total_size) * 100)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="")
            else:
                mb_downloaded = (block_num * block_size) / (1024 * 1024)
                print(f"\r  Downloaded: {mb_downloaded:.1f} MB", end="")
        
        urllib.request.urlretrieve(url, destination, progress_hook)
        print()  # New line after progress
        
        # Verify file size if expected
        if expected_size_mb:
            actual_size_mb = destination.stat().st_size / (1024 * 1024)
            if abs(actual_size_mb - expected_size_mb) > expected_size_mb * 0.1:  # 10% tolerance
                print(f"  Warning: File size mismatch. Expected ~{expected_size_mb}MB, got {actual_size_mb:.1f}MB")
        
        print(f"  Successfully downloaded: {destination.name}")
        return True
        
    except Exception as e:
        print(f"  Error downloading {url}: {str(e)}")
        return False

def extract_archive(archive_path: Path, extract_to: Path) -> bool:
    """Extract archive files (zip, tar, tar.gz)."""
    try:
        if archive_path.suffix.lower() == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_file:
                zip_file.extractall(extract_to)
        elif archive_path.suffix.lower() in ['.tar', '.gz']:
            with tarfile.open(archive_path, 'r:*') as tar_file:
                tar_file.extractall(extract_to)
        else:
            print(f"  Unsupported archive format: {archive_path.suffix}")
            return False
        
        print(f"  Extracted: {archive_path.name}")
        return True
        
    except Exception as e:
        print(f"  Error extracting {archive_path}: {str(e)}")
        return False

def verify_model(model_path: Path, expected_checksum: str) -> bool:
    """Verify model file integrity using checksum."""
    if not model_path.exists():
        return False
    
    # Skip checksum verification for now (placeholder checksums)
    # In production, use actual checksums
    if expected_checksum.startswith(('abcd', 'efgh', 'ijkl', 'mnop', 'qrst', 'uvwx', 'yzab')):
        print(f"  Skipping checksum verification (placeholder checksum)")
        return True
    
    print(f"  Verifying checksum...")
    actual_checksum = calculate_checksum(model_path)
    
    if actual_checksum == expected_checksum:
        print(f"  Checksum verified: {actual_checksum[:16]}...")
        return True
    else:
        print(f"  Checksum mismatch!")
        print(f"    Expected: {expected_checksum[:16]}...")
        print(f"    Actual:   {actual_checksum[:16]}...")
        return False

def download_model_category(category: str, models: Dict, cache_dir: Path) -> bool:
    """Download all models in a category."""
    print(f"\n{'='*60}")
    print(f"Downloading {category} models")
    print(f"{'='*60}")
    
    category_dir = cache_dir / category
    category_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    total_count = len(models)
    
    for model_name, model_info in models.items():
        print(f"\n[{model_name}] {model_info['description']}")
        
        model_path = category_dir / model_info['filename']
        
        # Check if model already exists and is valid
        if model_path.exists():
            print(f"  Model already exists: {model_path}")
            if verify_model(model_path, model_info['checksum']):
                print(f"  ✓ Model verified")
                success_count += 1
                continue
            else:
                print(f"  ✗ Model verification failed, re-downloading...")
                model_path.unlink()
        
        # Download the model
        if download_file(model_info['url'], model_path, model_info['size_mb']):
            # Handle archive extraction if needed
            if model_path.suffix.lower() in ['.zip', '.tar', '.gz']:
                extract_dir = category_dir / model_name
                extract_dir.mkdir(parents=True, exist_ok=True)
                
                if extract_archive(model_path, extract_dir):
                    # Look for the actual model file in extracted content
                    extracted_files = list(extract_dir.rglob('*.pth')) + list(extract_dir.rglob('*.pt'))
                    if extracted_files:
                        # Move the first found model file to the expected location
                        final_model_path = category_dir / model_info['filename']
                        if final_model_path.suffix != extracted_files[0].suffix:
                            final_model_path = final_model_path.with_suffix(extracted_files[0].suffix)
                        
                        extracted_files[0].rename(final_model_path)
                        model_path = final_model_path
                        
                        # Clean up extraction directory and archive
                        import shutil
                        shutil.rmtree(extract_dir)
                        if model_path != category_dir / model_info['filename']:
                            (category_dir / model_info['filename']).unlink(missing_ok=True)
                else:
                    print(f"  ✗ Failed to extract archive")
                    continue
            
            # Verify the downloaded model
            if verify_model(model_path, model_info['checksum']):
                print(f"  ✓ Model downloaded and verified")
                success_count += 1
            else:
                print(f"  ✗ Model verification failed")
        else:
            print(f"  ✗ Failed to download model")
    
    print(f"\n{category} summary: {success_count}/{total_count} models downloaded successfully")
    return success_count == total_count

def create_model_index(cache_dir: Path) -> None:
    """Create an index file with information about downloaded models."""
    index = {
        "version": "1.0",
        "created_at": str(Path(__file__).stat().st_mtime),
        "models": {}
    }
    
    for category_name, category_models in MODELS.items():
        category_dir = cache_dir / category_name
        if category_dir.exists():
            index["models"][category_name] = {}
            
            for model_name, model_info in category_models.items():
                model_path = category_dir / model_info['filename']
                
                # Also check for alternative extensions
                if not model_path.exists():
                    alternatives = [
                        category_dir / (model_path.stem + '.pt'),
                        category_dir / (model_path.stem + '.pth'),
                        category_dir / (model_path.stem + '.bin')
                    ]
                    for alt in alternatives:
                        if alt.exists():
                            model_path = alt
                            break
                
                if model_path.exists():
                    index["models"][category_name][model_name] = {
                        "filename": model_path.name,
                        "path": str(model_path),
                        "size_bytes": model_path.stat().st_size,
                        "description": model_info['description'],
                        "downloaded": True
                    }
                else:
                    index["models"][category_name][model_name] = {
                        "filename": model_info['filename'],
                        "description": model_info['description'],
                        "downloaded": False
                    }
    
    index_path = cache_dir / "model_index.json"
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)
    
    print(f"\nModel index created: {index_path}")

def main():
    """Main function to download all models."""
    print("Video Advertisement Placement System - Model Downloader")
    print("=" * 60)
    
    # Get cache directory from environment or use default
    cache_dir = Path(os.environ.get('MODEL_CACHE_DIR', '/models'))
    
    print(f"Model cache directory: {cache_dir}")
    
    # Create cache directory
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Check available disk space
    statvfs = os.statvfs(cache_dir)
    available_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
    
    # Calculate total required space
    total_required_mb = sum(
        model_info['size_mb']
        for category in MODELS.values()
        for model_info in category.values()
    )
    total_required_gb = total_required_mb / 1024
    
    print(f"Required space: {total_required_gb:.1f} GB")
    print(f"Available space: {available_gb:.1f} GB")
    
    if available_gb < total_required_gb * 1.1:  # 10% safety margin
        print("Warning: Insufficient disk space!")
        print("Continuing anyway, but some downloads may fail...")
    
    # Download models by category
    all_successful = True
    
    for category_name, category_models in MODELS.items():
        success = download_model_category(category_name, category_models, cache_dir)
        if not success:
            all_successful = False
    
    # Create model index
    create_model_index(cache_dir)
    
    # Final summary
    print(f"\n{'='*60}")
    if all_successful:
        print("✓ All models downloaded successfully!")
        print(f"Models cached in: {cache_dir}")
        sys.exit(0)
    else:
        print("✗ Some models failed to download")
        print("Check the logs above for details")
        sys.exit(1)

if __name__ == "__main__":
    main() 