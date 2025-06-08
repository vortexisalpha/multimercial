# AI-Powered Video Advertisement Placement System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![CUDA 11.8+](https://img.shields.io/badge/CUDA-11.8+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive AI-powered system for intelligent advertisement placement in video content using advanced computer vision techniques including depth estimation, object detection, 3D reconstruction, and photorealistic rendering.

## 🏗️ Architecture Overview

The system employs a modular architecture with the following key components:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Video Input    │───▶│ Depth Estimation│───▶│ Object Detection│
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ 3D Rendering    │◀───│Scene Understanding│◀───│Camera Estimation│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ▼
┌─────────────────┐    ┌─────────────────┐
│Temporal Consist.│───▶│  Final Output   │
└─────────────────┘    └─────────────────┘
```

### Core Technologies
- **Depth Estimation**: Marigold and Depth Pro models
- **Object Detection**: YOLOv9 with ByteTrack tracking
- **3D Reconstruction**: Plane detection and wall identification
- **Rendering**: PBR (Physically Based Rendering) pipeline
- **Temporal Consistency**: Kalman filtering and stabilization

## 📁 Project Structure

```
video_ad_placement/
├── src/video_ad_placement/          # Main source code
│   ├── core/                        # Core modules
│   │   ├── depth_estimation.py      # Depth estimation pipeline
│   │   ├── object_detection.py      # YOLOv9 + ByteTrack
│   │   ├── scene_understanding.py   # Plane detection & analysis
│   │   ├── camera_estimation.py     # Camera parameters estimation
│   │   ├── rendering_engine.py      # PBR rendering & compositing
│   │   ├── temporal_consistency.py  # Temporal stabilization
│   │   └── video_processor.py       # Main processing pipeline
│   ├── utils/                       # Utility functions
│   ├── models/                      # Model definitions
│   └── __init__.py
├── configs/                         # Hydra configuration files
├── tests/                          # Unit and integration tests
├── docs/                           # Sphinx documentation
├── scripts/                        # Utility scripts
├── docker/                         # Docker containerization
├── requirements.txt                # Python dependencies
├── setup.py                        # Package setup
├── pyproject.toml                  # Project configuration
├── .pre-commit-config.yaml         # Pre-commit hooks
├── .github/workflows/              # CI/CD pipeline
└── README.md                       # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- CUDA 11.8+ compatible GPU
- Docker (optional)
- Git

### Installation

#### Option 1: Local Installation
```bash
# Clone the repository
git clone <repository-url>
cd video_ad_placement

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install

# Install package in development mode
pip install -e .
```

#### Option 2: Docker Installation
```bash
# Build the Docker image
docker build -f docker/Dockerfile -t video-ad-placement .

# Run with GPU support
docker run --gpus all -v $(pwd):/workspace video-ad-placement
```

## 🎯 Usage Examples

### Basic Video Processing
```python
from video_ad_placement.core.video_processor import VideoProcessor
from video_ad_placement.utils.config import load_config

# Load configuration
config = load_config("configs/default_config.yaml")

# Initialize processor
processor = VideoProcessor(config)

# Process video
result = processor.process_video(
    input_path="input_video.mp4",
    output_path="output_video.mp4",
    ad_assets=["ad1.png", "ad2.jpg"]
)

print(f"Processing completed: {result.success}")
```

### Custom Configuration
```python
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    processor = VideoProcessor(cfg)
    # Your processing logic here

if __name__ == "__main__":
    main()
```

### Advanced Pipeline Usage
```python
from video_ad_placement.core import (
    DepthEstimator, ObjectDetector, SceneAnalyzer,
    CameraEstimator, RenderingEngine
)

# Initialize individual components
depth_estimator = DepthEstimator(model_type="marigold")
object_detector = ObjectDetector(model_type="yolov9")
scene_analyzer = SceneAnalyzer()
camera_estimator = CameraEstimator()
renderer = RenderingEngine()

# Process frame by frame
for frame in video_frames:
    depth_map = depth_estimator.estimate(frame)
    objects = object_detector.detect(frame)
    planes = scene_analyzer.detect_planes(depth_map, frame)
    camera_params = camera_estimator.estimate(frame, objects)
    
    rendered_frame = renderer.composite_ads(
        frame, depth_map, objects, planes, camera_params, ad_assets
    )
```

## 🔧 Configuration

The system uses Hydra for configuration management. Main configuration files:

- `configs/config.yaml`: Main configuration
- `configs/models/`: Model-specific configs
- `configs/processing/`: Processing pipeline configs
- `configs/rendering/`: Rendering engine configs

Example configuration:
```yaml
# config.yaml
defaults:
  - models: default
  - processing: standard
  - rendering: pbr

video:
  input_format: mp4
  output_format: mp4
  fps: 30

models:
  depth_estimation:
    type: marigold
    checkpoint_path: checkpoints/marigold.pth
  
  object_detection:
    type: yolov9
    confidence_threshold: 0.5
```

## 🧪 Testing

Run the test suite:
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/video_ad_placement

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
```

## 📚 Documentation

Build and view documentation:
```bash
cd docs
make html
open _build/html/index.html
```

## 🐳 Docker Support

### GPU-Enabled Container
```bash
# Build image with GPU support
docker build -f docker/Dockerfile.gpu -t video-ad-placement:gpu .

# Run with GPU
docker run --gpus all -v $(pwd):/workspace video-ad-placement:gpu
```

### Development Container
```bash
# Build development image
docker build -f docker/Dockerfile.dev -t video-ad-placement:dev .

# Run development container
docker run -it -v $(pwd):/workspace video-ad-placement:dev bash
```

## 🔄 CI/CD Pipeline

The project includes GitHub Actions workflows for:
- Code quality checks (linting, formatting)
- Unit and integration testing
- Documentation building
- Docker image building and publishing

## 🛠️ Development

### Code Quality
The project enforces code quality through:
- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **pre-commit**: Automated checks

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and quality checks
5. Submit a pull request

## 📈 Performance Benchmarks

| Component | GPU Memory | Processing Time (1080p) |
|-----------|------------|-------------------------|
| Depth Estimation | 4GB | 0.5s/frame |
| Object Detection | 2GB | 0.1s/frame |
| 3D Reconstruction | 1GB | 0.3s/frame |
| Rendering | 3GB | 0.8s/frame |

## 🔍 Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce batch size in config
   - Use model quantization
   - Process smaller chunks

2. **Model loading errors**
   - Check checkpoint paths
   - Verify CUDA compatibility
   - Update model versions

3. **Video codec issues**
   - Install ffmpeg properly
   - Check input format support
   - Use compatible codecs

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Marigold depth estimation team
- YOLOv9 authors
- ByteTrack developers
- Open3D community
- Hydra configuration framework

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the documentation
- Review existing discussions

---

Built with ❤️ for intelligent video advertisement placement 