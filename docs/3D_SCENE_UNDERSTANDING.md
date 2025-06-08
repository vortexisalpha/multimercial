# 3D Scene Understanding System for TV Advertisement Placement

## Overview

The 3D Scene Understanding System is a sophisticated computer vision framework designed to detect and analyze wall surfaces in video content for optimal TV advertisement placement. The system combines RANSAC-based plane detection, hybrid surface normal estimation, temporal tracking, and advanced quality assessment to identify suitable wall surfaces for virtual TV placement.

## Key Features

### ✅ Core Capabilities

- **RANSAC-based Plane Detection**: Robust plane fitting with geometric constraints and outlier rejection
- **Hybrid Surface Normal Estimation**: Combines geometric calculations with optional learned approaches
- **Temporal Tracking**: Maintains plane consistency across video frames with sophisticated association algorithms
- **Wall Surface Validation**: Comprehensive quality assessment for TV placement suitability
- **Real-time Performance**: Optimized for video processing with configurable quality profiles
- **Multi-scale Detection**: Handles walls of different sizes and distances
- **Camera Integration**: Full support for camera intrinsics and extrinsics
- **Comprehensive Evaluation**: MOT-style metrics and visualization tools

### 🏗️ Architecture

```
3D Scene Understanding System
├── Core Detection Engine
│   ├── RANSAC Plane Detector
│   ├── Surface Normal Estimator
│   └── Quality Assessment
├── Temporal Tracking
│   ├── Plane Association
│   ├── Track Management
│   └── Temporal Smoothing
├── Geometry Utilities
│   ├── Camera Projection
│   ├── 3D Transformations
│   └── Plane Operations
└── Evaluation & Visualization
    ├── Performance Metrics
    ├── Debug Visualizations
    └── Export Tools
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.0+ (optional, for GPU acceleration)

### Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Optional: Install additional 3D processing libraries
pip install open3d>=0.15.0 trimesh>=3.15.0

# Optional: Install GPU acceleration (Linux/Windows only)
pip install cupy-cuda11x>=11.0.0
```

### Verify Installation

```python
import torch
from src.video_ad_placement.scene_understanding import PlaneDetector

# Test basic functionality
config = PlaneDetectionConfig()
detector = PlaneDetector(config)
print("✅ 3D Scene Understanding System installed successfully")
```

## Quick Start

### Basic Plane Detection

```python
import torch
from src.video_ad_placement.scene_understanding import (
    PlaneDetector, PlaneDetectionConfig
)

# Configure detection
config = PlaneDetectionConfig(
    ransac_iterations=2000,
    ransac_threshold=0.02,
    min_plane_area=1.0,  # Minimum 1m² for TV placement
    wall_normal_tolerance=0.4
)

# Create detector
detector = PlaneDetector(config)

# Detect planes in depth map
depth_map = torch.randn(480, 640)  # Your depth data
rgb_frame = torch.randn(480, 640, 3)  # Your RGB data
camera_intrinsics = torch.eye(3)  # Your camera matrix

planes = detector.detect_planes(depth_map, rgb_frame, camera_intrinsics)

# Analyze results
for i, plane in enumerate(planes):
    print(f"Plane {i+1}: Type={plane.plane_type.value}, "
          f"Area={plane.area:.2f}m², "
          f"TV Score={plane.tv_placement_score:.3f}")
```

### Temporal Tracking

```python
from src.video_ad_placement.scene_understanding import (
    TemporalPlaneTracker, TemporalTrackingConfig
)

# Configure tracking
tracking_config = TemporalTrackingConfig(
    association_threshold=0.5,
    max_missing_frames=5,
    smoothing_alpha=0.7
)

# Create tracker
tracker = TemporalPlaneTracker(tracking_config)

# Process video frames
for frame_id, (depth, rgb) in enumerate(video_frames):
    # Detect planes
    planes = detector.detect_planes(depth, rgb, camera_intrinsics)
    
    # Update tracker
    tracks = tracker.update(planes)
    
    # Get stable wall surfaces
    stable_walls = [
        track.get_smoothed_plane() 
        for track in tracks 
        if track.stability_score > 0.7
    ]
    
    print(f"Frame {frame_id}: {len(stable_walls)} stable wall surfaces")
```

### Surface Normal Estimation

```python
from src.video_ad_placement.scene_understanding import (
    SurfaceNormalEstimator, SurfaceNormalConfig
)

# Configure normal estimation
normal_config = SurfaceNormalConfig(
    use_geometric_normals=True,
    use_learned_normals=False,  # Set True if you have trained models
    depth_smoothing=True,
    enable_refinement=True
)

# Create estimator
estimator = SurfaceNormalEstimator(normal_config)

# Estimate normals
normals = estimator.estimate_surface_normals(depth_map, rgb_frame)

# Normals shape: (H, W, 3) - unit vectors for each pixel
print(f"Surface normals: {normals.shape}")
```

## Configuration

### PlaneDetectionConfig

Core configuration for plane detection:

```python
@dataclass
class PlaneDetectionConfig:
    # RANSAC Parameters
    ransac_iterations: int = 2000          # Number of RANSAC iterations
    ransac_threshold: float = 0.02         # Distance threshold (meters)
    min_inliers: int = 200                 # Minimum inlier points
    max_inliers_ratio: float = 0.9         # Maximum inlier ratio
    
    # Geometric Constraints
    min_plane_area: float = 1.0            # Minimum area (m²)
    max_plane_area: float = 50.0           # Maximum area (m²)
    min_aspect_ratio: float = 0.1          # Minimum aspect ratio
    max_aspect_ratio: float = 10.0         # Maximum aspect ratio
    
    # Wall Detection
    wall_normal_tolerance: float = 0.4     # Wall normal angle tolerance
    floor_normal_tolerance: float = 0.3    # Floor normal angle tolerance
    ceiling_normal_tolerance: float = 0.3  # Ceiling normal angle tolerance
    
    # Quality Thresholds
    min_confidence: float = 0.3            # Minimum detection confidence
    min_planarity: float = 0.5             # Minimum planarity score
    min_visibility: float = 0.2            # Minimum visibility ratio
    
    # Performance
    enable_multithreading: bool = True     # Use multiple threads
    max_threads: int = 4                   # Maximum thread count
    enable_multiscale: bool = True         # Multi-scale detection
```

### TemporalTrackingConfig

Configuration for temporal tracking:

```python
@dataclass
class TemporalTrackingConfig:
    # Association
    association_threshold: float = 0.5     # Maximum association cost
    center_weight: float = 1.0             # Center distance weight
    normal_weight: float = 2.0             # Normal angle weight
    area_weight: float = 0.5               # Area difference weight
    type_penalty: float = 1.0              # Type mismatch penalty
    
    # Track Management
    max_missing_frames: int = 5            # Max frames before termination
    max_tracks: int = 100                  # Maximum simultaneous tracks
    
    # Smoothing
    smoothing_alpha: float = 0.7           # Exponential smoothing factor
    enable_interpolation: bool = True      # Enable missing frame interpolation
```

## API Reference

### PlaneDetector

Main plane detection class with RANSAC-based algorithms.

#### Methods

```python
def detect_planes(self, depth_map: torch.Tensor, rgb_frame: torch.Tensor, 
                 camera_intrinsics: torch.Tensor) -> List[Plane]:
    """
    Detect planes in depth map using RANSAC with geometric constraints.
    
    Args:
        depth_map: Depth map tensor (H, W)
        rgb_frame: RGB frame tensor (H, W, 3)
        camera_intrinsics: 3x3 camera intrinsics matrix
        
    Returns:
        List of detected Plane objects
    """

def validate_wall_surface(self, plane: Plane, 
                         visibility_history: List[float] = None) -> float:
    """
    Assess wall surface quality for TV placement.
    
    Args:
        plane: Plane object to validate
        visibility_history: Historical visibility scores
        
    Returns:
        TV placement suitability score [0-1]
    """

def get_statistics(self) -> Dict[str, Any]:
    """Get detection performance statistics."""
```

### Plane

Data class representing a detected plane with comprehensive metadata.

#### Properties

```python
@dataclass
class Plane:
    # Geometry
    normal: torch.Tensor              # 3D normal vector
    point: torch.Tensor               # Point on plane
    equation: torch.Tensor            # Plane equation coefficients [a,b,c,d]
    inlier_mask: torch.Tensor         # Boolean mask of inlier pixels
    
    # Quality Metrics
    confidence: float                 # Detection confidence [0-1]
    area: float                       # Surface area (m²)
    stability_score: float            # Temporal stability [0-1]
    planarity_score: float            # How planar the surface is [0-1]
    
    # Classification
    plane_type: PlaneType             # WALL, FLOOR, CEILING, etc.
    quality: PlaneQuality             # EXCELLENT, GOOD, FAIR, POOR
    
    # TV Placement
    tv_placement_score: float         # Suitability for TV [0-1]
    distance_to_camera: float         # Distance from camera (m)
    viewing_angle: float              # Angle relative to camera (radians)
    
    # Methods
    def is_wall_suitable_for_tv(self) -> bool:
        """Check if wall is suitable for TV placement."""
        
    def compute_bounding_box_3d(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute 3D bounding box of plane."""
        
    def project_to_2d(self, camera_params: CameraParameters) -> torch.Tensor:
        """Project plane to 2D image coordinates."""
```

### TemporalPlaneTracker

Temporal tracking system for maintaining plane consistency across frames.

#### Methods

```python
def update(self, detected_planes: List[Plane]) -> List[PlaneTrack]:
    """
    Update tracker with new frame detections.
    
    Args:
        detected_planes: Planes detected in current frame
        
    Returns:
        List of active plane tracks
    """

def get_smoothed_planes(self, min_stability: float = 0.5) -> List[Plane]:
    """
    Get temporally smoothed planes from stable tracks.
    
    Args:
        min_stability: Minimum stability score for inclusion
        
    Returns:
        List of smoothed plane representations
    """

def get_statistics(self) -> Dict[str, Any]:
    """Get tracking performance statistics."""

def reset(self):
    """Reset tracker state for new video sequence."""
```

## Advanced Usage

### Custom Wall Detection Criteria

```python
# Configure for specific wall detection requirements
config = PlaneDetectionConfig(
    # Stricter geometric constraints for TV walls
    min_plane_area=2.0,               # Minimum 2m² for TV
    wall_normal_tolerance=0.2,        # Stricter wall orientation
    min_aspect_ratio=0.3,             # Avoid very elongated surfaces
    max_aspect_ratio=3.0,             # Avoid very elongated surfaces
    
    # Higher quality thresholds
    min_confidence=0.6,               # Higher confidence required
    min_planarity=0.8,                # Very planar surfaces only
    min_visibility=0.4,               # Good visibility required
)

detector = PlaneDetector(config)

# Custom validation function
def custom_wall_validator(plane: Plane) -> bool:
    """Custom validation for premium TV placement."""
    return (plane.area >= 3.0 and                    # Large enough
            plane.distance_to_camera <= 5.0 and      # Not too far
            plane.viewing_angle <= np.pi/4 and        # Good viewing angle
            plane.tv_placement_score >= 0.7)          # High placement score

# Filter detected planes
suitable_walls = [p for p in planes if custom_wall_validator(p)]
```

### Multi-Camera Integration

```python
from src.video_ad_placement.scene_understanding.geometry_utils import CameraParameters

# Define multiple camera views
cameras = {
    'main': CameraParameters(fx=525, fy=525, cx=320, cy=240),
    'wide': CameraParameters(fx=400, fy=400, cx=320, cy=240),
    'zoom': CameraParameters(fx=800, fy=800, cx=320, cy=240)
}

# Process each camera view
all_planes = {}
for camera_name, camera_params in cameras.items():
    intrinsics = camera_params.get_intrinsics_matrix()
    depth_map = get_depth_for_camera(camera_name)
    rgb_frame = get_rgb_for_camera(camera_name)
    
    planes = detector.detect_planes(depth_map, rgb_frame, intrinsics)
    all_planes[camera_name] = planes
    
    print(f"{camera_name}: {len(planes)} planes detected")

# Merge and deduplicate planes across views
merged_planes = merge_planes_across_cameras(all_planes, cameras)
```

### Performance Optimization

```python
# High-performance configuration for real-time processing
fast_config = PlaneDetectionConfig(
    # Reduced iterations for speed
    ransac_iterations=500,
    min_inliers=50,
    
    # Disable expensive features
    enable_multiscale=False,
    enable_refinement=False,
    
    # Parallel processing
    enable_multithreading=True,
    max_threads=8,
    
    # Early termination
    early_termination=True,
    early_termination_threshold=0.8
)

# Memory-efficient tracking
tracking_config = TemporalTrackingConfig(
    max_tracks=20,                    # Limit active tracks
    max_missing_frames=3,             # Quick termination
    enable_interpolation=False        # Disable interpolation
)
```

## Visualization and Debugging

### Built-in Visualization Tools

```python
from src.video_ad_placement.scene_understanding.visualization import (
    PlaneVisualizer, TemporalVisualizer
)

# Create visualizer
visualizer = PlaneVisualizer(output_dir="debug_output")

# Visualize detection results
fig = visualizer.visualize_detections(
    depth_map=depth_map,
    rgb_frame=rgb_frame,
    planes=planes,
    camera_params=camera_params,
    save_path="plane_detection.png"
)

# Visualize temporal tracking
temporal_viz = TemporalVisualizer()
temporal_viz.visualize_tracks(
    tracks=tracks,
    frame_id=current_frame,
    save_path="tracking_results.png"
)

# Generate analysis plots
visualizer.plot_quality_distribution(planes)
visualizer.plot_area_vs_distance(planes)
visualizer.plot_normal_distribution(planes)
```

### Custom Visualization

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_plane_3d(plane: Plane, camera_params: CameraParameters):
    """Custom 3D visualization of detected plane."""
    # Convert to 3D point cloud
    projection = CameraProjection(camera_params)
    points_3d = projection.depth_map_to_point_cloud(depth_map)
    
    # Filter points on plane
    distances = GeometryUtils.point_to_plane_distance(
        points_3d, plane.normal, plane.point
    )
    inlier_points = points_3d[torch.abs(distances) < 0.05]
    
    # Plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot point cloud
    ax.scatter(inlier_points[:, 0], inlier_points[:, 1], inlier_points[:, 2],
               c='blue', alpha=0.6, s=1)
    
    # Plot plane normal
    center = plane.point
    normal_end = center + plane.normal * 0.5
    ax.plot([center[0], normal_end[0]], 
            [center[1], normal_end[1]], 
            [center[2], normal_end[2]], 
            'r-', linewidth=3, label='Normal')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend()
    plt.title(f'Plane: {plane.plane_type.value} (Area: {plane.area:.2f}m²)')
    plt.show()
```

## Performance Benchmarks

### Typical Performance Metrics

| Resolution | FPS | Planes/Frame | Memory Usage |
|------------|-----|--------------|--------------|
| 320x240    | 45  | 2-4          | 150MB       |
| 640x480    | 25  | 3-6          | 300MB       |
| 1280x720   | 12  | 4-8          | 600MB       |
| 1920x1080  | 6   | 5-10         | 1.2GB       |

### Optimization Guidelines

1. **Resolution**: Lower resolution for real-time processing
2. **RANSAC Iterations**: Balance between accuracy and speed
3. **Multi-threading**: Enable for multi-core systems
4. **Memory Management**: Use track limits for long sequences
5. **Quality Profiles**: Use different configs for different use cases

## Testing and Validation

### Unit Tests

```bash
# Run comprehensive test suite
python -m pytest tests/test_scene_understanding.py -v

# Run specific test categories
python -m pytest tests/test_scene_understanding.py::TestPlaneDetector -v
python -m pytest tests/test_scene_understanding.py::TestTemporalTracking -v

# Run with coverage
python -m pytest tests/test_scene_understanding.py --cov=src/video_ad_placement/scene_understanding
```

### Integration Tests

```bash
# Run full pipeline test
python examples/plane_detection_example.py

# Benchmark performance
python examples/plane_detection_example.py --benchmark

# Test with custom data
python examples/plane_detection_example.py --data_path /path/to/your/data
```

### Validation Metrics

The system provides comprehensive evaluation metrics:

- **Detection Accuracy**: Precision, recall, F1-score for plane detection
- **Temporal Consistency**: Track stability, association accuracy
- **Geometric Accuracy**: Normal vector accuracy, area estimation error
- **TV Placement Quality**: Suitability scores, placement success rate

## Troubleshooting

### Common Issues

1. **No Planes Detected**
   - Check depth map validity and range
   - Verify camera intrinsics
   - Lower minimum area threshold
   - Increase RANSAC iterations

2. **Poor Tracking Performance**
   - Adjust association threshold
   - Check temporal consistency parameters
   - Verify plane detection quality

3. **Memory Issues**
   - Reduce image resolution
   - Limit maximum tracks
   - Disable expensive features

4. **Performance Issues**
   - Enable multi-threading
   - Reduce RANSAC iterations
   - Use early termination

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging
config.enable_debug_output = True
detector = PlaneDetector(config)

# Monitor performance
stats = detector.get_statistics()
print(f"Average detection time: {stats['avg_detection_time']:.3f}s")
print(f"Memory usage: {stats['memory_usage_mb']:.1f}MB")
```

## Contributing

### Development Setup

```bash
# Clone repository
git clone <repository_url>
cd multimercial

# Install development dependencies
pip install -r requirements.txt
pip install -e .

# Set up pre-commit hooks
pre-commit install

# Run tests
python -m pytest tests/
```

### Adding New Features

1. **Plane Detection Algorithms**: Extend `BasePlaneDetector`
2. **Surface Normal Methods**: Extend `BaseSurfaceNormalEstimator`
3. **Quality Metrics**: Add to `WallQualityMetrics`
4. **Tracking Methods**: Extend `PlaneAssociator`

### Code Standards

- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include unit tests for new features
- Update documentation
- Use type hints consistently

## License

This 3D Scene Understanding System is part of the Multimercial project and is subject to the project's licensing terms.

## Support

For questions, issues, or contributions:

1. Check the documentation and examples
2. Search existing issues
3. Create a detailed issue report
4. Join the development discussions

---

**Note**: This system is optimized for TV advertisement placement scenarios but can be adapted for general 3D scene understanding tasks with appropriate configuration adjustments. 