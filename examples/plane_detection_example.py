"""
Comprehensive 3D Plane Detection Example for Wall Surface Analysis

This example demonstrates the complete plane detection system for detecting
and analyzing wall surfaces suitable for TV advertisement placement, including
RANSAC-based detection, surface normal estimation, and quality assessment.
"""

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import logging
import time
from pathlib import Path
from typing import List, Dict, Any

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from video_ad_placement.scene_understanding.plane_detector import PlaneDetector
from video_ad_placement.scene_understanding.surface_normal_estimator import SurfaceNormalEstimator
from video_ad_placement.scene_understanding.plane_models import (
    PlaneDetectionConfig, SurfaceNormalConfig, Plane, PlaneType, PlaneQuality
)
from video_ad_placement.scene_understanding.geometry_utils import (
    CameraParameters, CameraProjection, GeometryUtils
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockSceneGenerator:
    """Generate mock 3D scenes with walls for testing."""
    
    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        
        # Camera parameters
        self.camera_params = CameraParameters(
            fx=525.0, fy=525.0,
            cx=width / 2, cy=height / 2
        )
    
    def generate_room_scene(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate a simple room scene with walls, floor, and ceiling."""
        depth_map = torch.zeros((self.height, self.width))
        rgb_frame = torch.zeros((self.height, self.width, 3), dtype=torch.uint8)
        
        # Create a simple room geometry
        # Floor (y = bottom of image, depth varies)
        floor_depth = 3.0
        floor_region = slice(int(self.height * 0.7), self.height)
        depth_map[floor_region, :] = floor_depth
        rgb_frame[floor_region, :] = torch.tensor([100, 100, 100])  # Gray floor
        
        # Ceiling (y = top of image)
        ceiling_depth = 3.0
        ceiling_region = slice(0, int(self.height * 0.2))
        depth_map[ceiling_region, :] = ceiling_depth
        rgb_frame[ceiling_region, :] = torch.tensor([200, 200, 200])  # Light gray ceiling
        
        # Left wall
        left_wall_depth = 4.0
        left_wall_region = (slice(int(self.height * 0.2), int(self.height * 0.7)), 
                           slice(0, int(self.width * 0.3)))
        depth_map[left_wall_region] = left_wall_depth
        rgb_frame[left_wall_region] = torch.tensor([150, 120, 100])  # Brown wall
        
        # Right wall
        right_wall_depth = 3.5
        right_wall_region = (slice(int(self.height * 0.2), int(self.height * 0.7)), 
                            slice(int(self.width * 0.7), self.width))
        depth_map[right_wall_region] = right_wall_depth
        rgb_frame[right_wall_region] = torch.tensor([120, 150, 100])  # Green wall
        
        # Back wall (center)
        back_wall_depth = 5.0
        back_wall_region = (slice(int(self.height * 0.2), int(self.height * 0.7)), 
                           slice(int(self.width * 0.3), int(self.width * 0.7)))
        depth_map[back_wall_region] = back_wall_depth
        rgb_frame[back_wall_region] = torch.tensor([100, 100, 150])  # Blue wall
        
        # Add some noise for realism
        noise = torch.randn_like(depth_map) * 0.05
        depth_map = depth_map + noise
        depth_map = torch.clamp(depth_map, 0.1, 10.0)
        
        # Add RGB noise
        rgb_noise = torch.randint(-10, 10, rgb_frame.shape, dtype=torch.int16)
        rgb_frame = torch.clamp(rgb_frame.int() + rgb_noise, 0, 255).byte()
        
        return depth_map, rgb_frame
    
    def generate_complex_scene(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate a more complex scene with furniture and occlusions."""
        depth_map, rgb_frame = self.generate_room_scene()
        
        # Add furniture (closer objects that occlude walls)
        # Table
        table_region = (slice(int(self.height * 0.5), int(self.height * 0.65)), 
                       slice(int(self.width * 0.4), int(self.width * 0.6)))
        table_depth = 2.0
        depth_map[table_region] = table_depth
        rgb_frame[table_region] = torch.tensor([139, 69, 19])  # Brown table
        
        # Chair
        chair_region = (slice(int(self.height * 0.45), int(self.height * 0.6)), 
                       slice(int(self.width * 0.2), int(self.width * 0.35)))
        chair_depth = 2.2
        depth_map[chair_region] = chair_depth
        rgb_frame[chair_region] = torch.tensor([80, 80, 80])  # Gray chair
        
        # Picture frame on wall (slight depth variation)
        frame_region = (slice(int(self.height * 0.25), int(self.height * 0.4)), 
                       slice(int(self.width * 0.75), int(self.width * 0.9)))
        frame_depth = depth_map[frame_region] - 0.05  # Slightly closer than wall
        depth_map[frame_region] = frame_depth
        rgb_frame[frame_region] = torch.tensor([255, 215, 0])  # Gold frame
        
        return depth_map, rgb_frame


def create_plane_detection_config() -> PlaneDetectionConfig:
    """Create optimized plane detection configuration."""
    return PlaneDetectionConfig(
        # RANSAC parameters
        ransac_iterations=2000,
        ransac_threshold=0.02,
        min_inliers=200,
        max_inliers_ratio=0.9,
        
        # Multi-scale detection
        enable_multiscale=True,
        scale_factors=[1.0, 0.5],
        merge_threshold=0.15,
        
        # Geometric constraints
        min_plane_area=1.0,  # Minimum 1 m² for walls
        max_plane_area=20.0,
        min_aspect_ratio=0.2,
        max_aspect_ratio=5.0,
        
        # Normal vector constraints for room detection
        wall_normal_tolerance=0.4,
        floor_normal_tolerance=0.3,
        ceiling_normal_tolerance=0.3,
        
        # Quality thresholds
        min_confidence=0.4,
        min_planarity=0.6,
        min_visibility=0.3,
        
        # Performance optimization
        enable_multithreading=True,
        max_threads=2,
        early_termination=True,
        early_termination_threshold=0.9,
        
        # Memory management
        max_planes_per_frame=10,
        max_points_per_plane=5000
    )


def create_surface_normal_config() -> SurfaceNormalConfig:
    """Create surface normal estimation configuration."""
    return SurfaceNormalConfig(
        # Use geometric normals (no learned model for demo)
        use_geometric_normals=True,
        use_learned_normals=False,
        geometric_weight=1.0,
        learned_weight=0.0,
        
        # Geometric normal estimation
        neighborhood_size=5,
        depth_smoothing=True,
        smoothing_sigma=1.0,
        
        # Normal refinement
        enable_refinement=True,
        refinement_iterations=2,
        consistency_threshold=0.1,
        
        # Quality filtering
        confidence_threshold=0.5,
        gradient_threshold=0.2
    )


def demonstrate_basic_plane_detection():
    """Demonstrate basic plane detection on a simple room scene."""
    print("\n" + "="*60)
    print("BASIC PLANE DETECTION DEMONSTRATION")
    print("="*60)
    
    # Create mock scene
    scene_generator = MockSceneGenerator(width=640, height=480)
    depth_map, rgb_frame = scene_generator.generate_room_scene()
    
    print(f"Generated scene: {depth_map.shape} depth, {rgb_frame.shape} RGB")
    print(f"Depth range: {depth_map.min():.3f} - {depth_map.max():.3f} meters")
    
    # Create detector
    config = create_plane_detection_config()
    detector = PlaneDetector(config)
    
    # Get camera intrinsics
    camera_intrinsics = scene_generator.camera_params.get_intrinsics_matrix()
    
    # Detect planes
    print("Detecting planes...")
    start_time = time.time()
    planes = detector.detect_planes(depth_map, rgb_frame.float(), camera_intrinsics)
    detection_time = time.time() - start_time
    
    print(f"Detected {len(planes)} planes in {detection_time:.3f} seconds")
    
    # Analyze detected planes
    for i, plane in enumerate(planes):
        print(f"\nPlane {i+1}:")
        print(f"  Type: {plane.plane_type.value}")
        print(f"  Area: {plane.area:.2f} m²")
        print(f"  Confidence: {plane.confidence:.3f}")
        print(f"  Planarity: {plane.planarity_score:.3f}")
        print(f"  TV Placement Score: {plane.tv_placement_score:.3f}")
        print(f"  Distance to camera: {plane.distance_to_camera:.2f} m")
        print(f"  Normal vector: [{plane.normal[0]:.3f}, {plane.normal[1]:.3f}, {plane.normal[2]:.3f}]")
        
        # Check if suitable for TV placement
        if plane.is_wall_suitable_for_tv():
            print(f"  ✓ SUITABLE FOR TV PLACEMENT")
        else:
            print(f"  ✗ Not suitable for TV placement")
    
    # Get performance statistics
    stats = detector.get_statistics()
    print(f"\nPerformance Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    return planes, depth_map, rgb_frame, scene_generator


def demonstrate_surface_normal_estimation():
    """Demonstrate surface normal estimation."""
    print("\n" + "="*60)
    print("SURFACE NORMAL ESTIMATION DEMONSTRATION")
    print("="*60)
    
    # Create mock scene
    scene_generator = MockSceneGenerator(width=320, height=240)  # Smaller for faster processing
    depth_map, rgb_frame = scene_generator.generate_room_scene()
    
    # Create surface normal estimator
    config = create_surface_normal_config()
    estimator = SurfaceNormalEstimator(config)
    
    # Estimate surface normals
    print("Estimating surface normals...")
    start_time = time.time()
    normals = estimator.estimate_surface_normals(depth_map, rgb_frame.float())
    estimation_time = time.time() - start_time
    
    print(f"Estimated normals for {normals.shape[0]}x{normals.shape[1]} image in {estimation_time:.3f} seconds")
    
    # Analyze normal statistics
    valid_mask = torch.norm(normals, dim=2) > 0.5
    valid_normals = normals[valid_mask]
    
    if len(valid_normals) > 0:
        print(f"Valid normals: {len(valid_normals)}/{normals.numel()//3} ({100*len(valid_normals)/(normals.numel()//3):.1f}%)")
        print(f"Mean normal: [{valid_normals.mean(0)[0]:.3f}, {valid_normals.mean(0)[1]:.3f}, {valid_normals.mean(0)[2]:.3f}]")
        print(f"Normal std: [{valid_normals.std(0)[0]:.3f}, {valid_normals.std(0)[1]:.3f}, {valid_normals.std(0)[2]:.3f}]")
    
    return normals, depth_map, rgb_frame


def demonstrate_complex_scene_analysis():
    """Demonstrate plane detection on complex scene with furniture."""
    print("\n" + "="*60)
    print("COMPLEX SCENE ANALYSIS DEMONSTRATION")
    print("="*60)
    
    # Create complex scene
    scene_generator = MockSceneGenerator(width=640, height=480)
    depth_map, rgb_frame = scene_generator.generate_complex_scene()
    
    # Create detector with adjusted parameters for complex scenes
    config = create_plane_detection_config()
    config.min_inliers = 150  # Lower threshold due to occlusions
    config.ransac_iterations = 3000  # More iterations for robustness
    
    detector = PlaneDetector(config)
    camera_intrinsics = scene_generator.camera_params.get_intrinsics_matrix()
    
    # Detect planes
    print("Analyzing complex scene...")
    start_time = time.time()
    planes = detector.detect_planes(depth_map, rgb_frame.float(), camera_intrinsics)
    detection_time = time.time() - start_time
    
    print(f"Detected {len(planes)} planes in complex scene in {detection_time:.3f} seconds")
    
    # Categorize planes
    walls = [p for p in planes if p.plane_type == PlaneType.WALL]
    floors = [p for p in planes if p.plane_type == PlaneType.FLOOR]
    ceilings = [p for p in planes if p.plane_type == PlaneType.CEILING]
    furniture = [p for p in planes if p.plane_type == PlaneType.FURNITURE]
    unknown = [p for p in planes if p.plane_type == PlaneType.UNKNOWN]
    
    print(f"\nPlane categorization:")
    print(f"  Walls: {len(walls)}")
    print(f"  Floors: {len(floors)}")
    print(f"  Ceilings: {len(ceilings)}")
    print(f"  Furniture: {len(furniture)}")
    print(f"  Unknown: {len(unknown)}")
    
    # Find best wall for TV placement
    suitable_walls = [w for w in walls if w.is_wall_suitable_for_tv()]
    if suitable_walls:
        best_wall = max(suitable_walls, key=lambda w: w.tv_placement_score)
        print(f"\nBest wall for TV placement:")
        print(f"  Area: {best_wall.area:.2f} m²")
        print(f"  TV Placement Score: {best_wall.tv_placement_score:.3f}")
        print(f"  Distance: {best_wall.distance_to_camera:.2f} m")
        print(f"  Quality: {best_wall.quality.value}")
    else:
        print(f"\nNo suitable walls found for TV placement")
    
    return planes, depth_map, rgb_frame, scene_generator


def visualize_plane_detection_results(planes: List[Plane], depth_map: torch.Tensor, 
                                    rgb_frame: torch.Tensor, scene_generator: MockSceneGenerator):
    """Visualize plane detection results."""
    print("\n" + "="*60)
    print("VISUALIZATION")
    print("="*60)
    
    try:
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('3D Plane Detection Results', fontsize=16)
        
        # Original RGB
        axes[0, 0].imshow(rgb_frame.numpy())
        axes[0, 0].set_title('Original RGB')
        axes[0, 0].axis('off')
        
        # Depth map
        depth_vis = axes[0, 1].imshow(depth_map.numpy(), cmap='viridis')
        axes[0, 1].set_title('Depth Map')
        axes[0, 1].axis('off')
        plt.colorbar(depth_vis, ax=axes[0, 1], label='Depth (m)')
        
        # Plane segmentation
        plane_mask = torch.zeros_like(depth_map)
        colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'orange', 'purple']
        
        for i, plane in enumerate(planes):
            if torch.any(plane.inlier_mask):
                plane_mask[plane.inlier_mask] = i + 1
        
        plane_vis = axes[0, 2].imshow(plane_mask.numpy(), cmap='tab10')
        axes[0, 2].set_title(f'Detected Planes ({len(planes)})')
        axes[0, 2].axis('off')
        
        # Plane types
        type_mask = torch.zeros_like(depth_map)
        type_mapping = {
            PlaneType.WALL: 1,
            PlaneType.FLOOR: 2, 
            PlaneType.CEILING: 3,
            PlaneType.FURNITURE: 4,
            PlaneType.UNKNOWN: 5
        }
        
        for plane in planes:
            if torch.any(plane.inlier_mask):
                type_mask[plane.inlier_mask] = type_mapping[plane.plane_type]
        
        type_vis = axes[1, 0].imshow(type_mask.numpy(), cmap='Set1')
        axes[1, 0].set_title('Plane Types')
        axes[1, 0].axis('off')
        
        # Quality visualization
        quality_mask = torch.zeros_like(depth_map)
        quality_mapping = {
            PlaneQuality.EXCELLENT: 5,
            PlaneQuality.GOOD: 4,
            PlaneQuality.FAIR: 3,
            PlaneQuality.POOR: 2,
            PlaneQuality.UNUSABLE: 1
        }
        
        for plane in planes:
            if torch.any(plane.inlier_mask):
                quality_mask[plane.inlier_mask] = quality_mapping[plane.quality]
        
        quality_vis = axes[1, 1].imshow(quality_mask.numpy(), cmap='RdYlGn')
        axes[1, 1].set_title('Plane Quality')
        axes[1, 1].axis('off')
        
        # TV placement suitability
        tv_mask = torch.zeros_like(depth_map)
        for plane in planes:
            if torch.any(plane.inlier_mask):
                tv_mask[plane.inlier_mask] = plane.tv_placement_score
        
        tv_vis = axes[1, 2].imshow(tv_mask.numpy(), cmap='Reds')
        axes[1, 2].set_title('TV Placement Suitability')
        axes[1, 2].axis('off')
        plt.colorbar(tv_vis, ax=axes[1, 2], label='Placement Score')
        
        plt.tight_layout()
        
        # Save visualization
        output_dir = Path("plane_detection_output")
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / "plane_detection_results.png", dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_dir / 'plane_detection_results.png'}")
        
        # Show plot if in interactive environment
        try:
            plt.show()
        except:
            print("Unable to display plot interactively")
        
        return fig
        
    except Exception as e:
        print(f"Visualization failed: {e}")
        return None


def demonstrate_temporal_consistency():
    """Demonstrate temporal consistency across multiple frames."""
    print("\n" + "="*60)
    print("TEMPORAL CONSISTENCY DEMONSTRATION")
    print("="*60)
    
    scene_generator = MockSceneGenerator(width=480, height=360)
    config = create_plane_detection_config()
    detector = PlaneDetector(config)
    camera_intrinsics = scene_generator.camera_params.get_intrinsics_matrix()
    
    # Generate sequence of frames with slight variations
    num_frames = 5
    all_planes = []
    
    print(f"Processing {num_frames} frames for temporal analysis...")
    
    for frame_idx in range(num_frames):
        # Generate scene with slight noise variation
        depth_map, rgb_frame = scene_generator.generate_room_scene()
        
        # Add frame-specific noise
        noise_scale = 0.02 + frame_idx * 0.01
        depth_noise = torch.randn_like(depth_map) * noise_scale
        depth_map = torch.clamp(depth_map + depth_noise, 0.1, 10.0)
        
        # Detect planes
        planes = detector.detect_planes(depth_map, rgb_frame.float(), camera_intrinsics)
        all_planes.append(planes)
        
        print(f"  Frame {frame_idx + 1}: {len(planes)} planes detected")
    
    # Analyze temporal consistency
    print(f"\nTemporal Analysis:")
    print(f"  Frames processed: {num_frames}")
    
    # Count planes by type across frames
    type_counts = {ptype: [] for ptype in PlaneType}
    quality_scores = []
    
    for frame_planes in all_planes:
        frame_type_count = {ptype: 0 for ptype in PlaneType}
        frame_quality = []
        
        for plane in frame_planes:
            frame_type_count[plane.plane_type] += 1
            frame_quality.append(plane.tv_placement_score)
        
        for ptype, count in frame_type_count.items():
            type_counts[ptype].append(count)
        
        quality_scores.append(np.mean(frame_quality) if frame_quality else 0.0)
    
    # Report consistency metrics
    for ptype, counts in type_counts.items():
        if any(c > 0 for c in counts):
            mean_count = np.mean(counts)
            std_count = np.std(counts)
            print(f"  {ptype.value.title()} planes: {mean_count:.1f} ± {std_count:.1f}")
    
    print(f"  Average quality score: {np.mean(quality_scores):.3f} ± {np.std(quality_scores):.3f}")
    
    return all_planes


def benchmark_performance():
    """Benchmark detection performance on different scene sizes."""
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARK")
    print("="*60)
    
    config = create_plane_detection_config()
    
    # Test different image sizes
    sizes = [(320, 240), (640, 480), (960, 720)]
    results = {}
    
    for width, height in sizes:
        print(f"\nBenchmarking {width}x{height} resolution...")
        
        scene_generator = MockSceneGenerator(width=width, height=height)
        detector = PlaneDetector(config)
        camera_intrinsics = scene_generator.camera_params.get_intrinsics_matrix()
        
        # Run multiple iterations for stable timing
        times = []
        plane_counts = []
        
        for i in range(3):
            depth_map, rgb_frame = scene_generator.generate_complex_scene()
            
            start_time = time.time()
            planes = detector.detect_planes(depth_map, rgb_frame.float(), camera_intrinsics)
            end_time = time.time()
            
            times.append(end_time - start_time)
            plane_counts.append(len(planes))
        
        # Calculate statistics
        avg_time = np.mean(times)
        std_time = np.std(times)
        avg_planes = np.mean(plane_counts)
        
        results[f"{width}x{height}"] = {
            'avg_time': avg_time,
            'std_time': std_time,
            'avg_planes': avg_planes,
            'pixels': width * height
        }
        
        print(f"  Average time: {avg_time:.3f} ± {std_time:.3f} seconds")
        print(f"  Average planes: {avg_planes:.1f}")
        print(f"  Pixels per second: {(width * height) / avg_time:.0f}")
    
    # Summary
    print(f"\nPerformance Summary:")
    for resolution, metrics in results.items():
        print(f"  {resolution}: {metrics['avg_time']:.3f}s, "
              f"{metrics['pixels']/metrics['avg_time']:.0f} pixels/s")
    
    return results


def main():
    """Run all plane detection demonstrations."""
    print("3D Plane Detection System for Wall Surface Analysis")
    print("=" * 60)
    
    try:
        # Basic plane detection
        planes, depth_map, rgb_frame, scene_gen = demonstrate_basic_plane_detection()
        
        # Surface normal estimation
        normals, _, _ = demonstrate_surface_normal_estimation()
        
        # Complex scene analysis
        complex_planes, complex_depth, complex_rgb, complex_gen = demonstrate_complex_scene_analysis()
        
        # Visualization
        fig = visualize_plane_detection_results(complex_planes, complex_depth, complex_rgb, complex_gen)
        
        # Temporal consistency
        temporal_planes = demonstrate_temporal_consistency()
        
        # Performance benchmark
        perf_results = benchmark_performance()
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETE")
        print("="*60)
        print("Key Features Demonstrated:")
        print("✓ RANSAC-based plane detection with geometric constraints")
        print("✓ Surface normal estimation using hybrid approach")
        print("✓ Wall surface validation and quality assessment")
        print("✓ Temporal consistency across video frames")
        print("✓ Real-time performance optimization")
        print("✓ Comprehensive visualization and analysis")
        print("✓ Integration with camera parameters")
        print("✓ Multi-scale detection for different wall sizes")
        
        # Summary statistics
        if planes:
            wall_planes = [p for p in planes if p.plane_type == PlaneType.WALL]
            suitable_walls = [p for p in wall_planes if p.is_wall_suitable_for_tv()]
            
            print(f"\nFinal Results:")
            print(f"  Total planes detected: {len(planes)}")
            print(f"  Wall surfaces found: {len(wall_planes)}")
            print(f"  Suitable for TV placement: {len(suitable_walls)}")
            
            if suitable_walls:
                best_score = max(p.tv_placement_score for p in suitable_walls)
                print(f"  Best TV placement score: {best_score:.3f}")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    main() 