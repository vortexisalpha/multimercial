"""
Temporal Consistency System Example for TV Advertisement Placement

This example demonstrates the advanced temporal consistency system including:
- Kalman filter-based TV tracking
- Dynamic occlusion handling
- Lighting consistency management
- Quality assessment and artifact detection
- Prediction and interpolation for missing frames
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import logging
from typing import List, Dict, Any
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from video_ad_placement.scene_understanding.temporal_consistency import (
    TemporalConsistencyManager, TVPlacement, TVGeometry, 
    LightingEnvironment, QualityMetrics, TemporalConfig,
    TVPlacementState, QualityLevel
)
from video_ad_placement.scene_understanding.geometry_utils import CameraParameters

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockVideoSequence:
    """Mock video sequence generator for testing temporal consistency."""
    
    def __init__(self, num_frames: int = 100, resolution: tuple = (480, 640)):
        self.num_frames = num_frames
        self.height, self.width = resolution
        self.frame_id = 0
        
        # Generate trajectories and events
        self.tv_trajectory = self._generate_tv_trajectory()
        self.camera_trajectory = self._generate_camera_trajectory()
        self.lighting_changes = self._generate_lighting_changes()
        self.occlusion_events = self._generate_occlusion_events()
    
    def _generate_tv_trajectory(self) -> List[torch.Tensor]:
        """Generate realistic TV movement trajectory."""
        trajectory = []
        base_position = torch.tensor([0.0, 0.0, 2.5])
        
        for i in range(self.num_frames):
            # Simulate slight movement and noise
            noise_x = 0.02 * np.sin(i * 0.1) + 0.01 * np.random.randn()
            noise_y = 0.01 * np.cos(i * 0.15) + 0.005 * np.random.randn()
            noise_z = 0.01 * np.sin(i * 0.05) + 0.005 * np.random.randn()
            
            position = base_position + torch.tensor([noise_x, noise_y, noise_z])
            trajectory.append(position)
        
        return trajectory
    
    def _generate_camera_trajectory(self) -> List[Dict[str, Any]]:
        """Generate camera movement trajectory."""
        trajectory = []
        
        for i in range(self.num_frames):
            camera_pose = {
                'position': torch.tensor([0.05 * np.sin(i * 0.08), 0.0, 0.0]),
                'orientation': torch.tensor([1.0, 0.01 * np.sin(i * 0.1), 0.0, 0.0])
            }
            trajectory.append(camera_pose)
        
        return trajectory
    
    def _generate_lighting_changes(self) -> List[LightingEnvironment]:
        """Generate lighting changes over time."""
        lighting_changes = []
        
        for i in range(self.num_frames):
            lighting = LightingEnvironment()
            time_factor = i / self.num_frames
            lighting.scene_brightness = 0.4 + 0.3 * np.sin(time_factor * np.pi * 2)
            lighting.ambient_intensity = 0.2 + 0.2 * np.sin(time_factor * np.pi * 2)
            lighting.light_intensity = 0.6 + 0.2 * np.cos(time_factor * np.pi * 2)
            lighting.color_temperature = 6500 + 500 * np.sin(time_factor * np.pi)
            lighting.confidence = 0.8 + 0.1 * np.random.randn() * 0.1
            lighting_changes.append(lighting)
        
        return lighting_changes
    
    def _generate_occlusion_events(self) -> List[Dict[str, Any]]:
        """Generate occlusion events."""
        events = []
        occlusion_periods = [(20, 25), (50, 55), (80, 85)]
        
        for start, end in occlusion_periods:
            for frame_id in range(start, end + 1):
                event = {
                    'frame_id': frame_id,
                    'occlusion_type': 'person_walking',
                    'severity': 0.7 + 0.2 * np.random.randn()
                }
                events.append(event)
        
        return events
    
    def get_frame_data(self, frame_id: int) -> Dict[str, Any]:
        """Get data for a specific frame."""
        if frame_id >= self.num_frames:
            return None
        
        return {
            'frame_id': frame_id,
            'depth_map': self._generate_depth_map(frame_id),
            'rgb_frame': self._generate_rgb_frame(frame_id),
            'tv_detection': self._generate_tv_detection(frame_id),
            'camera_pose': self.camera_trajectory[frame_id],
            'lighting': self.lighting_changes[frame_id],
            'occlusions': [e for e in self.occlusion_events if e['frame_id'] == frame_id]
        }
    
    def _generate_depth_map(self, frame_id: int) -> torch.Tensor:
        """Generate synthetic depth map."""
        depth_map = torch.ones((self.height, self.width)) * 4.0
        
        # Wall region
        wall_depth = 2.5 + 0.1 * np.sin(frame_id * 0.1)
        depth_map[50:400, 100:500] = wall_depth
        
        # Floor region
        depth_map[400:, :] = 3.0
        
        # Add occlusion objects
        for occlusion in [e for e in self.occlusion_events if e['frame_id'] == frame_id]:
            if occlusion['occlusion_type'] == 'person_walking':
                x_pos = int(200 + 50 * np.sin(frame_id * 0.2))
                depth_map[200:400, x_pos:x_pos+100] = 1.5
        
        depth_map += torch.randn_like(depth_map) * 0.02
        return depth_map
    
    def _generate_rgb_frame(self, frame_id: int) -> torch.Tensor:
        """Generate synthetic RGB frame."""
        rgb_frame = torch.ones((self.height, self.width, 3)) * 0.3
        
        # Wall region
        rgb_frame[50:400, 100:500] = torch.tensor([0.6, 0.5, 0.4])
        
        # Floor region
        rgb_frame[400:, :] = torch.tensor([0.2, 0.2, 0.2])
        
        # Apply lighting
        lighting = self.lighting_changes[frame_id]
        rgb_frame *= lighting.scene_brightness
        
        # Add occlusion objects
        for occlusion in [e for e in self.occlusion_events if e['frame_id'] == frame_id]:
            if occlusion['occlusion_type'] == 'person_walking':
                x_pos = int(200 + 50 * np.sin(frame_id * 0.2))
                rgb_frame[200:400, x_pos:x_pos+100] = torch.tensor([0.1, 0.15, 0.1])
        
        rgb_frame += torch.randn_like(rgb_frame) * 0.02
        return torch.clamp(rgb_frame, 0, 1)
    
    def _generate_tv_detection(self, frame_id: int) -> Dict[str, Any]:
        """Generate TV detection data."""
        occlusions = [e for e in self.occlusion_events if e['frame_id'] == frame_id]
        
        if occlusions and any(e['severity'] > 0.8 for e in occlusions):
            return None
        
        position = self.tv_trajectory[frame_id]
        position += torch.randn(3) * 0.01
        
        confidence = 0.85
        if occlusions:
            max_severity = max(e['severity'] for e in occlusions)
            confidence *= (1.0 - max_severity * 0.5)
        
        return {
            'position': position,
            'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0]),
            'width': 1.2,
            'height': 0.8,
            'confidence': confidence
        }


class TemporalConsistencyExample:
    """Main example class demonstrating temporal consistency system."""
    
    def __init__(self):
        """Initialize the example."""
        self.config = TemporalConfig(
            process_noise_position=0.005,
            process_noise_orientation=0.002,
            measurement_noise_position=0.015,
            measurement_noise_orientation=0.008,
            max_tracking_distance=0.3,
            prediction_horizon=5,
            occlusion_threshold=0.15,
            min_visibility=0.3,
            occlusion_buffer_frames=10,
            lighting_smoothing_alpha=0.85,
            lighting_adaptation_frames=20,
            quality_window_size=20,
            stability_threshold=0.85,
            enable_multithreading=False,
            enable_debug_logging=True
        )
        
        self.consistency_manager = TemporalConsistencyManager(self.config)
        
        self.camera_params = CameraParameters(
            fx=500.0, fy=500.0, cx=320.0, cy=240.0
        )
        
        self.results = {
            'placements': [],
            'quality_metrics': [],
            'performance_stats': [],
            'predictions': [],
            'interpolations': []
        }
    
    def run_temporal_consistency_demo(self, num_frames: int = 100):
        """Run the main temporal consistency demonstration."""
        logger.info(f"Starting temporal consistency demo with {num_frames} frames")
        
        video_sequence = MockVideoSequence(num_frames=num_frames)
        
        for frame_id in range(num_frames):
            logger.info(f"Processing frame {frame_id + 1}/{num_frames}")
            
            frame_data = video_sequence.get_frame_data(frame_id)
            detections = [frame_data['tv_detection']] if frame_data['tv_detection'] else []
            object_tracks = self._create_mock_object_tracks(frame_data['occlusions'])
            
            start_time = time.time()
            
            placements = self.consistency_manager.track_tv_placement(
                tv_detections=detections,
                camera_poses=[frame_data['camera_pose']],
                depth_map=frame_data['depth_map'],
                rgb_frame=frame_data['rgb_frame'],
                object_tracks=object_tracks
            )
            
            processing_time = time.time() - start_time
            
            self.results['placements'].extend(placements)
            
            # Test prediction
            if frame_id % 10 == 0 and len(self.consistency_manager.tv_placement_history) > 0:
                predicted = self.consistency_manager.predict_placement(frames_ahead=3)
                if predicted:
                    self.results['predictions'].append({
                        'frame_id': frame_id,
                        'predicted_placement': predicted
                    })
            
            # Assess quality
            if frame_id % 20 == 0 and frame_id > 0:
                rendered_sequence = self._generate_mock_rendered_sequence()
                quality_metrics = self.consistency_manager.assess_quality(rendered_sequence)
                self.results['quality_metrics'].append({
                    'frame_id': frame_id,
                    'metrics': quality_metrics
                })
            
            # Store performance stats
            perf_stats = self.consistency_manager.get_performance_stats()
            perf_stats['frame_id'] = frame_id
            perf_stats['processing_time'] = processing_time
            self.results['performance_stats'].append(perf_stats)
            
            if frame_id % 20 == 0:
                logger.info(f"Processed {frame_id + 1} frames, "
                          f"avg processing time: {perf_stats['avg_processing_time']:.3f}s")
        
        logger.info("Temporal consistency demo completed")
    
    def demonstrate_interpolation(self):
        """Demonstrate interpolation capabilities."""
        logger.info("Demonstrating TV placement interpolation")
        
        if len(self.results['placements']) < 10:
            logger.warning("Not enough placements for interpolation demo")
            return
        
        start_placement = self.results['placements'][10]
        end_placement = self.results['placements'][20] if len(self.results['placements']) > 20 else self.results['placements'][-1]
        
        interpolated = self.consistency_manager.predictor.interpolate_tv_placements(
            start_placement, end_placement, num_intermediate=5
        )
        
        self.results['interpolations'] = interpolated
        logger.info(f"Generated {len(interpolated)} interpolated placements")
    
    def analyze_results(self):
        """Analyze and display results."""
        logger.info("Analyzing temporal consistency results")
        
        placement_positions = []
        placement_confidences = []
        visibility_scores = []
        quality_scores = []
        processing_times = []
        
        for placement in self.results['placements']:
            placement_positions.append(placement.geometry.position.numpy())
            placement_confidences.append(placement.confidence)
            visibility_scores.append(placement.visibility)
        
        for quality_data in self.results['quality_metrics']:
            quality_scores.append(quality_data['metrics'].overall_quality)
        
        for perf_data in self.results['performance_stats']:
            processing_times.append(perf_data['processing_time'])
        
        placement_positions = np.array(placement_positions)
        
        logger.info("\n=== TEMPORAL CONSISTENCY ANALYSIS ===")
        
        if len(placement_positions) > 0:
            logger.info(f"Total placements tracked: {len(placement_positions)}")
            logger.info(f"Average confidence: {np.mean(placement_confidences):.3f}")
            logger.info(f"Average visibility: {np.mean(visibility_scores):.3f}")
            
            if len(placement_positions) > 1:
                position_diffs = np.diff(placement_positions, axis=0)
                position_variation = np.mean(np.linalg.norm(position_diffs, axis=1))
                logger.info(f"Average position variation: {position_variation:.4f}m")
        
        if quality_scores:
            logger.info(f"Average quality score: {np.mean(quality_scores):.3f}")
            logger.info(f"Quality range: [{np.min(quality_scores):.3f}, {np.max(quality_scores):.3f}]")
        
        if processing_times:
            logger.info(f"Average processing time: {np.mean(processing_times):.3f}s")
            logger.info(f"Max processing time: {np.max(processing_times):.3f}s")
        
        if self.results['predictions']:
            logger.info(f"Generated {len(self.results['predictions'])} predictions")
            avg_pred_confidence = np.mean([p['predicted_placement'].prediction_confidence 
                                         for p in self.results['predictions']])
            logger.info(f"Average prediction confidence: {avg_pred_confidence:.3f}")
    
    def visualize_results(self, save_plots: bool = True):
        """Create visualizations of the results."""
        logger.info("Creating result visualizations")
        
        if len(self.results['placements']) == 0:
            logger.warning("No results to visualize")
            return
        
        # Extract data
        positions = np.array([p.geometry.position.numpy() for p in self.results['placements']])
        confidences = np.array([p.confidence for p in self.results['placements']])
        visibilities = np.array([p.visibility for p in self.results['placements']])
        frame_ids = np.arange(len(positions))
        
        # Create plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Temporal Consistency System Results', fontsize=16, fontweight='bold')
        
        # Position tracking
        ax1 = axes[0, 0]
        ax1.plot(frame_ids, positions[:, 0], 'b-', label='X position', alpha=0.8)
        ax1.plot(frame_ids, positions[:, 1], 'g-', label='Y position', alpha=0.8)
        ax1.plot(frame_ids, positions[:, 2], 'r-', label='Z position', alpha=0.8)
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Position (m)')
        ax1.set_title('TV Position Tracking')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Confidence and visibility
        ax2 = axes[0, 1]
        ax2.plot(frame_ids, confidences, 'b-', label='Confidence', linewidth=2)
        ax2.plot(frame_ids, visibilities, 'r-', label='Visibility', linewidth=2)
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Score')
        ax2.set_title('Confidence and Visibility')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Position stability
        ax3 = axes[0, 2]
        if len(positions) > 1:
            position_diffs = np.diff(positions, axis=0)
            position_magnitudes = np.linalg.norm(position_diffs, axis=1)
            ax3.plot(frame_ids[1:], position_magnitudes, 'purple', linewidth=2)
            ax3.set_xlabel('Frame')
            ax3.set_ylabel('Position Change (m)')
            ax3.set_title('Position Stability')
            ax3.grid(True, alpha=0.3)
        
        # Quality metrics
        ax4 = axes[1, 0]
        if self.results['quality_metrics']:
            quality_frames = [q['frame_id'] for q in self.results['quality_metrics']]
            overall_quality = [q['metrics'].overall_quality for q in self.results['quality_metrics']]
            temporal_stability = [q['metrics'].temporal_stability for q in self.results['quality_metrics']]
            
            ax4.plot(quality_frames, overall_quality, 'o-', label='Overall Quality', linewidth=2)
            ax4.plot(quality_frames, temporal_stability, 's-', label='Temporal Stability', linewidth=2)
            ax4.set_xlabel('Frame')
            ax4.set_ylabel('Quality Score')
            ax4.set_title('Quality Assessment')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim(0, 1)
        
        # Performance
        ax5 = axes[1, 1]
        if self.results['performance_stats']:
            perf_frames = [p['frame_id'] for p in self.results['performance_stats']]
            proc_times = [p['processing_time'] for p in self.results['performance_stats']]
            
            ax5.plot(perf_frames, proc_times, 'orange', linewidth=2)
            ax5.set_xlabel('Frame')
            ax5.set_ylabel('Processing Time (s)')
            ax5.set_title('Processing Performance')
            ax5.grid(True, alpha=0.3)
        
        # Predictions
        ax6 = axes[1, 2]
        if self.results['predictions']:
            pred_frames = [p['frame_id'] for p in self.results['predictions']]
            pred_confidences = [p['predicted_placement'].prediction_confidence for p in self.results['predictions']]
            
            ax6.bar(pred_frames, pred_confidences, alpha=0.7, color='skyblue')
            ax6.set_xlabel('Frame')
            ax6.set_ylabel('Prediction Confidence')
            ax6.set_title('Prediction Confidence')
            ax6.grid(True, alpha=0.3)
            ax6.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_plots:
            output_path = Path("temporal_consistency_results.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {output_path}")
        
        plt.show()
    
    def _create_mock_object_tracks(self, occlusions: List[Dict[str, Any]]) -> List[Any]:
        """Create mock object tracks for occlusion testing."""
        tracks = []
        
        for occlusion in occlusions:
            if occlusion['occlusion_type'] == 'person_walking':
                class MockTrack:
                    def __init__(self, frame_id):
                        self.position = torch.tensor([1.5, 0.0, 1.8])
                        self.bbox_2d = torch.tensor([180, 150, 280, 350])
                
                track = MockTrack(occlusion['frame_id'])
                tracks.append(track)
        
        return tracks
    
    def _generate_mock_rendered_sequence(self) -> List[torch.Tensor]:
        """Generate mock rendered sequence for quality assessment."""
        sequence = []
        
        for i in range(10):
            frame = torch.rand(240, 320, 3) * 0.5 + 0.25
            sequence.append(frame)
        
        return sequence


def main():
    """Run the temporal consistency example."""
    print("🎬 Temporal Consistency System for TV Advertisement Placement")
    print("=" * 70)
    
    example = TemporalConsistencyExample()
    
    # Run demonstrations
    example.run_temporal_consistency_demo(num_frames=100)
    example.demonstrate_interpolation()
    example.analyze_results()
    example.visualize_results()
    
    print("\n✅ Temporal consistency example completed successfully!")
    print("\nKey features demonstrated:")
    print("• Kalman filter-based 6-DOF TV tracking")
    print("• Dynamic occlusion detection and handling")
    print("• Temporal lighting consistency")
    print("• Quality assessment and artifact detection")
    print("• Prediction and interpolation for missing frames")
    print("• Real-time performance optimization")
    print("• Comprehensive quality metrics and validation")


if __name__ == "__main__":
    main() 