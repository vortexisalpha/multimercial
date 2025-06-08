"""
RANSAC-based Plane Detection for 3D Scene Understanding

This module implements sophisticated plane detection algorithms using RANSAC
with geometric constraints, multi-scale detection, and performance optimization
for real-time wall surface analysis.
"""

import torch
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict, Any
import logging
import time
import concurrent.futures
from abc import ABC, abstractmethod

from .plane_models import Plane, PlaneDetectionConfig, PlaneType, PlaneQuality, WallQualityMetrics
from .geometry_utils import GeometryUtils

logger = logging.getLogger(__name__)


class BasePlaneDetector(ABC):
    """Abstract base class for plane detection algorithms."""
    
    @abstractmethod
    def detect_planes(self, depth_map: torch.Tensor, 
                     rgb_frame: torch.Tensor,
                     camera_intrinsics: torch.Tensor) -> List[Plane]:
        """Detect planes in depth map and RGB frame."""
        pass


class RANSACPlaneDetector(BasePlaneDetector):
    """RANSAC-based plane detector with geometric constraints."""
    
    def __init__(self, config: PlaneDetectionConfig):
        """
        Initialize RANSAC plane detector.
        
        Args:
            config: Plane detection configuration
        """
        self.config = config
        self.config.validate()
        
        self.geometry_utils = GeometryUtils()
        self.plane_id_counter = 0
        
        # Performance tracking
        self.detection_times = []
        self.planes_detected = []
        
        logger.info(f"RANSACPlaneDetector initialized with {config.ransac_iterations} iterations")
    
    def detect_planes(self, depth_map: torch.Tensor, 
                     rgb_frame: torch.Tensor,
                     camera_intrinsics: torch.Tensor) -> List[Plane]:
        """
        Detect planes using RANSAC algorithm with geometric constraints.
        
        Args:
            depth_map: Depth map tensor (H, W)
            rgb_frame: RGB frame tensor (H, W, 3)
            camera_intrinsics: Camera intrinsics matrix (3, 3)
            
        Returns:
            List of detected planes
        """
        start_time = time.time()
        
        try:
            # Convert depth map to 3D points
            points_3d = self._depth_to_points_3d(depth_map, camera_intrinsics)
            
            if points_3d.shape[0] < self.config.min_inliers:
                logger.warning("Insufficient 3D points for plane detection")
                return []
            
            # Multi-scale detection if enabled
            if self.config.enable_multiscale:
                all_planes = self._multiscale_detection(
                    points_3d, depth_map, rgb_frame, camera_intrinsics
                )
            else:
                all_planes = self._single_scale_detection(
                    points_3d, depth_map, rgb_frame, camera_intrinsics
                )
            
            # Post-process and validate planes
            valid_planes = self._post_process_planes(all_planes, depth_map, camera_intrinsics)
            
            # Track performance
            detection_time = time.time() - start_time
            self.detection_times.append(detection_time)
            self.planes_detected.append(len(valid_planes))
            
            logger.debug(f"Detected {len(valid_planes)} planes in {detection_time:.3f}s")
            
            return valid_planes
            
        except Exception as e:
            logger.error(f"Plane detection failed: {e}")
            return []
    
    def _depth_to_points_3d(self, depth_map: torch.Tensor, 
                           camera_intrinsics: torch.Tensor) -> torch.Tensor:
        """Convert depth map to 3D point cloud."""
        height, width = depth_map.shape
        
        # Create pixel grid
        y_coords, x_coords = torch.meshgrid(
            torch.arange(height, device=depth_map.device),
            torch.arange(width, device=depth_map.device),
            indexing='ij'
        )
        
        # Flatten coordinates and depth
        x_flat = x_coords.flatten()
        y_flat = y_coords.flatten()
        depth_flat = depth_map.flatten()
        
        # Filter valid depth values
        valid_mask = (depth_flat > 0) & (depth_flat < 100)  # Reasonable depth range
        x_valid = x_flat[valid_mask]
        y_valid = y_flat[valid_mask]
        depth_valid = depth_flat[valid_mask]
        
        if len(depth_valid) == 0:
            return torch.empty((0, 3), device=depth_map.device)
        
        # Unproject to 3D
        fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
        cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
        
        x_3d = (x_valid - cx) * depth_valid / fx
        y_3d = (y_valid - cy) * depth_valid / fy
        z_3d = depth_valid
        
        points_3d = torch.stack([x_3d, y_3d, z_3d], dim=1)
        
        return points_3d
    
    def _multiscale_detection(self, points_3d: torch.Tensor,
                             depth_map: torch.Tensor,
                             rgb_frame: torch.Tensor,
                             camera_intrinsics: torch.Tensor) -> List[Plane]:
        """Perform multi-scale plane detection."""
        all_planes = []
        
        for scale_factor in self.config.scale_factors:
            if scale_factor == 1.0:
                # Full resolution
                scaled_points = points_3d
                scaled_depth = depth_map
                scaled_rgb = rgb_frame
                scaled_intrinsics = camera_intrinsics
            else:
                # Downsample
                scaled_points, scaled_depth, scaled_rgb, scaled_intrinsics = self._downsample_data(
                    points_3d, depth_map, rgb_frame, camera_intrinsics, scale_factor
                )
            
            # Detect planes at this scale
            scale_planes = self._single_scale_detection(
                scaled_points, scaled_depth, scaled_rgb, scaled_intrinsics
            )
            
            # Upscale plane parameters if needed
            if scale_factor != 1.0:
                scale_planes = self._upscale_planes(scale_planes, scale_factor)
            
            all_planes.extend(scale_planes)
        
        # Merge similar planes across scales
        merged_planes = self._merge_similar_planes(all_planes)
        
        return merged_planes
    
    def _single_scale_detection(self, points_3d: torch.Tensor,
                               depth_map: torch.Tensor,
                               rgb_frame: torch.Tensor,
                               camera_intrinsics: torch.Tensor) -> List[Plane]:
        """Detect planes at single scale using RANSAC."""
        planes = []
        remaining_points = points_3d.clone()
        remaining_indices = torch.arange(len(points_3d), device=points_3d.device)
        
        # Iteratively detect planes
        iteration = 0
        while (len(remaining_points) >= self.config.min_inliers and 
               len(planes) < self.config.max_planes_per_frame and
               iteration < 10):  # Max iterations to prevent infinite loop
            
            # Run RANSAC on remaining points
            best_plane, inlier_indices = self._ransac_plane_fitting(remaining_points)
            
            if best_plane is None or len(inlier_indices) < self.config.min_inliers:
                break
            
            # Create plane object with metadata
            plane = self._create_plane_object(
                best_plane, inlier_indices, remaining_indices, 
                depth_map, rgb_frame, camera_intrinsics
            )
            
            if self._validate_plane_geometry(plane):
                planes.append(plane)
            
            # Remove inlier points for next iteration
            mask = torch.ones(len(remaining_points), dtype=torch.bool, device=points_3d.device)
            mask[inlier_indices] = False
            remaining_points = remaining_points[mask]
            remaining_indices = remaining_indices[mask]
            
            iteration += 1
        
        return planes
    
    def _ransac_plane_fitting(self, points: torch.Tensor) -> Tuple[Optional[Dict], torch.Tensor]:
        """Fit plane using RANSAC algorithm."""
        if len(points) < 3:
            return None, torch.empty(0, device=points.device)
        
        best_inliers = torch.empty(0, device=points.device)
        best_plane = None
        best_inlier_count = 0
        
        # Multi-threaded RANSAC if enabled
        if self.config.enable_multithreading and self.config.max_threads > 1:
            return self._multithreaded_ransac(points)
        
        # Standard RANSAC
        for iteration in range(self.config.ransac_iterations):
            # Randomly sample 3 points
            if len(points) < 3:
                break
                
            indices = torch.randperm(len(points), device=points.device)[:3]
            sample_points = points[indices]
            
            # Fit plane to sample
            plane_params = self._fit_plane_to_points(sample_points)
            if plane_params is None:
                continue
            
            # Find inliers
            distances = self._point_to_plane_distances(points, plane_params)
            inlier_mask = distances < self.config.ransac_threshold
            inlier_indices = torch.where(inlier_mask)[0]
            
            # Check if this is the best plane so far
            inlier_count = len(inlier_indices)
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_inliers = inlier_indices
                best_plane = plane_params
                
                # Early termination if enough inliers found
                if (self.config.early_termination and 
                    inlier_count > len(points) * self.config.early_termination_threshold):
                    break
        
        # Refine plane using all inliers
        if best_plane is not None and len(best_inliers) >= self.config.min_inliers:
            inlier_points = points[best_inliers]
            refined_plane = self._fit_plane_to_points(inlier_points)
            if refined_plane is not None:
                best_plane = refined_plane
        
        return best_plane, best_inliers
    
    def _multithreaded_ransac(self, points: torch.Tensor) -> Tuple[Optional[Dict], torch.Tensor]:
        """Multi-threaded RANSAC implementation."""
        iterations_per_thread = self.config.ransac_iterations // self.config.max_threads
        
        def ransac_worker(start_iter: int, end_iter: int) -> Tuple[Optional[Dict], torch.Tensor, int]:
            best_plane = None
            best_inliers = torch.empty(0, device=points.device)
            best_count = 0
            
            for _ in range(start_iter, end_iter):
                if len(points) < 3:
                    break
                    
                indices = torch.randperm(len(points), device=points.device)[:3]
                sample_points = points[indices]
                
                plane_params = self._fit_plane_to_points(sample_points)
                if plane_params is None:
                    continue
                
                distances = self._point_to_plane_distances(points, plane_params)
                inlier_mask = distances < self.config.ransac_threshold
                inlier_indices = torch.where(inlier_mask)[0]
                
                if len(inlier_indices) > best_count:
                    best_count = len(inlier_indices)
                    best_inliers = inlier_indices
                    best_plane = plane_params
            
            return best_plane, best_inliers, best_count
        
        # Submit tasks to thread pool
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_threads) as executor:
            for i in range(self.config.max_threads):
                start_iter = i * iterations_per_thread
                end_iter = (i + 1) * iterations_per_thread
                if i == self.config.max_threads - 1:
                    end_iter = self.config.ransac_iterations
                
                future = executor.submit(ransac_worker, start_iter, end_iter)
                futures.append(future)
        
        # Collect results
        best_plane = None
        best_inliers = torch.empty(0, device=points.device)
        best_count = 0
        
        for future in concurrent.futures.as_completed(futures):
            plane, inliers, count = future.result()
            if count > best_count:
                best_count = count
                best_inliers = inliers
                best_plane = plane
        
        return best_plane, best_inliers
    
    def _fit_plane_to_points(self, points: torch.Tensor) -> Optional[Dict]:
        """Fit plane to set of 3D points using least squares."""
        if len(points) < 3:
            return None
        
        try:
            # Center the points
            centroid = torch.mean(points, dim=0)
            centered_points = points - centroid
            
            # SVD to find normal vector
            if len(points) == 3:
                # Exactly 3 points - use cross product
                v1 = centered_points[1] - centered_points[0]
                v2 = centered_points[2] - centered_points[0]
                normal = torch.cross(v1, v2, dim=0)
                
                if torch.norm(normal) < 1e-8:
                    return None  # Collinear points
                
                normal = torch.nn.functional.normalize(normal, dim=0)
            else:
                # More than 3 points - use SVD
                U, S, V = torch.svd(centered_points.T)
                normal = V[:, -1]  # Last column (smallest singular value)
            
            # Plane equation: normal · (point - centroid) = 0
            # Or: normal[0]*x + normal[1]*y + normal[2]*z + d = 0
            d = -torch.dot(normal, centroid)
            
            return {
                'normal': normal,
                'point': centroid,
                'equation': torch.cat([normal, d.unsqueeze(0)])
            }
            
        except Exception as e:
            logger.debug(f"Plane fitting failed: {e}")
            return None
    
    def _point_to_plane_distances(self, points: torch.Tensor, plane_params: Dict) -> torch.Tensor:
        """Compute distances from points to plane."""
        normal = plane_params['normal']
        point_on_plane = plane_params['point']
        
        # Distance = |normal · (point - point_on_plane)|
        diff = points - point_on_plane.unsqueeze(0)
        distances = torch.abs(torch.mv(diff, normal))
        
        return distances
    
    def _create_plane_object(self, plane_params: Dict, inlier_indices: torch.Tensor,
                           global_indices: torch.Tensor, depth_map: torch.Tensor,
                           rgb_frame: torch.Tensor, camera_intrinsics: torch.Tensor) -> Plane:
        """Create Plane object with comprehensive metadata."""
        # Map local inlier indices to global indices
        global_inlier_indices = global_indices[inlier_indices]
        
        # Create inlier mask
        height, width = depth_map.shape
        inlier_mask = torch.zeros((height, width), dtype=torch.bool, device=depth_map.device)
        
        # Convert 3D indices back to 2D coordinates
        y_coords, x_coords = torch.meshgrid(
            torch.arange(height, device=depth_map.device),
            torch.arange(width, device=depth_map.device),
            indexing='ij'
        )
        valid_depth_mask = (depth_map > 0) & (depth_map < 100)
        flat_indices = torch.arange(height * width, device=depth_map.device)[valid_depth_mask.flatten()]
        
        # Set inlier pixels
        for global_idx in global_inlier_indices:
            if global_idx < len(flat_indices):
                pixel_idx = flat_indices[global_idx]
                y, x = pixel_idx // width, pixel_idx % width
                inlier_mask[y, x] = True
        
        # Extract depth and RGB values for inliers
        inlier_depths = depth_map[inlier_mask]
        inlier_rgb = rgb_frame[inlier_mask] if rgb_frame is not None else None
        
        # Compute area
        area = self._compute_plane_area(inlier_mask, depth_map, camera_intrinsics)
        
        # Classify plane type
        plane_type = self._classify_plane_type(plane_params['normal'])
        
        # Compute quality metrics
        planarity_score = self._compute_planarity_score(inlier_indices, plane_params, depth_map)
        confidence = min(1.0, len(inlier_indices) / self.config.min_inliers)
        
        # Create plane object
        plane = Plane(
            normal=plane_params['normal'],
            point=plane_params['point'],
            equation=plane_params['equation'],
            inlier_mask=inlier_mask,
            depth_values=inlier_depths,
            rgb_values=inlier_rgb,
            confidence=confidence,
            area=area,
            planarity_score=planarity_score,
            plane_type=plane_type,
            plane_id=self._get_next_plane_id()
        )
        
        return plane
    
    def _classify_plane_type(self, normal: torch.Tensor) -> PlaneType:
        """Classify plane type based on normal vector."""
        # Normalize normal vector
        normal = torch.nn.functional.normalize(normal, dim=0)
        
        # Define reference vectors
        up_vector = torch.tensor([0.0, -1.0, 0.0], device=normal.device)  # Y up (camera coords)
        forward_vector = torch.tensor([0.0, 0.0, 1.0], device=normal.device)  # Z forward
        
        # Compute similarities
        up_similarity = torch.abs(torch.dot(normal, up_vector))
        forward_similarity = torch.abs(torch.dot(normal, forward_vector))
        
        # Classify based on dominant direction
        if up_similarity > self.config.floor_normal_tolerance:
            if normal[1] < 0:  # Normal points up
                return PlaneType.FLOOR
            else:  # Normal points down
                return PlaneType.CEILING
        elif forward_similarity < self.config.wall_normal_tolerance:
            return PlaneType.WALL
        else:
            return PlaneType.UNKNOWN
    
    def _compute_planarity_score(self, inlier_indices: torch.Tensor, 
                                plane_params: Dict, depth_map: torch.Tensor) -> float:
        """Compute how planar the detected surface is."""
        if len(inlier_indices) == 0:
            return 0.0
        
        # This is a simplified planarity measure
        # In practice, you might compute variance of distances to plane
        return min(1.0, len(inlier_indices) / (self.config.min_inliers * 3))
    
    def _compute_plane_area(self, inlier_mask: torch.Tensor, 
                           depth_map: torch.Tensor, 
                           camera_intrinsics: torch.Tensor) -> float:
        """Compute surface area of plane in square meters."""
        if not torch.any(inlier_mask):
            return 0.0
        
        # Get inlier coordinates
        y_coords, x_coords = torch.where(inlier_mask)
        
        if len(x_coords) == 0:
            return 0.0
        
        # Convert to 3D coordinates
        fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
        cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
        
        depths = depth_map[y_coords, x_coords]
        x_3d = (x_coords.float() - cx) * depths / fx
        y_3d = (y_coords.float() - cy) * depths / fy
        
        # Estimate area (simplified rectangular approximation)
        x_range = x_3d.max() - x_3d.min()
        y_range = y_3d.max() - y_3d.min()
        estimated_area = x_range * y_range
        
        return max(0.0, estimated_area.item())
    
    def _validate_plane_geometry(self, plane: Plane) -> bool:
        """Validate plane based on geometric constraints."""
        # Check area constraints
        if plane.area < self.config.min_plane_area or plane.area > self.config.max_plane_area:
            return False
        
        # Check confidence
        if plane.confidence < self.config.min_confidence:
            return False
        
        # Check planarity
        if plane.planarity_score < self.config.min_planarity:
            return False
        
        return True
    
    def _post_process_planes(self, planes: List[Plane], 
                            depth_map: torch.Tensor,
                            camera_intrinsics: torch.Tensor) -> List[Plane]:
        """Post-process detected planes."""
        if not planes:
            return []
        
        # Remove duplicate/overlapping planes
        unique_planes = self._remove_duplicate_planes(planes)
        
        # Compute additional quality metrics
        for plane in unique_planes:
            self._compute_wall_quality_metrics(plane, depth_map, camera_intrinsics)
        
        # Sort by quality
        unique_planes.sort(key=lambda p: p.tv_placement_score, reverse=True)
        
        return unique_planes[:self.config.max_planes_per_frame]
    
    def _remove_duplicate_planes(self, planes: List[Plane]) -> List[Plane]:
        """Remove duplicate or very similar planes."""
        if len(planes) <= 1:
            return planes
        
        unique_planes = []
        
        for plane in planes:
            is_duplicate = False
            
            for existing_plane in unique_planes:
                # Check normal similarity
                normal_similarity = torch.dot(plane.normal, existing_plane.normal).abs()
                
                # Check point distance
                point_distance = torch.norm(plane.point - existing_plane.point)
                
                if (normal_similarity > 0.9 and point_distance < self.config.merge_threshold):
                    is_duplicate = True
                    # Keep the better quality plane
                    if plane.confidence > existing_plane.confidence:
                        unique_planes.remove(existing_plane)
                        unique_planes.append(plane)
                    break
            
            if not is_duplicate:
                unique_planes.append(plane)
        
        return unique_planes
    
    def _compute_wall_quality_metrics(self, plane: Plane, 
                                     depth_map: torch.Tensor,
                                     camera_intrinsics: torch.Tensor):
        """Compute comprehensive quality metrics for wall surfaces."""
        metrics = WallQualityMetrics()
        
        # Basic geometric properties
        metrics.area = plane.area
        metrics.planarity = plane.planarity_score
        
        # Compute aspect ratio
        if torch.any(plane.inlier_mask):
            y_coords, x_coords = torch.where(plane.inlier_mask)
            height_pixels = y_coords.max() - y_coords.min() + 1
            width_pixels = x_coords.max() - x_coords.min() + 1
            metrics.aspect_ratio = width_pixels.float() / height_pixels.float()
        
        # Distance to camera
        metrics.distance_to_camera = torch.norm(plane.point).item()
        
        # Visibility score (simplified)
        total_pixels = plane.inlier_mask.numel()
        inlier_pixels = torch.sum(plane.inlier_mask).item()
        metrics.visibility_score = min(1.0, inlier_pixels / (total_pixels * 0.1))
        
        # TV placement score
        if plane.plane_type == PlaneType.WALL:
            size_factor = min(1.0, metrics.area / 5.0)  # Ideal wall size ~5 m²
            distance_factor = max(0.0, 1.0 - metrics.distance_to_camera / 10.0)
            planarity_factor = metrics.planarity
            
            metrics.tv_placement_score = (
                size_factor * 0.4 +
                distance_factor * 0.3 +
                planarity_factor * 0.3
            )
        
        plane.quality_metrics = metrics
        plane.tv_placement_score = metrics.tv_placement_score
        plane.distance_to_camera = metrics.distance_to_camera
    
    def _get_next_plane_id(self) -> int:
        """Get next unique plane ID."""
        self.plane_id_counter += 1
        return self.plane_id_counter
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if not self.detection_times:
            return {}
        
        return {
            'avg_detection_time': np.mean(self.detection_times),
            'max_detection_time': np.max(self.detection_times),
            'avg_planes_detected': np.mean(self.planes_detected),
            'total_detections': len(self.detection_times)
        }
    
    def _downsample_data(self, points_3d: torch.Tensor, depth_map: torch.Tensor,
                        rgb_frame: torch.Tensor, camera_intrinsics: torch.Tensor,
                        scale_factor: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Downsample input data by scale factor."""
        height, width = depth_map.shape
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)
        
        # Downsample depth map
        depth_downsampled = torch.nn.functional.interpolate(
            depth_map.unsqueeze(0).unsqueeze(0),
            size=(new_height, new_width),
            mode='bilinear',
            align_corners=False
        ).squeeze()
        
        # Downsample RGB frame
        rgb_downsampled = torch.nn.functional.interpolate(
            rgb_frame.permute(2, 0, 1).unsqueeze(0),
            size=(new_height, new_width),
            mode='bilinear',
            align_corners=False
        ).squeeze().permute(1, 2, 0)
        
        # Scale camera intrinsics
        scaled_intrinsics = camera_intrinsics.clone()
        scaled_intrinsics[0, 0] *= scale_factor  # fx
        scaled_intrinsics[1, 1] *= scale_factor  # fy
        scaled_intrinsics[0, 2] *= scale_factor  # cx
        scaled_intrinsics[1, 2] *= scale_factor  # cy
        
        # Generate new 3D points from downsampled depth
        points_downsampled = self._depth_to_points_3d(depth_downsampled, scaled_intrinsics)
        
        return points_downsampled, depth_downsampled, rgb_downsampled, scaled_intrinsics
    
    def _upscale_planes(self, planes: List[Plane], scale_factor: float) -> List[Plane]:
        """Upscale plane parameters back to original resolution."""
        upscaled_planes = []
        
        for plane in planes:
            # Scale the plane point (position in 3D space doesn't change much)
            # but we might want to adjust based on camera coordinate scaling
            upscaled_plane = Plane(
                normal=plane.normal,  # Normal vector doesn't change with scale
                point=plane.point,    # 3D point position relatively unchanged
                equation=plane.equation,
                inlier_mask=torch.nn.functional.interpolate(
                    plane.inlier_mask.float().unsqueeze(0).unsqueeze(0),
                    scale_factor=1.0/scale_factor,
                    mode='nearest'
                ).squeeze().bool(),
                depth_values=plane.depth_values,
                rgb_values=plane.rgb_values,
                confidence=plane.confidence,
                area=plane.area / (scale_factor ** 2),  # Scale area by square of scale factor
                planarity_score=plane.planarity_score,
                plane_type=plane.plane_type,
                quality=plane.quality,
                quality_metrics=plane.quality_metrics,
                plane_id=plane.plane_id
            )
            upscaled_planes.append(upscaled_plane)
        
        return upscaled_planes
    
    def _merge_similar_planes(self, planes: List[Plane]) -> List[Plane]:
        """Merge similar planes detected at different scales."""
        if len(planes) <= 1:
            return planes
        
        merged_planes = []
        used_indices = set()
        
        for i, plane1 in enumerate(planes):
            if i in used_indices:
                continue
                
            similar_planes = [plane1]
            used_indices.add(i)
            
            for j, plane2 in enumerate(planes[i+1:], i+1):
                if j in used_indices:
                    continue
                
                # Check similarity based on normal and position
                normal_similarity = torch.dot(plane1.normal, plane2.normal).abs()
                point_distance = torch.norm(plane1.point - plane2.point)
                
                if (normal_similarity > 0.9 and point_distance < self.config.merge_threshold):
                    similar_planes.append(plane2)
                    used_indices.add(j)
            
            # Merge similar planes by averaging parameters
            if len(similar_planes) > 1:
                merged_plane = self._average_planes(similar_planes)
                merged_planes.append(merged_plane)
            else:
                merged_planes.append(plane1)
        
        return merged_planes
    
    def _average_planes(self, planes: List[Plane]) -> Plane:
        """Average multiple similar planes into one."""
        # Weight by confidence
        weights = torch.tensor([p.confidence for p in planes])
        weights = weights / weights.sum()
        
        # Weighted average of normals
        normals = torch.stack([p.normal for p in planes])
        avg_normal = torch.sum(weights.unsqueeze(1) * normals, dim=0)
        avg_normal = torch.nn.functional.normalize(avg_normal, dim=0)
        
        # Weighted average of points
        points = torch.stack([p.point for p in planes])
        avg_point = torch.sum(weights.unsqueeze(1) * points, dim=0)
        
        # Use the highest quality plane as base
        best_plane = max(planes, key=lambda p: p.confidence)
        
        # Create merged plane
        merged_plane = Plane(
            normal=avg_normal,
            point=avg_point,
            equation=torch.cat([avg_normal, -torch.dot(avg_normal, avg_point).unsqueeze(0)]),
            inlier_mask=best_plane.inlier_mask,  # Use best plane's mask
            depth_values=best_plane.depth_values,
            rgb_values=best_plane.rgb_values,
            confidence=torch.sum(weights * torch.tensor([p.confidence for p in planes])).item(),
            area=torch.sum(weights * torch.tensor([p.area for p in planes])).item(),
            planarity_score=torch.sum(weights * torch.tensor([p.planarity_score for p in planes])).item(),
            plane_type=best_plane.plane_type,
            quality=best_plane.quality,
            quality_metrics=best_plane.quality_metrics,
            plane_id=best_plane.plane_id
        )
        
        return merged_plane


class PlaneDetector:
    """Main interface for plane detection with multiple algorithms."""
    
    def __init__(self, config: PlaneDetectionConfig):
        """Initialize plane detector with configuration."""
        self.config = config
        self.detector = RANSACPlaneDetector(config)
        
        logger.info("PlaneDetector initialized")
    
    def detect_planes(self, depth_map: torch.Tensor, 
                     rgb_frame: torch.Tensor,
                     camera_intrinsics: torch.Tensor) -> List[Plane]:
        """
        Detect planes in depth map and RGB frame.
        
        Args:
            depth_map: Depth map tensor (H, W)
            rgb_frame: RGB frame tensor (H, W, 3)
            camera_intrinsics: Camera intrinsics matrix (3, 3)
            
        Returns:
            List of detected planes
        """
        return self.detector.detect_planes(depth_map, rgb_frame, camera_intrinsics)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detection statistics."""
        return self.detector.get_performance_stats() 