"""
Geometry Utilities for 3D Scene Understanding

This module provides essential geometric utilities for plane detection,
camera projection, 3D transformations, and spatial calculations needed
for wall surface analysis and TV advertisement placement.
"""

import torch
import numpy as np
import cv2
from typing import Tuple, List, Optional, Dict, Any
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CameraParameters:
    """Camera intrinsic and extrinsic parameters."""
    
    # Intrinsic parameters
    fx: float  # Focal length in x
    fy: float  # Focal length in y
    cx: float  # Principal point x
    cy: float  # Principal point y
    
    # Distortion parameters
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0
    
    # Extrinsic parameters (camera pose)
    rotation: Optional[torch.Tensor] = None  # 3x3 rotation matrix
    translation: Optional[torch.Tensor] = None  # 3x1 translation vector
    
    def get_intrinsics_matrix(self) -> torch.Tensor:
        """Get 3x3 intrinsics matrix."""
        return torch.tensor([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ], dtype=torch.float32)
    
    def get_distortion_coefficients(self) -> torch.Tensor:
        """Get distortion coefficients."""
        return torch.tensor([self.k1, self.k2, self.p1, self.p2, self.k3], 
                           dtype=torch.float32)


class GeometryUtils:
    """Utility class for 3D geometry operations."""
    
    @staticmethod
    def normalize_vector(vector: torch.Tensor) -> torch.Tensor:
        """Normalize vector to unit length."""
        norm = torch.norm(vector, dim=-1, keepdim=True)
        return vector / (norm + 1e-8)
    
    @staticmethod
    def compute_angle_between_vectors(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        """Compute angle between two vectors in radians."""
        v1_norm = GeometryUtils.normalize_vector(v1)
        v2_norm = GeometryUtils.normalize_vector(v2)
        
        dot_product = torch.sum(v1_norm * v2_norm, dim=-1)
        # Clamp to handle numerical errors
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        
        angle = torch.acos(dot_product)
        return angle
    
    @staticmethod
    def point_to_plane_distance(points: torch.Tensor, plane_normal: torch.Tensor, 
                               plane_point: torch.Tensor) -> torch.Tensor:
        """
        Compute signed distance from points to plane.
        
        Args:
            points: Points tensor (N, 3)
            plane_normal: Plane normal vector (3,)
            plane_point: Point on plane (3,)
            
        Returns:
            Signed distances (N,)
        """
        # Vector from plane point to query points
        vectors = points - plane_point.unsqueeze(0)
        
        # Project onto normal
        distances = torch.mv(vectors, plane_normal)
        
        return distances
    
    @staticmethod
    def project_points_to_plane(points: torch.Tensor, plane_normal: torch.Tensor, 
                               plane_point: torch.Tensor) -> torch.Tensor:
        """
        Project 3D points onto plane.
        
        Args:
            points: Points tensor (N, 3)
            plane_normal: Plane normal vector (3,)
            plane_point: Point on plane (3,)
            
        Returns:
            Projected points (N, 3)
        """
        distances = GeometryUtils.point_to_plane_distance(points, plane_normal, plane_point)
        projected = points - distances.unsqueeze(-1) * plane_normal.unsqueeze(0)
        
        return projected
    
    @staticmethod
    def compute_plane_from_points(points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute plane normal and center from set of points using SVD.
        
        Args:
            points: Points tensor (N, 3)
            
        Returns:
            Tuple of (normal, center)
        """
        if len(points) < 3:
            raise ValueError("Need at least 3 points to fit plane")
        
        # Center the points
        center = torch.mean(points, dim=0)
        centered = points - center.unsqueeze(0)
        
        # SVD to find normal (direction of smallest variance)
        # We want SVD of the centered points matrix (N x 3)
        U, S, V = torch.svd(centered)
        normal = V[:, -1]  # Last column (smallest singular value) - this is the normal
        
        # Ensure normal points in consistent direction
        if normal[2] < 0:  # Point away from camera (positive Z)
            normal = -normal
        
        return normal, center
    
    @staticmethod
    def compute_convex_hull_2d(points: torch.Tensor) -> torch.Tensor:
        """
        Compute 2D convex hull using Graham scan algorithm.
        
        Args:
            points: 2D points tensor (N, 2)
            
        Returns:
            Convex hull points (M, 2)
        """
        if len(points) < 3:
            return points
        
        # Convert to numpy for processing
        points_np = points.cpu().numpy()
        
        # Find convex hull using cv2
        hull = cv2.convexHull(points_np.astype(np.float32))
        hull_points = torch.from_numpy(hull.squeeze()).to(points.device)
        
        return hull_points
    
    @staticmethod
    def compute_polygon_area(points: torch.Tensor) -> float:
        """
        Compute area of 2D polygon using shoelace formula.
        
        Args:
            points: 2D polygon vertices (N, 2)
            
        Returns:
            Polygon area
        """
        if len(points) < 3:
            return 0.0
        
        # Shoelace formula
        x = points[:, 0]
        y = points[:, 1]
        
        # Add first point at the end to close polygon
        x = torch.cat([x, x[0:1]])
        y = torch.cat([y, y[0:1]])
        
        area = 0.5 * torch.abs(torch.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))
        
        return area.item()
    
    @staticmethod
    def compute_bounding_box_3d(points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute 3D axis-aligned bounding box.
        
        Args:
            points: 3D points tensor (N, 3)
            
        Returns:
            Tuple of (min_bounds, max_bounds) each (3,)
        """
        if len(points) == 0:
            return torch.zeros(3), torch.zeros(3)
        
        min_bounds = torch.min(points, dim=0)[0]
        max_bounds = torch.max(points, dim=0)[0]
        
        return min_bounds, max_bounds
    
    @staticmethod
    def transform_points(points: torch.Tensor, rotation: torch.Tensor, 
                        translation: torch.Tensor) -> torch.Tensor:
        """
        Transform 3D points using rotation and translation.
        
        Args:
            points: Points tensor (N, 3)
            rotation: Rotation matrix (3, 3)
            translation: Translation vector (3,)
            
        Returns:
            Transformed points (N, 3)
        """
        # Apply rotation then translation
        transformed = torch.mm(points, rotation.T) + translation.unsqueeze(0)
        
        return transformed
    
    @staticmethod
    def compute_rotation_matrix_from_vectors(from_vec: torch.Tensor, 
                                           to_vec: torch.Tensor) -> torch.Tensor:
        """
        Compute rotation matrix that rotates from_vec to to_vec.
        
        Args:
            from_vec: Source vector (3,)
            to_vec: Target vector (3,)
            
        Returns:
            Rotation matrix (3, 3)
        """
        from_vec = GeometryUtils.normalize_vector(from_vec)
        to_vec = GeometryUtils.normalize_vector(to_vec)
        
        # Check if vectors are already aligned
        dot_product = torch.dot(from_vec, to_vec)
        if torch.abs(dot_product - 1.0) < 1e-6:
            return torch.eye(3, device=from_vec.device)
        
        # Check if vectors are opposite
        if torch.abs(dot_product + 1.0) < 1e-6:
            # Find perpendicular vector
            if torch.abs(from_vec[0]) < 0.9:
                perp = torch.tensor([1.0, 0.0, 0.0], device=from_vec.device)
            else:
                perp = torch.tensor([0.0, 1.0, 0.0], device=from_vec.device)
            
            # Make it perpendicular to from_vec
            perp = perp - torch.dot(perp, from_vec) * from_vec
            perp = GeometryUtils.normalize_vector(perp)
            
            # 180-degree rotation around perpendicular axis
            return 2.0 * torch.outer(perp, perp) - torch.eye(3, device=from_vec.device)
        
        # Rodrigues' rotation formula
        cross_product = torch.cross(from_vec, to_vec, dim=0)
        sin_angle = torch.norm(cross_product)
        cos_angle = dot_product
        
        if sin_angle < 1e-6:
            return torch.eye(3, device=from_vec.device)
        
        # Normalized axis
        axis = cross_product / sin_angle
        
        # Skew-symmetric matrix
        K = torch.tensor([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ], device=from_vec.device)
        
        # Rodrigues' formula
        R = (torch.eye(3, device=from_vec.device) + 
             sin_angle * K + 
             (1 - cos_angle) * torch.mm(K, K))
        
        return R


class CameraProjection:
    """Camera projection utilities for 3D to 2D transformations."""
    
    def __init__(self, camera_params: CameraParameters):
        """
        Initialize camera projection.
        
        Args:
            camera_params: Camera parameters
        """
        self.params = camera_params
        self.intrinsics = camera_params.get_intrinsics_matrix()
        self.distortion = camera_params.get_distortion_coefficients()
    
    def project_points_3d_to_2d(self, points_3d: torch.Tensor) -> torch.Tensor:
        """
        Project 3D points to 2D image coordinates.
        
        Args:
            points_3d: 3D points tensor (N, 3)
            
        Returns:
            2D image coordinates (N, 2)
        """
        if len(points_3d) == 0:
            return torch.empty((0, 2), device=points_3d.device)
        
        # Transform to camera coordinates if extrinsics available
        if self.params.rotation is not None and self.params.translation is not None:
            points_camera = GeometryUtils.transform_points(
                points_3d, self.params.rotation, self.params.translation
            )
        else:
            points_camera = points_3d
        
        # Project to image plane (homogeneous coordinates)
        points_homo = torch.mm(points_camera, self.intrinsics.T)
        
        # Convert to pixel coordinates
        points_2d = points_homo[:, :2] / points_homo[:, 2:3]
        
        return points_2d
    
    def unproject_2d_to_3d(self, points_2d: torch.Tensor, depths: torch.Tensor) -> torch.Tensor:
        """
        Unproject 2D image coordinates to 3D points using depth.
        
        Args:
            points_2d: 2D image coordinates (N, 2)
            depths: Depth values (N,)
            
        Returns:
            3D points tensor (N, 3)
        """
        if len(points_2d) == 0:
            return torch.empty((0, 3), device=points_2d.device)
        
        # Convert to normalized device coordinates
        x_norm = (points_2d[:, 0] - self.params.cx) / self.params.fx
        y_norm = (points_2d[:, 1] - self.params.cy) / self.params.fy
        
        # Create 3D points
        points_3d = torch.stack([
            x_norm * depths,
            y_norm * depths,
            depths
        ], dim=1)
        
        # Transform to world coordinates if extrinsics available
        if self.params.rotation is not None and self.params.translation is not None:
            # Inverse transformation
            rotation_inv = self.params.rotation.T
            translation_inv = -torch.mv(rotation_inv, self.params.translation)
            points_3d = GeometryUtils.transform_points(points_3d, rotation_inv, translation_inv)
        
        return points_3d
    
    def depth_map_to_point_cloud(self, depth_map: torch.Tensor) -> torch.Tensor:
        """
        Convert depth map to 3D point cloud.
        
        Args:
            depth_map: Depth map tensor (H, W)
            
        Returns:
            Point cloud tensor (N, 3) where N is number of valid depth pixels
        """
        height, width = depth_map.shape
        
        # Create pixel grid
        y_coords, x_coords = torch.meshgrid(
            torch.arange(height, device=depth_map.device),
            torch.arange(width, device=depth_map.device),
            indexing='ij'
        )
        
        # Flatten and filter valid depths
        x_flat = x_coords.flatten().float()
        y_flat = y_coords.flatten().float()
        depth_flat = depth_map.flatten()
        
        valid_mask = (depth_flat > 0) & (depth_flat < 100)
        x_valid = x_flat[valid_mask]
        y_valid = y_flat[valid_mask]
        depth_valid = depth_flat[valid_mask]
        
        if len(depth_valid) == 0:
            return torch.empty((0, 3), device=depth_map.device)
        
        # Stack 2D coordinates
        points_2d = torch.stack([x_valid, y_valid], dim=1)
        
        # Unproject to 3D
        points_3d = self.unproject_2d_to_3d(points_2d, depth_valid)
        
        return points_3d
    
    def compute_viewing_ray(self, pixel_coords: torch.Tensor) -> torch.Tensor:
        """
        Compute viewing ray direction for pixel coordinates.
        
        Args:
            pixel_coords: Pixel coordinates (N, 2)
            
        Returns:
            Ray directions (N, 3)
        """
        # Convert to normalized device coordinates
        x_norm = (pixel_coords[:, 0] - self.params.cx) / self.params.fx
        y_norm = (pixel_coords[:, 1] - self.params.cy) / self.params.fy
        
        # Ray directions (normalized)
        ray_dirs = torch.stack([x_norm, y_norm, torch.ones_like(x_norm)], dim=1)
        ray_dirs = GeometryUtils.normalize_vector(ray_dirs)
        
        return ray_dirs


class PlaneGeometry:
    """Specialized geometry utilities for plane operations."""
    
    @staticmethod
    def fit_plane_ransac(points: torch.Tensor, num_iterations: int = 1000,
                        distance_threshold: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fit plane to points using RANSAC.
        
        Args:
            points: 3D points tensor (N, 3)
            num_iterations: Number of RANSAC iterations
            distance_threshold: Distance threshold for inliers
            
        Returns:
            Tuple of (normal, center, inlier_mask)
        """
        if len(points) < 3:
            raise ValueError("Need at least 3 points for plane fitting")
        
        best_inliers = torch.zeros(len(points), dtype=torch.bool, device=points.device)
        best_normal = torch.zeros(3, device=points.device)
        best_center = torch.zeros(3, device=points.device)
        best_inlier_count = 0
        
        for _ in range(num_iterations):
            # Sample 3 random points
            indices = torch.randperm(len(points))[:3]
            sample_points = points[indices]
            
            # Fit plane to sample
            try:
                normal, center = GeometryUtils.compute_plane_from_points(sample_points)
            except:
                continue
            
            # Find inliers
            distances = GeometryUtils.point_to_plane_distance(points, normal, center)
            inliers = torch.abs(distances) < distance_threshold
            inlier_count = torch.sum(inliers)
            
            # Update best plane
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_inliers = inliers
                best_normal = normal
                best_center = center
        
        # Refine plane using all inliers
        if best_inlier_count >= 3:
            inlier_points = points[best_inliers]
            try:
                refined_normal, refined_center = GeometryUtils.compute_plane_from_points(inlier_points)
                return refined_normal, refined_center, best_inliers
            except:
                pass
        
        return best_normal, best_center, best_inliers
    
    @staticmethod
    def compute_plane_intersection_line(normal1: torch.Tensor, center1: torch.Tensor,
                                       normal2: torch.Tensor, center2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute intersection line between two planes.
        
        Args:
            normal1: First plane normal (3,)
            center1: First plane center (3,)
            normal2: Second plane normal (3,)
            center2: Second plane center (3,)
            
        Returns:
            Tuple of (line_direction, line_point)
        """
        # Line direction is cross product of normals
        line_direction = torch.cross(normal1, normal2, dim=0)
        
        # Check if planes are parallel
        if torch.norm(line_direction) < 1e-6:
            return torch.zeros(3), torch.zeros(3)
        
        line_direction = GeometryUtils.normalize_vector(line_direction)
        
        # Find a point on the line
        # Solve system: n1·(p-c1) = 0, n2·(p-c2) = 0, d·p = d·p0
        # where d is line direction and p0 is any point
        
        # Use coordinate with largest component of line_direction
        max_idx = torch.argmax(torch.abs(line_direction))
        
        # Set that coordinate to 0 and solve 2x2 system for other coordinates
        if max_idx == 0:  # x is largest
            # Solve for y, z with x = 0
            A = torch.stack([normal1[1:], normal2[1:]])
            b = torch.stack([torch.dot(normal1, center1), torch.dot(normal2, center2)])
            
            if torch.det(A).abs() > 1e-6:
                yz = torch.solve(b, A)[0]
                line_point = torch.tensor([0.0, yz[0], yz[1]], device=normal1.device)
            else:
                line_point = center1
        elif max_idx == 1:  # y is largest
            # Solve for x, z with y = 0
            A = torch.stack([normal1[[0, 2]], normal2[[0, 2]]])
            b = torch.stack([torch.dot(normal1, center1), torch.dot(normal2, center2)])
            
            if torch.det(A).abs() > 1e-6:
                xz = torch.solve(b, A)[0]
                line_point = torch.tensor([xz[0], 0.0, xz[1]], device=normal1.device)
            else:
                line_point = center1
        else:  # z is largest
            # Solve for x, y with z = 0
            A = torch.stack([normal1[:2], normal2[:2]])
            b = torch.stack([torch.dot(normal1, center1), torch.dot(normal2, center2)])
            
            if torch.det(A).abs() > 1e-6:
                xy = torch.solve(b, A)[0]
                line_point = torch.tensor([xy[0], xy[1], 0.0], device=normal1.device)
            else:
                line_point = center1
        
        return line_direction, line_point
    
    @staticmethod
    def compute_plane_area_from_mask(mask: torch.Tensor, depth_map: torch.Tensor, 
                                   camera_params: CameraParameters) -> float:
        """
        Compute surface area of plane from pixel mask and depth map.
        
        Args:
            mask: Binary mask of plane pixels (H, W)
            depth_map: Depth map (H, W)
            camera_params: Camera parameters
            
        Returns:
            Surface area in square meters
        """
        if not torch.any(mask):
            return 0.0
        
        # Get mask coordinates
        y_coords, x_coords = torch.where(mask)
        
        if len(x_coords) == 0:
            return 0.0
        
        # Convert to 3D points
        depths = depth_map[y_coords, x_coords]
        points_2d = torch.stack([x_coords.float(), y_coords.float()], dim=1)
        
        camera_proj = CameraProjection(camera_params)
        points_3d = camera_proj.unproject_2d_to_3d(points_2d, depths)
        
        # Compute area using convex hull approximation
        if len(points_3d) < 3:
            return 0.0
        
        # Project points to plane's local 2D coordinate system
        normal, center = GeometryUtils.compute_plane_from_points(points_3d)
        
        # Create local coordinate system
        # Choose two perpendicular vectors in the plane
        if torch.abs(normal[2]) < 0.9:
            u = torch.cross(normal, torch.tensor([0, 0, 1], device=normal.device, dtype=normal.dtype), dim=0)
        else:
            u = torch.cross(normal, torch.tensor([1, 0, 0], device=normal.device, dtype=normal.dtype), dim=0)
        
        u = GeometryUtils.normalize_vector(u)
        v = torch.cross(normal, u, dim=0)
        
        # Project 3D points to 2D plane coordinates
        points_relative = points_3d - center.unsqueeze(0)
        points_2d_plane = torch.stack([
            torch.mv(points_relative, u),
            torch.mv(points_relative, v)
        ], dim=1)
        
        # Compute convex hull and area
        hull_points = GeometryUtils.compute_convex_hull_2d(points_2d_plane)
        area = GeometryUtils.compute_polygon_area(hull_points)
        
        return area 