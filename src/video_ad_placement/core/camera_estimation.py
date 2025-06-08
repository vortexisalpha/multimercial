"""
Camera Estimation Module

Implements camera parameter estimation including focal length, pose estimation,
and camera calibration for accurate 3D scene reconstruction and advertisement placement.
"""

import logging
from typing import List, Dict, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import json

import numpy as np
import cv2
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

from ..utils.logging_utils import get_logger
from ..utils.geometry_utils import (
    estimate_focal_length_from_vanishing_points,
    estimate_pose_from_homography,
    triangulate_points_stereo,
    decompose_essential_matrix
)
from ..utils.camera_utils import (
    calibrate_camera_from_points,
    refine_camera_parameters,
    validate_camera_matrix
)

logger = get_logger(__name__)


class EstimationMethod(Enum):
    """Camera parameter estimation methods."""
    VANISHING_POINTS = "vanishing_points"
    FEATURE_MATCHING = "feature_matching"
    HOMOGRAPHY = "homography"
    ESSENTIAL_MATRIX = "essential_matrix"
    MULTI_VIEW = "multi_view"


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    fx: float  # Focal length x
    fy: float  # Focal length y
    cx: float  # Principal point x
    cy: float  # Principal point y
    k1: float = 0.0  # Radial distortion coefficient 1
    k2: float = 0.0  # Radial distortion coefficient 2
    p1: float = 0.0  # Tangential distortion coefficient 1
    p2: float = 0.0  # Tangential distortion coefficient 2
    k3: float = 0.0  # Radial distortion coefficient 3
    
    @property
    def matrix(self) -> np.ndarray:
        """Camera intrinsic matrix."""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)
    
    @property
    def distortion_coeffs(self) -> np.ndarray:
        """Distortion coefficients."""
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3], dtype=np.float32)


@dataclass
class CameraExtrinsics:
    """Camera extrinsic parameters (pose)."""
    rotation_matrix: np.ndarray  # 3x3 rotation matrix
    translation_vector: np.ndarray  # 3x1 translation vector
    
    @property
    def rotation_vector(self) -> np.ndarray:
        """Rotation as Rodrigues vector."""
        rvec, _ = cv2.Rodrigues(self.rotation_matrix)
        return rvec.flatten()
    
    @property
    def transformation_matrix(self) -> np.ndarray:
        """4x4 transformation matrix."""
        T = np.eye(4)
        T[:3, :3] = self.rotation_matrix
        T[:3, 3] = self.translation_vector.flatten()
        return T


@dataclass
class CameraParameters:
    """Complete camera parameters."""
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics
    image_size: Tuple[int, int]  # (width, height)
    estimation_method: EstimationMethod
    confidence: float
    reprojection_error: float
    timestamp: Optional[float] = None


class CameraEstimationConfig:
    """Configuration for camera parameter estimation."""
    
    def __init__(
        self,
        estimation_method: EstimationMethod = EstimationMethod.VANISHING_POINTS,
        enable_distortion_estimation: bool = True,
        enable_pose_estimation: bool = True,
        ransac_threshold: float = 1.0,
        ransac_iterations: int = 1000,
        feature_detector: str = "SIFT",  # SIFT, ORB, AKAZE
        max_features: int = 5000,
        confidence_threshold: float = 0.8,
        enable_bundle_adjustment: bool = False,
        use_temporal_smoothing: bool = True,
        temporal_window_size: int = 5,
        **kwargs
    ):
        self.estimation_method = estimation_method
        self.enable_distortion_estimation = enable_distortion_estimation
        self.enable_pose_estimation = enable_pose_estimation
        self.ransac_threshold = ransac_threshold
        self.ransac_iterations = ransac_iterations
        self.feature_detector = feature_detector
        self.max_features = max_features
        self.confidence_threshold = confidence_threshold
        self.enable_bundle_adjustment = enable_bundle_adjustment
        self.use_temporal_smoothing = use_temporal_smoothing
        self.temporal_window_size = temporal_window_size
        self.kwargs = kwargs


class CameraEstimator:
    """
    Advanced camera parameter estimation system for video processing.
    
    This class provides comprehensive camera calibration and pose estimation
    capabilities for accurate 3D reconstruction and advertisement placement.
    
    Attributes:
        config: Configuration object for estimation parameters
        feature_detector: Feature detection algorithm
        camera_history: Historical camera parameters for temporal consistency
        reference_frame: Reference frame for pose estimation
        is_initialized: Initialization status
    """

    def __init__(self, config: CameraEstimationConfig):
        """
        Initialize the camera estimator with specified configuration.
        
        Args:
            config: CameraEstimationConfig with estimation parameters
        """
        self.config = config
        self.feature_detector = self._initialize_feature_detector()
        self.camera_history: List[CameraParameters] = []
        self.reference_frame: Optional[np.ndarray] = None
        self.reference_features: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.is_initialized = True
        
        logger.info(f"CameraEstimator initialized with {config.estimation_method.value} method")

    def _initialize_feature_detector(self):
        """Initialize the feature detector based on configuration."""
        detector_name = self.config.feature_detector.upper()
        
        if detector_name == "SIFT":
            return cv2.SIFT_create(nfeatures=self.config.max_features)
        elif detector_name == "ORB":
            return cv2.ORB_create(nfeatures=self.config.max_features)
        elif detector_name == "AKAZE":
            return cv2.AKAZE_create()
        else:
            logger.warning(f"Unknown detector {detector_name}, using SIFT")
            return cv2.SIFT_create(nfeatures=self.config.max_features)

    def estimate(
        self,
        image: np.ndarray,
        depth_map: Optional[np.ndarray] = None,
        objects: Optional[List] = None,
        planes: Optional[List] = None
    ) -> CameraParameters:
        """
        Estimate camera parameters from input image and optional scene information.
        
        Args:
            image: Input RGB image
            depth_map: Optional depth map
            objects: Optional detected objects
            planes: Optional detected planes
            
        Returns:
            CameraParameters with intrinsics and extrinsics
            
        Raises:
            RuntimeError: If estimation fails
        """
        try:
            logger.info("Starting camera parameter estimation")
            
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            height, width = image_gray.shape
            image_size = (width, height)
            
            # Estimate intrinsic parameters
            intrinsics = self._estimate_intrinsics(image_gray, planes)
            
            # Estimate extrinsic parameters
            extrinsics = None
            if self.config.enable_pose_estimation:
                extrinsics = self._estimate_extrinsics(image_gray, intrinsics, depth_map, objects, planes)
            
            if extrinsics is None:
                # Default extrinsics (identity pose)
                extrinsics = CameraExtrinsics(
                    rotation_matrix=np.eye(3),
                    translation_vector=np.zeros(3)
                )
            
            # Compute confidence and reprojection error
            confidence = self._compute_estimation_confidence(image_gray, intrinsics, extrinsics)
            reprojection_error = self._compute_reprojection_error(image_gray, intrinsics, extrinsics)
            
            camera_params = CameraParameters(
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                image_size=image_size,
                estimation_method=self.config.estimation_method,
                confidence=confidence,
                reprojection_error=reprojection_error,
                timestamp=None
            )
            
            # Apply temporal smoothing if enabled
            if self.config.use_temporal_smoothing and len(self.camera_history) > 0:
                camera_params = self._apply_temporal_smoothing(camera_params)
            
            # Update history
            self.camera_history.append(camera_params)
            if len(self.camera_history) > self.config.temporal_window_size:
                self.camera_history.pop(0)
            
            logger.info(f"Camera estimation completed with confidence: {confidence:.3f}")
            return camera_params
            
        except Exception as e:
            logger.error(f"Camera estimation failed: {str(e)}")
            raise RuntimeError(f"Camera estimation error: {str(e)}")

    def _estimate_intrinsics(self, image: np.ndarray, planes: Optional[List] = None) -> CameraIntrinsics:
        """Estimate camera intrinsic parameters."""
        height, width = image.shape
        
        if self.config.estimation_method == EstimationMethod.VANISHING_POINTS:
            return self._estimate_intrinsics_vanishing_points(image, planes)
        elif self.config.estimation_method == EstimationMethod.FEATURE_MATCHING:
            return self._estimate_intrinsics_feature_matching(image)
        else:
            # Default estimation based on image size
            return self._estimate_intrinsics_default(width, height)

    def _estimate_intrinsics_vanishing_points(self, image: np.ndarray, planes: Optional[List] = None) -> CameraIntrinsics:
        """Estimate intrinsics using vanishing point analysis."""
        height, width = image.shape
        
        try:
            # Detect lines in the image
            edges = cv2.Canny(image, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is None or len(lines) < 10:
                return self._estimate_intrinsics_default(width, height)
            
            # Find vanishing points from lines
            vanishing_points = self._find_vanishing_points(lines, (width, height))
            
            if len(vanishing_points) >= 2:
                # Estimate focal length from vanishing points
                focal_length = self._compute_focal_from_vanishing_points(vanishing_points, (width, height))
                
                return CameraIntrinsics(
                    fx=focal_length,
                    fy=focal_length,
                    cx=width / 2,
                    cy=height / 2
                )
            else:
                return self._estimate_intrinsics_default(width, height)
                
        except Exception as e:
            logger.warning(f"Vanishing point estimation failed: {str(e)}")
            return self._estimate_intrinsics_default(width, height)

    def _estimate_intrinsics_feature_matching(self, image: np.ndarray) -> CameraIntrinsics:
        """Estimate intrinsics using feature matching with reference frame."""
        height, width = image.shape
        
        if self.reference_frame is None:
            # Set current frame as reference
            self.reference_frame = image.copy()
            kp, desc = self.feature_detector.detectAndCompute(image, None)
            self.reference_features = (kp, desc)
            return self._estimate_intrinsics_default(width, height)
        
        try:
            # Detect features in current frame
            kp_current, desc_current = self.feature_detector.detectAndCompute(image, None)
            
            if desc_current is None or self.reference_features[1] is None:
                return self._estimate_intrinsics_default(width, height)
            
            # Match features
            matcher = cv2.BFMatcher()
            matches = matcher.knnMatch(self.reference_features[1], desc_current, k=2)
            
            # Apply ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            
            if len(good_matches) < 50:
                return self._estimate_intrinsics_default(width, height)
            
            # Extract point correspondences
            pts_ref = np.float32([self.reference_features[0][m.queryIdx].pt for m in good_matches])
            pts_cur = np.float32([kp_current[m.trainIdx].pt for m in good_matches])
            
            # Estimate fundamental matrix
            F, mask = cv2.findFundamentalMat(pts_ref, pts_cur, cv2.FM_RANSAC)
            
            if F is None:
                return self._estimate_intrinsics_default(width, height)
            
            # Estimate focal length from fundamental matrix (simplified)
            focal_length = self._estimate_focal_from_fundamental(F, (width, height))
            
            return CameraIntrinsics(
                fx=focal_length,
                fy=focal_length,
                cx=width / 2,
                cy=height / 2
            )
            
        except Exception as e:
            logger.warning(f"Feature matching estimation failed: {str(e)}")
            return self._estimate_intrinsics_default(width, height)

    def _estimate_intrinsics_default(self, width: int, height: int) -> CameraIntrinsics:
        """Default intrinsic parameter estimation based on image size."""
        # Common heuristic: focal length ≈ image width
        focal_length = width * 0.7
        
        return CameraIntrinsics(
            fx=focal_length,
            fy=focal_length,
            cx=width / 2,
            cy=height / 2
        )

    def _find_vanishing_points(self, lines: np.ndarray, image_size: Tuple[int, int]) -> List[np.ndarray]:
        """Find vanishing points from detected lines."""
        width, height = image_size
        vanishing_points = []
        
        # Convert lines to line equations
        line_equations = []
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            line_equations.append([a, b, -rho])
        
        # Find intersections of lines (vanishing points)
        for i in range(len(line_equations)):
            for j in range(i + 1, len(line_equations)):
                intersection = self._line_intersection(line_equations[i], line_equations[j])
                
                if intersection is not None:
                    x, y = intersection
                    # Filter points within reasonable bounds
                    if -width < x < 2*width and -height < y < 2*height:
                        vanishing_points.append(np.array([x, y]))
        
        # Cluster similar vanishing points
        if len(vanishing_points) > 0:
            vanishing_points = self._cluster_vanishing_points(vanishing_points)
        
        return vanishing_points

    def _line_intersection(self, line1: List[float], line2: List[float]) -> Optional[Tuple[float, float]]:
        """Compute intersection of two lines."""
        a1, b1, c1 = line1
        a2, b2, c2 = line2
        
        denom = a1 * b2 - a2 * b1
        if abs(denom) < 1e-6:
            return None
        
        x = (b1 * c2 - b2 * c1) / denom
        y = (a2 * c1 - a1 * c2) / denom
        
        return (x, y)

    def _cluster_vanishing_points(self, points: List[np.ndarray], threshold: float = 50.0) -> List[np.ndarray]:
        """Cluster nearby vanishing points."""
        if len(points) <= 1:
            return points
        
        points_array = np.array(points)
        clustered_points = []
        used = np.zeros(len(points), dtype=bool)
        
        for i, point in enumerate(points_array):
            if used[i]:
                continue
            
            # Find nearby points
            distances = np.linalg.norm(points_array - point, axis=1)
            cluster_indices = np.where(distances < threshold)[0]
            
            # Compute cluster center
            cluster_center = np.mean(points_array[cluster_indices], axis=0)
            clustered_points.append(cluster_center)
            
            used[cluster_indices] = True
        
        return clustered_points

    def _compute_focal_from_vanishing_points(self, vanishing_points: List[np.ndarray], image_size: Tuple[int, int]) -> float:
        """Compute focal length from vanishing points."""
        width, height = image_size
        cx, cy = width / 2, height / 2
        
        if len(vanishing_points) < 2:
            return width * 0.7  # Default
        
        # Use the first two vanishing points
        vp1, vp2 = vanishing_points[:2]
        
        # Compute focal length using orthogonal vanishing points
        # f^2 = -(vp1 - pp) · (vp2 - pp)
        pp = np.array([cx, cy])
        v1 = vp1 - pp
        v2 = vp2 - pp
        
        dot_product = np.dot(v1, v2)
        if dot_product >= 0:
            return width * 0.7  # Default if not orthogonal
        
        focal_squared = -dot_product
        focal_length = np.sqrt(max(focal_squared, 0))
        
        # Clamp to reasonable range
        focal_length = max(min(focal_length, width * 2), width * 0.3)
        
        return focal_length

    def _estimate_focal_from_fundamental(self, F: np.ndarray, image_size: Tuple[int, int]) -> float:
        """Estimate focal length from fundamental matrix."""
        width, height = image_size
        
        # Simple estimation based on F matrix properties
        # This is a simplified approach
        focal_estimate = np.sqrt(abs(F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0])) * width
        
        # Clamp to reasonable range
        focal_estimate = max(min(focal_estimate, width * 2), width * 0.3)
        
        return focal_estimate

    def _estimate_extrinsics(
        self,
        image: np.ndarray,
        intrinsics: CameraIntrinsics,
        depth_map: Optional[np.ndarray] = None,
        objects: Optional[List] = None,
        planes: Optional[List] = None
    ) -> Optional[CameraExtrinsics]:
        """Estimate camera extrinsic parameters (pose)."""
        
        if self.config.estimation_method == EstimationMethod.HOMOGRAPHY and planes:
            return self._estimate_extrinsics_homography(image, intrinsics, planes)
        elif self.config.estimation_method == EstimationMethod.FEATURE_MATCHING:
            return self._estimate_extrinsics_feature_matching(image, intrinsics)
        else:
            # Default: assume camera at origin looking down negative Z axis
            return CameraExtrinsics(
                rotation_matrix=np.eye(3),
                translation_vector=np.zeros(3)
            )

    def _estimate_extrinsics_homography(
        self,
        image: np.ndarray,
        intrinsics: CameraIntrinsics,
        planes: List
    ) -> Optional[CameraExtrinsics]:
        """Estimate pose using homography from detected planes."""
        try:
            # Find the largest plane (likely floor or wall)
            if not planes:
                return None
            
            largest_plane = max(planes, key=lambda p: getattr(p, 'area', 0))
            
            # For simplicity, assume we can extract 4 corner points from the plane
            # In practice, you'd need to project the 3D plane to image coordinates
            # and find corresponding points
            
            # This is a placeholder implementation
            # Real implementation would require plane-to-image projection
            
            return CameraExtrinsics(
                rotation_matrix=np.eye(3),
                translation_vector=np.array([0, 1.5, 0])  # Assume camera at typical height
            )
            
        except Exception as e:
            logger.warning(f"Homography pose estimation failed: {str(e)}")
            return None

    def _estimate_extrinsics_feature_matching(
        self,
        image: np.ndarray,
        intrinsics: CameraIntrinsics
    ) -> Optional[CameraExtrinsics]:
        """Estimate pose using feature matching with reference frame."""
        if self.reference_frame is None or self.reference_features is None:
            return None
        
        try:
            # Detect and match features
            kp_current, desc_current = self.feature_detector.detectAndCompute(image, None)
            
            if desc_current is None:
                return None
            
            matcher = cv2.BFMatcher()
            matches = matcher.knnMatch(self.reference_features[1], desc_current, k=2)
            
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            
            if len(good_matches) < 50:
                return None
            
            # Extract point correspondences
            pts_ref = np.float32([self.reference_features[0][m.queryIdx].pt for m in good_matches])
            pts_cur = np.float32([kp_current[m.trainIdx].pt for m in good_matches])
            
            # Estimate essential matrix
            E, mask = cv2.findEssentialMat(
                pts_ref, pts_cur, intrinsics.matrix,
                method=cv2.RANSAC, prob=0.999, threshold=1.0
            )
            
            if E is None:
                return None
            
            # Recover pose from essential matrix
            _, R, t, mask = cv2.recoverPose(E, pts_ref, pts_cur, intrinsics.matrix)
            
            return CameraExtrinsics(
                rotation_matrix=R,
                translation_vector=t.flatten()
            )
            
        except Exception as e:
            logger.warning(f"Feature matching pose estimation failed: {str(e)}")
            return None

    def _compute_estimation_confidence(
        self,
        image: np.ndarray,
        intrinsics: CameraIntrinsics,
        extrinsics: CameraExtrinsics
    ) -> float:
        """Compute confidence score for the estimation."""
        # Simplified confidence computation
        base_confidence = 0.8
        
        # Factor in the estimation method
        method_confidence = {
            EstimationMethod.VANISHING_POINTS: 0.9,
            EstimationMethod.FEATURE_MATCHING: 0.8,
            EstimationMethod.HOMOGRAPHY: 0.85,
            EstimationMethod.ESSENTIAL_MATRIX: 0.9,
            EstimationMethod.MULTI_VIEW: 0.95
        }
        
        confidence = base_confidence * method_confidence.get(self.config.estimation_method, 0.7)
        
        return min(max(confidence, 0.0), 1.0)

    def _compute_reprojection_error(
        self,
        image: np.ndarray,
        intrinsics: CameraIntrinsics,
        extrinsics: CameraExtrinsics
    ) -> float:
        """Compute reprojection error for the estimation."""
        # Placeholder implementation
        # Real implementation would reproject known 3D points and compute error
        
        return 1.5  # pixels

    def _apply_temporal_smoothing(self, current_params: CameraParameters) -> CameraParameters:
        """Apply temporal smoothing to camera parameters."""
        if len(self.camera_history) == 0:
            return current_params
        
        # Simple averaging of recent parameters
        window_size = min(len(self.camera_history), self.config.temporal_window_size)
        recent_params = self.camera_history[-window_size:]
        
        # Smooth intrinsics
        avg_fx = np.mean([p.intrinsics.fx for p in recent_params] + [current_params.intrinsics.fx])
        avg_fy = np.mean([p.intrinsics.fy for p in recent_params] + [current_params.intrinsics.fy])
        avg_cx = np.mean([p.intrinsics.cx for p in recent_params] + [current_params.intrinsics.cx])
        avg_cy = np.mean([p.intrinsics.cy for p in recent_params] + [current_params.intrinsics.cy])
        
        smoothed_intrinsics = CameraIntrinsics(
            fx=avg_fx, fy=avg_fy, cx=avg_cx, cy=avg_cy,
            k1=current_params.intrinsics.k1,
            k2=current_params.intrinsics.k2,
            p1=current_params.intrinsics.p1,
            p2=current_params.intrinsics.p2,
            k3=current_params.intrinsics.k3
        )
        
        # For extrinsics, use current values (pose changes more rapidly)
        smoothed_params = CameraParameters(
            intrinsics=smoothed_intrinsics,
            extrinsics=current_params.extrinsics,
            image_size=current_params.image_size,
            estimation_method=current_params.estimation_method,
            confidence=current_params.confidence,
            reprojection_error=current_params.reprojection_error,
            timestamp=current_params.timestamp
        )
        
        return smoothed_params

    def get_camera_history(self) -> List[CameraParameters]:
        """Get historical camera parameters."""
        return self.camera_history.copy()

    def set_reference_frame(self, image: np.ndarray) -> None:
        """Set a new reference frame for pose estimation."""
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        self.reference_frame = image_gray.copy()
        
        kp, desc = self.feature_detector.detectAndCompute(image_gray, None)
        self.reference_features = (kp, desc)
        
        logger.info("Reference frame updated for camera estimation")

    def cleanup(self) -> None:
        """Clean up resources."""
        self.camera_history.clear()
        self.reference_frame = None
        self.reference_features = None
        logger.info("CameraEstimator cleanup completed") 