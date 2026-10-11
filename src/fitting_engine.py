"""Intelligent face fitting engine for MirrorFit XR.

Calculates 3D scale, rotation, and translation transformations to fit virtual accessories
onto estimated facial anatomy, decoupled from any specific graphics or rendering library.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from src.head_pose import HeadPoseResult


@dataclass
class FittingResult:
    """Transformation parameters to position a 3D accessory on the face."""

    model_matrix_4x4: np.ndarray  # 4x4 Homogeneous transformation matrix (Model -> Camera)
    rvec: np.ndarray              # (3, 1) Rotation vector for projection
    tvec: np.ndarray              # (3, 1) Translation vector for projection
    scale: float                  # Uniform scale factor
    anchor_center_2d: Tuple[int, int]
    is_valid: bool


class FittingEngine:
    """Computes geometric transformations for accessory placement.

    Mathematical Transformation Pipeline:
    1. Local Accessory Frame: Origin (0,0,0) is centered at the glasses nose saddle.
    2. Head Pose Alignment: Head pose estimation gives (R_head, t_head) relative to camera.
    3. Anatomical Anchor: Glasses anchor is positioned at the glabella / nose saddle (Y = -32mm, Z = +15mm).
       t_accessory = t_head + R_head * [offset_x, offset_y, offset_z]^T
    4. Proportional Scaling: Scales accessory matching detected face dimensions.
    5. Homogeneous Composition:
       T_model_to_cam = [ R_head * S   t_accessory ]
                        [    0              1      ]
    """

    # Anthropometric offset from nose tip (solvePnP origin) to glabella / glasses rest (in mm)
    # Aligned with canonical frame: -Y is up towards forehead, +Z is recessed into face
    DEFAULT_BRIDGE_OFFSET_MM = np.array([0.0, -32.0, 15.0], dtype=np.float64)

    def __init__(
        self,
        base_model_width_mm: float = 130.0,
        vertical_offset_mm: float = 0.0,
        depth_offset_mm: float = 0.0,
    ) -> None:
        """Initialize fitting engine.

        Args:
            base_model_width_mm: Natural width of the 3D model in millimeters (standard adult frames ~130mm).
            vertical_offset_mm: Fine vertical adjustment along face Y axis.
            depth_offset_mm: Fine depth adjustment along face Z axis.
        """
        self.base_model_width_mm = base_model_width_mm
        self.vertical_offset_mm = vertical_offset_mm
        self.depth_offset_mm = depth_offset_mm

    def compute_fitting(
        self,
        pose: HeadPoseResult,
        pixel_landmarks: np.ndarray,
        frame_shape: Tuple[int, int],
        model_scale_factor: float = 1.0,
        focal_length: Optional[float] = None,
    ) -> FittingResult:
        """Calculate transformation matrix and projection parameters for 3D accessory.

        Args:
            pose: Estimated HeadPoseResult containing rotation and translation vectors.
            pixel_landmarks: 2D pixel coordinates of facial landmarks.
            frame_shape: (height, width) of the video frame.
            model_scale_factor: User-adjustable scale multiplier.
            focal_length: Optional camera focal length in pixels for distance-invariant scaling.

        Returns:
            FittingResult with homogeneous matrix and OpenCV projection vectors.
        """
        if not pose.is_valid:
            return FittingResult(
                model_matrix_4x4=np.eye(4),
                rvec=np.zeros((3, 1)),
                tvec=np.zeros((3, 1)),
                scale=1.0,
                anchor_center_2d=(0, 0),
                is_valid=False,
            )

        R_head = pose.rotation_matrix
        t_head = pose.tvec.flatten()

        # Anatomical anchor offset in head-local coordinates (mm)
        local_offset = self.DEFAULT_BRIDGE_OFFSET_MM.copy()
        local_offset[1] += self.vertical_offset_mm
        local_offset[2] += self.depth_offset_mm

        # Transform offset into camera coordinate system:
        # t_acc_cam = t_head + R_head * local_offset
        t_accessory = t_head + (R_head @ local_offset)

        # Proportional scale based on user's eye and face dimensions:
        scale = float(model_scale_factor)
        if focal_length and pose.tvec[2, 0] > 100.0 and len(pixel_landmarks) > 263:
            p_left = pixel_landmarks[33][:2]
            p_right = pixel_landmarks[263][:2]
            d_eyes_px = float(np.linalg.norm(p_right - p_left))
            if d_eyes_px > 10.0:
                # 3D outer eye distance in mm (canonical adult model is ~95mm)
                eye_dist_3d = (d_eyes_px * pose.tvec[2, 0]) / focal_length
                scale_ratio = eye_dist_3d / 95.0
                scale = float(np.clip(scale_ratio * model_scale_factor, 0.85, 1.25))

        # Build 4x4 homogeneous transformation matrix
        model_matrix = np.eye(4, dtype=np.float64)
        model_matrix[:3, :3] = R_head * scale
        model_matrix[:3, 3] = t_accessory

        # 2D projection anchor: use nose bridge pixel coordinate (landmark 168)
        p_bridge = pixel_landmarks[168][:2].astype(int)
        anchor_2d = (int(p_bridge[0]), int(p_bridge[1]))

        return FittingResult(
            model_matrix_4x4=model_matrix,
            rvec=pose.rvec,
            tvec=t_accessory.reshape((3, 1)),
            scale=scale,
            anchor_center_2d=anchor_2d,
            is_valid=True,
        )
