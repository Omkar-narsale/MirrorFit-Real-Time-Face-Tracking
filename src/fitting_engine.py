"""Intelligent face fitting engine for MirrorFit XR Stage 5.

Features:
- Decoupled 3D transformation and anthropometric landmark fitting.
- Estimates eye positions, interpupillary distance (IPD), nose bridge root,
  and temporal face width from MediaPipe 478 landmarks.
- UserFittingAdjustments: Manual interactive adjustment of frame width, height,
  overall scale, horizontal/vertical offsets, and depth position.
- Anti-jitter exponential temporal filtering to eliminate sudden resizing or jumping.
- Clean separation between automatic landmark estimation and user-controlled offsets.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np

from src.head_pose import HeadPoseResult


@dataclass
class UserFittingAdjustments:
    """User-controlled manual adjustments applied on top of automatic fitting."""

    overall_scale: float = 1.0     # Overall size multiplier (0.65 to 1.60)
    width_scale: float = 1.0       # Horizontal frame width multiplier (0.70 to 1.45)
    height_scale: float = 1.0      # Vertical frame height multiplier (0.70 to 1.45)
    offset_x_mm: float = 0.0       # Lateral shift in mm (-35 to +35)
    offset_y_mm: float = 0.0       # Vertical shift in mm (-35 to +35, negative is up)
    offset_z_mm: float = 0.0       # Depth shift in mm (-45 to +45, positive is deeper)
    auto_fit_enabled: bool = True  # Whether dynamic anthropometric scaling is active

    def adjust_scale(self, delta: float) -> float:
        """Adjust overall scale multiplier."""
        self.overall_scale = float(np.clip(self.overall_scale + delta, 0.65, 1.60))
        return self.overall_scale

    def adjust_width(self, delta: float) -> float:
        """Adjust horizontal width multiplier."""
        self.width_scale = float(np.clip(self.width_scale + delta, 0.70, 1.45))
        return self.width_scale

    def adjust_height(self, delta: float) -> float:
        """Adjust vertical height multiplier."""
        self.height_scale = float(np.clip(self.height_scale + delta, 0.70, 1.45))
        return self.height_scale

    def adjust_offset(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> Tuple[float, float, float]:
        """Adjust local translation offsets in millimeters."""
        self.offset_x_mm = float(np.clip(self.offset_x_mm + dx, -35.0, 35.0))
        self.offset_y_mm = float(np.clip(self.offset_y_mm + dy, -35.0, 35.0))
        self.offset_z_mm = float(np.clip(self.offset_z_mm + dz, -45.0, 45.0))
        return self.offset_x_mm, self.offset_y_mm, self.offset_z_mm

    def toggle_autofit(self) -> bool:
        """Toggle dynamic facial proportion auto-fitting."""
        self.auto_fit_enabled = not self.auto_fit_enabled
        return self.auto_fit_enabled

    def reset(self) -> None:
        """Reset all manual fitting adjustments back to factory defaults."""
        self.overall_scale = 1.0
        self.width_scale = 1.0
        self.height_scale = 1.0
        self.offset_x_mm = 0.0
        self.offset_y_mm = 0.0
        self.offset_z_mm = 0.0
        self.auto_fit_enabled = True


@dataclass
class FittingResult:
    """Transformation parameters to position a 3D accessory on the face."""

    model_matrix_4x4: np.ndarray          # 4x4 Homogeneous transformation matrix (Model -> Camera)
    rvec: np.ndarray                      # (3, 1) Rotation vector for projection
    tvec: np.ndarray                      # (3, 1) Translation vector for projection
    scale: float                          # Uniform scale scalar
    scale_vec: np.ndarray                 # (3,) Anisotropic scale vector [scale_x, scale_y, scale_z]
    anchor_center_2d: Tuple[int, int]     # 2D projection bridge pixel coordinate
    is_valid: bool                        # Validity flag
    interpupillary_dist_mm: float = 64.0  # Estimated 3D interpupillary distance
    face_width_mm: float = 145.0          # Estimated 3D temporal face width
    auto_scale_ratio: float = 1.0         # Proportional scaling factor from face dimensions


class FittingEngine:
    """Computes geometric transformations for eyewear placement on detected facial anatomy.

    Mathematical Pipeline:
    1. Anatomical Anchor: Glasses rest across the nasal bridge (landmark 168 glabella).
    2. Anthropometric Proportions:
       - Distance between outer eye corners (landmarks 33 and 263).
       - Temporal face width between zygomatic/tragus landmarks (234 and 454).
       - Converts 2D pixel span to depth-invariant physical millimeter scale via camera focal length.
    3. User Adjustment Layer:
       Combines auto-scale with user overrides (width, height, overall scale, X/Y/Z offsets).
    4. Anti-Jitter Smoothing:
       Exponential temporal filter prevents micro-flickering and sudden jumps on face recovery.
    """

    # Anatomical offset from nose tip (solvePnP origin) to glabella / nasal saddle (in mm)
    # Aligned with canonical frame: -Y is up towards forehead, +Z is recessed into face
    DEFAULT_BRIDGE_OFFSET_MM = np.array([0.0, -32.0, 15.0], dtype=np.float64)

    def __init__(
        self,
        base_model_width_mm: float = 135.0,
        smoothing_factor: float = 0.35,
    ) -> None:
        """Initialize fitting engine.

        Args:
            base_model_width_mm: Standard adult eyewear width in millimeters (~135mm).
            smoothing_factor: Exponential smoothing weight beta in [0.05, 1.0].
        """
        self.base_model_width_mm = base_model_width_mm
        self.smoothing_factor = float(np.clip(smoothing_factor, 0.05, 1.0))
        self.adjustments = UserFittingAdjustments()

        # Smoothed state variables
        self._smoothed_scale: Optional[float] = None
        self._smoothed_tvec: Optional[np.ndarray] = None
        self._smoothed_ipd: float = 64.0
        self._smoothed_face_w: float = 145.0

    def reset_filters(self) -> None:
        """Reset temporal smoothing filters to prevent sudden jumping upon face re-entry."""
        self._smoothed_scale = None
        self._smoothed_tvec = None

    def compute_fitting(
        self,
        pose: HeadPoseResult,
        pixel_landmarks: np.ndarray,
        frame_shape: Tuple[int, int],
        base_model_scale: float = 1.0,
        base_bridge_offset: Optional[np.ndarray] = None,
        focal_length: Optional[float] = None,
    ) -> FittingResult:
        """Calculate 3D transformation matrix and projection parameters for eyewear fitting.

        Args:
            pose: Estimated HeadPoseResult containing rotation and translation vectors.
            pixel_landmarks: (N, 2) or (N, 3) pixel coordinates of facial landmarks.
            frame_shape: (height, width) of the video frame.
            base_model_scale: Style-specific baseline scale multiplier from catalogue.
            base_bridge_offset: Style-specific baseline bridge offset vector in mm.
            focal_length: Camera focal length in pixels for distance-invariant anthropometrics.

        Returns:
            FittingResult with homogeneous matrix, projection vectors, and scale vector.
        """
        if not pose.is_valid:
            self.reset_filters()
            return FittingResult(
                model_matrix_4x4=np.eye(4),
                rvec=np.zeros((3, 1)),
                tvec=np.zeros((3, 1)),
                scale=1.0,
                scale_vec=np.array([1.0, 1.0, 1.0]),
                anchor_center_2d=(0, 0),
                is_valid=False,
            )

        R_head = pose.rotation_matrix
        t_head = pose.tvec.flatten()

        # 1. Base Anatomical Anchor Position
        if base_bridge_offset is not None:
            local_offset = base_bridge_offset.copy().astype(np.float64)
        else:
            local_offset = self.DEFAULT_BRIDGE_OFFSET_MM.copy()

        # Apply user fine translation offsets
        local_offset[0] += self.adjustments.offset_x_mm
        local_offset[1] += self.adjustments.offset_y_mm
        local_offset[2] += self.adjustments.offset_z_mm

        # Transform offset into camera coordinate system:
        # t_acc_cam = t_head + R_head * local_offset
        t_accessory = t_head + (R_head @ local_offset)

        # 2. Dynamic Anthropometric Proportions & Scale
        auto_scale_ratio = 1.0
        ipd_mm = 64.0
        face_w_mm = 145.0

        if (
            self.adjustments.auto_fit_enabled
            and focal_length is not None
            and float(pose.tvec.flatten()[2]) > 100.0
            and len(pixel_landmarks) > 454
        ):
            # Outer eye corners: landmark 33 (user right) and 263 (user left)
            p_eye_r = pixel_landmarks[33][:2]
            p_eye_l = pixel_landmarks[263][:2]
            d_eyes_px = float(np.linalg.norm(p_eye_l - p_eye_r))

            # Temporal cheek margins: landmark 234 and 454
            p_face_r = pixel_landmarks[234][:2]
            p_face_l = pixel_landmarks[454][:2]
            d_face_px = float(np.linalg.norm(p_face_l - p_face_r))

            cam_z = float(pose.tvec.flatten()[2])
            if d_eyes_px > 10.0 and focal_length > 10.0:
                raw_ipd = (d_eyes_px * cam_z) / focal_length
                raw_face_w = (d_face_px * cam_z) / focal_length if d_face_px > 15.0 else 145.0

                # Temporal smoothing on measurements
                self._smoothed_ipd = 0.25 * raw_ipd + 0.75 * self._smoothed_ipd
                self._smoothed_face_w = 0.25 * raw_face_w + 0.75 * self._smoothed_face_w
                ipd_mm = self._smoothed_ipd
                face_w_mm = self._smoothed_face_w

                # Canonical outer-eye corner distance is ~95mm
                eye_ratio = ipd_mm / 95.0
                # Canonical temporal face width is ~145mm
                face_ratio = face_w_mm / 145.0

                # Blended anthropometric ratio (60% eyes, 40% face width)
                blended_ratio = 0.60 * eye_ratio + 0.40 * face_ratio
                auto_scale_ratio = float(np.clip(blended_ratio, 0.85, 1.25))

        # 3. Overall and Anisotropic Scale Vector
        raw_scale = (
            base_model_scale
            * (auto_scale_ratio if self.adjustments.auto_fit_enabled else 1.0)
            * self.adjustments.overall_scale
        )

        # Anti-jitter smoothing on scale and translation
        if self._smoothed_scale is None:
            self._smoothed_scale = raw_scale
            self._smoothed_tvec = t_accessory.copy()
        else:
            beta = self.smoothing_factor
            self._smoothed_scale = beta * raw_scale + (1.0 - beta) * self._smoothed_scale
            self._smoothed_tvec = beta * t_accessory + (1.0 - beta) * self._smoothed_tvec

        scale = float(self._smoothed_scale)
        t_final = self._smoothed_tvec

        # Anisotropic scale vector for width/height controls: [scale_x, scale_y, scale_z]
        scale_vec = np.array([
            scale * self.adjustments.width_scale,
            scale * self.adjustments.height_scale,
            scale,
        ], dtype=np.float64)

        # 4. Homogeneous 4x4 Model Transformation Matrix
        model_matrix = np.eye(4, dtype=np.float64)
        model_matrix[:3, :3] = R_head * scale
        model_matrix[:3, 3] = t_final

        # 2D projection anchor: use nose bridge pixel coordinate (landmark 168)
        p_bridge = pixel_landmarks[168][:2].astype(int)
        anchor_2d = (int(p_bridge[0]), int(p_bridge[1]))

        return FittingResult(
            model_matrix_4x4=model_matrix,
            rvec=pose.rvec,
            tvec=t_final.reshape((3, 1)),
            scale=scale,
            scale_vec=scale_vec,
            anchor_center_2d=anchor_2d,
            is_valid=True,
            interpupillary_dist_mm=round(ipd_mm, 1),
            face_width_mm=round(face_w_mm, 1),
            auto_scale_ratio=round(auto_scale_ratio, 2),
        )
