"""Glasses overlay module for MirrorFit.

Handles loading transparent PNG glasses assets, computing facial geometric anchors,
resizing according to eye distance, rotating based on the eye line, boundary clipping,
and alpha compositing onto live webcam frames.
"""

from dataclasses import dataclass
import os
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class OverlayState:
    """Smoothed geometric state to eliminate landmark detection jitter."""

    center_x: float
    center_y: float
    width: float
    angle_deg: float


class GlassesOverlay:
    """Manages virtual glasses transformation and alpha blending onto video frames.

    Key Features:
    - Safe 4-channel BGRA loading with alpha channel validation.
    - Resolution-independent proportional scaling based on Interpupillary Distance (IPD).
    - Unclipped affine rotation using expanded bounding boxes.
    - Vectorized NumPy alpha blending with safe viewport boundary clipping.
    - Exponential Moving Average (EMA) smoothing for jitter-free tracking.
    """

    def __init__(
        self,
        asset_path: str = "assets/glasses/default_glasses.png",
        scale_factor: float = 2.15,
        vertical_offset_ratio: float = 0.05,
        smoothing_factor: float = 0.35,
    ) -> None:
        """Initialize the glasses overlay engine.

        Args:
            asset_path: Path to transparent PNG glasses asset.
            scale_factor: Multiplier applied to eye distance to determine glasses width.
                          Typically 2.0 - 2.2 for human facial proportions.
            vertical_offset_ratio: Proportional vertical shift relative to glasses height.
                                   Positive values nudge glasses down towards nose bridge.
            smoothing_factor: Exponential smoothing weight (beta in [0.1, 1.0]).
                              Higher values react faster; lower values are smoother.

        Raises:
            FileNotFoundError: If the asset file cannot be located.
            ValueError: If the asset does not contain a genuine alpha channel.
        """
        self.scale_factor = scale_factor
        self.vertical_offset_ratio = vertical_offset_ratio
        self.smoothing_factor = smoothing_factor

        self.glasses_raw = self._load_glasses_asset(asset_path)
        self.orig_h, self.orig_w = self.glasses_raw.shape[:2]
        self.aspect_ratio = self.orig_h / self.orig_w

        # State filter for temporal smoothing (EMA)
        self.state: Optional[OverlayState] = None

    def _load_glasses_asset(self, path: str) -> np.ndarray:
        """Load and validate the transparent PNG asset."""
        resolved_path = os.path.abspath(path)

        if not os.path.exists(resolved_path):
            raise FileNotFoundError(
                f"\n[MirrorFit ERROR] Glasses asset not found at:\n  {resolved_path}\n"
                "Please place a valid transparent PNG glasses image at 'assets/glasses/default_glasses.png'."
            )

        # Read image with transparency intact (IMREAD_UNCHANGED retains alpha channel)
        glasses = cv2.imread(resolved_path, cv2.IMREAD_UNCHANGED)

        if glasses is None:
            raise ValueError(
                f"[MirrorFit ERROR] Failed to decode image at '{resolved_path}'. "
                "Ensure the file is a valid, uncorrupted PNG image."
            )

        # Verify genuine 4-channel (BGRA) format
        if len(glasses.shape) != 3 or glasses.shape[2] != 4:
            raise ValueError(
                f"[MirrorFit ERROR] The image at '{resolved_path}' has {glasses.shape[-1] if len(glasses.shape) == 3 else 1} channels. "
                "A genuine transparent PNG must have 4 channels (Blue, Green, Red, Alpha)."
            )

        # Verify the alpha channel is not completely opaque or completely transparent
        alpha_channel = glasses[:, :, 3]
        if np.all(alpha_channel == 255):
            print(
                f"[MirrorFit WARNING] '{path}' has no transparent pixels (fully opaque). "
                "The glasses may appear with an opaque bounding rectangle."
            )
        elif np.all(alpha_channel == 0):
            raise ValueError(
                f"[MirrorFit ERROR] '{path}' is completely transparent (alpha is zero everywhere)."
            )

        return glasses

    def reset_filter(self) -> None:
        """Reset temporal filter when face tracking is lost to prevent motion lag."""
        self.state = None

    def compute_geometry(
        self,
        left_eye: Tuple[int, int],
        right_eye: Tuple[int, int],
        nose_bridge: Optional[Tuple[int, int]] = None,
    ) -> Tuple[int, int, int, float]:
        """Calculate placement center, dimensions, and rotation angle from eye landmarks.

        Args:
            left_eye: Pixel (x, y) coordinates of one eye center.
            right_eye: Pixel (x, y) coordinates of the other eye center.
            nose_bridge: Optional pixel (x, y) coordinates of the nose saddle point.

        Returns:
            Tuple[int, int, int, float]: (center_x, center_y, glasses_width, rotation_angle_deg).
        """
        # Order landmarks in viewer's coordinate space:
        # p1 is on viewer's left (smaller x), p2 is on viewer's right (larger x)
        if left_eye[0] < right_eye[0]:
            p1, p2 = left_eye, right_eye
        else:
            p1, p2 = right_eye, left_eye

        # Interpupillary vector and Euclidean distance
        dx = float(p2[0] - p1[0])
        dy = float(p2[1] - p1[1])
        eye_distance = float(np.hypot(dx, dy))

        # Anchor center calculation:
        # Blend eye midpoint with the nose bridge (landmark 168) for natural nasal fit
        raw_mid_x = (p1[0] + p2[0]) / 2.0
        raw_mid_y = (p1[1] + p2[1]) / 2.0

        if nose_bridge is not None:
            # 70% eye midpoint, 30% nose bridge anchor
            center_x = 0.7 * raw_mid_x + 0.3 * float(nose_bridge[0])
            center_y = 0.7 * raw_mid_y + 0.3 * float(nose_bridge[1])
        else:
            center_x = raw_mid_x
            center_y = raw_mid_y

        # Target width based on eye span
        target_width = max(20.0, eye_distance * self.scale_factor)

        # Vertical offset (slight downward nudge towards nose bridge)
        target_height = target_width * self.aspect_ratio
        center_y += target_height * self.vertical_offset_ratio

        # Rotation angle:
        # arctan2(dy, dx) returns angle of eye line relative to horizontal.
        # In image space (+y downwards):
        # A positive dy means viewer's right eye is lower (head tilted clockwise).
        # In OpenCV, cv2.getRotationMatrix2D rotates COUNTER-CLOCKWISE with positive angle.
        # Therefore, rotation angle for OpenCV is -theta to achieve clockwise tilt!
        angle_rad = np.arctan2(dy, dx)
        angle_deg = float(np.degrees(angle_rad))
        opencv_rot_angle = -angle_deg

        # Apply Exponential Moving Average (EMA) to smooth out micro-jitter
        if self.state is None:
            self.state = OverlayState(
                center_x=center_x,
                center_y=center_y,
                width=target_width,
                angle_deg=opencv_rot_angle,
            )
        else:
            beta = self.smoothing_factor
            self.state.center_x = beta * center_x + (1.0 - beta) * self.state.center_x
            self.state.center_y = beta * center_y + (1.0 - beta) * self.state.center_y
            self.state.width = beta * target_width + (1.0 - beta) * self.state.width
            self.state.angle_deg = beta * opencv_rot_angle + (1.0 - beta) * self.state.angle_deg

        return (
            int(round(self.state.center_x)),
            int(round(self.state.center_y)),
            max(20, int(round(self.state.width))),
            self.state.angle_deg,
        )

    def transform_glasses(
        self, target_width: int, rotation_angle_deg: float
    ) -> np.ndarray:
        """Resize and rotate the glasses asset while preserving transparency without clipping.

        Args:
            target_width: Desired width in pixels.
            rotation_angle_deg: Rotation angle in degrees (OpenCV convention).

        Returns:
            np.ndarray: 4-channel BGRA image of resized, rotated glasses.
        """
        # 1. Proportional resizing
        target_height = max(10, int(round(target_width * self.aspect_ratio)))
        resized = cv2.resize(
            self.glasses_raw,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA if target_width < self.orig_w else cv2.INTER_LINEAR,
        )

        # 2. Rotation with expanded bounding box:
        # Standard rotation clips corners. Calculating new bounding box ensures
        # the entire frame and temples remain visible even at steep angles.
        h, w = resized.shape[:2]
        cx, cy = w / 2.0, h / 2.0

        M = cv2.getRotationMatrix2D((cx, cy), rotation_angle_deg, 1.0)
        cos_val = np.abs(M[0, 0])
        sin_val = np.abs(M[0, 1])

        # New bounding box dimensions
        new_w = int(np.ceil((h * sin_val) + (w * cos_val)))
        new_h = int(np.ceil((h * cos_val) + (w * sin_val)))

        # Adjust transformation matrix to account for translation to new center
        M[0, 2] += (new_w / 2.0) - cx
        M[1, 2] += (new_h / 2.0) - cy

        # Warp image with zero alpha (transparent background) border
        rotated = cv2.warpAffine(
            resized,
            M,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

        return rotated

    def blend_overlay(
        self,
        frame: np.ndarray,
        overlay: np.ndarray,
        center_x: int,
        center_y: int,
    ) -> np.ndarray:
        """Alpha-blend the 4-channel overlay onto the 3-channel frame with viewport clipping.

        Args:
            frame: Destination BGR video frame (modified in-place).
            overlay: 4-channel BGRA rotated glasses graphic.
            center_x: Target center X in frame pixels.
            center_y: Target center Y in frame pixels.

        Returns:
            np.ndarray: Frame with blended overlay.
        """
        frame_h, frame_w = frame.shape[:2]
        ov_h, ov_w = overlay.shape[:2]

        # Calculate bounding box on the background frame
        x1 = int(round(center_x - ov_w / 2.0))
        y1 = int(round(center_y - ov_h / 2.0))
        x2 = x1 + ov_w
        y2 = y1 + ov_h

        # Clip coordinates to frame viewport boundaries [0, frame_w] x [0, frame_h]
        x1_clip = max(0, x1)
        y1_clip = max(0, y1)
        x2_clip = min(frame_w, x2)
        y2_clip = min(frame_h, y2)

        # If overlay is completely outside the frame, skip rendering
        if x1_clip >= x2_clip or y1_clip >= y2_clip:
            return frame

        # Calculate corresponding slice inside the overlay image
        ov_x1 = x1_clip - x1
        ov_y1 = y1_clip - y1
        ov_x2 = ov_x1 + (x2_clip - x1_clip)
        ov_y2 = ov_y1 + (y2_clip - y1_clip)

        # Extract regions of interest
        overlay_roi = overlay[ov_y1:ov_y2, ov_x1:ov_x2]
        frame_roi = frame[y1_clip:y2_clip, x1_clip:x2_clip]

        # Normalize alpha channel to [0.0, 1.0] with shape (H, W, 1) for broadcasting
        alpha = overlay_roi[:, :, 3:4].astype(np.float32) / 255.0
        overlay_bgr = overlay_roi[:, :, :3].astype(np.float32)
        frame_bgr = frame_roi.astype(np.float32)

        # Alpha blending formula:
        # Out = alpha * Overlay + (1 - alpha) * Background
        blended = (alpha * overlay_bgr + (1.0 - alpha) * frame_bgr).astype(np.uint8)

        # Assign back to frame
        frame[y1_clip:y2_clip, x1_clip:x2_clip] = blended
        return frame

    def set_accessory_image(self, image: np.ndarray) -> None:
        """Update active accessory image and recompute dimensions.

        Args:
            image: 4-channel BGRA transparent accessory image.
        """
        if image is not None and len(image.shape) == 3 and image.shape[2] == 4:
            self.glasses_raw = image
            self.orig_h, self.orig_w = image.shape[:2]
            self.aspect_ratio = self.orig_h / self.orig_w

    def apply(
        self,
        frame: np.ndarray,
        left_eye: Tuple[int, int],
        right_eye: Tuple[int, int],
        nose_bridge: Optional[Tuple[int, int]] = None,
        accessory_image: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Process landmarks, transform glasses, and composite onto frame.

        Args:
            frame: Video frame in BGR format (modified in-place).
            left_eye: Left eye center (px, py).
            right_eye: Right eye center (px, py).
            nose_bridge: Optional nose bridge point (px, py).
            accessory_image: Optional 4-channel BGRA image of active accessory.

        Returns:
            np.ndarray: Annotated video frame.
        """
        if accessory_image is not None:
            self.set_accessory_image(accessory_image)

        # 1. Compute geometry with smoothing
        cx, cy, target_w, rot_deg = self.compute_geometry(
            left_eye, right_eye, nose_bridge
        )

        # 2. Resize and rotate glasses without clipping
        transformed_glasses = self.transform_glasses(target_w, rot_deg)

        # 3. Alpha blend onto video frame
        return self.blend_overlay(frame, transformed_glasses, cx, cy)
