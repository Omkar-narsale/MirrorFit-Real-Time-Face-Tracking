"""Head pose estimation module for MirrorFit XR.

Computes 6-DOF 3D head orientation (Yaw, Pitch, Roll) and translation
using OpenCV solvePnP and canonical anthropometric 3D facial landmarks.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class HeadPoseResult:
    """Estimated 3D head pose and orientation metrics."""

    rvec: np.ndarray          # (3, 1) Rotation vector
    tvec: np.ndarray          # (3, 1) Translation vector
    rotation_matrix: np.ndarray  # (3, 3) Rotation matrix
    yaw_deg: float            # Horizontal head turn (- left, + right)
    pitch_deg: float          # Vertical head tilt (- up, + down)
    roll_deg: float           # Lateral tilt (- CCW, + CW)
    is_valid: bool


class HeadPoseEstimator:
    """Estimates 6-DOF head pose from 2D facial landmarks using perspective-n-point."""

    # Canonical 3D anthropometric face model in millimeters.
    # Coordinate system aligned with camera frame (+X right, +Y down, +Z away into head):
    # Origin (0,0,0) is at the tip of the nose
    CANONICAL_3D_MODEL = np.array([
        [0.0, 0.0, 0.0],          # Nose tip (MediaPipe 4)
        [0.0, 75.0, 15.0],        # Chin (MediaPipe 152)
        [-65.0, -32.0, 30.0],     # Left eye outer corner
        [65.0, -32.0, 30.0],      # Right eye outer corner
        [-38.0, 40.0, 15.0],      # Left mouth corner
        [38.0, 40.0, 15.0],       # Right mouth corner
        [0.0, -32.0, 15.0],       # Glabella / nose saddle (MediaPipe 168)
    ], dtype=np.float64)

    # Corresponding MediaPipe landmark indices
    IDX_NOSE_TIP = 4
    IDX_CHIN = 152
    IDX_LEFT_EYE_OUTER = 263
    IDX_RIGHT_EYE_OUTER = 33
    IDX_LEFT_MOUTH_CORNER = 291
    IDX_RIGHT_MOUTH_CORNER = 61
    IDX_NOSE_BRIDGE = 168

    def __init__(self, smoothing_factor: float = 0.35) -> None:
        """Initialize head pose estimator.

        Args:
            smoothing_factor: Weight beta for exponential moving average filter.
        """
        self.smoothing_factor = smoothing_factor
        self.smoothed_rvec: Optional[np.ndarray] = None
        self.smoothed_tvec: Optional[np.ndarray] = None
        self.smoothed_yaw: float = 0.0
        self.smoothed_pitch: float = 0.0
        self.smoothed_roll: float = 0.0

    def reset(self) -> None:
        """Reset temporal smoothing filters when face tracking is lost."""
        self.smoothed_rvec = None
        self.smoothed_tvec = None
        self.smoothed_yaw = 0.0
        self.smoothed_pitch = 0.0
        self.smoothed_roll = 0.0

    @staticmethod
    def get_camera_matrix(width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
        """Construct intrinsic camera calibration matrix based on image dimensions."""
        focal_length = float(width)
        center_x = float(width) / 2.0
        center_y = float(height) / 2.0

        camera_matrix = np.array([
            [focal_length, 0.0, center_x],
            [0.0, focal_length, center_y],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        return camera_matrix, dist_coeffs

    def estimate_pose(
        self,
        pixel_landmarks: np.ndarray,
        frame_shape: Tuple[int, int],
    ) -> HeadPoseResult:
        """Estimate 3D head pose from 2D pixel coordinates.

        Args:
            pixel_landmarks: Array of shape (478, 3) containing 2D pixel (x, y) coordinates.
            frame_shape: (height, width) of the video frame.

        Returns:
            HeadPoseResult containing rotation vector, translation vector, and Euler angles.
        """
        h, w = frame_shape[:2]
        cam_matrix, dist_coeffs = self.get_camera_matrix(w, h)

        # Retrieve 2D image points corresponding to canonical 3D model
        p_nose = pixel_landmarks[self.IDX_NOSE_TIP][:2]
        p_chin = pixel_landmarks[self.IDX_CHIN][:2]
        p_eye_a = pixel_landmarks[self.IDX_LEFT_EYE_OUTER][:2]
        p_eye_b = pixel_landmarks[self.IDX_RIGHT_EYE_OUTER][:2]
        p_mouth_a = pixel_landmarks[self.IDX_LEFT_MOUTH_CORNER][:2]
        p_mouth_b = pixel_landmarks[self.IDX_RIGHT_MOUTH_CORNER][:2]
        p_bridge = pixel_landmarks[self.IDX_NOSE_BRIDGE][:2]

        # Order lateral features in screen space (screen left = smaller x, screen right = larger x)
        if p_eye_a[0] <= p_eye_b[0]:
            p_eye_left, p_eye_right = p_eye_a, p_eye_b
        else:
            p_eye_left, p_eye_right = p_eye_b, p_eye_a

        if p_mouth_a[0] <= p_mouth_b[0]:
            p_mouth_left, p_mouth_right = p_mouth_a, p_mouth_b
        else:
            p_mouth_left, p_mouth_right = p_mouth_b, p_mouth_a

        image_points = np.array([
            p_nose,
            p_chin,
            p_eye_left,
            p_eye_right,
            p_mouth_left,
            p_mouth_right,
            p_bridge,
        ], dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            self.CANONICAL_3D_MODEL,
            image_points,
            cam_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return HeadPoseResult(
                rvec=np.zeros((3, 1)),
                tvec=np.zeros((3, 1)),
                rotation_matrix=np.eye(3),
                yaw_deg=0.0,
                pitch_deg=0.0,
                roll_deg=0.0,
                is_valid=False,
            )

        # Convert rotation vector to 3x3 matrix
        R, _ = cv2.Rodrigues(rvec)

        # Extract Euler angles (in degrees)
        # Note: In our aligned frame (+X right, +Y down, +Z into scene):
        # Yaw: rotation around Y axis
        # Pitch: rotation around X axis
        # Roll: rotation around Z axis
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        if sy > 1e-6:
            pitch = float(np.degrees(np.arctan2(R[2, 1], R[2, 2])))
            yaw = float(np.degrees(np.arctan2(-R[2, 0], sy)))
            roll = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
        else:
            pitch = float(np.degrees(np.arctan2(-R[1, 2], R[1, 1])))
            yaw = float(np.degrees(np.arctan2(-R[2, 0], sy)))
            roll = 0.0

        # Anti-jitter Exponential Moving Average (EMA) filtering
        if self.smoothed_rvec is None:
            self.smoothed_rvec = rvec.copy()
            self.smoothed_tvec = tvec.copy()
            self.smoothed_yaw = yaw
            self.smoothed_pitch = pitch
            self.smoothed_roll = roll
        else:
            beta = self.smoothing_factor
            self.smoothed_rvec = beta * rvec + (1.0 - beta) * self.smoothed_rvec
            self.smoothed_tvec = beta * tvec + (1.0 - beta) * self.smoothed_tvec
            self.smoothed_yaw = beta * yaw + (1.0 - beta) * self.smoothed_yaw
            self.smoothed_pitch = beta * pitch + (1.0 - beta) * self.smoothed_pitch
            self.smoothed_roll = beta * roll + (1.0 - beta) * self.smoothed_roll

        smoothed_R, _ = cv2.Rodrigues(self.smoothed_rvec)

        return HeadPoseResult(
            rvec=self.smoothed_rvec,
            tvec=self.smoothed_tvec,
            rotation_matrix=smoothed_R,
            yaw_deg=round(self.smoothed_yaw, 1),
            pitch_deg=round(self.smoothed_pitch, 1),
            roll_deg=round(self.smoothed_roll, 1),
            is_valid=True,
        )

    @staticmethod
    def draw_axes(
        frame: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: Optional[np.ndarray] = None,
        length: float = 65.0,
    ) -> None:
        """Render 3D coordinate axes projecting from the face (X: Red, Y: Green, Z: Blue)."""
        if dist_coeffs is None:
            dist_coeffs = np.zeros((4, 1))

        # Origin and 3 axis endpoints in 3D (Z points forward out from nose towards camera)
        axis_3d = np.array([
            [0.0, 0.0, 0.0],
            [length, 0.0, 0.0],    # X axis: Red (Lateral Right)
            [0.0, length, 0.0],    # Y axis: Green (Vertical Down)
            [0.0, 0.0, -length],   # Z axis: Blue (Forward Towards Camera)
        ], dtype=np.float64)

        img_pts, _ = cv2.projectPoints(axis_3d, rvec, tvec, camera_matrix, dist_coeffs)
        origin = tuple(np.round(img_pts[0].ravel()).astype(int))
        pt_x = tuple(np.round(img_pts[1].ravel()).astype(int))
        pt_y = tuple(np.round(img_pts[2].ravel()).astype(int))
        pt_z = tuple(np.round(img_pts[3].ravel()).astype(int))

        # Draw lines: X (Red), Y (Green), Z (Cyan/Blue)
        cv2.line(frame, origin, pt_x, (0, 0, 255), 3, cv2.LINE_AA)      # Red
        cv2.line(frame, origin, pt_y, (0, 255, 0), 3, cv2.LINE_AA)      # Green
        cv2.line(frame, origin, pt_z, (255, 200, 0), 3, cv2.LINE_AA)    # Cyan/Blue
        cv2.circle(frame, origin, 4, (255, 255, 255), -1, cv2.LINE_AA)
