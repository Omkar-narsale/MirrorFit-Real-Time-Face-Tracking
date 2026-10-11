"""Face tracking module for MirrorFit.

Uses Google MediaPipe FaceLandmarker to detect 478 high-density 3D facial landmarks,
convert normalized coordinates to pixel coordinates, render visualizations,
and compute key geometric anchors for virtual glasses try-on.
"""

from dataclasses import dataclass
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import urllib.request

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import (
    FaceLandmarksConnections,
    drawing_styles,
    drawing_utils,
)


OFFICIAL_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)


@dataclass
class TryOnAnchors:
    """Key facial landmark pixel coordinates and geometric measurements for accessory fitting."""

    left_eye_center: Tuple[int, int]
    right_eye_center: Tuple[int, int]
    left_eye_outer: Tuple[int, int]
    left_eye_inner: Tuple[int, int]
    right_eye_outer: Tuple[int, int]
    right_eye_inner: Tuple[int, int]
    nose_bridge: Tuple[int, int]
    nose_tip: Tuple[int, int]
    left_temple: Tuple[int, int]
    right_temple: Tuple[int, int]
    interpupillary_distance_px: float
    glasses_width_px: float
    face_tilt_deg: float


class FaceTracker:
    """Manages MediaPipe FaceLandmarker inference, landmark extraction, and visualization.

    MediaPipe Face Landmarker produces 478 3D facial landmarks:
    - Landmarks 0 to 467: Standard canonical facial mesh surface.
    - Landmarks 468 to 472: Left eye iris landmarks (468 = iris center).
    - Landmarks 473 to 477: Right eye iris landmarks (473 = iris center).
    """

    # Canonical landmark index references
    IDX_NOSE_TIP = 4
    IDX_NOSE_BRIDGE = 168  # Glabella/saddle point between eyes (ideal glasses rest)
    IDX_NOSE_TOP = 6
    IDX_CHIN = 152
    IDX_FOREHEAD_CENTER = 10

    # Eye and Iris landmark indices
    IDX_RIGHT_EYE_OUTER = 33
    IDX_RIGHT_EYE_INNER = 133
    IDX_RIGHT_IRIS_CENTER = 473

    IDX_LEFT_EYE_OUTER = 263
    IDX_LEFT_EYE_INNER = 362
    IDX_LEFT_IRIS_CENTER = 468

    # Lateral temple anchors for glasses arms / frame width
    IDX_RIGHT_TEMPLE = 234
    IDX_LEFT_TEMPLE = 454

    def __init__(
        self,
        model_path: Optional[str] = None,
        num_faces: int = 1,
        min_face_detection_confidence: float = 0.5,
        min_face_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        """Initialize the MediaPipe Face Landmarker.

        Args:
            model_path: Path to the `face_landmarker.task` model file.
                        If None, searches standard directories.
            num_faces: Maximum number of faces to detect simultaneously.
            min_face_detection_confidence: Confidence threshold for face detection.
            min_face_presence_confidence: Confidence threshold for landmark presence.
            min_tracking_confidence: Confidence threshold for tracking between frames.

        Raises:
            FileNotFoundError: If the model asset is missing.
            RuntimeError: If MediaPipe fails to initialize.
        """
        self.model_path = self._resolve_model_path(model_path)
        self.num_faces = num_faces
        self.detector = self._initialize_detector(
            min_face_detection_confidence,
            min_face_presence_confidence,
            min_tracking_confidence,
        )

    @classmethod
    def download_model(cls, destination_path: str = "models/face_landmarker.task") -> str:
        """Download the official MediaPipe Face Landmarker model asset.

        Args:
            destination_path: Local file path where the model should be stored.

        Returns:
            str: Resolved absolute path to the downloaded model file.
        """
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        print(f"[MirrorFit] Downloading official MediaPipe model from:\n  {OFFICIAL_MODEL_URL}")
        urllib.request.urlretrieve(OFFICIAL_MODEL_URL, destination_path)
        print(f"[MirrorFit] Model successfully saved to: {destination_path}")
        return os.path.abspath(destination_path)

    def _resolve_model_path(self, model_path: Optional[str]) -> str:
        """Verify the model file exists, checking candidate locations."""
        candidate_paths: List[str] = []

        if model_path:
            candidate_paths.append(model_path)

        # Standard search paths in project
        candidate_paths.extend([
            os.path.join("models", "face_landmarker.task"),
            os.path.join("assets", "models", "face_landmarker.task"),
            "face_landmarker.task",
            os.path.join(os.path.dirname(__file__), "..", "models", "face_landmarker.task"),
        ])

        for path in candidate_paths:
            resolved = os.path.abspath(path)
            if os.path.exists(resolved) and os.path.getsize(resolved) > 0:
                return resolved

        # If not found anywhere, raise a descriptive, helpful error message
        error_msg = (
            "\n" + "=" * 70 + "\n"
            " [MirrorFit ERROR] MediaPipe Model Asset Missing!\n"
            "=" * 70 + "\n"
            "The Face Landmarker requires the official Google model file:\n"
            "  'face_landmarker.task'\n\n"
            "How to resolve:\n"
            "1. Download the official model from Google:\n"
            f"   {OFFICIAL_MODEL_URL}\n\n"
            "2. Save the file to your project at:\n"
            f"   {os.path.abspath(os.path.join('models', 'face_landmarker.task'))}\n\n"
            "3. Or run this one-line command in your terminal:\n"
            f"   curl -o models/face_landmarker.task {OFFICIAL_MODEL_URL}\n"
            "=" * 70 + "\n"
        )
        raise FileNotFoundError(error_msg)

    def _initialize_detector(
        self,
        detection_confidence: float,
        presence_confidence: float,
        tracking_confidence: float,
    ) -> vision.FaceLandmarker:
        """Create and configure the MediaPipe FaceLandmarker instance."""
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=self.num_faces,
            min_face_detection_confidence=detection_confidence,
            min_face_presence_confidence=presence_confidence,
            min_tracking_confidence=tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        try:
            return vision.FaceLandmarker.create_from_options(options)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize MediaPipe FaceLandmarker with model '{self.model_path}': {exc}"
            ) from exc

    def process_frame(
        self, frame_bgr: np.ndarray
    ) -> Tuple[bool, Optional[List[Any]], Optional[np.ndarray], Optional[TryOnAnchors]]:
        """Process a single camera frame to detect and extract face landmarks.

        Args:
            frame_bgr: Input video frame in OpenCV's BGR color space.

        Returns:
            Tuple containing:
            - is_detected (bool): True if at least one face was detected.
            - raw_landmarks: The raw MediaPipe NormalizedLandmark objects list (or None).
            - pixel_coords: NumPy array of shape (478, 3) with (px, py, depth_z) in integer pixels (or None).
            - tryon_anchors: TryOnAnchors dataclass with key fitting metrics (or None).
        """
        # MediaPipe expects RGB format, while OpenCV frames are in BGR format
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        result = self.detector.detect(mp_image)

        # Handle frames where no face is present
        if not result.face_landmarks:
            return False, None, None, None

        # Take primary face (index 0)
        first_face_raw = result.face_landmarks[0]
        h, w = frame_bgr.shape[:2]

        # Convert normalized coordinates to pixel space
        pixel_coords = self.convert_landmarks_to_pixels(first_face_raw, w, h)
        tryon_anchors = self.compute_tryon_anchors(pixel_coords)

        return True, first_face_raw, pixel_coords, tryon_anchors

    @staticmethod
    def convert_landmarks_to_pixels(
        landmarks: List[Any], width: int, height: int
    ) -> np.ndarray:
        """Convert normalized MediaPipe landmarks [0.0, 1.0] into pixel coordinates.

        Mathematical mapping:
            pixel_x = int(normalized_x * image_width)
            pixel_y = int(normalized_y * image_height)
            depth_z = normalized_z * image_width (relative depth scaling)

        Args:
            landmarks: List of NormalizedLandmark objects from MediaPipe.
            width: Image width in pixels.
            height: Image height in pixels.

        Returns:
            np.ndarray: Array of shape (N, 3) where columns are [px_x, px_y, depth_z].
        """
        num_pts = len(landmarks)
        coords = np.zeros((num_pts, 3), dtype=np.float32)

        for i, lm in enumerate(landmarks):
            px = int(np.clip(lm.x * width, 0, width - 1))
            py = int(np.clip(lm.y * height, 0, height - 1))
            coords[i] = [px, py, lm.z * width]

        return coords

    def compute_tryon_anchors(self, pixel_coords: np.ndarray) -> TryOnAnchors:
        """Extract key facial anchors and calculate geometry for glasses overlay (Phase 2).

        Args:
            pixel_coords: Array of shape (478, 3) with pixel coordinates.

        Returns:
            TryOnAnchors: Structured dataclass with coordinates, distances, and tilt.
        """
        # Eye centers: utilize iris center landmarks if available (468 and 473)
        left_eye_center = (
            int(pixel_coords[self.IDX_LEFT_IRIS_CENTER, 0]),
            int(pixel_coords[self.IDX_LEFT_IRIS_CENTER, 1]),
        )
        right_eye_center = (
            int(pixel_coords[self.IDX_RIGHT_IRIS_CENTER, 0]),
            int(pixel_coords[self.IDX_RIGHT_IRIS_CENTER, 1]),
        )

        left_eye_outer = (
            int(pixel_coords[self.IDX_LEFT_EYE_OUTER, 0]),
            int(pixel_coords[self.IDX_LEFT_EYE_OUTER, 1]),
        )
        left_eye_inner = (
            int(pixel_coords[self.IDX_LEFT_EYE_INNER, 0]),
            int(pixel_coords[self.IDX_LEFT_EYE_INNER, 1]),
        )

        right_eye_outer = (
            int(pixel_coords[self.IDX_RIGHT_EYE_OUTER, 0]),
            int(pixel_coords[self.IDX_RIGHT_EYE_OUTER, 1]),
        )
        right_eye_inner = (
            int(pixel_coords[self.IDX_RIGHT_EYE_INNER, 0]),
            int(pixel_coords[self.IDX_RIGHT_EYE_INNER, 1]),
        )

        nose_bridge = (
            int(pixel_coords[self.IDX_NOSE_BRIDGE, 0]),
            int(pixel_coords[self.IDX_NOSE_BRIDGE, 1]),
        )
        nose_tip = (
            int(pixel_coords[self.IDX_NOSE_TIP, 0]),
            int(pixel_coords[self.IDX_NOSE_TIP, 1]),
        )

        left_temple = (
            int(pixel_coords[self.IDX_LEFT_TEMPLE, 0]),
            int(pixel_coords[self.IDX_LEFT_TEMPLE, 1]),
        )
        right_temple = (
            int(pixel_coords[self.IDX_RIGHT_TEMPLE, 0]),
            int(pixel_coords[self.IDX_RIGHT_TEMPLE, 1]),
        )

        # Interpupillary Distance (IPD): Euclidean distance between eye centers
        # d = sqrt((x2 - x1)^2 + (y2 - y1)^2)
        dx_eyes = left_eye_center[0] - right_eye_center[0]
        dy_eyes = left_eye_center[1] - right_eye_center[1]
        ipd = float(np.hypot(dx_eyes, dy_eyes))

        # Glasses width based on temple-to-temple span
        dx_temple = left_temple[0] - right_temple[0]
        dy_temple = left_temple[1] - right_temple[1]
        glasses_width = float(np.hypot(dx_temple, dy_temple))

        # Face roll angle (head tilt) in degrees:
        # arctan2(dy, dx) returns angle of the eye line relative to horizontal
        face_tilt_deg = float(np.degrees(np.arctan2(dy_eyes, dx_eyes)))

        return TryOnAnchors(
            left_eye_center=left_eye_center,
            right_eye_center=right_eye_center,
            left_eye_outer=left_eye_outer,
            left_eye_inner=left_eye_inner,
            right_eye_outer=right_eye_outer,
            right_eye_inner=right_eye_inner,
            nose_bridge=nose_bridge,
            nose_tip=nose_tip,
            left_temple=left_temple,
            right_temple=right_temple,
            interpupillary_distance_px=round(ipd, 1),
            glasses_width_px=round(glasses_width, 1),
            face_tilt_deg=round(face_tilt_deg, 2),
        )

    def draw_landmarks(
        self,
        frame: np.ndarray,
        raw_landmarks: Optional[List[Any]],
        tryon_anchors: Optional[TryOnAnchors],
        show_mesh: bool = True,
        show_contours: bool = True,
        show_anchors: bool = True,
    ) -> np.ndarray:
        """Render facial landmarks and try-on geometric anchors onto the frame.

        Args:
            frame: OpenCV image frame (modified in-place).
            raw_landmarks: MediaPipe NormalizedLandmark list.
            tryon_anchors: TryOnAnchors computed from landmarks.
            show_mesh: If True, draws fine facial mesh tessellation.
            show_contours: If True, highlights eyes, lips, and face oval contours.
            show_anchors: If True, highlights virtual try-on anchor points.

        Returns:
            np.ndarray: Annotated frame.
        """
        if raw_landmarks is None:
            return frame

        # 1. Subtle Tessellation Mesh (soft wireframe)
        if show_mesh:
            drawing_utils.draw_landmarks(
                image=frame,
                landmark_list=raw_landmarks,
                connections=FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style(),
            )

        # 2. Key Facial Contours (face oval, eyes, eyebrows, lips)
        if show_contours:
            drawing_utils.draw_landmarks(
                image=frame,
                landmark_list=raw_landmarks,
                connections=FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style(),
            )
            # Irises
            drawing_utils.draw_landmarks(
                image=frame,
                landmark_list=raw_landmarks,
                connections=FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style(),
            )
            drawing_utils.draw_landmarks(
                image=frame,
                landmark_list=raw_landmarks,
                connections=FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style(),
            )

        # 3. Virtual Try-On Accessory Anchors (Phase 2 Preview)
        if show_anchors and tryon_anchors is not None:
            self._draw_tryon_overlay(frame, tryon_anchors)

        return frame

    def _draw_tryon_overlay(self, frame: np.ndarray, anchors: TryOnAnchors) -> None:
        """Draw high-contrast markers for glasses positioning anchors."""
        color_cyan = (255, 220, 0)      # BGR: Bright Cyan/Yellow
        color_magenta = (255, 0, 255)   # BGR: Magenta
        color_green = (50, 255, 100)    # BGR: Emerald Green

        # Draw line connecting eye centers (shows tilt orientation)
        cv2.line(
            frame,
            anchors.right_eye_center,
            anchors.left_eye_center,
            color_cyan,
            2,
            cv2.LINE_AA,
        )

        # Draw eye center circles (pupils)
        cv2.circle(frame, anchors.left_eye_center, 4, color_cyan, -1, cv2.LINE_AA)
        cv2.circle(frame, anchors.right_eye_center, 4, color_cyan, -1, cv2.LINE_AA)

        # Draw glasses bridge anchor (nose bridge)
        cv2.circle(frame, anchors.nose_bridge, 5, color_magenta, -1, cv2.LINE_AA)
        cv2.circle(frame, anchors.nose_bridge, 8, color_magenta, 1, cv2.LINE_AA)

        # Draw temple hinges
        cv2.circle(frame, anchors.left_temple, 4, color_green, -1, cv2.LINE_AA)
        cv2.circle(frame, anchors.right_temple, 4, color_green, -1, cv2.LINE_AA)

    def close(self) -> None:
        """Release MediaPipe detector resources."""
        if hasattr(self, "detector") and self.detector is not None:
            self.detector.close()
            self.detector = None
