"""Face and portrait segmentation module for MirrorFit XR.

Uses MediaPipe ImageSegmenter (selfie_segmenter.tflite) to extract high-accuracy
real-time foreground human masks and measure segmentation latency.
"""

import os
import time
from typing import Optional, Tuple
import urllib.request

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


OFFICIAL_SEGMENTER_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_segmenter/float16/latest/selfie_segmenter.tflite"
)


class FaceSegmentationModule:
    """Manages real-time portrait and face segmentation inference."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        """Initialize MediaPipe ImageSegmenter.

        Args:
            model_path: Path to selfie_segmenter.tflite model file.
        """
        self.model_path = self._resolve_model_path(model_path)
        self.segmenter: Optional[vision.ImageSegmenter] = None
        self.is_available: bool = False

        if self.model_path:
            self._init_detector()

    @classmethod
    def download_model(cls, destination: str = "models/selfie_segmenter.tflite") -> str:
        """Download official MediaPipe selfie segmentation model weights."""
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        print(f"[MirrorFit XR] Downloading official selfie segmenter from:\n  {OFFICIAL_SEGMENTER_URL}")
        urllib.request.urlretrieve(OFFICIAL_SEGMENTER_URL, destination)
        print(f"[MirrorFit XR] Segmentation model saved to: {destination}")
        return os.path.abspath(destination)

    def _resolve_model_path(self, custom_path: Optional[str]) -> Optional[str]:
        """Verify model file existence across standard paths."""
        candidates = []
        if custom_path:
            candidates.append(custom_path)
        candidates.extend([
            os.path.join("models", "selfie_segmenter.tflite"),
            os.path.join("assets", "models", "selfie_segmenter.tflite"),
            "selfie_segmenter.tflite",
        ])

        for path in candidates:
            resolved = os.path.abspath(path)
            if os.path.exists(resolved) and os.path.getsize(resolved) > 0:
                return resolved

        print(
            "\n[MirrorFit XR Warning] Segmentation model 'models/selfie_segmenter.tflite' not found.\n"
            "Face segmentation will operate in fallback mode (disabled).\n"
            f"To enable, download the official model from:\n  {OFFICIAL_SEGMENTER_URL}\n"
        )
        return None

    def _init_detector(self) -> None:
        """Instantiate MediaPipe ImageSegmenter."""
        try:
            base_options = python.BaseOptions(model_asset_path=self.model_path)
            options = vision.ImageSegmenterOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                output_category_mask=True,
                output_confidence_masks=True,
            )
            self.segmenter = vision.ImageSegmenter.create_from_options(options)
            self.is_available = True
            print("[MirrorFit XR] Face segmentation module initialized successfully.")
        except Exception as exc:
            print(f"[MirrorFit XR Warning] Failed to initialize segmentation module: {exc}")
            self.is_available = False

    def segment(self, frame_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """Perform segmentation on the frame and measure inference latency.

        Args:
            frame_bgr: Input video frame.

        Returns:
            Tuple[Optional[np.ndarray], float]: (binary_mask_uint8, latency_ms).
        """
        if not self.is_available or self.segmenter is None:
            return None, 0.0

        start_time = time.perf_counter()

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        try:
            result = self.segmenter.segment(mp_image)
            latency_ms = (time.perf_counter() - start_time) * 1000.0

            if result.confidence_masks:
                # Use first confidence mask (person probability)
                conf_mask = result.confidence_masks[0].numpy_view()
                binary_mask = (conf_mask > 0.45).astype(np.uint8) * 255
                return binary_mask, latency_ms
            elif result.category_mask:
                cat_mask = result.category_mask.numpy_view()
                binary_mask = (cat_mask > 0).astype(np.uint8) * 255
                return binary_mask, latency_ms
            else:
                return None, latency_ms
        except Exception as exc:
            print(f"[MirrorFit XR] Segmentation error: {exc}")
            return None, 0.0

    @staticmethod
    def render_debug_overlay(
        frame: np.ndarray,
        mask: np.ndarray,
        color: Tuple[int, int, int] = (0, 230, 110),
        alpha: float = 0.35,
    ) -> np.ndarray:
        """Render a semi-transparent color overlay highlighting the segmentation mask."""
        if mask is None:
            return frame

        # Squeeze trailing singleton dimension if present
        if len(mask.shape) == 3 and mask.shape[2] == 1:
            mask = mask[:, :, 0]

        # Resize mask to frame dimensions if necessary
        frame_h, frame_w = frame.shape[:2]
        if mask.shape[:2] != (frame_h, frame_w):
            mask = cv2.resize(mask, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)

        mask_bool = mask > 128
        if not np.any(mask_bool):
            return frame

        colored_mask = np.zeros_like(frame)
        colored_mask[:] = color

        frame[mask_bool] = (
            (1.0 - alpha) * frame[mask_bool].astype(np.float32)
            + alpha * colored_mask[mask_bool].astype(np.float32)
        ).astype(np.uint8)

        # Highlight contour edge
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (255, 255, 255), 1, cv2.LINE_AA)
        return frame

    def close(self) -> None:
        """Release segmentation resources."""
        if self.segmenter is not None:
            self.segmenter.close()
            self.segmenter = None
            self.is_available = False
