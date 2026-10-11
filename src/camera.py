"""Camera module for MirrorFit.

Manages video capture hardware, frame acquisition, horizontal mirroring,
and safe resource cleanup using OpenCV.
"""

from typing import Optional, Tuple
import cv2
import numpy as np


class Camera:
    """Manages webcam capture lifecycle and frame preprocessing.

    Features:
    - Robust device initialization with DirectShow support on Windows.
    - Automatic horizontal mirroring for natural physical-mirror UX.
    - Error detection for dropped or failed frames.
    - Python context manager protocol support (`with Camera() as cam:`).
    """

    def __init__(
        self,
        device_index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        mirror: bool = True,
    ) -> None:
        """Initialize camera parameters.

        Args:
            device_index: Webcam hardware index (default is 0).
            width: Desired frame width in pixels (e.g., 1280 for 720p).
            height: Desired frame height in pixels (e.g., 720 for 720p).
            fps: Desired frame rate target.
            mirror: If True, horizontally flip frames to emulate a real mirror.
        """
        self.device_index = device_index
        self.target_width = width
        self.target_height = height
        self.target_fps = fps
        self.mirror = mirror
        self._cap: Optional[cv2.VideoCapture] = None

    def start(self) -> None:
        """Open the camera device and configure frame properties.

        Raises:
            RuntimeError: If the webcam cannot be accessed or initialized.
        """
        if self.is_opened:
            return

        # Attempt opening with DirectShow backend (Windows-optimized) first
        # DirectShow avoids long startup latency often experienced with MSMF.
        self._cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)

        # Fallback to standard default backend if DirectShow fails to open
        if not self._cap.isOpened():
            self._cap.release()
            self._cap = cv2.VideoCapture(self.device_index)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Failed to open webcam (device index {self.device_index}). "
                "Ensure your camera is connected, not used by another application, "
                "and that camera access is enabled in Windows Privacy Settings."
            )

        # Configure preferred resolution and framerate
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)

    @property
    def is_opened(self) -> bool:
        """Check if the video capture object is currently open and valid."""
        return self._cap is not None and self._cap.isOpened()

    def get_resolution(self) -> Tuple[int, int]:
        """Return the actual active resolution (width, height) of the video feed."""
        if not self.is_opened:
            return 0, 0
        actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return actual_width, actual_height

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Capture the next video frame from the camera stream.

        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success_flag, bgr_frame).
                If capture fails, returns (False, None).
        """
        if not self.is_opened:
            return False, None

        ret, frame = self._cap.read()
        if not ret or frame is None:
            return False, None

        # Horizontal flipping (mirroring) creates an intuitive interaction:
        # moving your head to the right mirrors to the right of the screen.
        if self.mirror:
            frame = cv2.flip(frame, 1)

        return True, frame

    def release(self) -> None:
        """Release the camera hardware and clean up underlying capture handles."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "Camera":
        """Enter context manager, starting the camera automatically."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager, guaranteeing hardware resource release."""
        self.release()
