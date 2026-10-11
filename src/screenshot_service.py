"""Screenshot capture service for MirrorFit.

Manages saving timestamped rendered frames to an output directory
and providing confirmation feedback.
"""

from datetime import datetime
import os
from typing import Optional

import cv2
import numpy as np


class ScreenshotService:
    """Handles saving rendered try-on frames to disk."""

    def __init__(self, output_dir: str = "output/screenshots") -> None:
        """Initialize screenshot service and ensure destination directory exists.

        Args:
            output_dir: Destination folder path for screenshots.
        """
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    def capture(self, frame: np.ndarray, accessory_name: str = "") -> Optional[str]:
        """Save the given rendered video frame to disk with a unique timestamp.

        Args:
            frame: Rendered OpenCV BGR frame.
            accessory_name: Name of active accessory for filename tagging.

        Returns:
            Optional[str]: Relative path to the saved screenshot file, or None if failed.
        """
        if frame is None or frame.size == 0:
            print("[MirrorFit Error] Cannot save empty or invalid frame.")
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        clean_name = accessory_name.split("[")[0].strip().lower().replace(" ", "_")
        filename = f"mirrorfit_{clean_name}_{timestamp}.png" if clean_name else f"mirrorfit_{timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)

        success = cv2.imwrite(filepath, frame)
        if success:
            print(f"[MirrorFit] Screenshot saved successfully:\n  -> {filepath}")
            return filepath
        else:
            print(f"[MirrorFit Error] Failed to write image file: {filepath}")
            return None
