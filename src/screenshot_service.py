"""Screenshot and comparison studio service for MirrorFit XR Stage 5.

Manages saving timestamped rendered frames and generating side-by-side or 2x2
style comparison collages to compare different eyewear looks.
"""

from dataclasses import dataclass
from datetime import datetime
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class ComparisonLook:
    """Represents a captured eyewear look in the comparison studio."""

    image: np.ndarray
    style_name: str
    shape_name: str
    color_name: str
    timestamp: str


class ScreenshotService:
    """Handles saving single screenshots and creating multi-eyewear comparison sheets."""

    def __init__(self, output_dir: str = "output/screenshots") -> None:
        """Initialize screenshot service and ensure destination directory exists.

        Args:
            output_dir: Destination folder path for screenshots.
        """
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.comparison_looks: List[ComparisonLook] = []

    def capture(self, frame: np.ndarray, accessory_name: str = "") -> Optional[str]:
        """Save the given rendered video frame to disk with a unique timestamp.

        Args:
            frame: Rendered OpenCV BGR frame.
            accessory_name: Name of active accessory for filename tagging.

        Returns:
            Optional[str]: Absolute path to the saved screenshot file, or None if failed.
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
            print(f"[MirrorFit Studio] Screenshot saved successfully:\n  -> {filepath}")
            return filepath
        else:
            print(f"[MirrorFit Studio Error] Failed to write image file: {filepath}")
            return None

    def add_comparison_look(
        self,
        frame: np.ndarray,
        style_name: str,
        shape_name: str = "",
        color_name: str = "",
    ) -> int:
        """Add the current rendered frame into the comparison buffer (max 4 looks).

        Args:
            frame: Video frame with fitted glasses.
            style_name: Display name of the accessory.
            shape_name: Eyewear silhouette shape.
            color_name: Frame material color name.

        Returns:
            int: Number of looks currently in the comparison buffer.
        """
        if frame is None:
            return len(self.comparison_looks)

        if len(self.comparison_looks) >= 4:
            # Drop oldest look when full
            self.comparison_looks.pop(0)

        timestamp_str = datetime.now().strftime("%H:%M:%S")
        look = ComparisonLook(
            image=frame.copy(),
            style_name=style_name,
            shape_name=shape_name,
            color_name=color_name,
            timestamp=timestamp_str,
        )
        self.comparison_looks.append(look)
        print(f"[MirrorFit Studio] Added to Comparison Studio [{len(self.comparison_looks)}/4]: {style_name}")
        return len(self.comparison_looks)

    def clear_comparison(self) -> None:
        """Clear all stored looks from the comparison studio."""
        self.comparison_looks.clear()
        print("[MirrorFit Studio] Comparison studio buffer cleared.")

    def build_comparison_sheet(self) -> Optional[np.ndarray]:
        """Generate a comparison collage image containing the stored looks.

        Returns:
            Optional[np.ndarray]: Rendered comparison sheet (BGR image), or None if empty.
        """
        n = len(self.comparison_looks)
        if n == 0:
            return None

        sheet_w, sheet_h = 1280, 720
        sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)
        sheet[:] = (20, 20, 26)  # Dark charcoal studio background

        # Header Bar
        cv2.rectangle(sheet, (0, 0), (sheet_w, 64), (14, 14, 18), -1)
        cv2.line(sheet, (0, 64), (sheet_w, 64), (55, 55, 68), 1)
        cv2.putText(
            sheet,
            "MirrorFit XR | Virtual Eyewear Comparison Studio",
            (24, 40),
            cv2.FONT_HERSHEY_DUPLEX,
            0.75,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            sheet,
            f"Comparing {n} Look{'s' if n > 1 else ''}  |  Press [V] to Exit  |  [S] Save Sheet",
            (sheet_w - 480, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )

        content_y0 = 74
        content_h = sheet_h - content_y0 - 24

        # Tile Layout
        if n == 1:
            rows, cols = 1, 1
        elif n == 2:
            rows, cols = 1, 2
        else:
            rows, cols = 2, 2

        tile_w = (sheet_w - 32 - (cols - 1) * 16) // cols
        tile_h = (content_h - (rows - 1) * 16) // rows

        for idx, look in enumerate(self.comparison_looks[:4]):
            r = idx // cols
            c = idx % cols
            x0 = 16 + c * (tile_w + 16)
            y0 = content_y0 + r * (tile_h + 16)

            # Resize captured frame into tile thumbnail
            resized = cv2.resize(look.image, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            sheet[y0 : y0 + tile_h, x0 : x0 + tile_w] = resized

            # Tile border
            cv2.rectangle(sheet, (x0, y0), (x0 + tile_w, y0 + tile_h), (0, 200, 255), 1)

            # Look Info Tag at bottom of tile
            tag_h = 32
            tag_y = y0 + tile_h - tag_h
            tag_overlay = sheet[tag_y : y0 + tile_h, x0 : x0 + tile_w].copy()
            cv2.rectangle(tag_overlay, (0, 0), (tile_w, tag_h), (12, 12, 16), -1)
            cv2.addWeighted(tag_overlay, 0.85, sheet[tag_y : y0 + tile_h, x0 : x0 + tile_w], 0.15, 0, sheet[tag_y : y0 + tile_h, x0 : x0 + tile_w])

            tag_text = f"#{idx + 1}  {look.style_name}  ({look.shape_name})  |  {look.color_name}"
            cv2.putText(
                sheet,
                tag_text,
                (x0 + 12, tag_y + 21),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (230, 240, 255),
                1,
                cv2.LINE_AA,
            )

        return sheet

    def save_comparison_sheet(self) -> Optional[str]:
        """Generate and save the current comparison collage to disk."""
        sheet = self.build_comparison_sheet()
        if sheet is None:
            print("[MirrorFit Studio] No looks in comparison buffer to save.")
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mirrorfit_comparison_{timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)

        success = cv2.imwrite(filepath, sheet)
        if success:
            print(f"[MirrorFit Studio] Comparison sheet saved:\n  -> {filepath}")
            return filepath
        return None
