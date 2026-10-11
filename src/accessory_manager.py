"""Accessory manager module for MirrorFit.

Manages loading, cataloging, caching, and runtime selection of multiple virtual try-on accessories
(glasses and sunglasses) from transparent PNG assets.
"""

from dataclasses import dataclass
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class AccessoryItem:
    """Represents a single transparent PNG accessory."""

    id_key: int
    name: str
    filename: str
    image: np.ndarray
    width: int
    height: int


class AccessoryManager:
    """Manages the catalog of glasses and sunglasses assets.

    Features:
    - Loads transparent PNG images once during application startup.
    - Validates 4-channel BGRA format and genuine alpha channel.
    - Enables switching via direct keys (1, 2, 3, 4) or sequential cycling (Next/Prev).
    - Gracefully handles missing assets without crashing the application.
    """

    CATALOG_CONFIG = [
        (1, "Default Glasses", "default_glasses.png"),
        (2, "Classic Glasses", "classic_glasses.png"),
        (3, "Round Glasses", "round_glasses.png"),
        (4, "Sunglasses", "sunglasses.png"),
    ]

    def __init__(self, assets_dir: str = "assets/glasses") -> None:
        """Initialize the accessory catalog and load all available assets.

        Args:
            assets_dir: Directory containing transparent PNG accessory assets.
        """
        self.assets_dir = os.path.abspath(assets_dir)
        self.items: List[AccessoryItem] = []
        self.key_map: Dict[int, int] = {}  # Maps key number (1-4) to item index
        self.current_index: int = 0

        self._load_catalog()

    def _load_catalog(self) -> None:
        """Scan and load all configured accessories from the assets directory."""
        if not os.path.exists(self.assets_dir):
            os.makedirs(self.assets_dir, exist_ok=True)

        for key_num, name, filename in self.CATALOG_CONFIG:
            filepath = os.path.join(self.assets_dir, filename)

            if not os.path.exists(filepath):
                print(f"[MirrorFit Warning] Accessory '{filename}' not found at {filepath}. Skipping.")
                continue

            image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
            if image is None:
                print(f"[MirrorFit Warning] Failed to decode '{filename}'. Skipping.")
                continue

            if len(image.shape) != 3 or image.shape[2] != 4:
                print(
                    f"[MirrorFit Warning] '{filename}' does not have 4 channels (BGRA). Skipping."
                )
                continue

            h, w = image.shape[:2]
            item = AccessoryItem(
                id_key=key_num,
                name=name,
                filename=filename,
                image=image,
                width=w,
                height=h,
            )
            item_idx = len(self.items)
            self.items.append(item)
            self.key_map[key_num] = item_idx
            print(f"[MirrorFit] Loaded accessory [{key_num}]: {name} ({w}x{h})")

        if not self.items:
            raise FileNotFoundError(
                f"\n[MirrorFit ERROR] No valid transparent PNG accessories found in '{self.assets_dir}'!\n"
                "Please ensure at least one 4-channel PNG exists in 'assets/glasses/'."
            )

    @property
    def current_item(self) -> AccessoryItem:
        """Get the currently active accessory item."""
        return self.items[self.current_index]

    @property
    def current_name(self) -> str:
        """Get the name and index of the currently active accessory."""
        item = self.current_item
        return f"{item.name} [{self.current_index + 1}/{len(self.items)}]"

    @property
    def current_image(self) -> np.ndarray:
        """Get the active 4-channel BGRA image."""
        return self.current_item.image

    def select_by_key(self, key_num: int) -> bool:
        """Select accessory by direct number key (1, 2, 3, 4).

        Args:
            key_num: Key number (1 to 4).

        Returns:
            bool: True if selection was successful.
        """
        if key_num in self.key_map:
            self.current_index = self.key_map[key_num]
            print(f"[MirrorFit] Selected: {self.current_name}")
            return True
        else:
            print(f"[MirrorFit Warning] Accessory [{key_num}] is not available in catalog.")
            return False

    def next_accessory(self) -> str:
        """Cycle to the next available accessory."""
        self.current_index = (self.current_index + 1) % len(self.items)
        print(f"[MirrorFit] Switched to: {self.current_name}")
        return self.current_name

    def prev_accessory(self) -> str:
        """Cycle to the previous available accessory."""
        self.current_index = (self.current_index - 1) % len(self.items)
        print(f"[MirrorFit] Switched to: {self.current_name}")
        return self.current_name
