"""Eyewear accessory catalogue for MirrorFit XR Stage 5.

Manages 8 distinct 3D eyewear models and 2D transparent overlays:
1. Aviator Classic (Double brow bar teardrop rims)
2. Round Heritage (Circular wireframe rims with keyhole bridge)
3. Square Bold (Thick geometric square rims with beveled corners)
4. Rectangle Sleek (Slim modern rectangular profile)
5. Wayfarer Retro (Iconic trapezoidal browline rims)
6. Cat-Eye Glamour (Dramatic upswept winged corners)
7. Sports Shield (Aerodynamic wraparound face shield)
8. Polarized Sunglasses (Oversized luxury dark square-round frames)

Includes material color palettes for frames and tinted lenses.
"""

from dataclasses import dataclass, field
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class EyewearItem:
    """Represents a virtual eyewear accessory with 3D model and 2D overlay properties."""

    id_key: int
    name: str
    shape: str
    model_3d_file: str
    image_2d_file: str
    description: str
    default_frame_color: Tuple[int, int, int]    # BGR base frame color
    default_lens_color: Tuple[int, int, int]     # BGR lens tint color
    default_lens_alpha: float = 0.50             # Lens transparency
    default_scale: float = 1.0                   # Model scale multiplier
    bridge_offset_mm: np.ndarray = field(        # Anthropometric bridge offset in mm
        default_factory=lambda: np.array([0.0, -32.0, 15.0], dtype=np.float64)
    )
    width_scale: float = 1.0                     # Lateral proportion multiplier
    height_scale: float = 1.0                    # Vertical proportion multiplier


# Curated Material and Color Palettes (BGR format)
FRAME_COLOR_PALETTES = [
    ("Metallic Gold", (70, 175, 205)),     # Rich champagne gold
    ("Matte Black", (32, 34, 40)),         # Deep stealth carbon
    ("Chrome Silver", (195, 195, 205)),    # Polished titanium silver
    ("Tortoise Amber", (35, 70, 140)),     # Classic tortoiseshell amber
    ("Rose Gold", (150, 135, 210)),        # Warm rose gold
    ("Midnight Navy", (95, 45, 25)),       # Deep sophisticated navy
    ("Gunmetal Slate", (60, 62, 70)),      # Industrial dark slate
]

LENS_TINT_PALETTES = [
    ("Polarized Charcoal", (25, 30, 42), 0.55),     # Classic neutral sunglasses
    ("G-15 Forest Green", (32, 58, 38), 0.50),      # Iconic military pilot green
    ("Ocean Blue Mirror", (110, 55, 25), 0.60),     # Vibrant seaside blue tint
    ("Sunset Amber", (20, 65, 135), 0.52),          # Warm high-contrast amber
    ("Ruby Rose", (70, 45, 120), 0.45),             # Soft crimson tint
    ("Clear Optical", (190, 195, 205), 0.22),       # Prescription clear glass
]


class AccessoryCatalog:
    """Interactive Eyewear Catalogue for MirrorFit XR.

    Features:
    - Catalogs 8 distinct 3D eyewear models with dedicated geometry.
    - Manages customizable frame material colors and lens tints.
    - Supports direct keyboard selection (1 to 8) and sequential cycling.
    - Caches loaded 2D transparent PNGs and validates 4-channel BGRA.
    - Gracefully falls back when a specific asset is missing.
    """

    CATALOG_CONFIG: List[EyewearItem] = [
        EyewearItem(
            id_key=1,
            name="Aviator Classic",
            shape="Aviator",
            model_3d_file="aviator.obj",
            image_2d_file="default_glasses.png",
            description="Double brow bar with iconic teardrop lenses",
            default_frame_color=(70, 175, 205),    # Metallic Gold
            default_lens_color=(32, 58, 38),       # G-15 Green
            default_lens_alpha=0.52,
            default_scale=1.0,
            bridge_offset_mm=np.array([0.0, -32.0, 15.0]),
        ),
        EyewearItem(
            id_key=2,
            name="Round Heritage",
            shape="Round",
            model_3d_file="round.obj",
            image_2d_file="round_glasses.png",
            description="Circular wireframe rims with keyhole bridge saddle",
            default_frame_color=(195, 195, 205),   # Chrome Silver
            default_lens_color=(25, 30, 42),       # Polarized Charcoal
            default_lens_alpha=0.48,
            default_scale=1.0,
            bridge_offset_mm=np.array([0.0, -32.0, 15.0]),
        ),
        EyewearItem(
            id_key=3,
            name="Square Bold",
            shape="Square",
            model_3d_file="square.obj",
            image_2d_file="classic_glasses.png",
            description="Bold geometric square rims with beveled facets",
            default_frame_color=(32, 34, 40),      # Matte Black
            default_lens_color=(25, 30, 42),       # Polarized Charcoal
            default_lens_alpha=0.55,
            default_scale=1.02,
            bridge_offset_mm=np.array([0.0, -32.0, 15.0]),
        ),
        EyewearItem(
            id_key=4,
            name="Rectangle Sleek",
            shape="Rectangle",
            model_3d_file="rectangle.obj",
            image_2d_file="classic_glasses.png",
            description="Slim modern rectangular profile with low height",
            default_frame_color=(60, 62, 70),      # Gunmetal Slate
            default_lens_color=(190, 195, 205),    # Clear Optical
            default_lens_alpha=0.25,
            default_scale=0.98,
            bridge_offset_mm=np.array([0.0, -32.0, 15.0]),
        ),
        EyewearItem(
            id_key=5,
            name="Wayfarer Retro",
            shape="Wayfarer",
            model_3d_file="wayfarer.obj",
            image_2d_file="default_glasses.png",
            description="Trapezoidal browline rims with thick acetate hinges",
            default_frame_color=(35, 70, 140),     # Tortoise Amber
            default_lens_color=(20, 65, 135),      # Sunset Amber
            default_lens_alpha=0.54,
            default_scale=1.0,
            bridge_offset_mm=np.array([0.0, -32.0, 15.0]),
        ),
        EyewearItem(
            id_key=6,
            name="Cat-Eye Glamour",
            shape="Cat-Eye",
            model_3d_file="cat_eye.obj",
            image_2d_file="sunglasses.png",
            description="Dramatic winged upswept outer corners with flare",
            default_frame_color=(150, 135, 210),   # Rose Gold
            default_lens_color=(70, 45, 120),      # Ruby Rose
            default_lens_alpha=0.48,
            default_scale=1.0,
            bridge_offset_mm=np.array([0.0, -32.0, 15.0]),
        ),
        EyewearItem(
            id_key=7,
            name="Sports Shield",
            shape="Sports",
            model_3d_file="sports.obj",
            image_2d_file="sunglasses.png",
            description="Aerodynamic wraparound curvature with athletic temples",
            default_frame_color=(95, 45, 25),      # Midnight Navy
            default_lens_color=(110, 55, 25),      # Ocean Blue Mirror
            default_lens_alpha=0.60,
            default_scale=1.02,
            bridge_offset_mm=np.array([0.0, -32.0, 15.0]),
        ),
        EyewearItem(
            id_key=8,
            name="Polarized Sunglasses",
            shape="Sunglasses",
            model_3d_file="sunglasses.obj",
            image_2d_file="sunglasses.png",
            description="Oversized luxury dark square-round polarized lenses",
            default_frame_color=(32, 34, 40),      # Matte Black
            default_lens_color=(25, 30, 42),       # Polarized Charcoal
            default_lens_alpha=0.62,
            default_scale=1.04,
            bridge_offset_mm=np.array([0.0, -32.0, 15.0]),
        ),
    ]

    def __init__(
        self,
        models_dir: str = "assets/models_3d",
        images_dir: str = "assets/glasses",
    ) -> None:
        """Initialize the eyewear catalogue.

        Args:
            models_dir: Directory containing 3D Wavefront OBJ models.
            images_dir: Directory containing 2D transparent PNG accessory images.
        """
        self.models_dir = os.path.abspath(models_dir)
        self.images_dir = os.path.abspath(images_dir)

        self.items: List[EyewearItem] = list(self.CATALOG_CONFIG)
        self.key_map: Dict[int, int] = {item.id_key: idx for idx, item in enumerate(self.items)}
        self.current_index: int = 0

        # Active material color overrides (user selectable)
        self.active_frame_color_idx: int = 0
        self.active_lens_tint_idx: int = 0

        # 2D Transparent PNG Cache: maps filename -> 4-channel BGRA ndarray
        self._image_cache: Dict[str, np.ndarray] = {}
        self._load_2d_images()

    def _load_2d_images(self) -> None:
        """Preload 2D accessory PNG images into memory."""
        for item in self.items:
            img_path = os.path.join(self.images_dir, item.image_2d_file)
            if os.path.exists(img_path) and item.image_2d_file not in self._image_cache:
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if img is not None and len(img.shape) == 3 and img.shape[2] == 4:
                    self._image_cache[item.image_2d_file] = img

    @property
    def current_item(self) -> EyewearItem:
        """Get the currently selected EyewearItem."""
        return self.items[self.current_index]

    @property
    def current_name(self) -> str:
        """Get the display name and index of the active accessory."""
        item = self.current_item
        return f"{item.name} [{self.current_index + 1}/{len(self.items)}]"

    @property
    def current_3d_path(self) -> str:
        """Get the absolute filepath of the active 3D OBJ model."""
        return os.path.join(self.models_dir, self.current_item.model_3d_file)

    @property
    def current_image(self) -> Optional[np.ndarray]:
        """Get the active 4-channel BGRA image for 2D fallback rendering."""
        filename = self.current_item.image_2d_file
        if filename in self._image_cache:
            return self._image_cache[filename]
        # Attempt fallback to any loaded image
        if self._image_cache:
            return next(iter(self._image_cache.values()))
        return None

    @property
    def current_frame_color(self) -> Tuple[int, int, int]:
        """Get the active BGR frame material color."""
        if self.active_frame_color_idx == 0:
            return self.current_item.default_frame_color
        return FRAME_COLOR_PALETTES[self.active_frame_color_idx][1]

    @property
    def current_frame_color_name(self) -> str:
        """Get the name of the active frame color."""
        if self.active_frame_color_idx == 0:
            return "Default Style Color"
        return FRAME_COLOR_PALETTES[self.active_frame_color_idx][0]

    @property
    def current_lens_color(self) -> Tuple[int, int, int]:
        """Get the active BGR lens tint color."""
        if self.active_lens_tint_idx == 0:
            return self.current_item.default_lens_color
        return LENS_TINT_PALETTES[self.active_lens_tint_idx][1]

    @property
    def current_lens_alpha(self) -> float:
        """Get the active lens opacity."""
        if self.active_lens_tint_idx == 0:
            return self.current_item.default_lens_alpha
        return LENS_TINT_PALETTES[self.active_lens_tint_idx][2]

    @property
    def current_lens_tint_name(self) -> str:
        """Get the name of the active lens tint."""
        if self.active_lens_tint_idx == 0:
            return "Default Style Tint"
        return LENS_TINT_PALETTES[self.active_lens_tint_idx][0]

    def select_by_key(self, key_num: int) -> bool:
        """Select accessory by direct number key (1 to 8).

        Args:
            key_num: Key number (1 to 8).

        Returns:
            bool: True if selection was successful.
        """
        if key_num in self.key_map:
            self.current_index = self.key_map[key_num]
            # Reset active palette override to default for the newly selected style
            self.active_frame_color_idx = 0
            self.active_lens_tint_idx = 0
            print(f"[MirrorFit Studio] Selected: {self.current_name} ({self.current_item.description})")
            return True
        return False

    def next_accessory(self) -> str:
        """Cycle to the next available accessory."""
        self.current_index = (self.current_index + 1) % len(self.items)
        self.active_frame_color_idx = 0
        self.active_lens_tint_idx = 0
        print(f"[MirrorFit Studio] Switched to: {self.current_name}")
        return self.current_name

    def prev_accessory(self) -> str:
        """Cycle to the previous available accessory."""
        self.current_index = (self.current_index - 1) % len(self.items)
        self.active_frame_color_idx = 0
        self.active_lens_tint_idx = 0
        print(f"[MirrorFit Studio] Switched to: {self.current_name}")
        return self.current_name

    def cycle_frame_color(self) -> str:
        """Cycle to the next frame material color palette."""
        self.active_frame_color_idx = (self.active_frame_color_idx + 1) % len(FRAME_COLOR_PALETTES)
        color_name = self.current_frame_color_name
        print(f"[MirrorFit Studio] Frame Color: {color_name}")
        return color_name

    def cycle_lens_tint(self) -> str:
        """Cycle to the next lens tint palette."""
        self.active_lens_tint_idx = (self.active_lens_tint_idx + 1) % len(LENS_TINT_PALETTES)
        tint_name = self.current_lens_tint_name
        print(f"[MirrorFit Studio] Lens Tint: {tint_name}")
        return tint_name
