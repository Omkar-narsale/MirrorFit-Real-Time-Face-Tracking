"""Main application orchestrator for MirrorFit XR Stage 5.

Stage 5 — Smart AR Virtual Eyewear Try-On Studio:
- 8 Distinct 3D Eyewear Models (Aviator, Round, Square, Rectangle, Wayfarer, Cat-Eye, Sports, Sunglasses).
- Advanced anthropometric face-relative fitting (interpupillary distance, temporal face width, nose bridge).
- Interactive smart frame adjustments (Scale, Width, Height, X/Y/Z offsets, Auto-Fit).
- Real-time material and color palette selection (Frames and Polarized/Tinted Lenses).
- Eyewear Comparison Studio with multi-look collage generation.
- Interactive Modern Studio Interface with Top Bar, Collapsible Side Panel, and Mouse Button Controls.
- Real-time performance optimization benchmarked at 60+ FPS on CPU.
"""

import argparse
import os
from pathlib import Path
import sys
import time
from typing import Dict, Optional, Tuple

# Ensure project root directory is in sys.path for direct execution (e.g. python src/main.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from src.accessory_catalog import AccessoryCatalog, FRAME_COLOR_PALETTES, LENS_TINT_PALETTES
from src.accessory_manager import AccessoryManager
from src.camera import Camera
from src.face_segmentation import FaceSegmentationModule
from src.face_tracker import FaceTracker, TryOnAnchors
from src.fitting_engine import FittingEngine, FittingResult
from src.glasses_overlay import GlassesOverlay
from src.head_pose import HeadPoseEstimator, HeadPoseResult
from src.performance_monitor import PerformanceMetrics, PerformanceMonitor
from src.renderer_3d import Renderer3D
from src.screenshot_service import ScreenshotService


class MirrorFitApp:
    """Coordinates camera capture, head pose, 3D rendering, segmentation, and Studio HUD controls."""

    WINDOW_NAME = "MirrorFit XR | Smart AR Eyewear Studio"

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
        model_path: Optional[str] = None,
        segmenter_path: Optional[str] = None,
        model_3d_path: str = "assets/models_3d/aviator.obj",
        assets_dir: str = "assets/glasses",
        scale_factor: float = 2.15,
        vertical_offset: float = 0.05,
        smoothing_factor: float = 0.35,
    ) -> None:
        """Initialize the application components.

        Args:
            camera_index: Video capture hardware index (default: 0).
            width: Preferred frame width in pixels.
            height: Preferred frame height in pixels.
            model_path: Optional custom path to face_landmarker.task.
            segmenter_path: Optional custom path to selfie_segmenter.tflite.
            model_3d_path: Initial 3D OBJ model asset path.
            assets_dir: Directory containing 2D accessory PNG assets.
            scale_factor: Glasses width scale factor relative to eye distance.
            vertical_offset: Vertical offset ratio relative to glasses height.
            smoothing_factor: Exponential smoothing weight for jitter reduction.
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.model_path = model_path
        self.segmenter_path = segmenter_path
        self.model_3d_path = model_3d_path
        self.assets_dir = assets_dir
        self.scale_factor = scale_factor
        self.vertical_offset = vertical_offset
        self.smoothing_factor = smoothing_factor

        # Rendering & Visualization States
        self.use_3d_mode: bool = True            # Primary mode is 3D AR Model; Toggle via 'T'
        self.show_glasses: bool = True           # Toggle glasses on/off via 'G'
        self.show_pose_axes: bool = False        # Toggle 3D axes on/off via 'X'
        self.show_segmentation: bool = False     # Toggle segmentation mask on/off via 'B'
        self.show_mesh: bool = False             # Toggle face mesh on/off via 'M'
        self.show_contours: bool = False
        self.show_anchors: bool = False          # Toggle try-on anchors on/off via 'A'
        self.show_hud: bool = True               # Toggle HUD overlay on/off via 'H'
        self.show_side_panel: bool = True        # Toggle studio side panel via 'TAB' or 'K'
        self.in_comparison_mode: bool = False    # Toggle Comparison Studio via 'V'

        # Interactive Clickable UI Button Regions: name -> (x0, y0, x1, y1)
        self.ui_buttons: Dict[str, Tuple[int, int, int, int]] = {}

        # Notification banner tracking
        self.notification_message: Optional[str] = None
        self.notification_expiry: float = 0.0

        # Performance monitoring engine
        self.perf_monitor = PerformanceMonitor()

        # Cached modules initialized in run()
        self.catalog: Optional[AccessoryCatalog] = None
        self.fitting_engine: Optional[FittingEngine] = None
        self.renderer_3d: Optional[Renderer3D] = None
        self.screenshot_service: Optional[ScreenshotService] = None

    def show_notification(self, message: str, duration_sec: float = 2.4) -> None:
        """Display an on-screen notification badge on the HUD for a limited duration."""
        self.notification_message = message
        self.notification_expiry = time.perf_counter() + duration_sec

    def _sync_active_eyewear(self) -> None:
        """Synchronize the 3D renderer and materials with the active catalogue item."""
        if self.catalog and self.renderer_3d:
            item = self.catalog.current_item
            model_path = os.path.join(self.catalog.models_dir, item.model_3d_file)
            self.renderer_3d.set_model(model_path)
            self.renderer_3d.set_colors(
                primary_color=self.catalog.current_frame_color,
                lens_color=self.catalog.current_lens_color,
                lens_alpha=self.catalog.current_lens_alpha,
            )

    def handle_mouse_click(self, x: int, y: int) -> None:
        """Process interactive mouse clicks on HUD buttons."""
        for btn_name, (x0, y0, x1, y1) in self.ui_buttons.items():
            if x0 <= x <= x1 and y0 <= y <= y1:
                self._dispatch_button_action(btn_name)
                return

    def _dispatch_button_action(self, btn_name: str) -> None:
        """Execute action corresponding to a clicked UI button."""
        if btn_name.startswith("item_") and self.catalog:
            idx = int(btn_name.split("_")[1])
            self.catalog.select_by_key(idx)
            self._sync_active_eyewear()
            self.show_notification(f"Style [{idx}]: {self.catalog.current_item.name}")
        elif btn_name == "mode_toggle":
            self.use_3d_mode = not self.use_3d_mode
            active_label = "3D AR Model" if self.use_3d_mode else "2D Overlay"
            self.show_notification(f"Mode: {active_label}")
        elif btn_name == "wire_toggle" and self.renderer_3d:
            is_wf = self.renderer_3d.toggle_wireframe()
            self.show_notification(f"3D Wireframe: {'ON' if is_wf else 'OFF'}")
        elif btn_name == "color_cycle" and self.catalog:
            c_name = self.catalog.cycle_frame_color()
            self._sync_active_eyewear()
            self.show_notification(f"Frame Color: {c_name}")
        elif btn_name == "tint_cycle" and self.catalog:
            t_name = self.catalog.cycle_lens_tint()
            self._sync_active_eyewear()
            self.show_notification(f"Lens Tint: {t_name}")
        elif btn_name == "autofit_toggle" and self.fitting_engine:
            af_on = self.fitting_engine.adjustments.toggle_autofit()
            self.show_notification(f"Auto-Fit: {'ON' if af_on else 'OFF'}")
        elif btn_name == "scale_plus" and self.fitting_engine:
            s = self.fitting_engine.adjustments.adjust_scale(+0.05)
            self.show_notification(f"Scale: {s:.2f}x")
        elif btn_name == "scale_minus" and self.fitting_engine:
            s = self.fitting_engine.adjustments.adjust_scale(-0.05)
            self.show_notification(f"Scale: {s:.2f}x")
        elif btn_name == "width_plus" and self.fitting_engine:
            w = self.fitting_engine.adjustments.adjust_width(+0.05)
            self.show_notification(f"Width: {w:.2f}x")
        elif btn_name == "width_minus" and self.fitting_engine:
            w = self.fitting_engine.adjustments.adjust_width(-0.05)
            self.show_notification(f"Width: {w:.2f}x")
        elif btn_name == "reset_fit" and self.fitting_engine:
            self.fitting_engine.adjustments.reset()
            self.show_notification("Fitting Adjustments Reset")
        elif btn_name == "compare_toggle":
            self.in_comparison_mode = not self.in_comparison_mode
            self.show_notification(f"Comparison Studio: {'OPEN' if self.in_comparison_mode else 'CLOSED'}")

    def draw_hud(
        self,
        frame: np.ndarray,
        is_detected: bool,
        metrics: PerformanceMetrics,
        pose: HeadPoseResult,
        fitting: FittingResult,
    ) -> None:
        """Render modern eyewear studio HUD (top bar, side panel, bottom navigation)."""
        if not self.show_hud:
            return

        h, w = frame.shape[:2]
        self.ui_buttons.clear()

        # -------------------------------------------------------------
        # 1. Top Bar & Bottom Bar Overlays
        # -------------------------------------------------------------
        header_height = 56
        footer_height = 36
        overlay = frame.copy()

        cv2.rectangle(overlay, (0, 0), (w, header_height), (16, 16, 22), -1)
        cv2.rectangle(overlay, (0, h - footer_height), (w, h), (16, 16, 22), -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

        cv2.line(frame, (0, header_height), (w, header_height), (55, 55, 68), 1)
        cv2.line(frame, (0, h - footer_height), (w, h - footer_height), (55, 55, 68), 1)

        # -------------------------------------------------------------
        # 2. Top Bar: Branding & Active Mode Badge
        # -------------------------------------------------------------
        title_text = "MirrorFit XR"
        cv2.putText(frame, title_text, (20, 36), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)

        (t_w, _), _ = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_DUPLEX, 0.75, 1)
        pill_x = 20 + t_w + 14

        if self.use_3d_mode:
            if self.renderer_3d and self.renderer_3d.wireframe_only:
                mode_badge = "3D AR (WIREFRAME)"
                mode_color = (120, 220, 255)
            else:
                mode_badge = "3D AR (SOLID MESH)"
                mode_color = (0, 220, 255)
        else:
            mode_badge = f"2D: {self.catalog.current_item.name.upper() if self.catalog else 'OVERLAY'}"
            mode_color = (220, 165, 40)

        (mb_w, _), _ = cv2.getTextSize(mode_badge, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        pill_w = max(115, mb_w + 22)
        cv2.rectangle(frame, (pill_x, 15), (pill_x + pill_w, 41), (32, 36, 46), -1)
        cv2.rectangle(frame, (pill_x, 15), (pill_x + pill_w, 41), mode_color, 1)
        cv2.putText(frame, mode_badge, (pill_x + 11, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.44, mode_color, 1, cv2.LINE_AA)
        self.ui_buttons["mode_toggle"] = (pill_x, 15, pill_x + pill_w, 41)

        # -------------------------------------------------------------
        # 3. Top Bar: Face Status Badge (Center)
        # -------------------------------------------------------------
        badge_y = 14
        badge_h = 28
        if is_detected:
            status_text = "FACE DETECTED"
            badge_bg = (32, 125, 42)
            badge_border = (60, 220, 90)
            dot_color = (60, 255, 100)
        else:
            status_text = "NO FACE DETECTED"
            badge_bg = (30, 40, 140)
            badge_border = (50, 80, 230)
            dot_color = (60, 90, 255)

        (text_w, _), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        badge_w = text_w + 34
        badge_x = (w // 2) - (badge_w // 2)
        cv2.rectangle(frame, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), badge_bg, -1)
        cv2.rectangle(frame, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), badge_border, 1)
        cv2.circle(frame, (badge_x + 13, badge_y + 14), 4, dot_color, -1, cv2.LINE_AA)
        cv2.putText(frame, status_text, (badge_x + 25, badge_y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 4. Top Bar: Performance Telemetry (Top Right)
        # -------------------------------------------------------------
        perf_text = f"FPS: {metrics.fps:.1f}  |  Lat: {metrics.total_latency_ms:.1f}ms"
        fps_color = (80, 220, 100) if metrics.fps >= 25 else (50, 160, 255)
        (fps_w, _), _ = cv2.getTextSize(perf_text, cv2.FONT_HERSHEY_DUPLEX, 0.54, 1)
        cv2.putText(frame, perf_text, (w - fps_w - 20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.54, fps_color, 1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 5. Live Geometric Telemetry Bar (Below Header)
        # -------------------------------------------------------------
        telem_y = header_height + 24
        if is_detected and pose.is_valid:
            active_desc = self.catalog.current_name if self.catalog else "Default"
            af_str = f"AutoFit: {fitting.auto_scale_ratio:.2f}x" if self.fitting_engine and self.fitting_engine.adjustments.auto_fit_enabled else "AutoFit: OFF"
            ipd_str = f"IPD: {fitting.interpupillary_dist_mm:.0f}mm"
            seg_str = f"Seg: {metrics.segmentation_latency_ms:.1f}ms" if self.show_segmentation else "Seg: OFF"
            telem_text = (
                f"Pose: Yaw {pose.yaw_deg:+.1f}° Pitch {pose.pitch_deg:+.1f}° Roll {pose.roll_deg:+.1f}°  |  "
                f"{active_desc}  |  {ipd_str}  |  {af_str}  |  {seg_str}"
            )
        else:
            telem_text = "Tracking paused (waiting for face detection)"

        (t_w, _), _ = cv2.getTextSize(telem_text, cv2.FONT_HERSHEY_SIMPLEX, 0.41, 1)
        cv2.rectangle(frame, (16, telem_y - 17), (16 + t_w + 16, telem_y + 7), (18, 18, 24), -1)
        cv2.rectangle(frame, (16, telem_y - 17), (16 + t_w + 16, telem_y + 7), (55, 55, 68), 1)
        cv2.putText(frame, telem_text, (24, telem_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.41, (215, 230, 255), 1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 6. Interactive Studio Side Panel (Right Margin)
        # -------------------------------------------------------------
        if self.show_side_panel and self.catalog:
            panel_w = 265
            panel_x = w - panel_w
            panel_y0 = header_height + 1
            panel_h = h - footer_height - panel_y0

            panel_overlay = frame[panel_y0 : panel_y0 + panel_h, panel_x : w].copy()
            cv2.rectangle(panel_overlay, (0, 0), (panel_w, panel_h), (14, 14, 18), -1)
            cv2.addWeighted(panel_overlay, 0.82, frame[panel_y0 : panel_y0 + panel_h, panel_x : w], 0.18, 0, frame[panel_y0 : panel_y0 + panel_h, panel_x : w])
            cv2.line(frame, (panel_x, panel_y0), (panel_x, panel_y0 + panel_h), (50, 50, 62), 1)

            # Panel Title
            cv2.putText(frame, "EYEWEAR STUDIO CATALOGUE", (panel_x + 14, panel_y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)

            # 8 Catalogue Style Items
            btn_y = panel_y0 + 36
            btn_h = 24
            for item in self.catalog.items:
                is_sel = item.id_key == (self.catalog.current_index + 1)
                bg_col = (45, 95, 55) if is_sel else (24, 26, 32)
                border_col = (80, 240, 120) if is_sel else (45, 48, 58)
                text_col = (255, 255, 255) if is_sel else (190, 195, 210)

                cv2.rectangle(frame, (panel_x + 10, btn_y), (panel_x + panel_w - 10, btn_y + btn_h), bg_col, -1)
                cv2.rectangle(frame, (panel_x + 10, btn_y), (panel_x + panel_w - 10, btn_y + btn_h), border_col, 1)

                item_lbl = f"[{item.id_key}] {item.name}"
                cv2.putText(frame, item_lbl, (panel_x + 16, btn_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, text_col, 1, cv2.LINE_AA)
                self.ui_buttons[f"item_{item.id_key}"] = (panel_x + 10, btn_y, panel_x + panel_w - 10, btn_y + btn_h)
                btn_y += btn_h + 4

            # Material & Color Customization Section
            btn_y += 10
            cv2.line(frame, (panel_x + 12, btn_y), (panel_x + panel_w - 12, btn_y), (50, 50, 62), 1)
            btn_y += 14
            cv2.putText(frame, "MATERIAL & COLOR PALETTE", (panel_x + 14, btn_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 220, 255), 1, cv2.LINE_AA)

            btn_y += 10
            # Frame Color Button [C]
            cv2.rectangle(frame, (panel_x + 10, btn_y), (panel_x + panel_w - 10, btn_y + 24), (25, 28, 36), -1)
            cv2.rectangle(frame, (panel_x + 10, btn_y), (panel_x + panel_w - 10, btn_y + 24), (60, 65, 80), 1)
            # Color indicator swatch circle
            f_col = self.catalog.current_frame_color
            cv2.circle(frame, (panel_x + 22, btn_y + 12), 6, f_col, -1, cv2.LINE_AA)
            cv2.circle(frame, (panel_x + 22, btn_y + 12), 6, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, f"[C] Frame: {self.catalog.current_frame_color_name}", (panel_x + 36, btn_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 225, 240), 1, cv2.LINE_AA)
            self.ui_buttons["color_cycle"] = (panel_x + 10, btn_y, panel_x + panel_w - 10, btn_y + 24)

            btn_y += 30
            # Lens Tint Button [L]
            cv2.rectangle(frame, (panel_x + 10, btn_y), (panel_x + panel_w - 10, btn_y + 24), (25, 28, 36), -1)
            cv2.rectangle(frame, (panel_x + 10, btn_y), (panel_x + panel_w - 10, btn_y + 24), (60, 65, 80), 1)
            l_col = self.catalog.current_lens_color
            cv2.circle(frame, (panel_x + 22, btn_y + 12), 6, l_col, -1, cv2.LINE_AA)
            cv2.circle(frame, (panel_x + 22, btn_y + 12), 6, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, f"[L] Lens: {self.catalog.current_lens_tint_name}", (panel_x + 36, btn_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 225, 240), 1, cv2.LINE_AA)
            self.ui_buttons["tint_cycle"] = (panel_x + 10, btn_y, panel_x + panel_w - 10, btn_y + 24)

            # Smart Fitting Adjustment Section
            btn_y += 34
            cv2.line(frame, (panel_x + 12, btn_y), (panel_x + panel_w - 12, btn_y), (50, 50, 62), 1)
            btn_y += 14
            cv2.putText(frame, "SMART FIT ADJUSTMENTS", (panel_x + 14, btn_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 220, 255), 1, cv2.LINE_AA)

            if self.fitting_engine:
                adj = self.fitting_engine.adjustments
                btn_y += 12
                # Auto-Fit Toggle [F]
                af_bg = (30, 85, 45) if adj.auto_fit_enabled else (45, 30, 30)
                cv2.rectangle(frame, (panel_x + 10, btn_y), (panel_x + panel_w - 10, btn_y + 22), af_bg, -1)
                cv2.rectangle(frame, (panel_x + 10, btn_y), (panel_x + panel_w - 10, btn_y + 22), (60, 180, 90) if adj.auto_fit_enabled else (90, 60, 60), 1)
                cv2.putText(frame, f"[F] Auto-Fit Facial Ratio: {'ON' if adj.auto_fit_enabled else 'OFF'}", (panel_x + 16, btn_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
                self.ui_buttons["autofit_toggle"] = (panel_x + 10, btn_y, panel_x + panel_w - 10, btn_y + 22)

                btn_y += 28
                # Scale Controls [ [ / ] ]
                cv2.putText(frame, f"Scale: {adj.overall_scale:.2f}x", (panel_x + 14, btn_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 210, 230), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (panel_x + 120, btn_y), (panel_x + 150, btn_y + 20), (35, 40, 52), -1)
                cv2.putText(frame, "[-]", (panel_x + 128, btn_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1, cv2.LINE_AA)
                self.ui_buttons["scale_minus"] = (panel_x + 120, btn_y, panel_x + 150, btn_y + 20)
                cv2.rectangle(frame, (panel_x + 158, btn_y), (panel_x + 188, btn_y + 20), (35, 40, 52), -1)
                cv2.putText(frame, "[+]", (panel_x + 165, btn_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1, cv2.LINE_AA)
                self.ui_buttons["scale_plus"] = (panel_x + 158, btn_y, panel_x + 188, btn_y + 20)

                btn_y += 26
                # Width Controls [ - / + ]
                cv2.putText(frame, f"Width: {adj.width_scale:.2f}x", (panel_x + 14, btn_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 210, 230), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (panel_x + 120, btn_y), (panel_x + 150, btn_y + 20), (35, 40, 52), -1)
                cv2.putText(frame, "[-]", (panel_x + 128, btn_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1, cv2.LINE_AA)
                self.ui_buttons["width_minus"] = (panel_x + 120, btn_y, panel_x + 150, btn_y + 20)
                cv2.rectangle(frame, (panel_x + 158, btn_y), (panel_x + 188, btn_y + 20), (35, 40, 52), -1)
                cv2.putText(frame, "[+]", (panel_x + 165, btn_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1, cv2.LINE_AA)
                self.ui_buttons["width_plus"] = (panel_x + 158, btn_y, panel_x + 188, btn_y + 20)

                btn_y += 26
                # Reset Fit Button
                cv2.rectangle(frame, (panel_x + 10, btn_y), (panel_x + panel_w - 10, btn_y + 22), (28, 30, 38), -1)
                cv2.rectangle(frame, (panel_x + 10, btn_y), (panel_x + panel_w - 10, btn_y + 22), (70, 75, 90), 1)
                cv2.putText(frame, "[R] Reset All Fit Values", (panel_x + 28, btn_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 215, 235), 1, cv2.LINE_AA)
                self.ui_buttons["reset_fit"] = (panel_x + 10, btn_y, panel_x + panel_w - 10, btn_y + 22)

        # -------------------------------------------------------------
        # 7. Temporary On-Screen Notification Banner
        # -------------------------------------------------------------
        now = time.perf_counter()
        if self.notification_message and now < self.notification_expiry:
            banner_text = self.notification_message
            (n_w, _), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            banner_x = (w // 2) - (n_w // 2) - 16
            banner_y = h - footer_height - 38

            cv2.rectangle(frame, (banner_x, banner_y - 20), (banner_x + n_w + 32, banner_y + 12), (25, 120, 60), -1)
            cv2.rectangle(frame, (banner_x, banner_y - 20), (banner_x + n_w + 32, banner_y + 12), (80, 255, 120), 1)
            cv2.putText(frame, banner_text, (banner_x + 16, banner_y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 8. Bottom Navigation Footer Bar
        # -------------------------------------------------------------
        wf_stat = "ON" if (self.renderer_3d and self.renderer_3d.wireframe_only) else "OFF"
        footer_text = (
            f"[T] Mode: {'3D' if self.use_3d_mode else '2D'}  |  "
            f"[W] Wire: {wf_stat}  |  "
            f"[1-8] Styles  |  "
            f"[C] Color  |  "
            f"[L] Tint  |  "
            f"[F] AutoFit  |  "
            f"[[]/] Scale  |  "
            f"[V] Compare  |  "
            f"[S] Shot  |  "
            f"[R] Reset  |  "
            f"[Q] Exit"
        )
        cv2.putText(frame, footer_text, (18, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (195, 205, 220), 1, cv2.LINE_AA)

    def run(self) -> None:
        """Run the main camera, face tracking, 3D AR rendering, and event loop."""
        print("=" * 72)
        print("  MirrorFit XR | Smart AR Virtual Eyewear Studio (Stage 5)")
        print("=" * 72)
        print("Initializing webcam, MediaPipe Face Landmarker, and XR modules...")

        # 1. Initialize Face Tracker
        try:
            face_tracker = FaceTracker(model_path=self.model_path)
        except FileNotFoundError as err:
            print(err)
            sys.exit(1)
        except Exception as exc:
            print(f"[Error] Failed to initialize Face Tracker: {exc}")
            sys.exit(1)

        # 2. Initialize Head Pose Estimator
        head_pose_estimator = HeadPoseEstimator(smoothing_factor=self.smoothing_factor)

        # 3. Initialize Face Segmentation Module
        segmentation_module = FaceSegmentationModule(model_path=self.segmenter_path)

        # 4. Initialize Eyewear Catalog & 3D Renderer
        self.catalog = AccessoryCatalog(
            models_dir="assets/models_3d",
            images_dir=self.assets_dir,
        )
        self.renderer_3d = Renderer3D(self.catalog.current_3d_path)
        self._sync_active_eyewear()

        # 5. Initialize Fitting Engine
        self.fitting_engine = FittingEngine(smoothing_factor=self.smoothing_factor)

        # 6. Initialize 2D Glasses Overlay (Fallback path)
        try:
            accessory_mgr = AccessoryManager(assets_dir=self.assets_dir)
            glasses_overlay = GlassesOverlay(
                asset_path=os.path.join(self.assets_dir, accessory_mgr.current_item.filename),
                scale_factor=self.scale_factor,
                vertical_offset_ratio=self.vertical_offset,
                smoothing_factor=self.smoothing_factor,
            )
        except Exception as exc:
            accessory_mgr = None
            glasses_overlay = None

        # 7. Initialize Screenshot & Comparison Service
        self.screenshot_service = ScreenshotService()

        # 8. Initialize Camera
        try:
            camera = Camera(
                device_index=self.camera_index,
                width=self.width,
                height=self.height,
                mirror=True,
            )
            camera.start()
        except RuntimeError as err:
            print(f"\n[Camera Error] {err}")
            face_tracker.close()
            segmentation_module.close()
            sys.exit(1)

        actual_w, actual_h = camera.get_resolution()
        cam_matrix, dist_coeffs = HeadPoseEstimator.get_camera_matrix(actual_w, actual_h)

        print(f"[MirrorFit Studio] Camera started successfully at {actual_w}x{actual_h} resolution.")
        print("[MirrorFit Studio] Controls: [T] Mode 3D/2D, [W] Wireframe, [1-8] Styles, [C] Color, [L] Tint, [F] AutoFit, [V] Compare, [S] Shot, [Q] Exit.\n")

        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

        # Attach interactive mouse callback
        def on_mouse_event(event, mx, my, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.handle_mouse_click(mx, my)

        cv2.setMouseCallback(self.WINDOW_NAME, on_mouse_event)

        # Cached mask for throttled segmentation (every 3 frames)
        cached_seg_mask = None
        frame_idx = 0

        try:
            while True:
                frame_start_time = time.perf_counter()
                frame_idx += 1

                # 1. Capture video frame
                success, frame = camera.read_frame()
                if not success or frame is None:
                    continue

                # 2. Face Tracking (MediaPipe 478 landmarks)
                is_detected, raw_landmarks, pixel_coords, anchors = face_tracker.process_frame(frame)

                pose_res = HeadPoseResult(
                    rvec=np.zeros((3, 1)),
                    tvec=np.zeros((3, 1)),
                    rotation_matrix=np.eye(3),
                    yaw_deg=0.0,
                    pitch_deg=0.0,
                    roll_deg=0.0,
                    is_valid=False,
                )
                fitting_res = FittingResult(
                    model_matrix_4x4=np.eye(4),
                    rvec=np.zeros((3, 1)),
                    tvec=np.zeros((3, 1)),
                    scale=1.0,
                    scale_vec=np.array([1.0, 1.0, 1.0]),
                    anchor_center_2d=(0, 0),
                    is_valid=False,
                )
                pose_ms = 0.0
                seg_ms = 0.0
                render_ms = 0.0

                if is_detected and pixel_coords is not None:
                    # 3. Head Pose Estimation (solvePnP)
                    t0 = time.perf_counter()
                    pose_res = head_pose_estimator.estimate_pose(pixel_coords, (actual_h, actual_w))
                    pose_ms = (time.perf_counter() - t0) * 1000.0

                    # 4. Optional Face Segmentation Inference (throttled every 3 frames for 60+ FPS)
                    if self.show_segmentation and segmentation_module.is_available:
                        if frame_idx % 3 == 0 or cached_seg_mask is None:
                            cached_seg_mask, seg_ms = segmentation_module.segment(frame)
                        if cached_seg_mask is not None:
                            FaceSegmentationModule.render_debug_overlay(frame, cached_seg_mask)

                    # 5. Eyewear Fitting & 3D / 2D Rendering
                    t_rend = time.perf_counter()
                    if self.show_glasses and pose_res.is_valid:
                        # Anthropometric Fitting Calculation
                        active_item = self.catalog.current_item
                        fitting_res = self.fitting_engine.compute_fitting(
                            pose=pose_res,
                            pixel_landmarks=pixel_coords,
                            frame_shape=(actual_h, actual_w),
                            base_model_scale=active_item.default_scale,
                            base_bridge_offset=active_item.bridge_offset_mm,
                            focal_length=cam_matrix[0, 0],
                        )

                        if self.use_3d_mode and self.renderer_3d and self.renderer_3d.is_available:
                            # 3D AR Model Rendering Path
                            self.renderer_3d.render(
                                frame=frame,
                                fitting=fitting_res,
                                camera_matrix=cam_matrix,
                                dist_coeffs=dist_coeffs,
                                pixel_landmarks=pixel_coords,
                            )
                        elif glasses_overlay is not None and anchors is not None:
                            # 2D Transparent PNG Overlay Path
                            active_2d_img = self.catalog.current_image
                            glasses_overlay.apply(
                                frame=frame,
                                left_eye=anchors.left_eye_center,
                                right_eye=anchors.right_eye_center,
                                nose_bridge=anchors.nose_bridge,
                                accessory_image=active_2d_img,
                            )
                    render_ms = (time.perf_counter() - t_rend) * 1000.0

                    # 6. Optional 3D Coordinate Axes Visualization
                    if self.show_pose_axes and pose_res.is_valid:
                        HeadPoseEstimator.draw_axes(frame, pose_res.rvec, pose_res.tvec, cam_matrix, dist_coeffs)

                    # 7. Optional Landmark Mesh Visualization
                    if self.show_mesh or self.show_contours or self.show_anchors:
                        face_tracker.draw_landmarks(
                            frame=frame,
                            raw_landmarks=raw_landmarks,
                            tryon_anchors=anchors,
                            show_mesh=self.show_mesh,
                            show_contours=self.show_contours,
                            show_anchors=self.show_anchors,
                        )
                else:
                    # Reset smoothing filters when face is temporarily lost
                    head_pose_estimator.reset()
                    if self.fitting_engine:
                        self.fitting_engine.reset_filters()
                    if glasses_overlay:
                        glasses_overlay.reset_filter()

                # 8. Performance Monitoring & Latency Tracking
                total_frame_ms = (time.perf_counter() - frame_start_time) * 1000.0
                self.perf_monitor.update_fps()
                self.perf_monitor.record_latencies(
                    total_ms=total_frame_ms,
                    pose_ms=pose_ms,
                    seg_ms=seg_ms,
                    render_ms=render_ms,
                )
                metrics = self.perf_monitor.get_metrics()

                # 9. Comparison Mode vs Live Camera Preview
                if self.in_comparison_mode:
                    comp_sheet = self.screenshot_service.build_comparison_sheet()
                    display_frame = comp_sheet if comp_sheet is not None else frame
                else:
                    # Render Studio HUD Overlay
                    self.draw_hud(
                        frame=frame,
                        is_detected=is_detected,
                        metrics=metrics,
                        pose=pose_res,
                        fitting=fitting_res,
                    )
                    display_frame = frame

                # 10. Display Frame in Window
                cv2.imshow(self.WINDOW_NAME, display_frame)

                # 11. Handle Keyboard Controls
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):  # 27 is ESC key
                    print("\n[MirrorFit Studio] Exit requested by user. Shutting down...")
                    break
                elif key in (ord("t"), ord("T")):
                    self.use_3d_mode = not self.use_3d_mode
                    active_label = "3D AR Model" if self.use_3d_mode else "2D Overlay"
                    print(f"[MirrorFit Studio] Rendering mode switched to: {active_label}")
                    self.show_notification(f"Mode: {active_label}")
                elif key in (ord("w"), ord("W")):
                    if self.use_3d_mode and self.renderer_3d:
                        is_wf = self.renderer_3d.toggle_wireframe()
                        wf_label = "Wireframe Debug (ON)" if is_wf else "Solid Shaded (OFF)"
                        print(f"[MirrorFit Studio] 3D Wireframe mode: {wf_label}")
                        self.show_notification(f"3D Visual: {wf_label}")
                    else:
                        self.show_notification("Switch to 3D Mode ([T]) for Wireframe")
                elif key in (ord("1"), ord("2"), ord("3"), ord("4"), ord("5"), ord("6"), ord("7"), ord("8")):
                    idx = int(chr(key))
                    if self.catalog:
                        self.catalog.select_by_key(idx)
                        self._sync_active_eyewear()
                        self.show_notification(f"Selected: {self.catalog.current_item.name}")
                elif key in (ord("n"), ord("N")) and self.catalog:
                    self.catalog.next_accessory()
                    self._sync_active_eyewear()
                    self.show_notification(f"Selected: {self.catalog.current_item.name}")
                elif key in (ord("p"), ord("P")) and self.catalog:
                    self.catalog.prev_accessory()
                    self._sync_active_eyewear()
                    self.show_notification(f"Selected: {self.catalog.current_item.name}")
                elif key in (ord("c"), ord("C")) and self.catalog:
                    c_name = self.catalog.cycle_frame_color()
                    self._sync_active_eyewear()
                    self.show_notification(f"Frame Color: {c_name}")
                elif key in (ord("l"), ord("L")) and self.catalog:
                    t_name = self.catalog.cycle_lens_tint()
                    self._sync_active_eyewear()
                    self.show_notification(f"Lens Tint: {t_name}")
                elif key in (ord("f"), ord("F")) and self.fitting_engine:
                    af_on = self.fitting_engine.adjustments.toggle_autofit()
                    self.show_notification(f"Auto-Fit: {'ON' if af_on else 'OFF'}")
                elif key == ord("[") and self.fitting_engine:
                    s = self.fitting_engine.adjustments.adjust_scale(-0.05)
                    self.show_notification(f"Scale: {s:.2f}x")
                elif key == ord("]") and self.fitting_engine:
                    s = self.fitting_engine.adjustments.adjust_scale(+0.05)
                    self.show_notification(f"Scale: {s:.2f}x")
                elif key in (ord("-"), ord("_")) and self.fitting_engine:
                    w_s = self.fitting_engine.adjustments.adjust_width(-0.05)
                    self.show_notification(f"Width: {w_s:.2f}x")
                elif key in (ord("="), ord("+")) and self.fitting_engine:
                    w_s = self.fitting_engine.adjustments.adjust_width(+0.05)
                    self.show_notification(f"Width: {w_s:.2f}x")
                elif key == 82 and self.fitting_engine:  # Up arrow
                    self.fitting_engine.adjustments.adjust_offset(dy=-2.0)
                    self.show_notification(f"Offset Y: {self.fitting_engine.adjustments.offset_y_mm:+.1f}mm")
                elif key == 84 and self.fitting_engine:  # Down arrow
                    self.fitting_engine.adjustments.adjust_offset(dy=+2.0)
                    self.show_notification(f"Offset Y: {self.fitting_engine.adjustments.offset_y_mm:+.1f}mm")
                elif key == 81 and self.fitting_engine:  # Left arrow
                    self.fitting_engine.adjustments.adjust_offset(dx=-2.0)
                    self.show_notification(f"Offset X: {self.fitting_engine.adjustments.offset_x_mm:+.1f}mm")
                elif key == 83 and self.fitting_engine:  # Right arrow
                    self.fitting_engine.adjustments.adjust_offset(dx=+2.0)
                    self.show_notification(f"Offset X: {self.fitting_engine.adjustments.offset_x_mm:+.1f}mm")
                elif key in (ord(","), ord("<")) and self.fitting_engine:
                    self.fitting_engine.adjustments.adjust_offset(dz=-3.0)
                    self.show_notification(f"Depth Z: {self.fitting_engine.adjustments.offset_z_mm:+.1f}mm")
                elif key in (ord("."), ord(">")) and self.fitting_engine:
                    self.fitting_engine.adjustments.adjust_offset(dz=+3.0)
                    self.show_notification(f"Depth Z: {self.fitting_engine.adjustments.offset_z_mm:+.1f}mm")
                elif key in (ord("v"), ord("V")):
                    if not self.in_comparison_mode:
                        # Add current look to comparison studio
                        if self.catalog:
                            item = self.catalog.current_item
                            self.screenshot_service.add_comparison_look(
                                frame=frame,
                                style_name=item.name,
                                shape_name=item.shape,
                                color_name=self.catalog.current_frame_color_name,
                            )
                        self.in_comparison_mode = True
                        self.show_notification("Eyewear Comparison Studio: OPEN")
                    else:
                        self.in_comparison_mode = False
                        self.show_notification("Live AR Preview")
                elif key in (ord("s"), ord("S")):
                    if self.in_comparison_mode:
                        saved_path = self.screenshot_service.save_comparison_sheet()
                        if saved_path:
                            self.show_notification(f"Comparison Sheet Saved: {os.path.basename(saved_path)}")
                    else:
                        tag = self.catalog.current_item.shape if self.catalog else "glasses"
                        saved_path = self.screenshot_service.capture(frame, tag)
                        if saved_path:
                            self.show_notification(f"Screenshot Saved: {os.path.basename(saved_path)}")
                elif key in (ord("r"), ord("R")):
                    head_pose_estimator.reset()
                    if self.fitting_engine:
                        self.fitting_engine.adjustments.reset()
                        self.fitting_engine.reset_filters()
                    if glasses_overlay:
                        glasses_overlay.reset_filter()
                    print("[MirrorFit Studio] Tracking and fitting adjustments reset.")
                    self.show_notification("Fitting & Tracking Reset")
                elif key in (ord("x"), ord("X")):
                    self.show_pose_axes = not self.show_pose_axes
                    print(f"[MirrorFit Studio] 3D Pose Axes: {'ON' if self.show_pose_axes else 'OFF'}")
                    self.show_notification(f"Axes: {'ON' if self.show_pose_axes else 'OFF'}")
                elif key in (ord("b"), ord("B")):
                    self.show_segmentation = not self.show_segmentation
                    print(f"[MirrorFit Studio] Face Segmentation Mask: {'ON' if self.show_segmentation else 'OFF'}")
                    self.show_notification(f"Segmentation: {'ON' if self.show_segmentation else 'OFF'}")
                elif key in (ord("g"), ord("G")):
                    self.show_glasses = not self.show_glasses
                    print(f"[MirrorFit Studio] Glasses visibility: {'ON' if self.show_glasses else 'OFF'}")
                elif key in (ord("m"), ord("M")):
                    self.show_mesh = not self.show_mesh
                    self.show_contours = self.show_mesh
                    print(f"[MirrorFit Studio] Mesh wireframe: {'ON' if self.show_mesh else 'OFF'}")
                elif key in (ord("a"), ord("A")):
                    self.show_anchors = not self.show_anchors
                    print(f"[MirrorFit Studio] Try-On Anchors: {'ON' if self.show_anchors else 'OFF'}")
                elif key in (ord("h"), ord("H")):
                    self.show_hud = not self.show_hud
                    print(f"[MirrorFit Studio] HUD overlay: {'ON' if self.show_hud else 'OFF'}")
                elif key in (9, ord("k"), ord("K")):  # 9 is TAB key
                    self.show_side_panel = not self.show_side_panel
                    print(f"[MirrorFit Studio] Studio Side Panel: {'ON' if self.show_side_panel else 'OFF'}")

        except KeyboardInterrupt:
            print("\n[MirrorFit Studio] Interrupted by terminal (Ctrl+C). Exiting...")

        finally:
            print("[MirrorFit Studio] Releasing camera and cleaning up resources...")
            camera.release()
            face_tracker.close()
            segmentation_module.close()
            cv2.destroyAllWindows()
            print("[MirrorFit Studio] Application shut down cleanly.")


def main() -> None:
    """Parse command line arguments and execute MirrorFit XR."""
    parser = argparse.ArgumentParser(
        description="MirrorFit XR Stage 5: Smart AR Virtual Eyewear Try-On Studio"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index (default: 0)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Target video width in pixels (default: 1280)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Target video height in pixels (default: 720)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Custom path to face_landmarker.task model file",
    )
    parser.add_argument(
        "--segmenter",
        type=str,
        default=None,
        help="Custom path to selfie_segmenter.tflite model file",
    )
    parser.add_argument(
        "--model3d",
        type=str,
        default="assets/models_3d/aviator.obj",
        help="Path to initial 3D OBJ model asset (default: assets/models_3d/aviator.obj)",
    )
    parser.add_argument(
        "--assets",
        type=str,
        default="assets/glasses",
        help="Path to 2D accessory PNG directory (default: assets/glasses)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=2.15,
        help="Scale multiplier for glasses width (default: 2.15)",
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=0.05,
        help="Vertical offset ratio for glasses positioning (default: 0.05)",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.35,
        help="Exponential smoothing weight beta in [0.1, 1.0] (default: 0.35)",
    )

    args = parser.parse_args()

    app = MirrorFitApp(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        model_path=args.model,
        segmenter_path=args.segmenter,
        model_3d_path=args.model3d,
        assets_dir=args.assets,
        scale_factor=args.scale,
        vertical_offset=args.offset,
        smoothing_factor=args.smoothing,
    )
    app.run()


if __name__ == "__main__":
    main()
