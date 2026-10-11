"""Main application entry point for MirrorFit.

Phase 3: Real-time face tracking, virtual accessory catalog (glasses & sunglasses),
anti-jitter head tracking, screenshot capture, and advanced HUD interface.
"""

import argparse
from pathlib import Path
import sys
import time
from typing import Optional

# Ensure project root directory is in sys.path for direct execution (e.g. python src/main.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from src.accessory_manager import AccessoryManager
from src.camera import Camera
from src.face_tracker import FaceTracker, TryOnAnchors
from src.glasses_overlay import GlassesOverlay
from src.screenshot_service import ScreenshotService


class MirrorFitApp:
    """Coordinates camera capture, face tracking, accessory management, rendering, and UI."""

    WINDOW_NAME = "MirrorFit | Virtual Try-On Studio"

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
        model_path: Optional[str] = None,
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
            assets_dir: Directory containing accessory PNG assets.
            scale_factor: Glasses width scale factor relative to eye distance.
            vertical_offset: Vertical offset ratio relative to glasses height.
            smoothing_factor: Exponential smoothing weight for jitter reduction.
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.model_path = model_path
        self.assets_dir = assets_dir
        self.scale_factor = scale_factor
        self.vertical_offset = vertical_offset
        self.smoothing_factor = smoothing_factor

        # UI toggle states
        self.show_glasses = True
        self.show_mesh = False
        self.show_contours = False
        self.show_anchors = False
        self.show_hud = True

        # Screenshot and notification banner tracking
        self.notification_message: Optional[str] = None
        self.notification_expiry: float = 0.0

        # FPS calculation variables
        self.prev_frame_time = time.perf_counter()
        self.smoothed_fps = 0.0
        self.fps_alpha = 0.1  # Smoothing factor (Exponential Moving Average)

    def show_notification(self, message: str, duration_sec: float = 2.5) -> None:
        """Display an on-screen notification badge on the HUD for a limited duration."""
        self.notification_message = message
        self.notification_expiry = time.perf_counter() + duration_sec

    def draw_hud(
        self,
        frame: np.ndarray,
        is_detected: bool,
        fps: float,
        anchors: Optional[TryOnAnchors],
        accessory_name: str,
    ) -> None:
        """Render a clean, modern computer vision HUD on the frame.

        Args:
            frame: OpenCV image frame (modified in-place).
            is_detected: Whether a face was detected in this frame.
            fps: Current smoothed frames-per-second value.
            anchors: Geometric fitting metrics if face detected.
            accessory_name: Name and index of active accessory.
        """
        if not self.show_hud:
            return

        h, w = frame.shape[:2]

        # -------------------------------------------------------------
        # 1. Top Header Bar & Bottom Footer Bar (Semi-transparent overlay)
        # -------------------------------------------------------------
        header_height = 56
        footer_height = 36
        overlay = frame.copy()

        cv2.rectangle(overlay, (0, 0), (w, header_height), (18, 18, 24), -1)
        cv2.rectangle(overlay, (0, h - footer_height), (w, h), (18, 18, 24), -1)

        # Blend overlay with original frame (alpha = 0.72)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

        # Accent border lines
        cv2.line(frame, (0, header_height), (w, header_height), (60, 60, 75), 1)
        cv2.line(frame, (0, h - footer_height), (w, h - footer_height), (60, 60, 75), 1)

        # -------------------------------------------------------------
        # 2. Application Title
        # -------------------------------------------------------------
        title_text = "MirrorFit | Virtual Try-On Studio"
        cv2.putText(
            frame,
            title_text,
            (20, 36),
            cv2.FONT_HERSHEY_DUPLEX,
            0.75,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # -------------------------------------------------------------
        # 3. Real-Time FPS Counter (Top Right)
        # -------------------------------------------------------------
        fps_text = f"FPS: {fps:.1f}"
        fps_color = (80, 220, 100) if fps >= 25 else (50, 160, 255)
        (fps_w, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_DUPLEX, 0.65, 1)
        cv2.putText(
            frame,
            fps_text,
            (w - fps_w - 20, 36),
            cv2.FONT_HERSHEY_DUPLEX,
            0.65,
            fps_color,
            1,
            cv2.LINE_AA,
        )

        # -------------------------------------------------------------
        # 4. Face Status Badge (Top Center-Right)
        # -------------------------------------------------------------
        badge_y = 14
        badge_h = 28

        if is_detected:
            status_text = "FACE DETECTED"
            badge_bg = (35, 130, 45)      # Deep Emerald Green
            badge_border = (60, 220, 90)  # Bright Green
            dot_color = (60, 255, 100)
        else:
            status_text = "NO FACE DETECTED"
            badge_bg = (30, 40, 150)      # Deep Amber/Red
            badge_border = (50, 80, 240)  # Bright Red
            dot_color = (60, 90, 255)

        (text_w, text_h), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        badge_w = text_w + 36
        badge_x = (w // 2) - (badge_w // 2)

        cv2.rectangle(
            frame,
            (badge_x, badge_y),
            (badge_x + badge_w, badge_y + badge_h),
            badge_bg,
            -1,
        )
        cv2.rectangle(
            frame,
            (badge_x, badge_y),
            (badge_x + badge_w, badge_y + badge_h),
            badge_border,
            1,
        )
        cv2.circle(frame, (badge_x + 14, badge_y + (badge_h // 2)), 4, dot_color, -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            status_text,
            (badge_x + 26, badge_y + (badge_h // 2) + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # -------------------------------------------------------------
        # 5. Live Geometric Telemetry & Accessory Info
        # -------------------------------------------------------------
        telemetry_y = header_height + 26
        if is_detected and anchors is not None:
            telemetry_text = (
                f"Item: {accessory_name}  |  "
                f"Tilt: {anchors.face_tilt_deg:+.1f} deg  |  "
                f"IPD: {anchors.interpupillary_distance_px:.0f}px  |  "
                f"Span: {anchors.glasses_width_px:.0f}px"
            )
        else:
            telemetry_text = f"Item: {accessory_name}  |  Tracking paused (no face)"

        (t_w, _), _ = cv2.getTextSize(telemetry_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.rectangle(
            frame,
            (16, telemetry_y - 18),
            (16 + t_w + 16, telemetry_y + 8),
            (20, 20, 25),
            -1,
        )
        cv2.rectangle(
            frame,
            (16, telemetry_y - 18),
            (16 + t_w + 16, telemetry_y + 8),
            (60, 60, 75),
            1,
        )
        cv2.putText(
            frame,
            telemetry_text,
            (24, telemetry_y - 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (210, 225, 255),
            1,
            cv2.LINE_AA,
        )

        # -------------------------------------------------------------
        # 6. Temporary On-Screen Notification Banner (e.g. Screenshot Saved)
        # -------------------------------------------------------------
        now = time.perf_counter()
        if self.notification_message and now < self.notification_expiry:
            banner_text = self.notification_message
            (n_w, _), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            banner_x = (w // 2) - (n_w // 2) - 16
            banner_y = h - footer_height - 38

            cv2.rectangle(
                frame,
                (banner_x, banner_y - 20),
                (banner_x + n_w + 32, banner_y + 12),
                (25, 120, 60),
                -1,
            )
            cv2.rectangle(
                frame,
                (banner_x, banner_y - 20),
                (banner_x + n_w + 32, banner_y + 12),
                (80, 255, 120),
                1,
            )
            cv2.putText(
                frame,
                banner_text,
                (banner_x + 16, banner_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # -------------------------------------------------------------
        # 7. Bottom Navigation & Keyboard Controls Footer
        # -------------------------------------------------------------
        footer_text = (
            "Controls: [1-4] Styles  |  "
            "[N/P] Next/Prev  |  "
            "[S] Save Screenshot  |  "
            "[R] Reset Filter  |  "
            f"[G] Glasses: {'ON' if self.show_glasses else 'OFF'}  |  "
            "[Q/ESC] Exit"
        )
        cv2.putText(
            frame,
            footer_text,
            (20, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (195, 195, 210),
            1,
            cv2.LINE_AA,
        )

    def calculate_fps(self) -> float:
        """Calculate exponential moving average FPS for smooth rendering."""
        current_time = time.perf_counter()
        delta = current_time - self.prev_frame_time
        self.prev_frame_time = current_time

        if delta > 0.0:
            instant_fps = 1.0 / delta
            if self.smoothed_fps == 0.0:
                self.smoothed_fps = instant_fps
            else:
                self.smoothed_fps = (
                    self.fps_alpha * instant_fps + (1.0 - self.fps_alpha) * self.smoothed_fps
                )

        return self.smoothed_fps

    def run(self) -> None:
        """Run the main capture, tracking, overlay, rendering, and event loop."""
        print("=" * 70)
        print("  MirrorFit | Real-Time Virtual Try-On Studio (Phase 3)")
        print("=" * 70)
        print("Initializing webcam, MediaPipe Face Landmarker, and Accessory Catalog...")

        # 1. Initialize Face Tracker
        try:
            face_tracker = FaceTracker(model_path=self.model_path)
        except FileNotFoundError as err:
            print(err)
            sys.exit(1)
        except Exception as exc:
            print(f"[Error] Failed to initialize Face Tracker: {exc}")
            sys.exit(1)

        # 2. Initialize Accessory Manager (Loads all transparent PNGs)
        try:
            accessory_mgr = AccessoryManager(assets_dir=self.assets_dir)
        except FileNotFoundError as err:
            print(err)
            face_tracker.close()
            sys.exit(1)

        # 3. Initialize Glasses Overlay Engine
        glasses_overlay = GlassesOverlay(
            asset_path=os.path.join(self.assets_dir, accessory_mgr.current_item.filename),
            scale_factor=self.scale_factor,
            vertical_offset_ratio=self.vertical_offset,
            smoothing_factor=self.smoothing_factor,
        )

        # 4. Initialize Screenshot Service
        screenshot_service = ScreenshotService()

        # 5. Initialize Camera
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
            sys.exit(1)

        actual_w, actual_h = camera.get_resolution()
        print(f"[MirrorFit] Camera started successfully at {actual_w}x{actual_h} resolution.")
        print("[MirrorFit] Ready! Press [1-4] for styles, [S] for screenshot, [R] to reset smoothing, and [Q]/[ESC] to exit.\n")

        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

        try:
            while True:
                # 1. Capture video frame
                success, frame = camera.read_frame()
                if not success or frame is None:
                    print("[Warning] Dropped or empty video frame received. Continuing...")
                    continue

                # 2. Process frame with FaceTracker
                is_detected, raw_landmarks, pixel_coords, anchors = face_tracker.process_frame(
                    frame
                )

                if is_detected and anchors is not None:
                    # 3. Apply active virtual glasses overlay
                    if self.show_glasses:
                        glasses_overlay.apply(
                            frame=frame,
                            left_eye=anchors.left_eye_center,
                            right_eye=anchors.right_eye_center,
                            nose_bridge=anchors.nose_bridge,
                            accessory_image=accessory_mgr.current_image,
                        )

                    # 4. Optional landmark visualization overlay
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
                    # When face is absent, reset temporal filter so accessories don't lag on return
                    glasses_overlay.reset_filter()

                # 5. Calculate smoothed FPS
                current_fps = self.calculate_fps()

                # 6. Render HUD Overlay
                self.draw_hud(
                    frame=frame,
                    is_detected=is_detected,
                    fps=current_fps,
                    anchors=anchors,
                    accessory_name=accessory_mgr.current_name,
                )

                # 7. Display frame in window
                cv2.imshow(self.WINDOW_NAME, frame)

                # 8. Handle keyboard input (1 ms wait)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):  # 27 is ESC key
                    print("\n[MirrorFit] Exit requested by user. Shutting down...")
                    break
                elif key in (ord("1"), ord("2"), ord("3"), ord("4")):
                    accessory_mgr.select_by_key(int(chr(key)))
                elif key in (ord("n"), ord("N")):
                    accessory_mgr.next_accessory()
                elif key in (ord("p"), ord("P")):
                    accessory_mgr.prev_accessory()
                elif key in (ord("s"), ord("S")):
                    saved_path = screenshot_service.capture(frame, accessory_mgr.current_item.name)
                    if saved_path:
                        self.show_notification(f"Screenshot Saved: {os.path.basename(saved_path)}")
                elif key in (ord("r"), ord("R")):
                    glasses_overlay.reset_filter()
                    print("[MirrorFit] Tracking smoothing filter reset.")
                    self.show_notification("Tracking Smoothing Reset")
                elif key in (ord("g"), ord("G")):
                    self.show_glasses = not self.show_glasses
                    print(f"[MirrorFit] Glasses overlay: {'ON' if self.show_glasses else 'OFF'}")
                elif key in (ord("m"), ord("M")):
                    self.show_mesh = not self.show_mesh
                    self.show_contours = self.show_mesh
                    print(f"[MirrorFit] Mesh visualization: {'ON' if self.show_mesh else 'OFF'}")
                elif key in (ord("a"), ord("A")):
                    self.show_anchors = not self.show_anchors
                    print(f"[MirrorFit] Try-On Anchors: {'ON' if self.show_anchors else 'OFF'}")
                elif key in (ord("h"), ord("H")):
                    self.show_hud = not self.show_hud
                    print(f"[MirrorFit] HUD display: {'ON' if self.show_hud else 'OFF'}")

        except KeyboardInterrupt:
            print("\n[MirrorFit] Interrupted by terminal (Ctrl+C). Exiting...")

        finally:
            # Clean up all resources
            print("[MirrorFit] Releasing camera and cleaning up OpenCV windows...")
            camera.release()
            face_tracker.close()
            cv2.destroyAllWindows()
            print("[MirrorFit] Application shut down cleanly.")


def main() -> None:
    """Parse command line arguments and execute the application."""
    parser = argparse.ArgumentParser(
        description="MirrorFit Phase 3: Advanced Head Tracking and Accessory Selection"
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
        "--assets",
        type=str,
        default="assets/glasses",
        help="Path to directory containing accessory PNG assets (default: assets/glasses)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=2.15,
        help="Scale multiplier for glasses width based on eye distance (default: 2.15)",
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
        assets_dir=args.assets,
        scale_factor=args.scale,
        vertical_offset=args.offset,
        smoothing_factor=args.smoothing,
    )
    app.run()


if __name__ == "__main__":
    main()
