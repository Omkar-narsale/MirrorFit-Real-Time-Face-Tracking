"""Main application entry point for MirrorFit.

Phase 1: Real-time face detection, 478-point landmark tracking, and try-on geometric anchors.
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

from src.camera import Camera
from src.face_tracker import FaceTracker, TryOnAnchors


class MirrorFitApp:
    """Coordinates camera capture, face tracking, UI rendering, and user input."""

    WINDOW_NAME = "MirrorFit | Virtual Try-On Studio"

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
        model_path: Optional[str] = None,
    ) -> None:
        """Initialize the application components.

        Args:
            camera_index: Video capture hardware index (default: 0).
            width: Preferred frame width in pixels.
            height: Preferred frame height in pixels.
            model_path: Optional custom path to face_landmarker.task.
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.model_path = model_path

        # UI toggle states
        self.show_mesh = True
        self.show_contours = True
        self.show_anchors = True
        self.show_hud = True

        # FPS calculation variables
        self.prev_frame_time = time.perf_counter()
        self.smoothed_fps = 0.0
        self.fps_alpha = 0.1  # Smoothing factor (Exponential Moving Average)

    def draw_hud(
        self,
        frame: np.ndarray,
        is_detected: bool,
        fps: float,
        anchors: Optional[TryOnAnchors],
    ) -> None:
        """Render a clean, modern computer vision HUD (Head-Up Display) on the frame.

        Args:
            frame: OpenCV image frame (modified in-place).
            is_detected: Whether a face was detected in this frame.
            fps: Current smoothed frames-per-second value.
            anchors: Geometric fitting metrics if face detected.
        """
        if not self.show_hud:
            return

        h, w = frame.shape[:2]

        # -------------------------------------------------------------
        # 1. Top Header Bar (Semi-transparent dark overlay)
        # -------------------------------------------------------------
        header_height = 56
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, header_height), (18, 18, 24), -1)

        # Bottom Footer Bar
        footer_height = 36
        cv2.rectangle(overlay, (0, h - footer_height), (w, h), (18, 18, 24), -1)

        # Blend overlay with original frame (alpha = 0.7)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Accent top line (Cyber cyan)
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

        # Draw rounded badge background
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

        # Status pulse dot
        cv2.circle(frame, (badge_x + 14, badge_y + (badge_h // 2)), 4, dot_color, -1, cv2.LINE_AA)

        # Status text
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
        # 5. Live Geometric Telemetry (If Face Detected)
        # -------------------------------------------------------------
        if is_detected and anchors is not None:
            telemetry_text = (
                f"Landmarks: 478 pts  |  Tilt: {anchors.face_tilt_deg:+.1f} deg  |  "
                f"IPD: {anchors.interpupillary_distance_px:.0f}px  |  "
                f"Frame Span: {anchors.glasses_width_px:.0f}px"
            )
            # Small pill banner below the header
            telemetry_y = header_height + 26
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
                (200, 220, 255),
                1,
                cv2.LINE_AA,
            )

        # -------------------------------------------------------------
        # 6. Bottom Navigation and Keyboard Controls Footer
        # -------------------------------------------------------------
        footer_text = (
            "Controls: [Q/ESC] Exit  |  "
            f"[M] Mesh: {'ON' if self.show_mesh else 'OFF'}  |  "
            f"[A] Anchors: {'ON' if self.show_anchors else 'OFF'}  |  "
            "[H] Toggle HUD"
        )
        cv2.putText(
            frame,
            footer_text,
            (20, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (190, 190, 205),
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
        """Run the main capture, tracking, rendering, and event loop."""
        print("=" * 65)
        print("  MirrorFit | Real-Time Virtual Try-On Studio (Phase 1)")
        print("=" * 65)
        print("Initializing webcam and MediaPipe Face Landmarker...")

        try:
            face_tracker = FaceTracker(model_path=self.model_path)
        except FileNotFoundError as err:
            print(err)
            sys.exit(1)
        except Exception as exc:
            print(f"[Error] Failed to initialize Face Tracker: {exc}")
            sys.exit(1)

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
        print("[MirrorFit] Ready! Focus the webcam window and press 'Q' or 'ESC' to exit.\n")

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

                # 3. Draw facial landmarks on the live frame
                if is_detected:
                    face_tracker.draw_landmarks(
                        frame=frame,
                        raw_landmarks=raw_landmarks,
                        tryon_anchors=anchors,
                        show_mesh=self.show_mesh,
                        show_contours=self.show_contours,
                        show_anchors=self.show_anchors,
                    )

                # 4. Calculate smoothed FPS
                current_fps = self.calculate_fps()

                # 5. Render HUD Overlay
                self.draw_hud(
                    frame=frame,
                    is_detected=is_detected,
                    fps=current_fps,
                    anchors=anchors,
                )

                # 6. Display frame in window
                cv2.imshow(self.WINDOW_NAME, frame)

                # 7. Handle keyboard input (1 ms wait)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):  # 27 is ESC key
                    print("\n[MirrorFit] Exit requested by user. Shutting down...")
                    break
                elif key in (ord("m"), ord("M")):
                    self.show_mesh = not self.show_mesh
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
            # 8. Clean up all resources
            print("[MirrorFit] Releasing camera and cleaning up OpenCV windows...")
            camera.release()
            face_tracker.close()
            cv2.destroyAllWindows()
            print("[MirrorFit] Application shut down cleanly.")


def main() -> None:
    """Parse command line arguments and execute the application."""
    parser = argparse.ArgumentParser(
        description="MirrorFit Phase 1: Real-Time Face Detection and Tracking"
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

    args = parser.parse_args()
    app = MirrorFitApp(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        model_path=args.model,
    )
    app.run()


if __name__ == "__main__":
    main()
