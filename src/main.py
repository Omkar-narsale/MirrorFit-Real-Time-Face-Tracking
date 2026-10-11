"""Main application entry point for MirrorFit XR.

Phase 4: Real-time 6-DOF head pose estimation, 3D accessory rendering,
face segmentation, intelligent fitting engine, performance monitoring, and multi-accessory try-on.
"""

import argparse
import os
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
from src.face_segmentation import FaceSegmentationModule
from src.face_tracker import FaceTracker, TryOnAnchors
from src.fitting_engine import FittingEngine
from src.glasses_overlay import GlassesOverlay
from src.head_pose import HeadPoseEstimator, HeadPoseResult
from src.performance_monitor import PerformanceMetrics, PerformanceMonitor
from src.renderer_3d import Renderer3D
from src.screenshot_service import ScreenshotService


class MirrorFitApp:
    """Coordinates camera capture, head pose, 3D rendering, segmentation, and HUD controls."""

    WINDOW_NAME = "MirrorFit XR | 3D Virtual Try-On Studio"

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
        model_path: Optional[str] = None,
        segmenter_path: Optional[str] = None,
        model_3d_path: str = "assets/models_3d/glasses_3d.obj",
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
            model_3d_path: Path to 3D OBJ model asset.
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
        self.use_3d_mode = True            # Primary mode is 3D AR Model; Toggle via 'T'
        self.show_glasses = True           # Toggle glasses on/off via 'G'
        self.show_pose_axes = False        # Toggle 3D axes on/off via 'X'
        self.show_segmentation = False     # Toggle segmentation mask on/off via 'B'
        self.show_mesh = False             # Toggle face mesh on/off via 'M'
        self.show_contours = False
        self.show_anchors = False          # Toggle try-on anchors on/off via 'A'
        self.show_hud = True               # Toggle HUD overlay on/off via 'H'

        # Screenshot and notification banner tracking
        self.notification_message: Optional[str] = None
        self.notification_expiry: float = 0.0

        # Performance monitoring engine
        self.perf_monitor = PerformanceMonitor()

    def show_notification(self, message: str, duration_sec: float = 2.5) -> None:
        """Display an on-screen notification badge on the HUD for a limited duration."""
        self.notification_message = message
        self.notification_expiry = time.perf_counter() + duration_sec

    def draw_hud(
        self,
        frame: np.ndarray,
        is_detected: bool,
        metrics: PerformanceMetrics,
        pose: HeadPoseResult,
        anchors: Optional[TryOnAnchors],
        accessory_name: str,
        renderer_3d: Optional[Renderer3D] = None,
    ) -> None:
        """Render a clean, modern computer vision HUD on the frame."""
        if not self.show_hud:
            return

        h, w = frame.shape[:2]

        # -------------------------------------------------------------
        # 1. Top Header Bar & Bottom Footer Bar (Semi-transparent overlay)
        # -------------------------------------------------------------
        header_height = 58
        footer_height = 36
        overlay = frame.copy()

        cv2.rectangle(overlay, (0, 0), (w, header_height), (18, 18, 24), -1)
        cv2.rectangle(overlay, (0, h - footer_height), (w, h), (18, 18, 24), -1)

        cv2.addWeighted(overlay, 0.74, frame, 0.26, 0, frame)

        # Accent border lines
        cv2.line(frame, (0, header_height), (w, header_height), (60, 60, 75), 1)
        cv2.line(frame, (0, h - footer_height), (w, h - footer_height), (60, 60, 75), 1)

        # -------------------------------------------------------------
        # 2. Application Title & Mode Tag
        # -------------------------------------------------------------
        if self.use_3d_mode:
            if renderer_3d and renderer_3d.wireframe_only:
                mode_badge = "3D AR (WIREFRAME)"
                mode_color = (120, 220, 255)
            else:
                mode_badge = "3D AR (SOLID MESH)"
                mode_color = (0, 220, 255)
        else:
            mode_badge = f"2D: {accessory_name.upper()}"
            mode_color = (220, 160, 40)
        title_text = "MirrorFit XR"

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

        # Mode pill
        (t_w, _), _ = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_DUPLEX, 0.75, 1)
        (mb_w, _), _ = cv2.getTextSize(mode_badge, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        pill_x = 20 + t_w + 14
        pill_w = max(110, mb_w + 20)
        cv2.rectangle(frame, (pill_x, 16), (pill_x + pill_w, 42), (35, 40, 50), -1)
        cv2.rectangle(frame, (pill_x, 16), (pill_x + pill_w, 42), mode_color, 1)
        cv2.putText(
            frame,
            mode_badge,
            (pill_x + 10, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            mode_color,
            1,
            cv2.LINE_AA,
        )

        # -------------------------------------------------------------
        # 3. Real-Time Performance Counter (Top Right)
        # -------------------------------------------------------------
        perf_text = f"FPS: {metrics.fps:.1f}  |  Lat: {metrics.total_latency_ms:.1f}ms"
        fps_color = (80, 220, 100) if metrics.fps >= 25 else (50, 160, 255)
        (fps_w, _), _ = cv2.getTextSize(perf_text, cv2.FONT_HERSHEY_DUPLEX, 0.58, 1)
        cv2.putText(
            frame,
            perf_text,
            (w - fps_w - 20, 36),
            cv2.FONT_HERSHEY_DUPLEX,
            0.58,
            fps_color,
            1,
            cv2.LINE_AA,
        )

        # -------------------------------------------------------------
        # 4. Face Status Badge (Top Center)
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
        # 5. Live Geometric Telemetry & 6-DOF Head Pose (Below Header)
        # -------------------------------------------------------------
        telemetry_y = header_height + 26
        if is_detected and pose.is_valid:
            if self.use_3d_mode:
                wf_mode = "Wireframe" if (renderer_3d and renderer_3d.wireframe_only) else "Solid"
                active_item = f"3D Aviator ({wf_mode})"
                num_v = len(renderer_3d.model.vertices) if (renderer_3d and renderer_3d.model) else 0
                num_f = len(renderer_3d.model.faces) if (renderer_3d and renderer_3d.model) else 0
                model_stat = f"3D Mesh: OK ({num_v}v/{num_f}f)" if (renderer_3d and renderer_3d.is_available) else "3D: Fallback"
            else:
                active_item = f"2D Style ({accessory_name})"
                model_stat = "2D PNG"

            seg_info = f"Seg: {metrics.segmentation_latency_ms:.1f}ms" if self.show_segmentation else "Seg: OFF"
            telemetry_text = (
                f"Pose: Yaw {pose.yaw_deg:+.1f} deg  Pitch {pose.pitch_deg:+.1f} deg  Roll {pose.roll_deg:+.1f} deg  |  "
                f"{active_item}  |  {model_stat}  |  {seg_info}"
            )
        else:
            telemetry_text = "Tracking paused (waiting for face)"

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
        # 6. Temporary On-Screen Notification Banner
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
        wf_stat = "ON" if (renderer_3d and renderer_3d.wireframe_only) else "OFF"
        footer_text = (
            f"[T] Mode: {'3D' if self.use_3d_mode else '2D'}  |  "
            f"[W] Wire: {wf_stat}  |  "
            f"[X] Axes: {'ON' if self.show_pose_axes else 'OFF'}  |  "
            f"[B] Mask: {'ON' if self.show_segmentation else 'OFF'}  |  "
            "[1-4/N/P] 2D Styles  |  "
            "[S] Screenshot  |  "
            "[R] Reset  |  "
            "[Q/ESC] Exit"
        )
        cv2.putText(
            frame,
            footer_text,
            (20, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.41,
            (195, 195, 210),
            1,
            cv2.LINE_AA,
        )

    def run(self) -> None:
        """Run the main capture, tracking, 3D rendering, segmentation, and event loop."""
        print("=" * 72)
        print("  MirrorFit XR | 3D Augmented Reality Studio (Phase 4)")
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

        # 4. Initialize Fitting Engine & 3D Renderer
        fitting_engine = FittingEngine()
        renderer_3d = Renderer3D(model_path=self.model_3d_path)

        # 5. Initialize 2D Accessory Manager & Overlay Engine (Fallback / Mode Switching)
        try:
            accessory_mgr = AccessoryManager(assets_dir=self.assets_dir)
            glasses_overlay = GlassesOverlay(
                asset_path=os.path.join(self.assets_dir, accessory_mgr.current_item.filename),
                scale_factor=self.scale_factor,
                vertical_offset_ratio=self.vertical_offset,
                smoothing_factor=self.smoothing_factor,
            )
        except Exception as exc:
            print(f"[MirrorFit XR Warning] 2D Accessory initialization warning: {exc}")
            accessory_mgr = None
            glasses_overlay = None

        # 6. Initialize Screenshot Service
        screenshot_service = ScreenshotService()

        # 7. Initialize Camera
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

        print(f"[MirrorFit XR] Camera started successfully at {actual_w}x{actual_h} resolution.")
        print("[MirrorFit XR] Controls: [T] Toggle 3D/2D, [W] Wireframe, [X] 3D Axes, [B] Segmentation, [1-4/N/P] 2D Styles, [S] Screenshot, [R] Reset, [Q] Exit.\n")

        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

        try:
            while True:
                frame_start_time = time.perf_counter()

                # 1. Capture video frame
                success, frame = camera.read_frame()
                if not success or frame is None:
                    continue

                # 2. Face Tracking (MediaPipe 478 landmarks)
                is_detected, raw_landmarks, pixel_coords, anchors = face_tracker.process_frame(
                    frame
                )

                pose_res = HeadPoseResult(
                    rvec=np.zeros((3, 1)),
                    tvec=np.zeros((3, 1)),
                    rotation_matrix=np.eye(3),
                    yaw_deg=0.0,
                    pitch_deg=0.0,
                    roll_deg=0.0,
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

                    # 4. Optional Face Segmentation Inference
                    if self.show_segmentation and segmentation_module.is_available:
                        mask, seg_ms = segmentation_module.segment(frame)
                        if mask is not None:
                            FaceSegmentationModule.render_debug_overlay(frame, mask)

                    # 5. Accessory Fitting & Rendering (3D vs 2D)
                    t_rend = time.perf_counter()
                    if self.show_glasses:
                        if self.use_3d_mode:
                            if renderer_3d.is_available and pose_res.is_valid:
                                # 3D AR Model Rendering Path with distance-invariant scaling
                                fitting_res = fitting_engine.compute_fitting(
                                    pose=pose_res,
                                    pixel_landmarks=pixel_coords,
                                    frame_shape=(actual_h, actual_w),
                                    focal_length=cam_matrix[0, 0],
                                )
                                renderer_3d.render(frame, fitting_res, cam_matrix, dist_coeffs)
                        else:
                            if glasses_overlay is not None and anchors is not None:
                                # 2D Image Overlay Path
                                current_img = accessory_mgr.current_image if accessory_mgr else None
                                glasses_overlay.apply(
                                    frame=frame,
                                    left_eye=anchors.left_eye_center,
                                    right_eye=anchors.right_eye_center,
                                    nose_bridge=anchors.nose_bridge,
                                    accessory_image=current_img,
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
                    # Reset smoothing filters when face is lost
                    head_pose_estimator.reset()
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

                # 9. Render HUD Overlay
                active_2d_name = accessory_mgr.current_name if accessory_mgr else "2D Default"
                self.draw_hud(
                    frame=frame,
                    is_detected=is_detected,
                    metrics=metrics,
                    pose=pose_res,
                    anchors=anchors,
                    accessory_name=active_2d_name,
                    renderer_3d=renderer_3d,
                )

                # 10. Display Frame in Window
                cv2.imshow(self.WINDOW_NAME, frame)

                # 11. Handle Keyboard Controls
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):  # 27 is ESC key
                    print("\n[MirrorFit XR] Exit requested by user. Shutting down...")
                    break
                elif key in (ord("t"), ord("T")):
                    self.use_3d_mode = not self.use_3d_mode
                    active_label = "3D AR Model" if self.use_3d_mode else f"2D ({accessory_mgr.current_item.name if accessory_mgr else 'Default'})"
                    print(f"[MirrorFit XR] Rendering mode switched to: {active_label}")
                    self.show_notification(f"Mode: {active_label}")
                elif key in (ord("w"), ord("W")):
                    if self.use_3d_mode:
                        is_wf = renderer_3d.toggle_wireframe()
                        wf_label = "Wireframe Debug (ON)" if is_wf else "Solid Shaded (OFF)"
                        print(f"[MirrorFit XR] 3D Wireframe mode: {wf_label}")
                        self.show_notification(f"3D Visual: {wf_label}")
                    else:
                        self.show_notification("Switch to 3D Mode ([T]) for Wireframe")
                elif key in (ord("x"), ord("X")):
                    self.show_pose_axes = not self.show_pose_axes
                    print(f"[MirrorFit XR] 3D Pose Axes: {'ON' if self.show_pose_axes else 'OFF'}")
                    self.show_notification(f"Axes: {'ON' if self.show_pose_axes else 'OFF'}")
                elif key in (ord("b"), ord("B")):
                    self.show_segmentation = not self.show_segmentation
                    print(f"[MirrorFit XR] Face Segmentation Mask: {'ON' if self.show_segmentation else 'OFF'}")
                    self.show_notification(f"Segmentation: {'ON' if self.show_segmentation else 'OFF'}")
                elif key in (ord("1"), ord("2"), ord("3"), ord("4")) and accessory_mgr:
                    idx = int(chr(key))
                    accessory_mgr.select_by_key(idx)
                    self.use_3d_mode = False
                    self.show_notification(f"2D Style: {accessory_mgr.current_item.name}")
                    print(f"[MirrorFit XR] Selected 2D style [{idx}]: {accessory_mgr.current_item.name}")
                elif key in (ord("n"), ord("N")) and accessory_mgr:
                    accessory_mgr.next_accessory()
                    self.use_3d_mode = False
                    self.show_notification(f"2D Style: {accessory_mgr.current_item.name}")
                elif key in (ord("p"), ord("P")) and accessory_mgr:
                    accessory_mgr.prev_accessory()
                    self.use_3d_mode = False
                    self.show_notification(f"2D Style: {accessory_mgr.current_item.name}")
                elif key in (ord("s"), ord("S")):
                    tag = "3d_model" if self.use_3d_mode else (accessory_mgr.current_item.name if accessory_mgr else "frame")
                    saved_path = screenshot_service.capture(frame, tag)
                    if saved_path:
                        self.show_notification(f"Screenshot Saved: {os.path.basename(saved_path)}")
                elif key in (ord("r"), ord("R")):
                    head_pose_estimator.reset()
                    if glasses_overlay:
                        glasses_overlay.reset_filter()
                    print("[MirrorFit XR] Tracking smoothing filters reset.")
                    self.show_notification("Smoothing Reset")
                elif key in (ord("g"), ord("G")):
                    self.show_glasses = not self.show_glasses
                    print(f"[MirrorFit XR] Glasses visibility: {'ON' if self.show_glasses else 'OFF'}")
                elif key in (ord("m"), ord("M")):
                    self.show_mesh = not self.show_mesh
                    self.show_contours = self.show_mesh
                    print(f"[MirrorFit XR] Mesh wireframe: {'ON' if self.show_mesh else 'OFF'}")
                elif key in (ord("a"), ord("A")):
                    self.show_anchors = not self.show_anchors
                    print(f"[MirrorFit XR] Try-On Anchors: {'ON' if self.show_anchors else 'OFF'}")
                elif key in (ord("h"), ord("H")):
                    self.show_hud = not self.show_hud
                    print(f"[MirrorFit XR] HUD overlay: {'ON' if self.show_hud else 'OFF'}")

        except KeyboardInterrupt:
            print("\n[MirrorFit XR] Interrupted by terminal (Ctrl+C). Exiting...")

        finally:
            print("[MirrorFit XR] Releasing camera and cleaning up resources...")
            camera.release()
            face_tracker.close()
            segmentation_module.close()
            cv2.destroyAllWindows()
            print("[MirrorFit XR] Application shut down cleanly.")


def main() -> None:
    """Parse command line arguments and execute MirrorFit XR."""
    parser = argparse.ArgumentParser(
        description="MirrorFit XR Phase 4: 3D Augmented Reality & Intelligent Face Fitting"
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
        default="assets/models_3d/glasses_3d.obj",
        help="Path to 3D OBJ model asset (default: assets/models_3d/glasses_3d.obj)",
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
