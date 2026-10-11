"""MirrorFit XR - 3D Augmented Reality & Intelligent Face Fitting.

Phase 4: Real-time 6-DOF head pose estimation, face segmentation, 3D accessory rendering,
fitting engine, performance monitoring, and multi-accessory management.
"""

from src.accessory_manager import AccessoryManager
from src.camera import Camera
from src.face_segmentation import FaceSegmentationModule
from src.face_tracker import FaceTracker
from src.fitting_engine import FittingEngine, FittingResult
from src.glasses_overlay import GlassesOverlay
from src.head_pose import HeadPoseEstimator, HeadPoseResult
from src.performance_monitor import PerformanceMonitor
from src.renderer_3d import Renderer3D
from src.screenshot_service import ScreenshotService

__all__ = [
    "AccessoryManager",
    "Camera",
    "FaceSegmentationModule",
    "FaceTracker",
    "FittingEngine",
    "FittingResult",
    "GlassesOverlay",
    "HeadPoseEstimator",
    "HeadPoseResult",
    "PerformanceMonitor",
    "Renderer3D",
    "ScreenshotService",
]
