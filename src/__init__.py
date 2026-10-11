"""MirrorFit - Real-Time Virtual Try-On Studio.

Phase 1: Real-time face detection and 478-point landmark tracking.
"""

from src.camera import Camera
from src.face_tracker import FaceTracker

__all__ = ["Camera", "FaceTracker"]
