"""Performance monitoring module for MirrorFit XR.

Tracks real-time framerate (FPS), frame processing latency, head pose estimation latency,
face segmentation inference latency, and accessory rendering latency.
"""

from dataclasses import dataclass
import time


@dataclass
class PerformanceMetrics:
    """Consolidated performance telemetry."""

    fps: float
    total_latency_ms: float
    pose_latency_ms: float
    segmentation_latency_ms: float
    render_latency_ms: float


class PerformanceMonitor:
    """Calculates smoothed FPS and detailed component latencies."""

    def __init__(self, fps_alpha: float = 0.1) -> None:
        """Initialize performance monitor.

        Args:
            fps_alpha: Exponential moving average smoothing factor.
        """
        self.fps_alpha = fps_alpha
        self.prev_time = time.perf_counter()
        self.smoothed_fps = 0.0

        # Latency records in milliseconds
        self.total_latency_ms = 0.0
        self.pose_latency_ms = 0.0
        self.segmentation_latency_ms = 0.0
        self.render_latency_ms = 0.0

    def update_fps(self) -> float:
        """Update and return smoothed FPS."""
        now = time.perf_counter()
        delta = now - self.prev_time
        self.prev_time = now

        if delta > 0.0:
            instant_fps = 1.0 / delta
            if self.smoothed_fps == 0.0:
                self.smoothed_fps = instant_fps
            else:
                self.smoothed_fps = (
                    self.fps_alpha * instant_fps + (1.0 - self.fps_alpha) * self.smoothed_fps
                )

        return self.smoothed_fps

    def record_latencies(
        self,
        total_ms: float,
        pose_ms: float = 0.0,
        seg_ms: float = 0.0,
        render_ms: float = 0.0,
    ) -> None:
        """Record per-stage execution times."""
        self.total_latency_ms = total_ms
        self.pose_latency_ms = pose_ms
        self.segmentation_latency_ms = seg_ms
        self.render_latency_ms = render_ms

    def get_metrics(self) -> PerformanceMetrics:
        """Retrieve current performance metrics snapshot."""
        return PerformanceMetrics(
            fps=round(self.smoothed_fps, 1),
            total_latency_ms=round(self.total_latency_ms, 1),
            pose_latency_ms=round(self.pose_latency_ms, 1),
            segmentation_latency_ms=round(self.segmentation_latency_ms, 1),
            render_latency_ms=round(self.render_latency_ms, 1),
        )
