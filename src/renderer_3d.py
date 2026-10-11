"""3D accessory rendering engine for MirrorFit XR.

Parses 3D Wavefront OBJ models, applies 6-DOF head pose transformations,
performs camera perspective projection, depth-sorts polygonal faces (Painter's Algorithm),
culls occluded head-depth geometry, and renders solid shaded or wireframe 3D geometry
composited directly onto the video feed at 60+ FPS.
"""

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.fitting_engine import FittingResult


class Model3D:
    """Represents a loaded 3D polygonal mesh with precomputed geometry."""

    def __init__(self, obj_path: str) -> None:
        """Parse Wavefront OBJ file."""
        self.vertices: np.ndarray = np.empty((0, 3), dtype=np.float64)
        self.faces: List[List[int]] = []
        self.face_groups: List[str] = []
        self.face_centroids: np.ndarray = np.empty((0, 3), dtype=np.float64)
        self.face_normals: np.ndarray = np.empty((0, 3), dtype=np.float64)
        self.face_local_z: np.ndarray = np.empty((0,), dtype=np.float64)

        self._load_obj(obj_path)

    def _load_obj(self, path: str) -> None:
        """Parse vertex, group, and face lines from OBJ file and precompute geometry."""
        v_list = []
        f_list = []
        fg_list = []
        current_group = "frame"

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                if parts[0] in ("g", "group") and len(parts) >= 2:
                    current_group = parts[1].lower()
                elif parts[0] == "v" and len(parts) >= 4:
                    v_list.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif parts[0] == "f" and len(parts) >= 4:
                    face_v = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                    f_list.append(face_v)
                    fg_list.append(current_group)

        self.vertices = np.array(v_list, dtype=np.float64)
        self.faces = f_list
        self.face_groups = fg_list

        # Precompute model-space centroids, face normals, and local Z coordinates for 60+ FPS vectorization
        centroids = []
        normals = []
        local_z = []

        for f in self.faces:
            if len(f) >= 3:
                pts = self.vertices[f]
                centroids.append(np.mean(pts, axis=0))
                local_z.append(float(np.mean(pts[:, 2])))

                edge1 = pts[1] - pts[0]
                edge2 = pts[2] - pts[0]
                n = np.cross(edge1, edge2)
                norm_len = np.linalg.norm(n)
                normals.append(n / norm_len if norm_len > 1e-6 else np.array([0.0, 0.0, 1.0]))
            else:
                centroids.append(self.vertices[f[0]] if f else np.zeros(3))
                local_z.append(0.0)
                normals.append(np.array([0.0, 0.0, 1.0]))

        self.face_centroids = np.array(centroids, dtype=np.float64)
        self.face_normals = np.array(normals, dtype=np.float64)
        self.face_local_z = np.array(local_z, dtype=np.float64)


class Renderer3D:
    """Projective 3D Augmented Reality renderer.

    Why Projective Mesh Rendering over Heavy Headless OpenGL on Windows:
    1. Zero External Drivers: PyOpenGL on headless Windows often encounters WGL/EGL context failures.
    2. Direct Frame Buffer Compositing: Eliminates expensive GPU-to-CPU texture readbacks (glReadPixels).
    3. Mathematical Determinism: Complete control over the camera projection matrix K and lighting.
    4. Real-Time Performance: Runs at 60+ FPS on CPU with vectorized math, bounding-box ROI alpha blending,
       and depth-sorted polygon shading.
    """

    def __init__(
        self,
        model_path: str = "assets/models_3d/glasses_3d.obj",
        primary_color: Tuple[int, int, int] = (32, 34, 40),        # Dark Matte Charcoal / Titanium
        accent_color: Tuple[int, int, int] = (195, 170, 70),       # Subtle Metallic Gold Trim
        temple_color: Tuple[int, int, int] = (42, 44, 52),         # Dark Slate Frame Arms
        lens_color: Tuple[int, int, int] = (25, 30, 42),           # Polarized Charcoal Tint
        lens_alpha: float = 0.50,                                  # Semi-transparency for lenses
    ) -> None:
        """Initialize the 3D renderer.

        Args:
            model_path: Path to Wavefront OBJ 3D model asset.
            primary_color: BGR base frame color.
            accent_color: BGR rim highlight color.
            temple_color: BGR temple arm color.
            lens_color: BGR lens tint color.
            lens_alpha: Alpha blending factor for semi-transparent lenses (0.0 to 1.0).
        """
        self.primary_color = primary_color
        self.accent_color = accent_color
        self.temple_color = temple_color
        self.lens_color = lens_color
        self.lens_alpha = lens_alpha

        self.wireframe_only: bool = False
        self.model: Optional[Model3D] = None
        self.is_available: bool = False

        self._load_model(model_path)

    def toggle_wireframe(self) -> bool:
        """Toggle between solid shaded surfaces and wireframe debug mode."""
        self.wireframe_only = not self.wireframe_only
        return self.wireframe_only

    def _load_model(self, path: str) -> None:
        """Load 3D model file and validate geometry."""
        resolved = os.path.abspath(path)
        if not os.path.exists(resolved):
            print(
                f"[MirrorFit XR Warning] 3D Model file not found at '{resolved}'.\n"
                "3D rendering will operate in fallback mode."
            )
            self.is_available = False
            return

        try:
            self.model = Model3D(resolved)
            self.is_available = True
            print(
                f"[MirrorFit XR] Loaded 3D model '{os.path.basename(resolved)}' "
                f"({len(self.model.vertices)} vertices, {len(self.model.faces)} faces)."
            )
        except Exception as exc:
            print(f"[MirrorFit XR Warning] Failed to parse 3D model '{resolved}': {exc}")
            self.is_available = False

    def render(
        self,
        frame: np.ndarray,
        fitting: FittingResult,
        camera_matrix: np.ndarray,
        dist_coeffs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Render the 3D accessory onto the frame using the estimated head pose.

        Applies 6-DOF transformation, perspective projection, geometric temple
        occlusion culling, and Painter's algorithm depth-sorted rendering.

        Args:
            frame: Video frame (modified in-place).
            fitting: FittingResult containing 6-DOF pose and transformation matrix.
            camera_matrix: 3x3 intrinsic camera matrix.
            dist_coeffs: Lens distortion coefficients.

        Returns:
            np.ndarray: Rendered video frame with 3D model composited.
        """
        if not self.is_available or self.model is None or not fitting.is_valid:
            return frame

        if dist_coeffs is None:
            dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # 1. Transform vertices and centroids to Camera Coordinate Space
        R, _ = cv2.Rodrigues(fitting.rvec)
        scaled_vertices = self.model.vertices * fitting.scale
        cam_centroids = (self.model.face_centroids * fitting.scale) @ R.T + fitting.tvec.flatten()
        depths = cam_centroids[:, 2]

        # 2. Perspective Projection to Image Space (2D pixels)
        img_pts, _ = cv2.projectPoints(
            scaled_vertices,
            fitting.rvec,
            fitting.tvec,
            camera_matrix,
            dist_coeffs,
        )
        pts_2d = img_pts.reshape(-1, 2)

        # 3. Geometric Temple Occlusion & Far-Side Culling
        # Normal pointing laterally outward from left/right side of head:
        # Left normal in head coordinates: [-1, 0, 0]; Right normal: [1, 0, 0]
        # In camera coordinates, view direction from object towards camera is [0, 0, -1]
        # If normal points away from camera (Z component > 0.16 in camera space),
        # that side of the head is facing away from camera and is occluded by the skull.
        n_left_cam = R @ np.array([-1.0, 0.0, 0.0], dtype=np.float64)
        n_right_cam = R @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
        left_temple_occluded = n_left_cam[2] > 0.16
        right_temple_occluded = n_right_cam[2] > 0.16

        # 4. Vectorized Directional Lighting (Key Light upper-front)
        light_dir = np.array([0.25, -0.65, -0.71], dtype=np.float64)
        light_dir /= np.linalg.norm(light_dir)
        N_cam = self.model.face_normals @ R.T
        diffuse_all = np.maximum(0.0, N_cam @ light_dir)

        # 5. Depth Sorting (Painter's Algorithm: farthest first)
        sorted_order = np.argsort(depths)[::-1]
        h, w = frame.shape[:2]

        # 6. Wireframe Mode (Debug Visualizer)
        if self.wireframe_only:
            for face_idx in sorted_order:
                group = self.model.face_groups[face_idx]

                # Cull occluded temple arms behind the head volume
                if group == "left_temple" and left_temple_occluded and self.model.face_local_z[face_idx] > 15.0:
                    continue
                if group == "right_temple" and right_temple_occluded and self.model.face_local_z[face_idx] > 15.0:
                    continue

                face = self.model.faces[face_idx]
                poly_pts = np.round(pts_2d[face]).astype(np.int32)
                if np.all(poly_pts[:, 0] < 0) or np.all(poly_pts[:, 0] >= w) or \
                   np.all(poly_pts[:, 1] < 0) or np.all(poly_pts[:, 1] >= h):
                    continue

                if group == "lenses":
                    wire_color = (255, 180, 50)     # Sky blue for lenses
                elif "temple" in group:
                    wire_color = (120, 210, 255)    # Light gold/yellow for temples
                else:
                    wire_color = (80, 240, 160)     # Bright cyan for front frame

                cv2.polylines(frame, [poly_pts], isClosed=True, color=wire_color, thickness=1, lineType=cv2.LINE_AA)
            return frame

        # 7. Solid Shaded Mode (Default High-Fidelity AR)
        lens_render_queue = []

        for face_idx in sorted_order:
            group = self.model.face_groups[face_idx]

            # Cull occluded temple arms behind the head volume
            if group == "left_temple" and left_temple_occluded and self.model.face_local_z[face_idx] > 15.0:
                continue
            if group == "right_temple" and right_temple_occluded and self.model.face_local_z[face_idx] > 15.0:
                continue

            face = self.model.faces[face_idx]
            poly_pts = np.round(pts_2d[face]).astype(np.int32)
            if np.all(poly_pts[:, 0] < 0) or np.all(poly_pts[:, 0] >= w) or \
               np.all(poly_pts[:, 1] < 0) or np.all(poly_pts[:, 1] >= h):
                continue

            if group == "lenses":
                lens_render_queue.append((poly_pts, diffuse_all[face_idx]))
                continue

            # Frame & Temple Shading
            base_color = self.temple_color if "temple" in group else self.primary_color
            accent_edge = (60, 60, 75) if "temple" in group else self.accent_color

            # Lambertian Shading: Base Ambient (0.38) + Diffuse (0.62)
            intensity = 0.38 + 0.62 * diffuse_all[face_idx]
            shaded_color = (
                int(np.clip(base_color[0] * intensity, 0, 255)),
                int(np.clip(base_color[1] * intensity, 0, 255)),
                int(np.clip(base_color[2] * intensity, 0, 255)),
            )

            # Draw solid filled polygon and crisp edge contour
            cv2.fillPoly(frame, [poly_pts], shaded_color, lineType=cv2.LINE_AA)
            cv2.polylines(frame, [poly_pts], isClosed=True, color=accent_edge, thickness=1, lineType=cv2.LINE_AA)

        # 8. High-Performance Bounding-Box ROI Lens Alpha Blending
        if lens_render_queue:
            all_lens_pts = np.vstack([pts for pts, _ in lens_render_queue])
            x0 = max(0, int(np.min(all_lens_pts[:, 0])))
            y0 = max(0, int(np.min(all_lens_pts[:, 1])))
            x1 = min(w, int(np.max(all_lens_pts[:, 0])) + 1)
            y1 = min(h, int(np.max(all_lens_pts[:, 1])) + 1)

            if x1 > x0 and y1 > y0:
                roi = frame[y0:y1, x0:x1]
                sub_overlay = np.zeros_like(roi)
                sub_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
                roi_offset = np.array([x0, y0], dtype=np.int32)

                for poly_pts, diffuse in lens_render_queue:
                    sub_pts = poly_pts - roi_offset
                    tint_intensity = 0.50 + 0.50 * diffuse
                    tint_color = (
                        int(np.clip(self.lens_color[0] * tint_intensity, 0, 255)),
                        int(np.clip(self.lens_color[1] * tint_intensity, 0, 255)),
                        int(np.clip(self.lens_color[2] * tint_intensity, 0, 255)),
                    )
                    cv2.fillPoly(sub_overlay, [sub_pts], tint_color, lineType=cv2.LINE_AA)
                    cv2.fillPoly(sub_mask, [sub_pts], 255, lineType=cv2.LINE_AA)

                mb = sub_mask > 0
                alpha = self.lens_alpha
                roi[mb] = (
                    (1.0 - alpha) * roi[mb].astype(np.float32)
                    + alpha * sub_overlay[mb].astype(np.float32)
                ).astype(np.uint8)

                for poly_pts, _ in lens_render_queue:
                    cv2.polylines(frame, [poly_pts], isClosed=True, color=(140, 150, 165), thickness=1, lineType=cv2.LINE_AA)

        return frame
