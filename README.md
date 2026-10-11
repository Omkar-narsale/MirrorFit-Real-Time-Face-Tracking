# MirrorFit XR | 3D Augmented Reality & Intelligent Face Fitting (Phase 4)

**MirrorFit XR** is a real-time computer vision and augmented reality application that tracks 3D head pose, performs real-time face segmentation, and renders shaded 3D polygonal accessories using **Python**, **OpenCV**, **MediaPipe**, and **NumPy**.

* **Phase 1:** High-speed webcam capture, 478-point 3D facial landmark detection, and HUD telemetry.
* **Phase 2:** Virtual glasses overlay with dynamic scaling, head tilt rotation, boundary clipping, and alpha compositing.
* **Phase 3:** Anti-jitter temporal smoothing, multi-accessory 2D catalog, and timestamped screenshot service.
* **Phase 4.1 (Stage 4.1 - Current):** **Enhanced 3D Augmented Reality & Realistic Fitting**:
  * **Robust 2D/3D Mode Switching:** Reliable `[T]` toggle between 3D AR Model Mode and 2D Overlay Mode. Selecting 2D styles `1-4` or `N`/`P` switches to 2D without disabling 3D, and pressing `T` returns to 3D mode seamlessly.
  * **Realistic Solid 3D Glasses Geometry:** 162 vertices, 180 faces featuring solid volumetric front rims, anatomical bridge saddle, tinted semi-transparent lenses, and volumetric temple arms extending back toward the ears ($Z = +95\text{ mm}$) with curved earpieces ($Z = +115\text{ mm}$).
  * **Solid Shaded Surfaces Default with Wireframe Debug:** Renders shaded solid surfaces with directional Lambertian lighting and anti-aliased contours by default; press `[W]` to toggle wireframe debug mode.
  * **Geometric Temple Arm Occlusion:** Dynamically evaluates head yaw and outward temple normals in camera space to cull far-side geometry behind the skull, preventing temple arms from drawing across the eyes or nose.
  * **Proportional Anthropometric Scaling:** Dynamically scales glasses width according to detected eye and face dimensions, remaining stable when the user moves closer or farther.
  * **60+ FPS Real-Time Optimization:** Precomputes face normals and centroids, vectorizes lighting and depth calculations, utilizes bounding-box ROI lens alpha blending, and downsamples segmentation input to native $256 \times 256$, boosting frame rates from ~10 FPS to 60+ FPS.

---

## 1. Project Structure

```
MirrorFit/
├── assets/
│   ├── glasses/                # 2D Transparent PNG accessory assets
│   │   ├── default_glasses.png
│   │   ├── classic_glasses.png
│   │   ├── round_glasses.png
│   │   └── sunglasses.png
│   └── models_3d/              # 3D Accessory models
│       └── glasses_3d.obj      # Realistic 3D OBJ glasses mesh (162v / 180f)
├── models/
│   ├── face_landmarker.task    # Google MediaPipe 478-point Face Landmarker (~3.76 MB)
│   └── selfie_segmenter.tflite # Google MediaPipe Image Segmenter (~250 KB)
├── output/
│   └── screenshots/            # Timestamped screenshot captures
├── scratch/
│   └── test_stage4_1.py        # Automated Stage 4.1 test and benchmark suite
├── src/
│   ├── __init__.py             # Package exports
│   ├── accessory_manager.py    # 2D Multi-accessory catalog & switcher
│   ├── camera.py               # DirectShow camera acquisition & mirroring
│   ├── face_segmentation.py    # MediaPipe face segmentation module (optimized 256x256)
│   ├── face_tracker.py         # 478-point landmark tracker & geometric anchors
│   ├── fitting_engine.py       # Decoupled 3D transformation & anthropometric scaling
│   ├── glasses_overlay.py      # 2D Scaling, rotation, & alpha compositing
│   ├── head_pose.py            # 6-DOF solvePnP head pose estimator & axes
│   ├── performance_monitor.py  # Latency tracking and smoothed FPS calculation
│   ├── renderer_3d.py          # Vectorized 3D OBJ mesh renderer, shader, & culling
│   ├── screenshot_service.py   # High-resolution screenshot capture service
│   └── main.py                 # Application orchestrator, HUD, & hotkey controls
├── requirements.txt            # Compatible dependencies
├── README.md                   # Full documentation, math theory, & testing plan
└── .gitignore                  # Ignored environments, cache, & artifacts
```

---

## 2. Windows & VS Code Setup Instructions

Follow these step-by-step commands in **PowerShell** or **Command Prompt** inside Visual Studio Code.

### Step 1: Open Project in VS Code
```powershell
cd c:\Users\Omkar\Desktop\MirrorFit
code .
```

### Step 2: Create and Activate Virtual Environment
```powershell
python -m venv venv
```

Activate the virtual environment:
* In **PowerShell**:
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
* In **Command Prompt (CMD)**:
  ```cmd
  .\venv\Scripts\activate.bat
  ```

### Step 3: Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Run MirrorFit XR
Launch the application:
```powershell
python src/main.py
```
or as a module:
```powershell
python -m src.main
```

#### Optional CLI Customization Flags:
```powershell
# Adjust 3D/2D glasses scale multiplier (default: 2.15)
python src/main.py --scale 2.20

# Adjust vertical offset ratio relative to glasses height (default: 0.05)
python src/main.py --offset 0.06

# Adjust anti-jitter smoothing factor beta in [0.1, 1.0] (default: 0.35)
python src/main.py --smoothing 0.30

# Specify custom 3D model asset
python src/main.py --model3d assets/models_3d/glasses_3d.obj

# Specify custom camera index and resolution
python src/main.py --camera 0 --width 1280 --height 720
```

---

## 3. Keyboard Controls Reference

| Key | Action |
| :--- | :--- |
| **T** | Toggle between **3D AR Model Mode** and **2D Overlay Mode** |
| **W** | Toggle **3D Wireframe Debug Mode** on/off (solid surfaces by default) |
| **X** | Toggle **3D Head Pose Coordinate Axes** on/off (Red=X, Green=Y, Blue=Z) |
| **B** | Toggle **Face Segmentation Mask Overlay** on/off |
| **1** | Select **Default Glasses** (switches to 2D style mode) |
| **2** | Select **Classic Glasses** (switches to 2D style mode) |
| **3** | Select **Round Glasses** (switches to 2D style mode) |
| **4** | Select **Sunglasses** (switches to 2D style mode) |
| **N** / **P** | Cycle to Next / Previous 2D accessory |
| **S** | Capture and save a timestamped screenshot to `output/screenshots/` |
| **R** | Reset temporal tracking smoothing filters (both head pose and 2D overlay) |
| **G** | Toggle glasses visibility on/off |
| **M** | Toggle facial mesh wireframe on/off |
| **A** | Toggle try-on geometric anchor points on/off |
| **H** | Toggle HUD overlay (header, metrics, and badges) |
| **Q** or **ESC** | Safely exit application and release camera |

---

## 4. Technical Analysis & Mathematical Architecture

### 1. 6-DOF Head Pose Estimation via OpenCV solvePnP
Instead of approximating head orientation using a 2D line between eyes, MirrorFit XR computes full 3D rotation and translation vectors $[R \mid \mathbf{t}]$ using perspective-n-point geometry (`cv2.solvePnP`):

#### A. Canonical 3D Anthropometric Model
We define canonical anatomical reference landmarks in millimeters, aligned with OpenCV's camera coordinate frame ($+X$ right, $+Y$ down, $+Z$ forward):
* **Nose Tip:** $(0, 0, 0)$ (Landmark 4)
* **Chin:** $(0, 85, -25)$ (Landmark 152)
* **Left Eye Outer:** $(-65, -40, -40)$ (Landmark 263)
* **Right Eye Outer:** $(65, -40, -40)$ (Landmark 33)
* **Left Mouth Corner:** $(-45, 45, -35)$ (Landmark 291)
* **Right Mouth Corner:** $(45, 45, -35)$ (Landmark 61)
* **Glabella / Nose Bridge:** $(0, -45, -30)$ (Landmark 168)

#### B. Camera Intrinsic Calibration Matrix
We build the pinhole camera matrix $K$ from the active video resolution $(W, H)$:
$$K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$
where $f_x = f_y = W$, $c_x = W / 2$, $c_y = H / 2$, and distortion coefficients $D = [0, 0, 0, 0]^T$.

#### C. Euler Angle Decomposition (Yaw, Pitch, Roll)
`solvePnP` returns rotation vector $\mathbf{r}$ and translation vector $\mathbf{t}$. Using `cv2.Rodrigues`, we recover rotation matrix $R \in \mathbb{R}^{3 \times 3}$. We extract Euler angles:
$$\text{Pitch} = \arctan2(R_{21}, R_{22}) \times \frac{180}{\pi}$$
$$\text{Yaw} = \arctan2(-R_{20}, \sqrt{R_{00}^2 + R_{10}^2}) \times \frac{180}{\pi}$$
$$\text{Roll} = \arctan2(R_{10}, R_{00}) \times \frac{180}{\pi}$$

* **Yaw:** Horizontal head turn (left/right).
* **Pitch:** Vertical nod (up/down).
* **Roll:** Lateral head tilt (clockwise/counter-clockwise).

---

### 2. Intelligent Fitting Engine
The fitting engine is strictly decoupled from the rendering pipeline. Its responsibility is calculating the homogeneous transformation:
$$T_{\text{accessory}\to\text{cam}} = \begin{bmatrix} R_{\text{head}} \cdot S & \mathbf{t}_{\text{accessory}} \\ \mathbf{0} & 1 \end{bmatrix}$$
where:
$$\mathbf{t}_{\text{accessory}} = \mathbf{t}_{\text{head}} + R_{\text{head}} \cdot \mathbf{offset}_{\text{local}}$$
This positions the 3D model directly on the nose saddle while allowing independent scaling ($S$) and fine depth adjustments without modifying camera or pose tracking code.

---

### 3. Projective 3D Mesh Rendering
The 3D renderer parses Wavefront OBJ models (`assets/models_3d/glasses_3d.obj`) and composites them onto the video feed:
1. **Model Coordinates:** Vertices $V \in \mathbb{R}^{N \times 3}$ define front rims, bridge, and 3D temple arms extending back towards the ears ($Z \approx -100\text{ mm}$).
2. **Pose Transformation:** Vertices are transformed to camera space:
   $$\mathbf{v}_{\text{cam}} = R (S \mathbf{v}) + \mathbf{t}_{\text{acc}}$$
3. **Perspective Projection:** Projected into 2D screen coordinates using $K$ (`cv2.projectPoints`).
4. **Painter's Algorithm (Depth Sorting):** Faces are sorted by average camera depth $Z_{\text{cam}}$ from back to front (farthest rendered first, nearest rendered last).
5. **Lambertian Directional Shading:** For key light $\mathbf{L}$:
   $$I = I_{\text{ambient}} + I_{\text{diffuse}} \max(0, \mathbf{n} \cdot \mathbf{L})$$
   Polygons are filled using anti-aliased shading (`cv2.fillPoly` and `cv2.polylines`).

---

### 4. Real-Time Face Segmentation
* **Model:** Google MediaPipe Image Segmenter (`selfie_segmenter.tflite`).
* **Official URL:** `https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite` (~250 KB).
* **Latency:** Benchmarked at ~10–25 ms on CPU.
* **Fallback:** If the model file is absent, segmentation disables cleanly while 3D and 2D accessory rendering continues uninterrupted.

---

## 5. Conceptual Differences in Computer Vision Techniques

| Concept | Dimensionality | Description | Usage in MirrorFit XR |
| :--- | :--- | :--- | :--- |
| **2D Image Overlay** | 2D $(u, v)$ | Flat 4-channel PNG resized and rotated in image space. | Fast fallback mode (`[T]` key) with classic sunglasses styles. |
| **Head Pose Estimation** | 3D $(X, Y, Z, \theta, \phi, \psi)$ | Calculates 6-DOF camera pose ($R, \mathbf{t}$) and Euler angles via `solvePnP`. | Computes Yaw, Pitch, Roll to drive true 3D spatial orientation. |
| **3D Model Rendering** | 3D $\rightarrow$ 2D | Projects a 3D polygonal mesh with depth sorting and directional lighting. | Primary AR mode (`[T]` key) showing 3D perspective and temple arms. |
| **Face Segmentation** | 2D Pixel Mask | Neural network classifying each pixel as foreground person vs background. | Debug visualization (`[B]` key) and background matting foundation. |
| **Depth Occlusion** | 3D Depth Buffer | Per-pixel Z-buffering where foreground face geometry occludes accessory parts. | Documented below as an active research area for Phase 5. |

---

## 6. Known Limitations

1. **Approximate Camera Calibration:** In the absence of a per-camera checkerboard calibration file, the focal length is estimated as $f = W$ (approx. 65° field of view). Minor radial distortion from wide-angle lenses may introduce slight perspective deviations at extreme frame borders.
2. **Ear Occlusion:** Currently, 3D temple arms render using Painter's depth sorting. Behind-the-ear occlusion is approximated; full physical ear occlusion requires depth-map volumetric compositing (scheduled for Phase 5).

---

## 7. Manual Testing Checklist (Tests A through K)

| Test | Action | Expected Result | Troubleshooting / Failure Check |
| :--- | :--- | :--- | :--- |
| **A. Upright View** | Look straight at the camera. | Yaw $\approx 0^\circ$, Pitch $\approx -6^\circ$, Roll $\approx 0^\circ$; 3D glasses sit level across the nose saddle. | Check that `dx = eye_right - eye_left > 0`. |
| **B. Head Tilt** | Tilt head left and right. | 3D glasses rotate smoothly matching Roll angle. | Verify rotation sign in `HeadPoseEstimator`. |
| **C. Turn Sideways (Yaw)** | Turn head 30° to the left and right. | The 3D model exhibits genuine perspective foreshortening: the far rim shrinks and the near temple arm extends back. | Press `X` to inspect 3D pose axes. |
| **D. Move Close / Far** | Step closer and farther from camera. | Fitting engine scales the 3D model dynamically matching face size. | Verify `compute_fitting` scale calculation. |
| **E. Enable Segmentation** | Press `B`. | A green semi-transparent segmentation mask highlights the user's face and body; segmentation latency displays on the HUD. | Ensure `models/selfie_segmenter.tflite` is present. |
| **F. Disable Segmentation** | Press `B` again. | Segmentation mask disappears, latency reduces, application runs smoothly. | Verify `show_segmentation = False` skips inference. |
| **G. Leave Camera Frame** | Step out of camera view. | Glasses disappear immediately; status badge turns red (`NO FACE DETECTED`). When returning, tracking recovers without jumping. | Verify `reset()` clears smoothing filters. |
| **H. Enable Pose Axes** | Press `X`. | Red (X), Green (Y), and Blue (Z) 3D coordinate axes project directly from the tip of the nose. | Verify `draw_axes` in `HeadPoseEstimator`. |
| **I. Performance Telemetry** | Inspect top-right HUD banner. | FPS ($\sim 30\text{--}60$), total frame latency, and segmentation latency update in real time. | Verify `PerformanceMonitor` records. |
| **J. Run Without 3D Model** | Rename `glasses_3d.obj`. | App displays warning on startup and falls back gracefully to 2D overlay mode without crashing. | Verify `Renderer3D.is_available = False` fallback. |
| **K. Exit Application** | Press `Q` or `ESC`. | OpenCV window closes, camera and segmentation resources release cleanly, application shuts down. | Verify `finally:` cleanup block in `main.py`. |
