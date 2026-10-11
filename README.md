# MirrorFit | Real-Time Virtual Try-On Studio (Phase 1 & Phase 2)

MirrorFit is a real-time computer vision application that tracks facial landmarks and overlays virtual accessories using **Python**, **OpenCV**, **MediaPipe**, and **NumPy**.

* **Phase 1:** High-speed webcam capture, 478-point 3D facial landmark detection, and HUD telemetry.
* **Phase 2:** Real-time **Virtual Glasses Overlay** with proportional scaling, head tilt rotation, boundary clipping, and vectorized alpha compositing.

---

## 1. Project Structure

```
MirrorFit/
├── assets/
│   └── glasses/
│       └── default_glasses.png # 4-channel RGBA transparent glasses graphic
├── models/
│   └── face_landmarker.task    # Google MediaPipe Face Landmarker model bundle
├── src/
│   ├── __init__.py             # Package exports (Camera, FaceTracker, GlassesOverlay)
│   ├── camera.py               # Camera capture, DirectShow backend, & mirroring
│   ├── face_tracker.py         # 478-point landmark tracker & geometric anchors
│   ├── glasses_overlay.py      # Scaling, unclipped rotation, & alpha blending
│   └── main.py                 # Application loop, HUD overlay, & keyboard controls
├── requirements.txt            # Compatible Python dependencies
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
Create an isolated Python virtual environment named `venv`:
```powershell
python -m venv venv
```

Activate the virtual environment:
* In **PowerShell**:
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
  *(If execution policy errors occur, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and rerun the activate command).*
* In **Command Prompt (CMD)**:
  ```cmd
  .\venv\Scripts\activate.bat
  ```

### Step 3: Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify Assets
* **MediaPipe Model:** Ensure `models/face_landmarker.task` is present (~3.76 MB).
* **Glasses Graphic:** Ensure `assets/glasses/default_glasses.png` is present (4-channel RGBA with alpha transparency).

### Step 5: Run the Application
Launch MirrorFit Phase 2:
```powershell
python src/main.py
```

#### Optional CLI Customization Flags:
```powershell
# Adjust glasses scale multiplier (default: 2.15)
python src/main.py --scale 2.25

# Adjust vertical offset ratio relative to glasses height (default: 0.05)
python src/main.py --offset 0.08

# Specify custom glasses PNG asset
python src/main.py --glasses assets/glasses/default_glasses.png

# Specify camera index and resolution
python src/main.py --camera 0 --width 1280 --height 720
```

---

## 3. Keyboard Controls

| Key | Action |
| :--- | :--- |
| **G** | Toggle Virtual Glasses Overlay on/off |
| **M** | Toggle facial mesh wireframe on/off |
| **A** | Toggle try-on geometric anchor points on/off |
| **H** | Toggle HUD overlay (header, metrics, and badges) |
| **Q** or **ESC** | Safely exit application and release camera |

---

## 4. Code Architecture & Modular Design

### `src/camera.py` — Hardware Video Capture
* **DirectShow Acceleration:** Uses `cv2.CAP_DSHOW` on Windows to eliminate multi-second initialization lag.
* **Horizontal Mirroring:** Automatically executes `cv2.flip(frame, 1)`. In a virtual try-on mirror, moving your head right must move right on-screen.
* **Safe Resource Lifecycle:** Implements context management and `release()` to guarantee hardware handles are freed under all exit paths.

### `src/face_tracker.py` — Face Mesh & Coordinate Engine
* **478-Point Tracking:** Leverages Google MediaPipe `FaceLandmarker` with XNNPACK CPU acceleration.
* **Coordinate Conversion:** Transforms normalized landmarks $(x, y \in [0.0, 1.0])$ into image pixel coordinates $(px_x, px_y)$.
* **Try-On Anchors:**
  * Left & Right Eye / Iris centers (`landmarks 468 & 473`).
  * Nose saddle point / bridge (`landmark 168`).
  * Temple width span (`landmarks 234 & 454`).

### `src/glasses_overlay.py` — Augmented Reality Overlay Engine
* **Asset Validation:** Ensures the image is a valid 4-channel BGRA PNG with a non-trivial alpha channel.
* **Dynamic Proportional Scaling:** Calculates glasses width from Interpupillary Distance (IPD) multiplied by `scale_factor` (default `2.15`).
* **Unclipped Bounding-Box Rotation:** Computes new expanded bounding dimensions $(W_{\text{new}}, H_{\text{new}})$ during affine rotation so glasses temples are never truncated.
* **Viewport Clipping:** Slices intersecting regions between the rotated glasses and the video viewport $[0, W] \times [0, H]$, preventing crashes when users lean near edges.
* **Vectorized Alpha Blending:** Uses NumPy broadcasting to blend RGB channels weighted by normalized alpha channel values.
* **Jitter Reduction:** Applies Exponential Moving Average (EMA) filtering to position, scale, and angle.

### `src/main.py` — Application Orchestrator
* **Inference & Render Pipeline:** Captures mirrored frames $\rightarrow$ runs face tracking $\rightarrow$ renders glasses overlay $\rightarrow$ updates HUD $\rightarrow$ handles keyboard input.
* **No-Face Handling:** When a face is absent, glasses are immediately hidden and the filter is reset so no residual position lag occurs when a face re-enters.

---

## 5. Mathematical & Computer Vision Concepts Explained

### 1. Determining Placement Center
To place the glasses naturally across the nose saddle rather than floating artificially:
$$X_{\text{mid}} = \frac{X_{\text{left\_eye}} + X_{\text{right\_eye}}}{2}, \quad Y_{\text{mid}} = \frac{Y_{\text{left\_eye}} + Y_{\text{right\_eye}}}{2}$$
We blend the eye midpoint with the physical nose bridge (landmark 168) and add a proportional vertical offset:
$$C_X = 0.7 \times X_{\text{mid}} + 0.3 \times X_{\text{bridge}}$$
$$C_Y = (0.7 \times Y_{\text{mid}} + 0.3 \times Y_{\text{bridge}}) + (H_{\text{glasses}} \times \text{offset\_ratio})$$

### 2. Proportional Scaling Based on Eye Distance
As a user moves closer or farther from the camera, the face size changes. We compute the Interpupillary Distance (IPD) using Euclidean distance:
$$\text{IPD} = \sqrt{(X_2 - X_1)^2 + (Y_2 - Y_1)^2}$$
$$\text{Width}_{\text{glasses}} = \text{IPD} \times \text{scale\_factor}$$
$$\text{Height}_{\text{glasses}} = \text{Width}_{\text{glasses}} \times \left(\frac{H_{\text{orig}}}{W_{\text{orig}}}\right)$$
Because human temple width is approximately $2.1\times$ to $2.2\times$ the interpupillary distance, this yields anatomically accurate sizing across all camera distances.

### 3. Eye Line Rotation & OpenCV Sign Convention
Let $(X_1, Y_1)$ be the viewer's left eye and $(X_2, Y_2)$ be the viewer's right eye on screen ($X_1 < X_2$). The tilt angle of the eye line relative to horizontal is:
$$\theta = \arctan2(Y_2 - Y_1, X_2 - X_1) \times \frac{180}{\pi}$$
* In screen coordinates, $+Y$ points downward. If the user tilts their head clockwise, the right eye drops lower ($Y_2 > Y_1$), producing $\theta > 0$.
* OpenCV's `cv2.getRotationMatrix2D` rotates **counter-clockwise** for positive angles.
* Therefore, to rotate the horizontal glasses graphic clockwise to match the head tilt, the rotation angle passed to OpenCV is:
$$\text{Angle}_{\text{OpenCV}} = -\theta$$

### 4. Unclipped Affine Rotation (Expanded Bounding Box)
Rotating an image of width $w$ and height $h$ by angle $\phi$ inside its original boundary cuts off the corners. To preserve the entire frame and temples:
$$W_{\text{new}} = \lceil |h \sin(\phi)| + |w \cos(\phi)| \rceil$$
$$H_{\text{new}} = \lceil |h \cos(\phi)| + |w \sin(\phi)| \rceil$$
The transformation matrix translation terms are shifted to the expanded center:
$$M[0, 2] += \frac{W_{\text{new}}}{2} - \frac{w}{2}, \quad M[1, 2] += \frac{H_{\text{new}}}{2} - \frac{h}{2}$$
and warped using `cv2.warpAffine(..., (W_{\text{new}}, H_{\text{new}}), borderValue=(0, 0, 0, 0))`.

### 5. Alpha Compositing (Transparency Blending)
For overlapping pixels between the 4-channel BGRA glasses and the 3-channel BGR webcam frame:
$$\alpha = \frac{\text{Overlay}_{\text{alpha}}}{255.0} \in [0.0, 1.0]$$
$$\text{Pixel}_{\text{out}} = \alpha \times \text{Overlay}_{\text{BGR}} + (1.0 - \alpha) \times \text{Frame}_{\text{BGR}}$$
Transparent areas $(\alpha = 0)$ leave the webcam frame completely untouched, semi-transparent areas $(\alpha \approx 0.4)$ show tinted sunglasses lenses with specular reflection, and opaque areas $(\alpha = 1.0)$ render the solid glasses frames.

### 6. Viewport Boundary Clipping
When the user moves close to the camera borders, the glasses bounding box may exceed $[0, W] \times [0, H]$. We clip the target coordinates:
$$X_{1,\text{clip}} = \max(0, X_1), \quad Y_{1,\text{clip}} = \max(0, Y_1)$$
$$X_{2,\text{clip}} = \min(W_{\text{frame}}, X_2), \quad Y_{2,\text{clip}} = \min(H_{\text{frame}}, Y_2)$$
and slice the exact corresponding sub-region of the glasses graphic. If $X_{1,\text{clip}} \ge X_{2,\text{clip}}$ or $Y_{1,\text{clip}} \ge Y_{2,\text{clip}}$, rendering is cleanly skipped without memory access errors.

### 7. Temporal Smoothing (Jitter Suppression)
Micro-movements and sensor noise cause raw landmark coordinates to oscillate by 1–2 pixels between frames. We apply an Exponential Moving Average (EMA) filter:
$$S_t = \beta \cdot M_t + (1 - \beta) \cdot S_{t-1}$$
where $\beta = 0.35$. This eliminates jitter while maintaining responsive tracking during rapid head motion.

---

## 6. Manual Testing Plan (Tests A through J)

| Test | Objective & Action | Expected Result | What to Check If Failed |
| :--- | :--- | :--- | :--- |
| **Test A** | Start app via `python src/main.py`. | App launches immediately without errors; logs indicate glasses asset loaded. | Check `assets/glasses/default_glasses.png` exists and is a 4-channel PNG. |
| **Test B** | Position face in front of webcam. | Glasses appear centered directly over both eyes and resting on the nose bridge. | Check `--scale` and `--offset` flags or press `G` to ensure glasses are toggled ON. |
| **Test C** | Step closer and farther from the webcam. | Glasses scale up smoothly when moving closer and scale down when moving farther away. | Ensure `eye_distance * scale_factor` is calculating properly in `compute_geometry`. |
| **Test D** | Move face left and right across the frame. | Glasses follow the face seamlessly without falling behind or detaching. | Verify webcam mirroring is handled consistently before landmark detection. |
| **Test E** | Tilt head to the left and to the right. | Glasses rotate smoothly to match the eye line tilt in both directions. | Verify the rotation sign (`-angle_deg`) is applied in `compute_geometry`. |
| **Test F** | Move head towards the extreme edges of the camera frame. | Glasses clip cleanly against frame borders without crashes or index errors. | Verify `x1_clip` and `y1_clip` clamping in `blend_overlay`. |
| **Test G** | Step completely out of camera view. | Glasses disappear immediately; status badge turns amber/red (`NO FACE DETECTED`). | Verify `is_detected` check in `main.py` hides glasses when no face is detected. |
| **Test H** | Re-enter camera view. | Glasses reappear instantly in the correct position without lag or jumping. | Verify `reset_filter()` is called when face tracking is lost. |
| **Test I** | Inspect transparent lens areas and background. | Webcam background is visible around the frames and through tinted lens areas. | Verify PNG alpha channel is decoded via `cv2.IMREAD_UNCHANGED`. |
| **Test J** | Press **Q** or **ESC**. | Window closes immediately, camera is released, and terminal exits cleanly. | Verify `finally:` block in `main.py` calls `camera.release()`. |

---

## 7. Troubleshooting Guide

| Symptom | Probable Cause | Recommended Resolution |
| :--- | :--- | :--- |
| **Glasses too high or too low** | Facial geometry differences | Run with custom vertical offset: `python src/main.py --offset 0.08` (lower) or `--offset 0.0` (higher). |
| **Glasses too large or too small** | Proportions relative to camera FOV | Run with custom scale factor: `python src/main.py --scale 2.0` (smaller) or `--scale 2.3` (larger). |
| **Glasses rotate in reverse** | Inverted trigonometric angle sign | Ensure `opencv_rot_angle = -angle_deg` is used in `src/glasses_overlay.py`. |
| **Opaque black/white box around glasses** | Image saved without alpha channel | Ensure image is saved with 4 channels (BGRA). Run the validation in `GlassesOverlay._load_glasses_asset`. |
| **Temples clipped during head tilt** | Standard rotation cuts off bounding box | Verify `transform_glasses` computes `new_w` and `new_h` from the rotation matrix bounding expansion. |
| **Glasses jittering during slight movement** | Landmark detection noise | Adjust `smoothing_factor` (e.g. `0.25` for smoother, `0.5` for faster reaction). |
| **Low FPS (< 20 FPS)** | Dense wireframe mesh rendering overhead | Press **M** to toggle off the 478-point mesh wireframe. |
