# MirrorFit | Real-Time Virtual Try-On Studio (Phases 1, 2 & 3)

MirrorFit is a real-time computer vision application that tracks facial landmarks and overlays virtual accessories using **Python**, **OpenCV**, **MediaPipe**, and **NumPy**.

* **Phase 1:** High-speed webcam capture, 478-point 3D facial landmark detection, and HUD telemetry.
* **Phase 2:** Real-time **Virtual Glasses Overlay** with proportional scaling, head tilt rotation, boundary clipping, and vectorized alpha compositing.
* **Phase 3:** **Advanced Head Tracking & Accessory Selection**:
  * Corrected head tilt rotation geometry (upright head strictly $\approx 0.0^\circ$, no $\pm 180^\circ$ flips).
  * Anti-jitter Exponential Moving Average (EMA) smoothing for position, width, and rotation.
  * Multi-accessory catalog supporting transparent PNG glasses and sunglasses.
  * Direct numeric key selection (`1`–`4`) and sequential cycling (`N`/`P`).
  * On-demand tracking filter reset (`R`).
  * High-resolution timestamped screenshot capture (`S`) to `output/screenshots/`.

---

## 1. Project Structure

```
MirrorFit/
├── assets/
│   └── glasses/
│       ├── default_glasses.png # Standard dark-rim glasses
│       ├── classic_glasses.png # Rectangular smart browline frames
│       ├── round_glasses.png   # Retro circular spectacles
│       └── sunglasses.png      # Dark polarized Aviator sunglasses
├── models/
│   └── face_landmarker.task    # Google MediaPipe Face Landmarker model bundle
├── output/
│   └── screenshots/            # Timestamped screenshots directory
├── src/
│   ├── __init__.py             # Package exports
│   ├── accessory_manager.py    # Multi-accessory catalog & runtime switcher
│   ├── camera.py               # Camera capture, DirectShow backend, & mirroring
│   ├── face_tracker.py         # 478-point landmark tracker & geometric anchors
│   ├── glasses_overlay.py      # Scaling, unclipped rotation, & alpha blending
│   ├── main.py                 # Application loop, HUD overlay, & keyboard controls
│   └── screenshot_service.py   # High-resolution screenshot capture service
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

### Step 4: Run MirrorFit Phase 3
Launch the application:
```powershell
python src/main.py
```

#### Optional Customization Flags:
```powershell
# Adjust glasses scale multiplier (default: 2.15)
python src/main.py --scale 2.20

# Adjust vertical offset ratio relative to glasses height (default: 0.05)
python src/main.py --offset 0.06

# Adjust anti-jitter smoothing factor beta in [0.1, 1.0] (default: 0.35)
python src/main.py --smoothing 0.30

# Specify custom camera index and resolution
python src/main.py --camera 0 --width 1280 --height 720
```

---

## 3. Keyboard Controls

| Key | Action |
| :--- | :--- |
| **1** | Select **Default Glasses** |
| **2** | Select **Classic Glasses** (Rectangular smart frames) |
| **3** | Select **Round Glasses** (Retro circular spectacles) |
| **4** | Select **Sunglasses** (Dark Aviator frames) |
| **N** | Cycle to the **Next** available accessory |
| **P** | Cycle to the **Previous** available accessory |
| **S** | Capture and save a timestamped screenshot to `output/screenshots/` |
| **R** | Reset temporal tracking smoothing filter |
| **G** | Toggle Virtual Glasses Overlay on/off |
| **M** | Toggle facial mesh wireframe on/off |
| **A** | Toggle try-on geometric anchor markers on/off |
| **H** | Toggle HUD overlay (header, metrics, and badges) |
| **Q** or **ESC** | Safely exit application and release camera |

---

## 4. Technical Analysis & Mathematical Foundations

### 1. The Head Rotation Fix (Why -177.8° Occurred and How It Was Solved)
* **The Root Cause:** In a mirrored video feed (`cv2.flip(frame, 1)`), the subject's anatomical left eye (MediaPipe landmark 468) appears on the viewer's left side of the screen ($X \approx 500$), while the subject's anatomical right eye (landmark 473) appears on the viewer's right side ($X \approx 700$).
* When computing $dx = X_{\text{left\_eye}} - X_{\text{right\_eye}}$, the result was $500 - 700 = -200$ (a **negative** number).
* In trigonometry, $\arctan2(dy, dx)$ with a negative $dx$ and near-zero $dy$ points backwards along the negative X-axis ($180^\circ$ or $-180^\circ$). If $dy$ fluctuated slightly negative (e.g. $-7\text{px}$), the angle calculated to:
  $$\arctan2(-7, -200) \times \frac{180}{\pi} = -177.99^\circ$$
  Furthermore, minor vertical micro-movements caused the angle to violently jump between $+178^\circ$ and $-178^\circ$.
* **The Solution:** We order eye positions in viewer screen space before computing the displacement vector:
  $$P_L = \text{eye with smaller } X \quad (\text{screen left})$$
  $$P_R = \text{eye with larger } X \quad (\text{screen right})$$
  $$\Delta X = X_R - X_L > 0 \quad (\text{strictly positive!})$$
  $$\Delta Y = Y_R - Y_L$$
  Because $\Delta X > 0$, $\arctan2(\Delta Y, \Delta X)$ is mathematically guaranteed to fall strictly within:
  $$\theta \in (-90^\circ, +90^\circ)$$
  * Upright, level head $\implies \Delta Y \approx 0 \implies \mathbf{\theta \approx 0.0^\circ}$.
  * Clockwise tilt (right eye lower) $\implies \Delta Y > 0 \implies \mathbf{\theta > 0^\circ}$.
  * Counter-clockwise tilt (right eye higher) $\implies \Delta Y < 0 \implies \mathbf{\theta < 0^\circ}$.

### 2. OpenCV Rotation Angle Convention
In standard 2D image coordinates, the $+Y$ axis points **downward**.
* When a user tilts clockwise, the right eye drops downward ($\Delta Y > 0 \implies \theta > 0$).
* In OpenCV, `cv2.getRotationMatrix2D(center, angle, 1.0)` rotates **counter-clockwise** for positive angle arguments.
* Therefore, to rotate horizontal glasses clockwise to match the head tilt, the angle passed to OpenCV is:
  $$\text{Angle}_{\text{OpenCV}} = -\theta$$

### 3. Anti-Jitter Temporal Smoothing (Exponential Moving Average)
Sensory camera noise causes landmark pixel coordinates to fluctuate by 1–2 pixels every frame. We apply an Exponential Moving Average (EMA) filter:
$$S_t = \beta \cdot M_t + (1 - \beta) \cdot S_{t-1}$$
where $\beta = 0.35$ (configurable via `--smoothing`).
* When face tracking is lost, `reset_filter()` clears $S_{t-1}$ so stale coordinates are never used.
* Pressing `R` manually resets the filter state on demand.

### 4. Vectorized Alpha Compositing & Viewport Boundary Clipping
For 4-channel BGRA overlays on 3-channel BGR frames:
$$\alpha = \frac{\text{Overlay}_{\text{alpha}}}{255.0} \in [0.0, 1.0]$$
$$\text{Pixel}_{\text{out}} = \alpha \times \text{Overlay}_{\text{BGR}} + (1.0 - \alpha) \times \text{Frame}_{\text{BGR}}$$
When users move close to the edge of the webcam frame, target coordinates are clamped to $[0, W_{\text{frame}}] \times [0, H_{\text{frame}}]$, and the overlapping region of the accessory is sliced. This prevents array shape mismatches and OpenCV indexing crashes.

---

## 5. Manual Testing Checklist (Tests A through L)

| Test | Action | Expected Behavior | Failure Check |
| :--- | :--- | :--- | :--- |
| **A. Upright View** | Look straight at camera. | Tilt angle displays near $0.0^\circ$ ($\pm 1.5^\circ$), glasses sit level on nose bridge. | Check that `dx = eye_right - eye_left > 0`. |
| **B. Move Left / Right** | Move head horizontally across frame. | Glasses follow smoothly without lag or spatial offset. | Verify webcam mirroring happens before landmark inference. |
| **C. Move Up / Down** | Move head vertically up and down. | Glasses maintain alignment on the nose saddle. | Verify nose bridge anchor blending. |
| **D. Move Closer / Farther** | Step close to and away from the camera. | Glasses smoothly scale up and down proportional to eye distance. | Verify `target_width = IPD * scale_factor`. |
| **E. Tilt Head Left** | Tilt head counter-clockwise. | Tilt displays negative degrees (e.g. $-12^\circ$), glasses rotate to match eye line. | Check OpenCV rotation sign (`-angle`). |
| **F. Tilt Head Right** | Tilt head clockwise. | Tilt displays positive degrees (e.g. $+14^\circ$), glasses rotate to match eye line. | Check OpenCV rotation sign (`-angle`). |
| **G. Turn Head Sideways** | Turn head moderately left and right. | Glasses remain centered over visible eye span. | Ensure minimum scale boundary clamp ($W \ge 20$). |
| **H. Leave Camera View** | Step completely out of camera frame. | Glasses disappear instantly; status displays `NO FACE DETECTED`. | Verify `is_detected` skips overlay when no face is found. |
| **I. Return to View** | Step back into camera frame. | Glasses reappear at the new position without jumping or lagging. | Verify `reset_filter()` was called during absence. |
| **J. Switch Accessories** | Press `1`, `2`, `3`, `4`, and `N` / `P`. | Active accessory changes immediately; name updates on HUD. | Check that all 4 PNG assets exist in `assets/glasses/`. |
| **K. Save Screenshot** | Press `S`. | Screenshot saved to `output/screenshots/`; on-screen confirmation banner flashes. | Verify write permissions in `output/screenshots/`. |
| **L. Exit Cleanly** | Press `Q` or `ESC`. | Window closes immediately, camera hardware handle is released, clean exit. | Verify `finally:` cleanup block in `main.py`. |

---

## 6. Troubleshooting Guide

| Symptom | Probable Cause | Recommended Resolution |
| :--- | :--- | :--- |
| **Tilt angle reads near $\pm 180^\circ$** | Eye coordinate order inverted | Ensure `eye_left_screen` and `eye_right_screen` are sorted by X coordinate so $\Delta X > 0$ strictly. |
| **Accessory PNG not found** | File moved or renamed | Verify `assets/glasses/` contains `default_glasses.png`, `classic_glasses.png`, `round_glasses.png`, and `sunglasses.png`. |
| **Glasses rotate in reverse** | Inverted rotation sign | Ensure OpenCV rotation angle is set to `-angle_deg`. |
| **Glasses jitter during slight motion** | Landmark detection noise | Lower smoothing factor: `python src/main.py --smoothing 0.25` for smoother response. |
| **Screenshot directory error** | Permissions issue | Run VS Code terminal as standard user; directory `output/screenshots/` is created automatically. |
| **Low FPS (< 20 FPS)** | Dense mesh wireframe rendering overhead | Press **M** to toggle off the 478-point wireframe mesh. |
