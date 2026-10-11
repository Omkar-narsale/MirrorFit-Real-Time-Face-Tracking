# MirrorFit | Real-Time Virtual Try-On Studio (Phase 1)

MirrorFit is a real-time computer vision application that tracks facial landmarks using **Python**, **OpenCV**, **MediaPipe**, and **NumPy**.

Phase 1 establishes the foundational computer vision pipeline: high-speed webcam frame capture, 478-point 3D facial landmark detection, geometric anchor extraction for virtual glasses positioning, and a clean HUD interface.

---

## 1. Project Structure

```
MirrorFit/
├── assets/
│   └── glasses/             # Reserved for Phase 2 transparent glasses PNGs
├── models/
│   └── face_landmarker.task # Google MediaPipe Face Landmarker model bundle
├── src/
│   ├── __init__.py          # Package initialization
│   ├── camera.py            # OpenCV camera capture & mirroring manager
│   ├── face_tracker.py      # MediaPipe landmark detector & try-on geometry
│   └── main.py              # Application loop, HUD overlay, & keyboard controls
├── requirements.txt         # Compatible Python dependencies
├── README.md                # Comprehensive project documentation
└── .gitignore               # Ignored environments, cache, & artifacts
```

---

## 2. Windows & VS Code Setup Instructions

Follow these step-by-step commands in **PowerShell** or **Command Prompt** inside Visual Studio Code.

### Step 1: Open Project in VS Code
Open your terminal and navigate to the project directory:
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
  *(If you encounter an execution policy error in PowerShell, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and then run the activation script again).*
* In **Command Prompt (CMD)**:
  ```cmd
  .\venv\Scripts\activate.bat
  ```

### Step 3: Install Dependencies
Upgrade pip and install the required dependencies:
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify MediaPipe Model Asset
The modern MediaPipe Tasks API requires Google's official `face_landmarker.task` model bundle.
The model is stored inside `models/face_landmarker.task`. If you need to download it manually:
```powershell
# Using PowerShell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" -OutFile "models\face_landmarker.task"
```
Or using `curl`:
```bash
curl -o models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

### Step 5: Run the Application
Launch MirrorFit Phase 1:
```powershell
python src/main.py
```

Optional CLI flags:
```powershell
# Specify camera index (default: 0)
python src/main.py --camera 0

# Specify custom resolution (default: 1280x720)
python src/main.py --width 1920 --height 1080
```

---

## 3. Keyboard Controls

| Key | Action |
| :--- | :--- |
| **Q** or **ESC** | Safely exit application and release camera |
| **M** | Toggle fine facial mesh wireframe on/off |
| **A** | Toggle virtual try-on anchor markers (eye line, pupils, nose bridge) |
| **H** | Toggle HUD overlay (header, metrics, and badges) |

---

## 4. MediaPipe Model Requirements

* **Model Name:** Google MediaPipe Face Landmarker (Float16 task bundle)
* **Official URL:** `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`
* **Local Destination:** `models/face_landmarker.task`
* **File Verification:** `src/face_tracker.py` verifies the model file exists and is non-empty before instantiating the detector. If missing, it halts with an explicit error detailing the download link and placement steps.

---

## 5. Code Architecture (File-by-File Explanation)

### `src/camera.py` — Hardware Abstraction Layer
* **`Camera` class:** Encapsulates `cv2.VideoCapture`.
* **DirectShow Initialization:** On Windows, `cv2.CAP_DSHOW` opens webcams immediately without multi-second DirectShow/MSMF negotiation delays. If unsupported, it gracefully falls back to `cv2.CAP_ANY`.
* **Mirroring:** Uses `cv2.flip(frame, 1)` to flip horizontally. In a virtual try-on mirror, moving your head to the right must move right on-screen.
* **Failure Detection:** `read_frame()` returns a tuple `(success, frame)`. If frames drop or the camera disconnects, the pipeline detects it without crashing.
* **Resource Release:** Implements `release()` and the context manager protocol (`__enter__` and `__exit__`) to guarantee hardware handles are freed even if an exception occurs.

### `src/face_tracker.py` — Computer Vision & Geometry Engine
* **`FaceTracker` class:** Manages `mediapipe.tasks.python.vision.FaceLandmarker`.
* **Preprocessing:** Converts OpenCV frames from BGR format to RGB format (`cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`), wraps them into `mp.Image`, and feeds them to the detector.
* **Coordinate Mapping:** Transforms normalized landmarks $(x, y \in [0.0, 1.0])$ into image pixel coordinates:
  $$\text{pixel\_x} = \text{int}(x \times \text{width}), \quad \text{pixel\_y} = \text{int}(y \times \text{height})$$
* **Anchor Extraction (`compute_tryon_anchors`):**
  * Computes **Interpupillary Distance (IPD)** in pixels via Euclidean distance between iris centers ($d = \sqrt{\Delta x^2 + \Delta y^2}$).
  * Computes **Face Roll / Tilt Angle** in degrees ($\theta = \arctan2(\Delta y, \Delta x) \times \frac{180}{\pi}$) between eye centers.
  * Identifies the **Nose Saddle Point** (landmark 168) where glasses rests.
  * Identifies the **Temple Hinges** (landmarks 234 and 454) for glasses frame width.
* **Visualization:** Draws the fine 478-landmark tessellation, facial contours (eyes, lips, brows, oval), irises, and high-visibility try-on anchor markers.

### `src/main.py` — Orchestration & User Interface
* **`MirrorFitApp` class:** Orchestrates the capture $\rightarrow$ tracking $\rightarrow$ rendering loop.
* **HUD Overlay:** Renders a semi-transparent HUD header and footer using `cv2.addWeighted`, title text, dynamic face status badges, real-time FPS, and live geometric telemetry.
* **FPS Calculation:** Calculates instantaneous FPS via high-resolution `time.perf_counter()` and applies an Exponential Moving Average (EMA) to keep the display readable and jitter-free.
* **Clean Shutdown:** Catches `Q`, `ESC`, and `SIGINT` (Ctrl+C), ensuring OpenCV windows close and camera resources are released.

---

## 6. Learning Objectives (Beginner-Friendly Computer Vision)

### 1. How OpenCV Captures Webcam Frames
OpenCV uses `cv2.VideoCapture` to communicate with the OS camera driver (DirectShow or Media Foundation on Windows). The camera hardware continuously digitizes light sensor data into raw frames. When you call `cap.read()`, OpenCV grabs the latest frame buffer and decodes it into a NumPy array of shape `(height, width, 3)`.

### 2. What MediaPipe Does
MediaPipe is Google's high-performance machine learning framework for perceptual computing. For face tracking, it runs a two-step deep learning pipeline:
1. **Face Detector:** Detects the bounding box of the face in the image.
2. **Face Landmarker:** Crops and normalizes the face region, then runs a lightweight convolutional neural network that estimates 478 3D landmark points and surface blendshapes in just 1–3 milliseconds on CPU.

### 3. What Facial Landmarks Represent
Facial landmarks are specific anatomical points on the human face:
* Point 4 is the tip of the nose.
* Point 168 is the glabella/bridge of the nose between the eyes.
* Points 33 and 263 are the outer corners of the eyes.
* Points 468 and 473 are the exact centers of the left and right irises.
Together, these 478 points define a 3D surface mesh topology covering the face oval, lips, eyes, eyebrows, and irises.

### 4. How Normalized Coordinates Map to Pixel Coordinates
Neural networks perform best when input coordinates are normalized between $0.0$ and $1.0$, regardless of camera resolution.
* If a camera feeds an image of size $1280 \times 720$, and a landmark has normalized coordinates $x = 0.5$ and $y = 0.4$:
  $$\text{pixel\_x} = 0.5 \times 1280 = 640\text{ px}$$
  $$\text{pixel\_y} = 0.4 \times 720 = 288\text{ px}$$
This resolution-independent representation allows the same model to work across webcams of any resolution.

### 5. How NumPy Helps with Image Processing
In OpenCV, every image frame is represented as a NumPy array of unsigned 8-bit integers (`np.uint8`) with shape `(height, width, channels)`. NumPy enables:
* High-speed vectorized math (e.g. Euclidean distances, trigonometric rotations, matrix scaling).
* Instant slicing and masking without slow Python `for` loops.
* Efficient color space transformations and memory sharing.

### 6. How FPS is Calculated
Frames Per Second (FPS) measures performance. We measure the elapsed time $\Delta t$ between consecutive frames using `time.perf_counter()`:
$$\text{Instantaneous FPS} = \frac{1}{\Delta t}$$
To prevent the number on screen from flickering wildly, we apply an Exponential Moving Average (EMA):
$$\text{FPS}_{\text{smoothed}} = \alpha \times \text{FPS}_{\text{instant}} + (1 - \alpha) \times \text{FPS}_{\text{prev}}$$
where $\alpha = 0.1$.

### 7. How the Application Handles Missing Faces
When a user steps away or covers the camera, MediaPipe returns an empty landmark list `[]`. The tracker checks this condition and returns `is_detected = False`. The application immediately updates the HUD badge to `"NO FACE DETECTED"`, skips landmark rendering, and waits for a face to re-enter without crashing or throwing null-reference exceptions.

### 8. How Webcam Resources Are Released
Webcam devices are shared hardware resources locked by the operating system. If an application exits abruptly without releasing the camera, the OS lock remains active, preventing other apps from using the camera. MirrorFit uses a `finally:` block and `camera.release()`, ensuring `cv2.VideoCapture` is closed and hardware locks are freed under all exit conditions (Q key, ESC key, window close, or Ctrl+C).

### 9. How Coordinates Will Position Glasses in Phase 2
In Phase 2, virtual glasses (transparent PNGs) will be positioned using the anchors calculated in Phase 1:
1. **Scale:** The width of the glasses is computed from the temple-to-temple span (`glasses_width_px`) or scaled Interpupillary Distance (`IPD * 2.2`).
2. **Anchor Position:** The center bridge of the glasses frame is translated to the nose bridge anchor (`landmark 168`).
3. **Rotation:** The glasses image is rotated around its center using OpenCV's affine transform (`cv2.getRotationMatrix2D`) matching the head roll angle (`face_tilt_deg`).
4. **Alpha Blending:** The 4-channel BGRA glasses image is seamlessly overlaid using pixel-wise alpha compositing.

---

## 7. Troubleshooting Guide

### 1. Webcam Not Opening
* **Symptom:** `[Camera Error] Failed to open webcam (device index 0)`.
* **Fix:**
  1. Verify the webcam is connected and the lens cover is open.
  2. If using an external USB camera or multiple cameras, pass `--camera 1` or `--camera 2`.
  3. Close other applications (Zoom, Microsoft Teams, Skype, Windows Camera app) that might have locked the webcam.

### 2. Windows Camera Permission Issues
* **Symptom:** Webcam opens but returns black frames or fails immediately.
* **Fix:**
  1. Open Windows **Settings** $\rightarrow$ **Privacy & Security** $\rightarrow$ **Camera**.
  2. Ensure **"Camera access"** is turned **On**.
  3. Ensure **"Let desktop apps access your camera"** is turned **On**.

### 3. MediaPipe Installation or Import Errors
* **Symptom:** `ModuleNotFoundError: No module named 'mediapipe'`.
* **Fix:**
  1. Ensure your virtual environment is activated (`(venv)` appears in terminal prompt).
  2. Run `pip install -r requirements.txt`.
  3. Note: Python 3.10 through 3.13 are supported.

### 4. Missing Model Asset (`face_landmarker.task`)
* **Symptom:** `[MirrorFit ERROR] MediaPipe Model Asset Missing!`.
* **Fix:**
  Download the model using the command in Step 4:
  ```powershell
  Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" -OutFile "models\face_landmarker.task"
  ```

### 5. Black or Frozen Webcam Output
* **Symptom:** Window appears with title but image is pitch black.
* **Fix:**
  * Try specifying the resolution explicitly: `python src/main.py --width 640 --height 480`.
  * Ensure your room has adequate lighting.

### 6. Low FPS (Laggy Video)
* **Symptom:** FPS is under 15 FPS.
* **Fix:**
  * Press **M** to disable the full dense mesh wireframe (rendering hundreds of lines every frame adds drawing overhead).
  * Reduce resolution to standard 720p or 480p: `python src/main.py --width 640 --height 480`.

---

## 8. Acceptance Criteria Checklist (Phase 1)

- [x] Webcam opens successfully via OpenCV.
- [x] Live video feed appears in high-resolution window.
- [x] Video feed is horizontally mirrored for natural user interaction.
- [x] MediaPipe Face Landmarker tracks 478 3D landmarks in real time.
- [x] Normalized coordinates map accurately to image pixel space.
- [x] Eye regions, nose bridge, temple hinges, and contours are identified.
- [x] Interpupillary distance (IPD) and face tilt angle are computed.
- [x] Dynamic face status (`FACE DETECTED` vs `NO FACE DETECTED`) updates cleanly.
- [x] Real-time smoothed FPS counter is visible on the HUD.
- [x] Pressing `Q` or `ESC` cleanly terminates the app and releases all resources.
- [x] Windows virtual environment and VS Code compatibility verified.

---

## 9. Future Development Roadmap

* **Phase 2: Virtual Glasses Overlay**
  * Load transparent PNG glasses models from `assets/glasses/`.
  * Compute affine rotation matrix matching `face_tilt_deg`.
  * Scale frame width to match interpupillary and temple span.
  * Alpha-composite glasses onto nose bridge (`landmark 168`).
* **Phase 3: Multiple Accessories & Interactive Catalog**
  * Switch between sunglasses, reading glasses, and hats using keyboard keys (1–5) or on-screen buttons.
  * Real-time color customization (lens tint, frame finish).
* **Phase 4: 3D Head Pose & Perspective Warping**
  * Use solvePnP with canonical 3D facial points to calculate yaw, pitch, and roll.
  * Apply perspective warping or 3D rendering for glasses arms when turning head sideways.
* **Phase 5: Advanced Computer Vision Experiments**
  * Try-on lighting simulation, lens reflection rendering, and segmentation using PyTorch models.
