# MirrorFit XR | Smart AR Virtual Eyewear Try-On Studio (Stage 5)

**MirrorFit XR** is a real-time computer vision and augmented reality application that tracks 3D head pose, performs real-time face segmentation, and renders shaded 3D polygonal eyewear using **Python**, **OpenCV**, **MediaPipe**, and **NumPy**.

* **Phase 1:** High-speed webcam capture, 478-point 3D facial landmark detection, and HUD telemetry.
* **Phase 2:** Virtual glasses overlay with dynamic scaling, head tilt rotation, boundary clipping, and alpha compositing.
* **Phase 3:** Anti-jitter temporal smoothing, multi-accessory 2D catalog, and timestamped screenshot service.
* **Phase 4:** 6-DOF solvePnP head-pose tracking, initial 3D model rendering, and face segmentation.
* **Stage 5 (Current):** **Smart AR Virtual Eyewear Try-On Studio**:
  * **8 Distinct 3D Eyewear Models:** Full authentic 3D geometry for **Aviator**, **Round**, **Square**, **Rectangle**, **Wayfarer**, **Cat-Eye**, **Sports**, and **Sunglasses**.
  * **Advanced Anthropometric Fitting:** Measures physical interpupillary distance (IPD), temporal face width, and nasal saddle anchor to auto-scale glasses to facial proportions.
  * **Smart Frame Adjustments:** Interactive controls for overall scale, horizontal frame width, vertical height, lateral/vertical shift, and forward/backward depth offset.
  * **Curated Material & Color Palettes:** Dynamic cycling of metallic and matte frame finishes (Gold, Matte Black, Chrome, Tortoise, Rose Gold) and polarized lens tints (G-15 Green, Charcoal, Ocean Blue, Sunset Amber, Clear Optical).
  * **Eyewear Comparison Studio:** Capture up to 4 eyewear looks and view an elegant side-by-side or 2x2 comparison board (`[V]` key), with instant export to PNG.
  * **Modern Studio Interface:** Dark charcoal aesthetic, electric-cyan accents, top status bar, collapsible catalogue side panel, and clickable on-screen buttons.
  * **60+ FPS Real-Time Optimization:** Vectorized 6-DOF projection, multi-model RAM caching, and throttled segmentation.

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
│   └── models_3d/              # 8 Distinct 3D Wavefront OBJ models
│       ├── aviator.obj         # Teardrop rims & double brow bar (170v / 184f)
│       ├── round.obj           # Circular wireframe rims (162v / 180f)
│       ├── square.obj          # Bold geometric square rims (162v / 180f)
│       ├── rectangle.obj       # Slim modern rectangular profile (162v / 180f)
│       ├── wayfarer.obj        # Trapezoidal browline rims (162v / 180f)
│       ├── cat_eye.obj         # Upswept winged outer corners (162v / 180f)
│       ├── sports.obj          # Aerodynamic wraparound shield (162v / 180f)
│       └── sunglasses.obj      # Luxury oversized polarized dark frames (162v / 180f)
├── models/
│   ├── face_landmarker.task    # Google MediaPipe 478-point Face Landmarker (~3.76 MB)
│   └── selfie_segmenter.tflite # Google MediaPipe Image Segmenter (~250 KB)
├── output/
│   └── screenshots/            # Saved screenshots & comparison studio collages
├── scratch/
│   ├── test_stage4_1.py        # Stage 4.1 verification test suite
│   ├── test_stage5.py          # Stage 5 verification & benchmark test suite
│   └── generate_8_models.py    # Parametric 3D eyewear mesh generator
├── src/
│   ├── __init__.py             # Package exports
│   ├── accessory_catalog.py    # 8-Style eyewear catalogue & material palettes
│   ├── accessory_manager.py    # 2D Multi-accessory catalog & switcher
│   ├── camera.py               # DirectShow camera acquisition & mirroring
│   ├── face_segmentation.py    # MediaPipe face segmentation module (optimized)
│   ├── face_tracker.py         # 478-point landmark tracker & geometric anchors
│   ├── fitting_engine.py       # Anthropometric fitting & smart adjustments
│   ├── glasses_overlay.py      # 2D Scaling, rotation, & alpha compositing
│   ├── head_pose.py            # 6-DOF solvePnP head pose estimator & axes
│   ├── performance_monitor.py  # Latency tracking and smoothed FPS calculation
│   ├── renderer_3d.py          # Vectorized 3D OBJ mesh renderer, materials, & culling
│   ├── screenshot_service.py   # Screenshots and multi-look Comparison Studio
│   └── main.py                 # Studio orchestrator, HUD, mouse & key controls
├── requirements.txt            # Compatible dependencies
├── README.md                   # Full documentation, controls, & testing plan
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

### Step 4: Run MirrorFit XR Studio
Launch the application using either direct script or module syntax:
```powershell
python -m src.main
```
or:
```powershell
python src/main.py
```

#### Optional CLI Customization Flags:
```powershell
# Specify starting 3D model asset
python -m src.main --model3d assets/models_3d/round.obj

# Adjust starting glasses width scale multiplier (default: 2.15)
python -m src.main --scale 2.20

# Adjust vertical offset ratio relative to glasses height (default: 0.05)
python -m src.main --offset 0.06

# Adjust anti-jitter smoothing factor beta in [0.1, 1.0] (default: 0.35)
python -m src.main --smoothing 0.30

# Specify custom camera device index and resolution
python -m src.main --camera 0 --width 1280 --height 720
```

---

## 3. Interactive Studio Controls Reference

MirrorFit XR supports both **keyboard shortcuts** and **direct mouse clicks** on HUD buttons.

### Eyewear Selection & Catalogue
| Control | Action |
| :--- | :--- |
| **1** | Select **Aviator Classic** (Teardrop profile with double brow bar) |
| **2** | Select **Round Heritage** (Circular rims with keyhole bridge) |
| **3** | Select **Square Bold** (Thick geometric square rims) |
| **4** | Select **Rectangle Sleek** (Slim modern rectangular profile) |
| **5** | Select **Wayfarer Retro** (Trapezoidal browline rims) |
| **6** | Select **Cat-Eye Glamour** (Dramatic upswept winged corners) |
| **7** | Select **Sports Shield** (Aerodynamic wraparound curvature) |
| **8** | Select **Polarized Sunglasses** (Oversized luxury dark frames) |
| **N** / **P** | Cycle to Next / Previous eyewear style |
| **Mouse Click** | Click on any catalogue item in the right-hand studio side panel |

### Visual Modes & Materials
| Control | Action |
| :--- | :--- |
| **T** | Toggle between **3D AR Model Mode** and **2D Overlay Mode** |
| **W** | Toggle **3D Wireframe Debug Mode** on/off (solid shaded by default) |
| **C** | Cycle **Frame Material Color** (Gold, Matte Black, Chrome, Tortoise, Rose Gold, Navy, Slate) |
| **L** | Cycle **Lens Tint Palette** (Polarized Charcoal, G-15 Green, Ocean Blue, Sunset Amber, Rose, Clear) |
| **X** | Toggle **3D Head Pose Coordinate Axes** on/off (Red=X, Green=Y, Blue=Z) |
| **B** | Toggle **Face Segmentation Mask Overlay** on/off |
| **TAB** or **K** | Toggle **Studio Side Panel** visibility on/off |

### Smart Frame Adjustments
| Control | Action |
| :--- | :--- |
| **F** | Toggle **Dynamic Anthropometric Auto-Fit** on/off |
| **[** / **]** | Decrease / Increase **Overall Frame Scale** ($\pm 0.05\text{x}$) |
| **-** / **+** (or `=`) | Narrow / Widen **Horizontal Frame Width** ($\pm 0.05\text{x}$) |
| **Up** / **Down** | Shift frame **Up / Down vertically** ($\pm 2.0\text{ mm}$) |
| **Left** / **Right** | Shift frame **Left / Right laterally** ($\pm 2.0\text{ mm}$) |
| **,** / **.** | Shift frame **Forward / Backward in depth** ($\pm 3.0\text{ mm}$) |
| **R** | **Reset All Adjustments** and temporal smoothing filters |

### Comparison Studio & Screenshots
| Control | Action |
| :--- | :--- |
| **V** | Open **Comparison Studio** / Add current look / Toggle comparison collage |
| **S** | Capture and save a timestamped screenshot or comparison collage to `output/screenshots/` |
| **Q** or **ESC** | Safely exit application and release webcam hardware |

---

## 4. Technical Analysis & Mathematical Architecture

### 1. 6-DOF Head Pose Tracking via OpenCV solvePnP
MirrorFit XR tracks head rotation ($R$) and translation ($\mathbf{t}$) in 3D camera space using perspective-n-point geometry:
* **Canonical 3D Model:** Anthropometric facial landmarks in millimeters ($+X$ right, $+Y$ down, $+Z$ into skull).
* **Pinhole Camera Calibration:** $f_x = f_y = 1.2 \times W$, $c_x = W/2$, $c_y = H/2$.
* **Euler Decomposition:** Computes continuous Yaw, Pitch, and Roll angles with anti-jitter Exponential Moving Average (EMA) filtering.

### 2. Anthropometric Proportions & Dynamic Scaling
The fitting engine extracts real-time physical dimensions from MediaPipe 478 landmarks:
$$\text{IPD}_{\text{3D}} = \frac{\|\mathbf{p}_{\text{left\_eye}} - \mathbf{p}_{\text{right\_eye}}\|_{\text{px}} \times Z_{\text{cam}}}{f_x}$$
$$\text{Width}_{\text{3D}} = \frac{\|\mathbf{p}_{\text{left\_zygomatic}} - \mathbf{p}_{\text{right\_zygomatic}}\|_{\text{px}} \times Z_{\text{cam}}}{f_x}$$
This blended ratio scales the 3D model proportionally, keeping fitting stable as the user moves closer or farther from the lens.

### 3. Vectorized 3D Shaded Rendering & Depth Occlusion
* **Painter's Algorithm:** Precomputed model centroids and normals rotate via $R @ \mathbf{N}^T$ in a single matrix multiplication, sorting all 180 faces by camera depth $Z_{\text{cam}}$ in under $0.05\text{ ms}$.
* **Lambertian Lighting:** $I = 0.38 + 0.62 \max(0, \mathbf{n} \cdot \mathbf{L})$ with metallic edge highlights.
* **Temple Arm Occlusion:** Evaluates lateral temple normal vectors in camera coordinates. When head yaw causes the far temple to turn away from the camera ($n_z > 0.16$), its rear shaft is geometrically culled, preventing it from projecting across the cheek, nose, or eyes.
* **Bounding-Box Lens ROI Blending:** Blends polarized lens tints exclusively within the $300 \times 150$ pixel eye ROI instead of the full $1280 \times 720$ frame, maintaining **60+ FPS**.

---

## 5. Automated Verification Test Suite

Run the automated test suite verifying all Stage 5 features:
```powershell
$env:PYTHONPATH="."
.\venv\Scripts\python.exe scratch/test_stage5.py
```

### Verified Test Cases:
1. `TEST 1: Eyewear Catalogue & 8 Distinct 3D Model Geometries` — PASSED
2. `TEST 2: Catalogue Selection & Cycling (Keys 1-8, Next/Prev)` — PASSED
3. `TEST 3: Anthropometric Fitting & Smart Frame Adjustments` — PASSED
4. `TEST 4: 3D Fitting Calculation with Landmark Proportions` — PASSED
5. `TEST 5: Missing Asset Handling & Graceful Fallback` — PASSED
6. `TEST 6: Screenshot & Comparison Studio Collage Generation` — PASSED
7. `TEST 7: Multi-Model Performance Benchmark (All 8 Models)` — PASSED
