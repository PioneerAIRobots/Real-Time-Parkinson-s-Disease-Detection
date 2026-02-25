# 🧠 NeuroVision PD — Real-Time Parkinson's Disease Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=flat-square&logo=flask&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED?style=flat-square&logo=onnx&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Research](https://img.shields.io/badge/Use-Research%20Only-orange?style=flat-square)

**A non-invasive, computer vision–based screening tool for early-stage Parkinson's Disease detection through real-time facial tremor analysis.**

[Features](#-features) • [How It Works](#-how-it-works) • [Installation](#-installation) • [Usage](#-usage) • [Web Interface](#-web-interface) • [Project Structure](#-project-structure) • [Disclaimer](#-disclaimer)

</div>

---


![](parkinson.gif)

## 📌 Overview

NeuroVision PD analyzes resting head tremor — a hallmark symptom of Parkinson's Disease — using only a standard camera. The system tracks 68 facial landmarks per frame, estimates 3D head pose, extracts tremor frequency from the yaw (left-right rotation) signal using Welch's Power Spectral Density method, and classifies the result using a machine learning model.

> **Key clinical insight:** Parkinsonian resting tremor characteristically occurs in the **3–6 Hz frequency band** and is absent during voluntary movement. This system captures that signature non-invasively in under 5 seconds of video.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎯 **Face Detection** | ONNX-accelerated neural network, >99% confidence at 30 FPS |
| 📍 **68-Point Landmarks** | Full facial landmark map with anatomically grouped overlays |
| 📐 **3D Pose Estimation** | Pitch, Yaw, Roll via `solvePnP` + Rodrigues decomposition |
| 🌊 **Tremor Waveform** | Live yaw signal scrolling display with 3–6 Hz band highlighted |
| 📊 **Welch PSD Analysis** | Dominant tremor frequency, band power ratio, amplitude σ |
| 🤖 **ML Classifier** | Logistic regression on [dominant freq, amplitude, mean velocity] |
| 📈 **Probability Gauge** | Real-time circular gauge with calibrated PD likelihood score |
| 🌐 **Web Dashboard** | Flask-based live MJPEG stream + polling stats panel |
| 📷 **Webcam Support** | Works with any standard USB/built-in webcam |

---

## 🔬 How It Works

The detection pipeline mirrors the clinical assessment of resting tremor:

```
        Video Frame
            │
            ▼
┌─────────────────────────────┐
│  1. Face Detection (ONNX)   │  → Bounding box + confidence score
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  2. Landmark Detection      │  → 68 facial keypoints (eyes, nose,
│     (face_landmarks.onnx)   │    jaw, brows, mouth)
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  3. 3D Pose Estimation      │  → Pitch, Yaw, Roll angles via
│     (solvePnP + Rodrigues)  │    PnP algorithm + rotation matrix
└────────────┬────────────────┘
             │  Yaw signal buffered over 5s window (150 frames)
             ▼
┌─────────────────────────────┐
│  4. Welch's PSD Analysis    │  → Dominant frequency in 3–6 Hz band
│     (scipy.signal.welch)    │    Band power ratio, amplitude σ
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  5. ML Classification       │  → PD probability score [0–1]
│     (Logistic Regression)   │    Threshold: 0.70
└────────────┬────────────────┘
             │
             ▼
        Diagnosis:
    🟢 Healthy  /  🔴 PD Detected
```

### Signal Features Used

| Feature | Clinical Relevance |
|---|---|
| `dominant_freq` | PD tremor peaks at 3–6 Hz; healthy motion is slower |
| `tremor_amplitude` | Standard deviation of yaw signal — PD shows higher variance |
| `mean_velocity` | Mean absolute frame-to-frame change — elevated in tremor |

---

## 🖥️ Demo

### Command-Line Mode (`main.py`)
Runs the full OpenCV pipeline in a local window — bounding box, landmarks, pose axes, waveform, and diagnosis overlaid directly on the video.

### Web Interface (`app.py` + `webapp.html`)
A clinical-grade Flask web dashboard with:
- **Live MJPEG video stream** with all overlays
- **Real-time statistics panel** — pose angles, signal features, tremor waveform
- **Animated probability gauge** — color shifts cyan → red above threshold
- **Video source selector** — switch between `.mp4` files or webcam

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip

### 1. Clone the repository

```bash
git clone https://github.com/your-username/neurovision-pd.git
cd neurovision-pd
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**
```
opencv-python
numpy
scipy
scikit-learn
flask
onnxruntime
```

> **Note:** If you have a CUDA-capable GPU, replace `onnxruntime` with `onnxruntime-gpu` for faster inference.

### 3. Verify assets

Ensure the following model files are present:

```
assets/
├── face_detector.onnx      ← Face detection model
├── face_landmarks.onnx     ← 68-point landmark model
└── model.txt               ← Model configuration
```
Pre-trained models provided in the assets directory. Download them with Git LFS:

git lfs pull
Or, download manually from the release page. https://github.com/yinguobing/head-pose-estimation/releases
and put in assests


---

## 🚀 Usage

### Option A — Command-Line (OpenCV Window)

```bash
# Run on webcam (default)
python main.py

# Run on a video file
python main.py --video 1.mp4

# Specify webcam index
python main.py --cam 1
```

**Controls:**
- `ESC` — Exit

### Option B — Web Interface (Recommended for Demos)

```bash
python app.py
```

Then open your browser at: **`http://localhost:5000`**

The web interface will:
1. Auto-detect any `.mp4` files in the project directory
2. Show them in a dropdown — click **Load & Analyze**
3. Start streaming the processed video with live stats

---

## 🌐 Web Interface

The Flask backend (`app.py`) exposes three endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serves the dashboard HTML |
| `/video_feed` | GET | MJPEG stream of processed frames |
| `/api/stats` | GET | JSON blob of all real-time metrics |
| `/api/set_source` | POST | Switch video source (file or webcam) |
| `/api/videos` | GET | Lists available `.mp4` files |

**Stats JSON example:**
```json
{
  "prob": 0.82,
  "diagnosis": "Parkinson's Disease Detected",
  "yaw": 4.31,
  "pitch": -1.20,
  "roll": 0.87,
  "dominant_freq": 4.5,
  "tremor_amplitude": 2.18,
  "mean_velocity": 0.31,
  "band_power_ratio": 0.74,
  "face_detected": true,
  "fps": 28.4,
  "frame_count": 1420,
  "buffer_size": 150
}
```

---

## 📁 Project Structure

```
neurovision-pd/
│
├── assets/
│   ├── face_detector.onnx      # ONNX face detector
│   ├── face_landmarks.onnx     # ONNX 68-point landmark model
│   └── model.txt
│
├── templates/
│   └── index.html              # Flask web dashboard template
│
├── main.py                     # Standalone OpenCV pipeline
├── app.py                      # Flask web server
├── face_detection.py           # FaceDetector class (ONNX wrapper)
├── mark_detection.py           # MarkDetector class (68 landmarks)
├── pose_estimation.py          # PoseEstimator class (solvePnP)
├── utils.py                    # Helper: refine(), etc.
├── real_time.py                # Webcam real-time entry point
├── webapp.html                 # Standalone HTML demo page
├── requirements.txt
├── LICENSE
└── README.md
```

---

## ⚙️ Configuration

Key parameters are defined at the top of `main.py` and `app.py`:

```python
WINDOW_SECONDS = 5      # Analysis buffer duration (seconds)
TREMBAND = (3, 6)       # Parkinsonian tremor frequency band (Hz)
MAX_FPS = 30            # Target frame rate
THRESHOLD = 0.70        # PD classification threshold (0–1)
```

### Replacing the Classifier

The current model is a placeholder logistic regression. To use a trained model:

```python
# Replace in main.py / app.py:
import joblib
clf = joblib.load('your_trained_model.pkl')
```

The classifier receives a feature vector: `[dominant_freq, tremor_amplitude, mean_velocity]`

---

## 📊 Technical Specifications

| Parameter | Value |
|---|---|
| Landmark model | 68-point facial landmark (dlib-compatible layout) |
| Face detector | Lightweight ONNX CNN |
| Pose solver | OpenCV `solvePnP` with `SOLVEPNP_ITERATIVE` |
| PSD method | Welch's method (`scipy.signal.welch`) |
| Tremor band | 3–6 Hz (Parkinsonian resting tremor range) |
| Buffer window | 5 seconds at 30 FPS = 150 samples |
| Classification | Logistic regression on 3 signal features |
| Streaming | MJPEG over HTTP (Flask) |
| Stats polling | 150 ms interval (~6 updates/sec) |

---

## 🧪 Related Research

This system was developed as part of research in AI-assisted clinical screening at the **National Science & Technology Park (NSTP), Islamabad**.

If you use this work, please cite:

```bibtex
@misc{neurovisionpd2025,
  title   = {NeuroVision PD: Real-Time Parkinson's Disease Detection via Facial Tremor Analysis},
  author  = {Mansoor},
  year    = {2025},
  url     = {https://github.com/your-username/neurovision-pd}
}
```

---

## 🤝 Contributing

Contributions are welcome. Areas of interest:

- Training a proper classifier on a clinical dataset (e.g., [mPower](https://www.synapse.org/#!Synapse:syn4993293))
- Replacing the logistic regression with a deep learning approach (LSTM on the yaw sequence)
- Adding audio-based tremor analysis
- Exporting session reports as PDF

```bash
# Fork the repo, then:
git checkout -b feature/your-feature
git commit -m "Add your feature"
git push origin feature/your-feature
# Open a Pull Request
```

---

## ⚠️ Disclaimer

> **This is a research and educational tool only.**
>
> NeuroVision PD is **not a medical device** and has **not been clinically validated** for diagnostic use. It must not be used as a substitute for professional neurological examination or diagnosis.
>
> All outputs should be interpreted only by qualified healthcare professionals. Parkinson's Disease diagnosis requires comprehensive clinical assessment including motor examination, patient history, and where indicated, imaging or specialist referral.
>
> The authors accept no liability for any clinical decisions made based on this software.

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

