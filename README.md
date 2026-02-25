# Scoliosis Posture Visualizer — Demo

A lightweight Streamlit demo that uses pose detection to visualize posture and compute a **demo** curvature/asymmetry score.  
**Not a medical device or diagnosis.**

## Features list

- **Upload or capture** a back photo (file upload or camera).
- **Pose detection** (MediaPipe): shoulders, hips, torso midline with overlaid lines.
- **Curvature score** (0–100) and label: Normal range / Mild asymmetry / Moderate asymmetry.
- **Exercise suggestions** (rule-based, by label).
- **History & compare**: save scans locally and compare two images side-by-side.

## Setup

### Option A — with [uv](https://docs.astral.sh/uv/)

```powershell
# From project root (ensure uv is installed: https://docs.astral.sh/uv/getting-started/installation/)
uv venv
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

Or use the sync command (installs from `pyproject.toml`):

```powershell
uv venv
uv sync
.\.venv\Scripts\Activate.ps1
```

### Option B — with pip

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Requirements

- Python 3.8+
- `streamlit`, `opencv-python`, `mediapipe`, `numpy`, `Pillow`

## Disclaimer

This app is for **demo and educational purposes only**. It is not a medical device and does not provide a diagnosis. Always consult a healthcare professional for posture or spine-related concerns.
