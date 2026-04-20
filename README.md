# NeuralCensor

An AI-powered tool for fully automatic, pixel-precise image and video anonymization. NeuralCensor combines state-of-the-art models into a unified pipeline:

**Images:**
1. **SAM 3 (Segment Anything 3)** — direct text-prompt segmentation: pass 1 + safety pass
2. **Falcon Perception (0.6B)** — additional detector running alongside SAM3 for higher detection coverage (IoU-merged)
3. **Ollama Vision LLM** — paranoid verification with automatic SAM 3 re-segmentation

**Videos (Beta):**
1. **SAM 3 (Segment Anything 3)** — direct text-prompt segmentation on every frame (no YOLO needed)
2. **Ollama Vision LLM** — spot-check verification on sampled frames after rendering

Every step runs **100% locally** on your machine. No images are ever uploaded to any cloud.

---

## Key Features

- **Privacy First** — All processing happens on-device. Your images and videos never leave your machine.
- **Comprehensive Detection** — SAM 3 + Falcon Perception text-prompts detect persons, cars, trucks, buses, motorcycles, and license plates. Two independent architectures are merged via IoU filtering, catching blind spots of each model.
- **Pixel-Perfect Masking** — SAM 3 replaces crude bounding-box blurs with exact contour masks that follow each object's outline.
- **Multi-Pass Gaussian Blur** — 3x Gaussian blur + pixel quantization makes reconstruction practically impossible.
- **Ollama Retry Loop** — If the LLM finds missed objects, SAM 3 re-runs on the original image at a lower confidence threshold (`0.15`). This repeats up to **3 times**.
- **Runtime Settings** — Click the gear icon in the header to adjust all pipeline parameters (blur strength, SAM 3 confidence, Falcon Perception, Ollama limits, text prompts) directly in the UI — no code editing required. Code defaults are always shown for reference.
- **Batch Processing** — Process single files, flat folders, or nested folder structures via a modern dark-themed GUI.
- **Flexible Input** — Click **Browse Input** and choose between a **folder** (auto-detects flat/subfolder structure) or **individual files** (select any mix of images and videos).
- **Video Support** — Process `.mp4`, `.mov`, `.avi`, `.mkv`, and `.webm` videos using SAM 3 text-prompt segmentation on **every single frame** for perfect, flicker-free anonymization. Audio tracks are automatically preserved using FFmpeg.
- **Video ETA** — The status bar shows how many key frames have been processed, ETA, and filename while a video is processing.
- **Dual-GPU Acceleration** — If a second NVIDIA GPU is available (`cuda:1`), NeuralCensor automatically loads a second SAM3 instance and distributes frames between both GPUs — roughly doubling video throughput.
- **Toggle Ollama** — The Ollama verification step can be toggled on/off in the UI. When enabled, images get full verification with retry loop, and videos get an automatic **spot-check** (10 sampled frames verified after rendering). When disabled, neither images nor videos use Ollama.
- **Fully Automated Setup** — `start.bat` installs everything on first run: Python venv, PyTorch + CUDA, SAM 3, and the SAM 3 checkpoint. **No manual steps required.**
- **Automatic Ollama Management** — `start.bat` checks whether Ollama is installed and whether the required vision model is already downloaded. If the model pull fails due to an outdated Ollama version, it **automatically downloads and installs the latest Ollama**, then retries the model download — all without any user interaction.

---

## How the Pipeline Works

<p align="center">
  <img src="neuralcensor_flowchart_en.svg" alt="NeuralCensor Pipeline Flowchart" width="680">
</p>

### Image Pipeline

| Step | What happens |
|---|---|
| **SAM 3 text-search** | Text-prompt search on original image → pixel-precise contour masks |
| **Falcon Perception** | Additional AI detector (0.6B) runs text-prompt search; new masks are IoU-filtered against SAM3 results and merged |
| **Blur** | 3× Gaussian blur + quantization applied to every mask |
| **Ollama verify** | Vision LLM reviews result; triggers SAM 3 re-passes if needed (up to 3x) |

> **Why two detectors?** SAM 3 and Falcon Perception use fundamentally different architectures — each has different blind spots. By running both and merging their masks via IoU overlap filtering, NeuralCensor achieves significantly higher detection coverage than either model alone. Falcon adds ~1 GB VRAM and runs only on images (too slow per-frame for video).

### Video Pipeline

| Step | What happens |
|---|---|
| **Pre-read** | All frames loaded into RAM for parallel processing |
| **SAM3 text-prompt** | Every frame segmented for "person", "car", "truck" — no skipping |
| **Dual-GPU** | Frames distributed across available GPUs automatically |
| **Sequential drop-check** | If a frame loses >50% of the previous frame's mask, SAM3 re-runs that frame at lower confidence (0.15) to recover the object |
| **Temporal smoothing** | Single-frame gaps where SAM3 missed an object are filled from the adjacent real detection (max 1 frame propagation) |
| **Blur** | All frames rendered in parallel on CPU cores |
| **Audio** | Original audio merged back via FFmpeg |
| **Ollama spot-check** | 10 evenly distributed frames verified by Vision LLM (if Ollama enabled) |

> **Why SAM3 on every frame?** Frame-skipping + interpolation causes flickering when objects move. Processing every frame guarantees pixel-perfect, flicker-free anonymization throughout the entire video.
>
> **Sequential mask-drop detection** acts as a safety net: if an object is tracked but suddenly loses more than 50% of its mask area in the very next frame, NeuralCensor automatically re-evaluates that frame with higher sensitivity (0.15 confidence) to ensure the object isn't lost. 
>
> **Temporal smoothing** then catches the remaining edge cases: videos with non-integer frame rates (e.g. 29.97, 23.976) can cause SAM3 to completely miss an object on an isolated frame. The smoothing pass fills those gaps by copying masks from the direct neighbor frame — but only if that neighbor has a real SAM3 detection. Propagation is strictly limited to 1 frame so masks are never invented in regions where nothing was detected.

### Why does SAM 3 scan the original image — won't it find already-blurred areas again?

SAM 3 always scans the **original unblurred image** — intentionally: pixelated areas are harder for AI to detect reliably, so the original gives SAM 3 the best chance to find missed objects clearly.

An **IoU overlap filter** (threshold: 0.3) discards any mask that significantly overlaps with an already-covered area. Only genuinely new, uncovered regions are returned and then blurred.

---

## Prerequisites

| Requirement | Details |
|---|---|
| **NVIDIA GPU** | Required — minimum **18 GB VRAM** (SAM 3 + Falcon Perception + Ollama vision model run simultaneously), tested on RTX 4090 (24 GB) |
| **FFmpeg** | Highly Recommended — required for copying original audio into pixelated videos. If not installed, videos are saved without audio. |
| **Python 3.12+** | Recommended for SAM 3 compatibility |
| **Ollama** | Must be installed and running. `start.bat` handles installation and model download automatically. |
| **Git** | Required for SAM 3 installation from GitHub |
| **HuggingFace Account** | Required for downloading the SAM 3 checkpoint (free, gated access) |

> **⚠️ GPU Requirement:** NeuralCensor loads SAM 3 (~5 GB VRAM), Falcon Perception (~1 GB VRAM), and an Ollama vision model (e.g. `gemma4:e4b` ~7 GB VRAM) simultaneously. A GPU with **at least 18 GB VRAM** is required. **Tested on NVIDIA RTX 4090 (24 GB VRAM).**

---

## Installation & Quick Start

NeuralCensor handles the entire setup automatically via `start.bat`.

1. Clone or download this repository.
2. Ensure you have sufficient disk space (~12 GB for PyTorch, SAM 3, Falcon Perception, and model checkpoints).
3. **Double-click `start.bat`**.

> **The only manual input required** is your **HuggingFace access token** — needed once to download the SAM 3 checkpoint. Everything else is fully automatic.
>
> **How to get it (free):**
> 1. Create a free account at [huggingface.co](https://huggingface.co)
> 2. Accept the model terms at [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3) → click "Agree and access repository"
> 3. Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → "Create new token" and **enable all 3 checkboxes**:
>    - ☑ Read access to contents of all public gated repos
>    - ☑ Read access to contents of all repos you can access
>    - ☑ Read access to your personal info (email, username)
> 4. Click "Create token", copy it, and paste it when `start.bat` asks — stored **only locally** on your machine, never sent anywhere else

### What `start.bat` does on first run:

1. Creates an isolated Python virtual environment (`venv`)
2. Installs PyTorch 2.10 with CUDA 12.8
3. Installs all dependencies from `requirements.txt` (including `transformers` for Falcon Perception)
4. Installs SAM 3 from the official GitHub repository
5. **Asks once for your HuggingFace token** — then downloads the SAM 3 checkpoint (~5 GB)
6. Checks Ollama and the vision model:
   - If Ollama is not installed → shows download link
   - If the model pull fails (e.g. Ollama too old) → **automatically downloads and installs the latest Ollama silently**, then retries
   - If the model is missing → **automatically pulls `gemma4:e4b`** (~7 GB)
7. Launches the NeuralCensor GUI (Falcon Perception is auto-downloaded from HuggingFace on first use, ~1.2 GB)

> **Entering the HuggingFace token is the only manual step.** After that, simply wait for setup to finish.

> On subsequent runs, `start.bat` skips installation and launches the GUI directly.

---

## Usage

1. Run `start.bat` (or activate the venv and run `python neuralcensor.py`).
2. Select the **Verification Model** in the settings panel (e.g. `gemma4:e4b`).
3. Click **📂 Browse Input** and choose:
   - **📂 Folder** — select a folder, auto-detected mode:
     - Folder with images/videos directly → flat mode
     - Folder with subfolders → subfolder mode (each subfolder gets its own output)
     - Mixed → both handled automatically
   - **📎 Files** — select individual images and/or videos in any combination
4. *(Optional)* Select an output folder. If left blank, a `NeuralCensor_Blurry` folder is created automatically next to the input.
5. Click **▶ Start Processing**.

### Live Processing Log

The status bar shows which file is currently being processed:

```
🎥 video.mp4 (2/5)  →  🎥 SAM3: 120/532 key frames | ~3m 14s remaining
🖼 photo.jpg (3/5)
```

The log panel shows detailed progress. At the end of each image, a compact summary is printed:

```
┌─ photo.jpg ───────────────────────── 12.3s ─┐
│  SAM3:    27 objects (text-search)           │
│  Falcon:  +5 new objects                     │
│  Ollama:  1 re-pass -> +3 new objects        │
│  Total:   35 masked regions saved            │
└─────────────────────────────────────────────┘
```

---

## Configuration

All pipeline parameters can be adjusted **live in the UI** — click the **gear icon** in the top-right corner of the header. Changes take effect on the next processing run. The code defaults (shown in grey inside the dialog) serve as the baseline and can be restored at any time via **Reset to Defaults**.

The settings dialog has four tabs:

### Blur

| Setting | Default | Description |
|---|---|---|
| Blur Kernel Base | 101 | Gaussian blur kernel size (odd number, larger = stronger) |
| Blur Passes | 3 | Number of blur passes per mask region |
| Quantize Step | 8 | Pixel quantization grid step (anti-reconstruction) |
| Padding Fraction | 0.04 | Mask edge padding as fraction of mask size |

### SAM3

| Setting | Default | Description |
|---|---|---|
| Confidence | 0.15 | Image detection threshold (lower = more detections) |
| Video Confidence | 0.20 | Video frame detection threshold |
| Confidence (Retry) | 0.12 | Lower confidence for Ollama re-passes and video mask-drop recovery |
| Min Mask Pixels | 100 | Masks smaller than this are discarded |
| Overlap IoU Threshold | 0.3 | Masks overlapping existing ones above this ratio are skipped |
| Image SAM3 Prompts | person, car, truck, bus, motorcycle, license plate | Text prompts for image segmentation |
| Video SAM3 Prompts | person, car, truck | Text prompts for video segmentation |

### Falcon Perception

| Setting | Default | Description |
|---|---|---|
| Enable | On | Toggle Falcon Perception as additional detector (~1 GB VRAM) |
| Falcon Text Prompts | person, car, truck, bus, motorcycle, license plate | Text prompts for Falcon segmentation |
| Max Dimension | 1024 | Maximum image edge length for Falcon inference |

### Ollama

| Setting | Default | Description |
|---|---|---|
| Max Re-passes | 3 | Max Ollama-triggered SAM 3 re-passes per image |
| Max Send Size | 1536 px | Maximum image edge length sent to Ollama |
| Spot-Check Frames | 10 | Frames sampled from rendered video for Ollama verification |

> The constants at the top of `neuralcensor.py` define the code defaults. Any values changed in the UI override them at runtime without modifying the source file.

---

## Open Source & Licensing

NeuralCensor is open-source under the **MIT License with Commons Clause**.

- ✅ **Free for** personal use, research, education, and non-profit organizations
- ❌ **Commercial use** (paid products, SaaS, commercial workflows) requires a written agreement — [contact via GitHub Issues](https://github.com/AIhdproduction/NeuralCensor/issues)

The underlying AI models and libraries carry their own licenses:

| Component | License |
|---|---|
| **SAM 3 (Meta)** | Apache 2.0 |
| **Falcon Perception (TII/UAE)** | Apache 2.0 |
| **Ollama** | MIT |
| **Gemma 4 (Google)** | Gemma License — allows commercial use with specific redistribution terms |
| **OpenCV** | Apache 2.0 |
| **PyTorch** | BSD-3-Clause |

---

## Project Structure

```
NeuralCensor/
├── neuralcensor.py      # Main application (GUI + processing pipeline)
├── start.bat            # Automated setup & launch script
├── requirements.txt     # Python dependencies
├── README.md
├── LICENSE
└── checkpoints/
    └── sam3/             # SAM 3 model checkpoint (downloaded via HuggingFace)
```

> **Falcon Perception** is automatically downloaded from HuggingFace on first use (~1.2 GB) — no manual setup required.
