# NeuralCensor

An AI-powered tool for fully automatic, pixel-precise image anonymization. NeuralCensor combines three state-of-the-art models into a single pipeline:

1. **YOLOv8** — blazing-fast object detection for persons & vehicles  
2. **SAM 3 (Segment Anything 3)** — pixel-perfect contour masks  
3. **Ollama Vision LLM** — paranoid verification with automatic SAM 3 re-segmentation  

Every step runs **100% locally** on your machine. No images are ever uploaded to any cloud.

---

## Key Features

- **Privacy First** — All processing happens on-device. Your images never leave your machine.
- **Comprehensive Detection** — YOLOv8 detects persons, cars, trucks, buses, motorcycles, and bicycles — even in dense crowds or far backgrounds.
- **Pixel-Perfect Masking** — SAM 3 replaces crude bounding-box blurs with exact contour masks that follow each object's outline.
- **Multi-Pass Gaussian Blur** — 3× Gaussian blur + pixel quantization makes reconstruction practically impossible.
- **SAM 3 Safety Pass** — Before Ollama verification, SAM 3 runs a text-prompt scan on the original image to catch anything YOLO missed. Only genuinely new regions are added.
- **Strict LLM Verification** — After pixelation and the SAM 3 safety pass, a local Vision LLM (e.g. Gemma 4) reviews the result. It is instructed to be *paranoid*: even a single visible head, arm, or silhouette counts as a missed person.
- **Ollama Retry Loop** — If the LLM finds missed objects, SAM 3 re-runs on the original image at a lower confidence threshold (0.12). This repeats up to **3 times**. If objects are still detected after 3 re-passes, the current result is saved and processing continues with the next image.
- **Batch Processing** — Process single images, flat folders, or nested folder structures via a modern dark-themed GUI.
- **Fully Automated Setup** — `start.bat` installs everything on first run: Python venv, PyTorch + CUDA, YOLO, SAM 3, and the SAM 3 checkpoint. **No manual steps required.**
- **Automatic Ollama Management** — `start.bat` checks whether Ollama is installed and whether the required vision model is already downloaded. If the model pull fails due to an outdated Ollama version, it **automatically downloads and installs the latest Ollama**, then retries the model download — all without any user interaction.

---

## How the Pipeline Works

<p align="center">
  <img src="neuralcensor_flowchart_en.svg" alt="NeuralCensor Pipeline Flowchart" width="680">
</p>

### Why SAM 3 instead of YOLO for the second pass?

### Why does SAM 3 scan the original image — won't it find already-blurred areas again?

SAM 3 always scans the **original unblurred image** — intentionally: pixelated areas are harder for AI to detect reliably, so the original gives SAM 3 the best chance to find missed objects clearly.

To prevent re-processing regions **already blurred**, an **IoU overlap filter** (threshold: 0.3) discards any mask that significantly overlaps with an already-covered area. Only genuinely new, uncovered regions are returned and then blurred. Each newly found mask is immediately added to the "covered" union, so subsequent text prompts also skip it.

SAM 3 text-search runs in up to **three stages**:
1. **Safety Pass (always)** — automatically before the Ollama check, catches what YOLO missed
2. **Ollama-triggered re-pass** — only if Ollama flags remaining unblurred objects; SAM 3 re-runs at a lower confidence threshold (`0.12` instead of `0.15`)
3. This re-pass repeats up to **3 times** — after 3 failed attempts the image is saved as-is and processing continues

---

## Prerequisites

| Requirement | Details |
|---|---|
| **NVIDIA GPU** | Required — minimum **12 GB VRAM** (SAM 3 ~5 GB + Ollama vision model ~7 GB), tested on 24 GB |
| **Python 3.12+** | Recommended for SAM 3 compatibility |
| **Ollama** | Must be installed and running. `start.bat` handles installation and model download automatically. |
| **Git** | Required for SAM 3 installation from GitHub |
| **HuggingFace Account** | Required for downloading the SAM 3 checkpoint (free, gated access) |

> **⚠️ GPU Requirement:** NeuralCensor loads SAM 3 (~5 GB VRAM) and an Ollama vision model (e.g. `gemma4:e4b` ~7 GB VRAM) simultaneously. A GPU with **at least 12 GB VRAM** is required. GPUs with less VRAM will cause out-of-memory errors or fall back to CPU, which is extremely slow. **Tested on NVIDIA RTX with 24 GB VRAM.**

---

## Installation & Quick Start

NeuralCensor handles the entire setup automatically via `start.bat`.

1. Clone or download this repository.
2. Ensure you have sufficient disk space (~10 GB for PyTorch, SAM 3, and model checkpoints).
3. **Double-click `start.bat`**.

> **The only manual input required** is your **HuggingFace access token** — needed once to download the SAM 3 checkpoint. Everything else is fully automatic.
>
> **How to get it (free):**
> 1. Create a free account at [huggingface.co](https://huggingface.co)
> 2. Accept the model terms at [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3)
> 3. Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → "Create new token" (read access is sufficient)
> 4. Paste it when `start.bat` asks — stored **only locally** on your machine, never sent anywhere else

### What `start.bat` does on first run:

1. Creates an isolated Python virtual environment (`venv`)
2. Installs PyTorch 2.10 with CUDA 12.8
3. Installs all dependencies from `requirements.txt`
4. Installs SAM 3 from the official GitHub repository
5. **Asks once for your HuggingFace token** — then downloads the SAM 3 checkpoint (~5 GB)
6. Checks Ollama and the vision model:
   - If Ollama is not installed → shows download link
   - If the model pull fails (e.g. Ollama too old) → **automatically downloads and installs the latest Ollama silently**, then retries
   - If the model is missing → **automatically pulls `gemma4:e4b`** (~7 GB)
7. Launches the NeuralCensor GUI

> **Entering the HuggingFace token is the only manual step.** After that, simply wait for setup to finish.

> On subsequent runs, `start.bat` skips installation and launches the GUI directly.

---

## Usage

1. Run `start.bat` (or activate the venv and run `python neuralcensor.py`).
2. Select the **Verification Model** in the settings panel (e.g. `gemma4:e4b`).
3. Click **📂 Browse Input** and select a folder — the mode is detected automatically:
   - Folder with images directly → flat mode
   - Folder with subfolders containing images → subfolder mode (each subfolder gets its own output)
   - Mixed → both handled automatically
4. *(Optional)* Select an output folder. If left blank, a `NeuralCensor_Blurry` folder is created automatically next to the input.
5. Click **▶ Start Processing**.

### Live Processing Log

The log panel shows detailed progress for every image. At the end of each image, a compact summary block is printed:

```
┌─ photo.jpg ───────────────────────── 22.5s ─┐
│  YOLO:    17 persons + 10 vehicles  (27 detected)  │
│  SAM3:    27 masks (pass 1)                        │
│  Safety:  +84 new objects  ✓                       │
│  Ollama:  1 re-pass → +105 new objects  ✓          │
│  Total:   216 masked regions saved                 │
└────────────────────────────────────────────────────┘
```

| Field | Meaning |
|---|---|
| **YOLO** | How many persons and vehicles YOLO detected |
| **SAM3** | Pixel-precise masks generated from YOLO boxes (pass 1) |
| **Safety** | Additional objects found by the SAM 3 text-search safety pass (always runs before Ollama) |
| **Ollama** | `all clear ✓` if everything is blurred — or `N re-pass(es) → +X new objects ✓` if SAM 3 had to correct something — or `⚠ ABORTED` after 3 failed re-passes |
| **Total** | Total number of masked regions saved to disk |
| *(header right)* | Total processing time for this image |

### Anonymization Report (`Anonymization_Report.txt`)

After processing, a report file is created in the output folder. Each line documents one image:

```
photo.jpg | 22.5s | YOLO: 17 persons + 10 vehicles → SAM3: 27 masks | Safety: +84 | Ollama: 1 re-pass(es) → +105 new | Total masks: 216
```

| Column | Meaning |
|---|---|
| `filename` | Name of the processed image |
| `Xs` | Total processing time in seconds |
| `YOLO: N persons + M vehicles → SAM3: K masks` | YOLO detections split by type, and how many SAM 3 masks were refined from them |
| `Safety: +N` | New objects found by the automatic SAM 3 safety pass |
| `Ollama: OK` | Ollama verified the result — nothing missed |
| `Ollama: N re-pass(es) → +X new` | Ollama found missed objects; SAM 3 re-ran N times and found X new regions |
| `Ollama: ABORTED after N re-pass(es)` | After 3 re-passes Ollama still detected objects — image saved as-is |
| `Total masks: N` | Sum of all masked regions across all passes |

---

## YOLO Detection Classes

| Class ID | Label |
|---|---|
| 0 | Person |
| 2 | Car |
| 3 | Motorcycle |
| 5 | Bus |
| 7 | Truck |

---

## Configuration

Key constants can be adjusted at the top of `neuralcensor.py`:

| Constant | Default | Description |
|---|---|---|
| `YOLO_CONF` | 0.15 | YOLO confidence threshold (lower = more detections) |
| `SAM3_CONFIDENCE` | 0.15 | SAM 3 confidence for pass 1 and safety pass |
| `SAM3_CONFIDENCE_RETRY` | 0.12 | Lower SAM 3 confidence used during Ollama-triggered re-passes |
| `MAX_OLLAMA_PASSES` | 3 | Maximum number of Ollama-triggered SAM 3 re-passes per image |
| `BLUR_KERNEL_BASE` | 101 | Gaussian blur kernel size (must be odd) |
| `BLUR_PASSES` | 3 | Number of blur passes per mask |
| `QUANTIZE_STEP` | 8 | Pixel quantization step (anti-reconstruction) |
| `PADDING_FRACTION` | 0.04 | Mask edge padding (contour dilation) |
| `OLLAMA_MAX_SIZE` | 1536 | Max image edge length sent to Ollama |

---

## Open Source & Licensing

NeuralCensor is open-source. The underlying AI models and libraries carry their own licenses:

| Component | License |
|---|---|
| **YOLOv8 (Ultralytics)** | AGPL-3.0 — public distribution requires open-sourcing modifications |
| **SAM 3 (Meta)** | Apache 2.0 |
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
├── yolov8n.pt           # YOLOv8 nano model (auto-downloaded)
└── checkpoints/
    └── sam3/             # SAM 3 model checkpoint (downloaded via HuggingFace)
```
