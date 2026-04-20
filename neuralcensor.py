"""
NeuralCensor – Automatic Image Anonymization
Persons & vehicles are detected and segmented via SAM3 text-prompts (precise pixel masks),
pixelated multiple times with OpenCV (non-reconstructable), and verified via Ollama.
If Ollama finds missed objects, SAM3 runs again at a lower confidence threshold to catch
any remaining objects – yielding a unified pipeline for both images and videos.
"""

import io
import os
import queue
import threading
import time
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk
import cv2
import numpy as np
import requests
from PIL import Image

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".jfif", ".png", ".bmp", ".tiff", ".tif", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
CV2_WRITABLE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}  # .jfif etc. → .jpg

SAM3_CHECKPOINT_DIR = Path(__file__).parent / "checkpoints" / "sam3"

BLUR_KERNEL_BASE    = 101    # Gaussian-Blur base kernel size (odd, large!)
BLUR_PASSES         = 3      # Number of blur passes (security)
QUANTIZE_STEP       = 8      # Pixel quantization after blur (prevents reconstruction)
PADDING_FRACTION    = 0.04   # Edge padding around each mask
OLLAMA_MAX_SIZE     = 1536   # Maximum edge length for Ollama input
AUTO_OUTPUT_NAME    = "NeuralCensor_Blurry"

SAM3_MIN_MASK_PX    = 100               # min mask pixels before falling back to bbox
SAM3_CONFIDENCE     = 0.20              # confidence for SAM3 text-search (higher = fewer false positives)
SAM3_CONFIDENCE_RETRY = 0.15            # lower confidence for Ollama-triggered re-passes
SAM3_OVERLAP_IOU    = 0.3               # IoU threshold: new mask vs 1st-pass masks
MAX_OLLAMA_PASSES   = 3                 # max Ollama-triggered SAM3 re-passes per image
VIDEO_SPOT_CHECK_FRAMES = 10            # frames to sample for Ollama spot-check after video rendering

NEURALCENSOR_VERSION = "1.0"
NEURALCENSOR_URL     = "https://github.com/AIhdproduction/NeuralCensor"

# Text prompts for SAM3 2nd pass (searched independently on the original image)
SAM3_TEXT_PROMPTS   = ["person", "car", "truck", "bus", "motorcycle", "license plate"]

# Falcon Perception (additional detector alongside SAM3)
FALCON_TEXT_PROMPTS  = ["person", "car", "truck", "bus", "motorcycle", "license plate"]
FALCON_MAX_DIMENSION = 1024

# ── Video pipeline constants ───────────────────────────────────────────────
# Text prompts used for SAM3 video segmentation (keep minimal – fewer = faster)
VIDEO_SAM3_PROMPTS = ["person", "car", "truck"]

OLLAMA_VERIFY_PROMPT = (
    "This image has been anonymized. Your job is to check if ANY person or vehicle was MISSED. "
    "Be EXTREMELY strict and paranoid. Check every pixel of the image.\n\n"
    "PERSONS: Even a single visible head, face, hair, arm, hand, leg, foot, "
    "silhouette, or ANY recognizable human body part counts as a missed person. "
    "A person partially hidden behind an object, in a window, in a mirror, "
    "or barely visible in the background is STILL a person. "
    "If you can tell it is human in any way, it counts as missed.\n\n"
    "VEHICLES: Any car, truck, bus, motorcycle or license plate "
    "that is clearly unblurred counts as missed, even if partially occluded.\n\n"
    "Objects that are already properly blurred/pixelated are NOT missed.\n\n"
    "Answer ONLY whether you found any missed (unblurred) persons or vehicles. "
    "When in doubt, answer YES – false positives are acceptable, false negatives are not."
)

APPEARANCE = {
    "fg_color_primary":   "#1a1a2e",
    "fg_color_secondary": "#16213e",
    "accent":             "#0f3460",
    "highlight":          "#e94560",
    "text_primary":       "#eaeaea",
    "text_secondary":     "#a0a0b0",
    "success":            "#4ade80",
    "warning":            "#fbbf24",
    "error":              "#f87171",
    "button_hover":       "#c73652",
}

# ──────────────────────────────────────────────────────────────
# Runtime configuration (overrides constants at runtime via settings dialog)
# Defaults always match the module constants above.
# ──────────────────────────────────────────────────────────────

class _Cfg:
    """Mutable runtime config – all values start at their code defaults."""
    blur_kernel_base        = BLUR_KERNEL_BASE
    blur_passes             = BLUR_PASSES
    quantize_step           = QUANTIZE_STEP
    padding_fraction        = PADDING_FRACTION
    ollama_max_size         = OLLAMA_MAX_SIZE
    sam3_min_mask_px        = SAM3_MIN_MASK_PX
    sam3_confidence         = SAM3_CONFIDENCE
    sam3_confidence_retry   = SAM3_CONFIDENCE_RETRY
    sam3_overlap_iou        = SAM3_OVERLAP_IOU
    max_ollama_passes       = MAX_OLLAMA_PASSES
    video_spot_check_frames = VIDEO_SPOT_CHECK_FRAMES
    sam3_text_prompts       = list(SAM3_TEXT_PROMPTS)
    video_sam3_prompts      = list(VIDEO_SAM3_PROMPTS)
    falcon_enabled          = True
    falcon_text_prompts     = list(FALCON_TEXT_PROMPTS)
    falcon_max_dimension    = FALCON_MAX_DIMENSION

    def reset(self):
        self.blur_kernel_base        = BLUR_KERNEL_BASE
        self.blur_passes             = BLUR_PASSES
        self.quantize_step           = QUANTIZE_STEP
        self.padding_fraction        = PADDING_FRACTION
        self.ollama_max_size         = OLLAMA_MAX_SIZE
        self.sam3_min_mask_px        = SAM3_MIN_MASK_PX
        self.sam3_confidence         = SAM3_CONFIDENCE
        self.sam3_confidence_retry   = SAM3_CONFIDENCE_RETRY
        self.sam3_overlap_iou        = SAM3_OVERLAP_IOU
        self.max_ollama_passes       = MAX_OLLAMA_PASSES
        self.video_spot_check_frames = VIDEO_SPOT_CHECK_FRAMES
        self.sam3_text_prompts       = list(SAM3_TEXT_PROMPTS)
        self.video_sam3_prompts      = list(VIDEO_SAM3_PROMPTS)
        self.falcon_enabled          = True
        self.falcon_text_prompts     = list(FALCON_TEXT_PROMPTS)
        self.falcon_max_dimension    = FALCON_MAX_DIMENSION

cfg = _Cfg()



def pad_mask(mask: np.ndarray, pad: float | None = None) -> np.ndarray:
    """Extends a binary mask by dilating its contour (preserves shape)."""
    if pad is None:
        pad = cfg.padding_fraction
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return mask
    # Dilation kernel size based on mask extent
    h_extent = ys.max() - ys.min()
    w_extent = xs.max() - xs.min()
    k = max(3, int(min(h_extent, w_extent) * pad))
    k = k if k % 2 == 1 else k + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask, kernel, iterations=1)


def blur_region(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Applies multiple Gaussian blur passes to masked regions (in-place on ROI).
    Followed by pixel quantization for maximum security.
    The pixelation is practically non-reconstructable.
    """
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return image

    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())

    roi = image[y_min:y_max + 1, x_min:x_max + 1].copy()
    if roi.size == 0:
        return image

    roi_h, roi_w = roi.shape[:2]

    # Kernel size scales with ROI, capped for performance
    k = max(cfg.blur_kernel_base, int(min(roi_h, roi_w) * 0.4))
    k = k if k % 2 == 1 else k + 1
    k = min(k, 301)  # cap to avoid extreme kernel sizes

    blurred = roi
    for _ in range(cfg.blur_passes):
        blurred = cv2.GaussianBlur(blurred, (k, k), sigmaX=0, sigmaY=0)

    # Pixel quantization: round values to QUANTIZE_STEP grid
    blurred = (blurred // cfg.quantize_step * cfg.quantize_step).astype(np.uint8)

    local_mask = mask[y_min:y_max + 1, x_min:x_max + 1]
    image[y_min:y_max + 1, x_min:x_max + 1][local_mask > 0] = blurred[local_mask > 0]

    return image


def box_to_mask(box_xyxy, img_h: int, img_w: int) -> np.ndarray:
    """Creates a rectangular mask as fallback."""
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w, x2), min(img_h, y2)
    mask[y1:y2, x1:x2] = 255
    return mask


def embed_image_metadata(image_path: Path) -> None:
    """
    Embeds metadata into a saved image to mark it as processed by NeuralCensor.
    - JPEG/TIFF/WebP: EXIF via piexif
    - PNG: tEXt chunks via Pillow (PngInfo)
    - BMP: no standard metadata mechanism – silently skipped
    """
    from datetime import datetime

    ext = image_path.suffix.lower()
    timestamp = datetime.now().strftime("%Y:%m:%d %H:%M:%S")

    # ── PNG: Pillow tEXt chunks (no extra dependency) ────────────────
    if ext == ".png":
        try:
            from PIL.PngImagePlugin import PngInfo
            img = Image.open(image_path)
            meta = PngInfo()
            meta.add_text("Software", f"NeuralCensor {NEURALCENSOR_VERSION}")
            meta.add_text("Author", NEURALCENSOR_URL)
            meta.add_text("Description", "Anonymized with NeuralCensor \u2013 persons and vehicles blurred")
            meta.add_text("Copyright", "Processed by NeuralCensor (MIT + Commons Clause)")
            meta.add_text("Creation Time", timestamp)
            img.save(image_path, pnginfo=meta)
        except Exception:
            pass
        return

    # ── JPEG / TIFF / WebP: EXIF via piexif ──────────────────────────
    if ext in {".jpg", ".jpeg", ".tiff", ".tif", ".webp"}:
        try:
            import piexif
            exif_dict = {
                "0th": {
                    piexif.ImageIFD.Software:         f"NeuralCensor {NEURALCENSOR_VERSION}".encode("utf-8"),
                    piexif.ImageIFD.Artist:            NEURALCENSOR_URL.encode("utf-8"),
                    piexif.ImageIFD.ImageDescription:  "Anonymized with NeuralCensor \u2013 persons and vehicles blurred".encode("utf-8"),
                    piexif.ImageIFD.Copyright:         b"Processed by NeuralCensor (MIT + Commons Clause)",
                    piexif.ImageIFD.DateTime:          timestamp.encode("utf-8"),
                }
            }
            exif_bytes = piexif.dump(exif_dict)
            piexif.insert(exif_bytes, str(image_path))
        except ImportError:
            pass  # piexif nicht installiert – Schritt überspringen
        except Exception:
            pass  # Fehler nie die Hauptverarbeitung unterbrechen lassen


# ──────────────────────────────────────────────────────────────
# Processing class
# ──────────────────────────────────────────────────────────────

class Processor:
    """
    Backend: SAM3 text-search → OpenCV multi-pass pixelation → Ollama verification.
    Runs entirely in its own thread.
    """

    def __init__(self, msg_queue: queue.Queue):
        self.msg_queue     = msg_queue
        self.sam3_proc     = None
        self.sam3_proc2    = None    # second SAM3 instance (GPU 1 if available)
        self.sam3_loaded   = False
        self.ollama_ready  = False
        self.falcon_model  = None
        self.falcon_loaded = False
        self._stop_event   = threading.Event()

    # ── Message helpers ──────────────────────────────────────
    def _emit(self, kind: str, **kwargs):
        self.msg_queue.put({"kind": kind, **kwargs})

    def _log(self, text: str):
        self._emit("log", text=text)

    def _progress(self, value: float, current: int = 0, total: int = 0):
        self._emit("progress", value=value, current=current, total=total)

    def _status(self, text: str):
        """Update the status label in the GUI (below the progress bar)."""
        self._emit("status", text=text)

    def _done(self, success: bool, message: str = ""):
        self._emit("done", success=success, message=message)

    def stop(self):
        self._stop_event.set()

    # -- Load SAM3 (mask refiner) -------------------------------------------
    def _load_sam3(self) -> bool:
        if self.sam3_loaded:
            return True
        try:
            import torch
            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            # Monkey-patch fused addmm_act: the original forces bfloat16 for a
            # fused CUDA kernel that may not be available.  Replace with standard
            # float32 linear + activation to avoid dtype mismatches.
            import sam3.perflib.fused as _fused_mod
            import sam3.model.vitdet as _vitdet_mod

            def _addmm_act_f32(activation, linear, mat1):
                x = torch.nn.functional.linear(mat1, linear.weight, linear.bias)
                return activation()(x)  # activation is a class, instantiate then call

            _fused_mod.addmm_act = _addmm_act_f32
            _vitdet_mod.addmm_act = _addmm_act_f32

            if not torch.cuda.is_available():
                self._log("[WARN] CUDA not available - SAM3 runs on CPU (slow).")

            import sam3 as _sam3_pkg
            bpe_path = Path(_sam3_pkg.__file__).parent / "assets" / "bpe_simple_vocab_16e6.txt.gz"
            if not bpe_path.exists():
                bpe_path = Path(_sam3_pkg.__file__).parent.parent / "assets" / "bpe_simple_vocab_16e6.txt.gz"

            self._log("[LOAD] Loading SAM3 mask refiner ...")
            device = "cuda" if torch.cuda.is_available() else "cpu"

            ckpt_dir  = SAM3_CHECKPOINT_DIR
            # build_sam3_image_model expects a FILE path, not a directory.
            # Resolve to the actual checkpoint file.
            ckpt_path = None
            if ckpt_dir.exists():
                for candidate in ("sam3.pt", "model.safetensors"):
                    if (ckpt_dir / candidate).exists():
                        ckpt_path = str(ckpt_dir / candidate)
                        break

            if ckpt_path is None:
                self._log("  Checkpoint not local - loading from HuggingFace ...")

            model = build_sam3_image_model(
                checkpoint_path=ckpt_path,
                bpe_path=str(bpe_path) if bpe_path.exists() else None,
                device=device,
                load_from_HF=(ckpt_path is None),
            )
            # Ensure float32 – checkpoint may load as bfloat16
            model = model.float()

            self.sam3_proc = Sam3Processor(
                model,
                confidence_threshold=cfg.sam3_confidence,
                device=device,
            )
            self.sam3_loaded = True
            self._log("[OK] SAM3 loaded.")
            return True

        except ImportError:
            self._log("[ERROR] SAM3 not installed. Please run start.bat to install it.")
            return False
        except Exception as exc:
            self._log(f"[ERROR] SAM3 error: {exc}")
            return False

    def _load_sam3_extra(self, device: str) -> bool:
        """
        Load a second SAM3 instance on `device` (e.g. 'cuda:1').
        Sets self.sam3_proc2. Returns True on success, False on error.
        """
        try:
            import torch
            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            import sam3.perflib.fused as _fused_mod
            import sam3.model.vitdet as _vitdet_mod

            def _addmm_act_f32(activation, linear, mat1):
                x = torch.nn.functional.linear(mat1, linear.weight, linear.bias)
                return activation()(x)
            _fused_mod.addmm_act = _addmm_act_f32
            _vitdet_mod.addmm_act = _addmm_act_f32

            import sam3 as _sam3_pkg
            bpe_path = Path(_sam3_pkg.__file__).parent / "assets" / "bpe_simple_vocab_16e6.txt.gz"
            if not bpe_path.exists():
                bpe_path = Path(_sam3_pkg.__file__).parent.parent / "assets" / "bpe_simple_vocab_16e6.txt.gz"

            ckpt_dir  = SAM3_CHECKPOINT_DIR
            ckpt_path = None
            if ckpt_dir.exists():
                for candidate in ("sam3.pt", "model.safetensors"):
                    if (ckpt_dir / candidate).exists():
                        ckpt_path = str(ckpt_dir / candidate)
                        break

            model2 = build_sam3_image_model(
                checkpoint_path=ckpt_path,
                bpe_path=str(bpe_path) if bpe_path.exists() else None,
                device=device,
                load_from_HF=(ckpt_path is None),
            )
            # Ensure weights are float32 AND on the target device
            model2 = model2.float().to(device)
            self.sam3_proc2 = Sam3Processor(
                model2,
                confidence_threshold=cfg.sam3_confidence,
                device=device,
            )
            self._log(f"[OK] Second SAM3 instance loaded on {device}.")
            return True
        except Exception as exc:
            self._log(f"  [WARN] Second SAM3 instance on {device} failed: {exc}")
            return False

    # -- Load Falcon Perception (additional detector) -------------------------
    def _load_falcon(self) -> bool:
        """Load Falcon Perception model for additional detection."""
        if self.falcon_loaded:
            return True
        if not cfg.falcon_enabled:
            return False
        try:
            import torch
            from transformers import AutoModelForCausalLM

            self._log("[LOAD] Loading Falcon Perception (0.6B) ...")
            device = "cuda" if torch.cuda.is_available() else "cpu"

            self.falcon_model = AutoModelForCausalLM.from_pretrained(
                "tiiuae/falcon-perception",
                trust_remote_code=True,
                device_map={"": device},
                attn_implementation="eager",
                torch_dtype=torch.bfloat16,
            )
            self.falcon_loaded = True
            self._log("[OK] Falcon Perception loaded (~1 GB VRAM).")
            return True
        except Exception as exc:
            self._log(f"[WARN] Falcon Perception not available: {exc}")
            self._log("       Continuing with SAM3 only.")
            return False

    # -- Falcon Perception text-prompt search ---------------------------------
    def _falcon_search(
        self,
        cv_image_original: np.ndarray,
        existing_masks: list[np.ndarray],
    ) -> list[np.ndarray]:
        """
        Run Falcon Perception text-prompt search on the original image.
        Returns only NEW masks that don't overlap with existing_masks (IoU filter).
        """
        import torch
        from pycocotools import mask as mask_utils

        h, w = cv_image_original.shape[:2]
        new_masks: list[np.ndarray] = []

        # Build union of existing masks for IoU check
        existing_union = np.zeros((h, w), dtype=np.uint8)
        for m in existing_masks:
            existing_union = np.maximum(existing_union, m)

        rgb = cv2.cvtColor(cv_image_original, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        prompts = cfg.falcon_text_prompts
        self._log(f"  -> Falcon Perception search ({len(prompts)} prompts) ...")
        t_falcon = time.perf_counter()

        # Batch all prompts in a single generate call (image encoded once)
        try:
            all_preds = self.falcon_model.generate(
                [pil_img] * len(prompts),
                prompts,
                max_dimension=cfg.falcon_max_dimension,
                compile=False,
            )
        except Exception as exc:
            self._log(f"    Falcon batch error - {exc}")
            self._log(f"    Retrying prompts individually with reduced dimension (768) ...")
            all_preds = []
            for prompt in prompts:
                try:
                    preds = self.falcon_model.generate(
                        [pil_img],
                        [prompt],
                        max_dimension=768,
                        compile=False,
                    )
                    all_preds.extend(preds)
                except Exception as e:
                    self._log(f"    Falcon single prompt error '{prompt}': {e}")
                    all_preds.append([])

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for prompt, preds in zip(prompts, all_preds):
            if not preds:
                self._log(f"    Falcon '{prompt}': 0 detections")
                continue

            found_count = 0
            for p in preds:
                # Decode RLE mask to binary
                rle = p["mask_rle"]
                rle_coco = {"size": rle["size"], "counts": rle["counts"].encode("utf-8")}
                binary = mask_utils.decode(rle_coco).astype(np.uint8) * 255

                # Resize if dimensions don't match original
                if binary.shape != (h, w):
                    binary = cv2.resize(
                        binary.astype(np.float32), (w, h),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    binary = (binary > 127).astype(np.uint8) * 255

                px = int(np.count_nonzero(binary))
                if px < cfg.sam3_min_mask_px:
                    continue

                # IoU overlap check against existing masks
                intersection = int(np.count_nonzero(np.bitwise_and(binary, existing_union)))
                union_area = int(np.count_nonzero(np.bitwise_or(binary, existing_union)))
                iou = intersection / union_area if union_area > 0 else 0.0

                if iou > cfg.sam3_overlap_iou:
                    continue  # Already covered by SAM3

                self._log(f"    Falcon '{prompt}': NEW mask {px:,} px (IoU={iou:.2f})")
                padded = pad_mask(binary)
                new_masks.append(padded)
                existing_union = np.maximum(existing_union, padded)
                found_count += 1

            if found_count == 0:
                self._log(f"    Falcon '{prompt}': 0 new detections")

        dt_total = time.perf_counter() - t_falcon
        self._log(f"  Falcon Perception: {len(new_masks)} new mask(s) in {dt_total:.2f}s")

        # Free VRAM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return new_masks

    # -- Warmup Ollama (pre-load model into GPU memory) ---------------------
    def _warmup_ollama(self, model: str) -> bool:
        if self.ollama_ready:
            return True
        try:
            import ollama as ollama_client
            self._log("[LOAD] Warming up Ollama verification model ...")
            # Send a minimal request to force the model into memory
            ollama_client.chat(
                model=model,
                messages=[{"role": "user", "content": "Hi"}],
                options={"num_predict": 1},
            )
            self.ollama_ready = True
            self._log("[OK] Ollama model loaded and ready.")
            return True
        except Exception as exc:
            self._log(f"[ERROR] Ollama warmup failed: {exc}")
            return False

    # ── Ollama verification (yes/no only) ────────────────────
    def _verify_with_ollama(
        self, cv_image: np.ndarray, model: str
    ) -> bool:
        """
        Sends the (already pixelated) image to Ollama for verification.
        Returns True if Ollama still sees unblurred persons/vehicles,
        False if everything looks properly anonymized.
        Ollama does NOT return coordinates – only a yes/no answer.
        """
        import json
        import ollama as ollama_client

        orig_h, orig_w = cv_image.shape[:2]
        scale   = min(cfg.ollama_max_size / orig_w, cfg.ollama_max_size / orig_h, 1.0)
        send_w  = int(orig_w * scale)
        send_h  = int(orig_h * scale)
        send_img = cv2.resize(cv_image, (send_w, send_h), interpolation=cv2.INTER_AREA) \
                   if scale < 1.0 else cv_image

        rgb     = cv2.cvtColor(send_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        buf     = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=92)
        img_bytes = buf.getvalue()

        schema = {
            "type": "object",
            "properties": {
                "found": {
                    "type": "boolean",
                    "description": "true if any unblurred person or vehicle is still visible",
                },
            },
            "required": ["found"],
        }

        try:
            response = ollama_client.chat(
                model=model,
                messages=[{
                    "role":    "user",
                    "content": OLLAMA_VERIFY_PROMPT,
                    "images":  [img_bytes],
                }],
                format=schema,
            )
            raw  = response.message.content.strip()
            data = json.loads(raw)
            found = bool(data.get("found", False))

            if found:
                self._log("  Ollama: FOUND unblurred objects – triggering 2nd pass")
            else:
                self._log("  Ollama: all clear – no missed objects")

            return found

        except Exception as exc:
            self._log(f"  [ERROR] Ollama error: {exc}")
            # On error, assume nothing was missed (don't block pipeline)
            return False

    # -- SAM3 text-prompt search for missed objects (2nd pass) ---------------
    def _sam3_text_search(
        self,
        cv_image_original: np.ndarray,
        first_pass_masks: list[np.ndarray],
        confidence: float | None = None,
    ) -> list[np.ndarray]:
        """
        Uses SAM3 text prompts to search for ALL persons/vehicles on the
        ORIGINAL (unblurred) image.  Filters out masks that overlap with
        already-handled 1st-pass masks (IoU check) so only genuinely NEW
        detections are returned.
        """
        import torch

        # Temporarily lower confidence threshold if requested (Ollama re-pass)
        original_conf = None
        if confidence is not None and hasattr(self.sam3_proc, "confidence_threshold"):
            original_conf = self.sam3_proc.confidence_threshold
            self.sam3_proc.confidence_threshold = confidence
            self._log(f"    [conf overridden: {original_conf} → {confidence}]")

        h, w = cv_image_original.shape[:2]
        new_masks: list[np.ndarray] = []

        self._log(f"  -> SAM3 text-prompt search on original image ({len(cfg.sam3_text_prompts)} prompts) ...")
        rgb     = cv2.cvtColor(cv_image_original, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        t_sam = time.perf_counter()
        try:
            state = self.sam3_proc.set_image(pil_img)
        except Exception as exc:
            self._log(f"  [ERROR] SAM3 set_image error (2nd pass): {exc}")
            return new_masks

        dt_encode = time.perf_counter() - t_sam
        self._log(f"    SAM3 set_image (2nd pass): {dt_encode:.2f}s")

        # Combine all 1st-pass masks into a single union mask for fast overlap check
        first_pass_union = np.zeros((h, w), dtype=np.uint8)
        for m in first_pass_masks:
            first_pass_union = np.maximum(first_pass_union, m)

        for prompt in cfg.sam3_text_prompts:
            t_p = time.perf_counter()
            try:
                self.sam3_proc.reset_all_prompts(state)
                result_state = self.sam3_proc.set_text_prompt(prompt, state)
            except Exception as exc:
                self._log(f"    '{prompt}': error – {exc}")
                continue
            dt_p = time.perf_counter() - t_p

            raw_masks = result_state.get("masks")
            scores    = result_state.get("scores")

            if raw_masks is None or (hasattr(raw_masks, "__len__") and len(raw_masks) == 0):
                self._log(f"    '{prompt}': 0 detections ({dt_p:.2f}s)")
                continue

            if hasattr(raw_masks, "cpu"):
                raw_masks = raw_masks.cpu().numpy()
            if hasattr(scores, "cpu"):
                scores = scores.cpu().numpy()

            found_count = 0
            for i, m in enumerate(raw_masks):
                if m.ndim == 4:     # (1, 1, H, W)
                    m = m[0, 0]
                elif m.ndim == 3:   # (1, H, W)
                    m = m[0]
                if m.shape != (h, w):
                    m = cv2.resize(
                        m.astype(np.float32), (w, h),
                        interpolation=cv2.INTER_LINEAR,
                    )
                binary = (m > 0.5).astype(np.uint8) * 255
                px = int(np.count_nonzero(binary))
                if px < cfg.sam3_min_mask_px:
                    continue

                # Overlap check: how much of this mask was already covered?
                intersection = int(np.count_nonzero(np.bitwise_and(binary, first_pass_union)))
                union = int(np.count_nonzero(np.bitwise_or(binary, first_pass_union)))
                iou = intersection / union if union > 0 else 0.0

                if iou > cfg.sam3_overlap_iou:
                    # Already handled in 1st pass – skip
                    continue

                score_str = f" score={scores[i]:.2f}" if scores is not None and i < len(scores) else ""
                self._log(f"    '{prompt}': NEW mask {px:,} px (IoU={iou:.2f}{score_str})")
                padded = pad_mask(binary)
                new_masks.append(padded)
                # Add to union so subsequent prompts also see this as covered
                first_pass_union = np.maximum(first_pass_union, padded)
                found_count += 1

            if found_count == 0:
                self._log(f"    '{prompt}': 0 new detections ({dt_p:.2f}s)")

        dt_total = time.perf_counter() - t_sam
        self._log(f"  SAM3 2nd pass: {len(new_masks)} new mask(s) in {dt_total:.2f}s")

        # Free SAM3 state tensors to release VRAM
        del state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Restore original confidence threshold
        if original_conf is not None:
            self.sam3_proc.confidence_threshold = original_conf

        return new_masks

    # ── Video helpers (SAM3 text-prompt pipeline) ─────────────────────────────

    def _video_sam3_frame(self, frame: np.ndarray, proc=None) -> list[np.ndarray]:
        """
        Run SAM3 text-prompt segmentation on a single video frame.
        Returns a list of binary masks (uint8, 255 = object).
        Uses the same text prompts as the image safety pass.
        """
        import torch
        h, w = frame.shape[:2]
        masks: list[np.ndarray] = []

        _proc = proc if proc is not None else self.sam3_proc
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        try:
            state = _proc.set_image(pil_img)
        except Exception as exc:
            self._log(f"    [ERROR] SAM3 set_image: {exc}")
            return masks

        covered_union = np.zeros((h, w), dtype=np.uint8)

        for prompt in cfg.video_sam3_prompts:
            try:
                _proc.reset_all_prompts(state)
                rs = _proc.set_text_prompt(prompt, state)
            except Exception:
                continue
            raw = rs.get("masks")
            if raw is None or (hasattr(raw, "__len__") and len(raw) == 0):
                continue
            if hasattr(raw, "cpu"):
                raw = raw.cpu().numpy()
            for m in raw:
                if m.ndim == 4:   m = m[0, 0]
                elif m.ndim == 3: m = m[0]
                if m.shape != (h, w):
                    m = cv2.resize(m.astype(np.float32), (w, h),
                                   interpolation=cv2.INTER_LINEAR)
                binary = (m > 0.5).astype(np.uint8) * 255
                if int(np.count_nonzero(binary)) < cfg.sam3_min_mask_px:
                    continue
                # Skip if heavily overlapping with already-found masks
                inter = int(np.count_nonzero(np.bitwise_and(binary, covered_union)))
                uni   = int(np.count_nonzero(np.bitwise_or(binary, covered_union)))
                if (inter / uni if uni > 0 else 0.0) > cfg.sam3_overlap_iou:
                    continue
                padded = pad_mask(binary)
                masks.append(padded)
                covered_union = np.maximum(covered_union, padded)

        del state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return masks

    # -- Process one video file -----------------------------------------------
    def _process_video(
        self, video_path: Path, output_dir: Path,
        model: str = "", use_ollama: bool = False,
        file_idx: int = 0, total_files: int = 1,
    ) -> bool:
        import subprocess
        from concurrent.futures import ThreadPoolExecutor as _TPE
        t_total = time.perf_counter()
        self._log(f">> [VIDEO] {video_path.name}")

        out_name      = video_path.name
        out_path      = output_dir / out_name
        temp_out_path = output_dir / f"temp_{out_name}"

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self._log("  [ERROR] Could not load video.")
            return False

        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out    = cv2.VideoWriter(str(temp_out_path), fourcc, fps, (width, height))
        if not out.isOpened():
            self._log("  [ERROR] Could not create video writer.")
            cap.release()
            return False

        frames_str = str(total_frames) if total_frames > 0 else "unknown"

        # ── Auto-detect GPUs – try to load second SAM3 instance ───────────
        import torch
        sam3_instances = [self.sam3_proc]
        try:
            n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            if n_gpus >= 2 and self.sam3_proc2 is None:
                self._log("  [TURBO] Loading SAM3 on second GPU (cuda:1) ...")
                self._load_sam3_extra("cuda:1")
            if self.sam3_proc2 is not None:
                sam3_instances.append(self.sam3_proc2)
        except Exception:
            pass

        n_sam3 = len(sam3_instances)
        gpu_str = f" | {n_sam3}× SAM3 (GPU 0+1)" if n_sam3 > 1 else ""
        self._log(f"  {frames_str} frames | SAM3 every frame{gpu_str}")
        self._status(f"Pre-reading video: {video_path.name} ...")

        # ── Phase 1: Pre-read all frames into RAM ─────────────────────────
        # Pre-reading lets SAM3 on both GPUs run without waiting for disk I/O.
        # Memory limit: skip pre-read if estimated RAM > 16 GB.
        est_bytes = width * height * 3 * max(total_frames, 1)
        RAM_LIMIT = 16 * 1024 ** 3
        all_frames: list[np.ndarray] = []

        if est_bytes <= RAM_LIMIT:
            while cap.isOpened():
                ret, f = cap.read()
                if not ret:
                    break
                all_frames.append(f)
            cap.release()
            actual_total = len(all_frames)
        else:
            # Too large to pre-read; fall back to reading all frames anyway
            # (user has 192 GB RAM, so this branch rarely fires)
            self._log(f"  [INFO] Video ~{est_bytes/1024**3:.1f} GB – reading sequentially.")
            while cap.isOpened():
                ret, f = cap.read()
                if not ret:
                    break
                all_frames.append(f)
            cap.release()
            actual_total = len(all_frames)

        if self._stop_event.is_set():
            out.release()
            if temp_out_path.exists(): temp_out_path.unlink()
            return False

        # ── Phase 2: All frames are key frames ─────────────────────────
        # SAM3 runs on every frame – no skipping, no interpolation.
        kf_indices = list(range(actual_total))

        self._status(f"SAM3: processing {len(kf_indices)} frames ({n_sam3}x GPU) ...")

        kf_masks_map: dict[int, list[np.ndarray]] = {}
        kf_done = 0

        if n_sam3 > 1:
            # Dual-GPU: submit key frames alternately to both SAM3 instances
            def _sam3_worker(args):
                kf_idx, inst = args
                return kf_idx, self._video_sam3_frame(all_frames[kf_idx], proc=inst)

            with _TPE(max_workers=n_sam3) as sam_pool:
                work = [(kf_indices[i], sam3_instances[i % n_sam3])
                        for i in range(len(kf_indices))]
                for kf_idx, masks in sam_pool.map(_sam3_worker, work):
                    kf_masks_map[kf_idx] = masks
                    kf_done += 1
                    pct = kf_done / len(kf_indices)
                    interp = (file_idx + pct) / total_files
                    self._progress(interp, file_idx, total_files)
                    self._status(
                        f"SAM3: {kf_done}/{len(kf_indices)} frames "
                        f"| {video_path.name}"
                    )
        else:
            # Single GPU: sequential SAM3
            for kf_idx in kf_indices:
                if self._stop_event.is_set():
                    break
                kf_masks_map[kf_idx] = self._video_sam3_frame(all_frames[kf_idx])
                kf_done += 1
                pct = kf_done / len(kf_indices)
                interp = (file_idx + pct) / total_files
                self._progress(interp, file_idx, total_files)
                elapsed = time.perf_counter() - t_total
                if kf_done >= 3 and elapsed > 0:
                    fps_proc  = kf_done / elapsed
                    remaining = (len(kf_indices) - kf_done) / fps_proc
                    m, s      = divmod(int(remaining), 60)
                    h_val, m  = divmod(m, 60)
                    if h_val > 0:   eta = f"{h_val}h {m:02d}m"
                    elif m > 0:     eta = f"{m}m {s:02d}s"
                    else:           eta = f"{s}s"
                    self._status(
                        f"SAM3: {kf_done}/{len(kf_indices)} frames "
                        f"| ~{eta} remaining | {video_path.name}"
                    )
                else:
                    self._status(f"SAM3: {kf_done}/{len(kf_indices)} frames | {video_path.name}")

        if self._stop_event.is_set():
            out.release()
            if temp_out_path.exists(): temp_out_path.unlink()
            return False

        # ── Phase 3: all_frame_masks = kf_masks_map (every frame was segmented) ──
        all_frame_masks: list[list[np.ndarray]] = [
            kf_masks_map.get(i, []) for i in range(actual_total)
        ]

        # ── Phase 3.1: Sequential mask-drop detection & re-pass ─────────────
        # If frame i loses > 50% of the mask area from frame i-1, re-run with lower confidence.
        if actual_total > 1 and all_frames:
            self._status("Checking for missed objects (sequential re-pass) ...")
            repass_count = 0
            h, w = all_frames[0].shape[:2]
            original_conf = self.sam3_proc.confidence_threshold
            
            for i in range(1, actual_total):
                if self._stop_event.is_set():
                    break
                prev_masks = all_frame_masks[i - 1]
                curr_masks = all_frame_masks[i]
                
                if not prev_masks:
                    continue
                    
                prev_union = np.zeros((h, w), dtype=np.uint8)
                for m in prev_masks:
                    prev_union = np.maximum(prev_union, m)
                    
                prev_area = int(np.count_nonzero(prev_union))
                if prev_area == 0:
                    continue
                    
                curr_union = np.zeros((h, w), dtype=np.uint8)
                for m in curr_masks:
                    curr_union = np.maximum(curr_union, m)
                    
                intersection = int(np.count_nonzero(np.bitwise_and(prev_union, curr_union)))
                
                # If we lost more than 50% of the previous mask area
                if (intersection / prev_area) < 0.5:
                    self.sam3_proc.confidence_threshold = cfg.sam3_confidence_retry
                    new_masks = self._video_sam3_frame(all_frames[i], proc=self.sam3_proc)
                    if new_masks:
                        all_frame_masks[i] = new_masks
                        repass_count += 1
                    self.sam3_proc.confidence_threshold = original_conf
                    
            if repass_count > 0:
                self._log(f"  [RE-PASS] Recovered masks in {repass_count} frame(s) using lower confidence.")

        # ── Phase 3.5: Temporal smoothing – fill single-frame gaps ─────────
        # Prevents flickering where SAM3 misses objects on isolated frames
        # (common with non-integer FPS like 29.97 or 23.976).
        # Rule: propagate masks at most 1 frame from a REAL detection.
        # has_real_masks is a snapshot BEFORE smoothing, so propagated masks
        # can never chain beyond 1 frame.
        has_real_masks = [bool(all_frame_masks[i]) for i in range(actual_total)]
        filled_count = 0
        for i in range(actual_total):
            if all_frame_masks[i]:
                continue  # already has real masks
            prev_real = has_real_masks[i - 1] if i > 0 else False
            next_real = has_real_masks[i + 1] if i < actual_total - 1 else False
            if prev_real and next_real:
                # Gap between two real detections – carry previous masks
                all_frame_masks[i] = list(all_frame_masks[i - 1])
                filled_count += 1
            elif prev_real:
                # Extend forward by 1 frame from last real detection
                all_frame_masks[i] = list(all_frame_masks[i - 1])
                filled_count += 1
            elif next_real:
                # Extend backward by 1 frame from next real detection
                all_frame_masks[i] = list(all_frame_masks[i + 1])
                filled_count += 1

        if filled_count > 0:
            self._log(f"  [SMOOTH] Filled {filled_count} frame gap(s) via temporal smoothing.")

        self._status(f"Rendering {actual_total} frames (parallel) ...")

        # ── Phase 4: Parallel blur rendering on all CPU cores ──────────────
        n_render_workers = min(os.cpu_count() or 4, 8)

        def _render_one(args):
            frm, masks = args
            res = frm.copy()
            for mask in masks:
                res = blur_region(res, mask)
            return res

        with _TPE(max_workers=n_render_workers) as render_pool:
            rendered_frames = list(render_pool.map(
                _render_one, zip(all_frames, all_frame_masks)
            ))

        # Free RAM
        all_frames.clear()
        all_frame_masks.clear()

        # ── Phase 5: Write rendered frames in order ──────────────────────
        self._status(f"Writing output ...")
        for r in rendered_frames:
            out.write(r)

        # ── Sample frames for Ollama spot-check (before freeing RAM) ─────
        spot_frames: list[np.ndarray] = []
        if use_ollama and model and len(rendered_frames) > 0:
            n_check = min(cfg.video_spot_check_frames, len(rendered_frames))
            if n_check > 0:
                step = max(1, len(rendered_frames) // n_check)
                indices = [i * step for i in range(n_check)]
                indices = [i for i in indices if i < len(rendered_frames)]
                spot_frames = [rendered_frames[i].copy() for i in indices]
                self._log(f"  [SPOT-CHECK] Sampled {len(spot_frames)} frames for Ollama verification.")

        rendered_frames.clear()

        out.release()

        if self._stop_event.is_set():
            if temp_out_path.exists():
                temp_out_path.unlink()
            return False

        self._log("  Video processing done. Merging audio using ffmpeg...")
        try:
            subprocess.run([
                "ffmpeg", "-y",
                "-i", str(temp_out_path), "-i", str(video_path),
                "-c:v", "copy", "-c:a", "aac",
                "-map", "0:v:0", "-map", "1:a:0?",
                "-metadata", f"software=NeuralCensor {NEURALCENSOR_VERSION}",
                "-metadata", f"author={NEURALCENSOR_URL}",
                "-metadata", f"comment={NEURALCENSOR_URL}",
                "-metadata", "copyright=Processed by NeuralCensor (MIT + Commons Clause)",
                "-metadata", "description=Anonymized with NeuralCensor - persons and vehicles blurred",
                "-shortest", str(out_path),
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if temp_out_path.exists():
                temp_out_path.unlink()
            self._log("  [OK] Merged audio successfully.")
        except (subprocess.CalledProcessError, FileNotFoundError):
            self._log("  [WARN] ffmpeg failed or not found. Saving video without audio.")
            if out_path.exists():
                out_path.unlink()
            temp_out_path.rename(out_path)

        # ── Phase 6: Ollama spot-check on sampled frames ─────────────────
        spot_total = len(spot_frames)
        spot_ok = True
        spot_failed_count = 0
        if spot_frames:
            self._status(f"Ollama spot-check: 0/{spot_total} frames | {video_path.name}")
            if not self.ollama_ready:
                self._warmup_ollama(model)
            for sc_idx, sc_frame in enumerate(spot_frames):
                if self._stop_event.is_set():
                    break
                self._status(
                    f"Ollama spot-check: {sc_idx+1}/{spot_total} frames | {video_path.name}"
                )
                found = self._verify_with_ollama(sc_frame, model)
                if found:
                    spot_failed_count += 1
                    spot_ok = False
            spot_frames.clear()
            if spot_ok:
                self._log(f"  [SPOT-CHECK] \u2713 All {spot_total} frames passed Ollama verification.")
            else:
                self._log(
                    f"  [SPOT-CHECK] \u26a0 {spot_failed_count}/{spot_total} frame(s) "
                    f"may still contain unblurred objects!"
                )

        dt_total = time.perf_counter() - t_total
        self._log(f"  [DONE] Video completed in {dt_total:.2f}s")

        # ── Write report ─────────────────────────────────────────────────
        report_file = output_dir / "Anonymization_Report.txt"
        if spot_total > 0:
            if spot_ok:
                ollama_str = f"Ollama spot-check: OK ({spot_total}/{spot_total} frames clear)"
            else:
                ollama_str = f"Ollama spot-check: WARNING ({spot_failed_count}/{spot_total} frames suspicious)"
        else:
            ollama_str = "Ollama: skipped"
        with open(report_file, "a", encoding="utf-8") as f:
            f.write(
                f"[VIDEO] {video_path.name} | {dt_total:.1f}s"
                f" | {actual_total} frames"
                f" | {ollama_str}\n"
            )

        return True

    def _process_image(
        self, image_path: Path, output_dir: Path, model: str, use_ollama: bool,
        file_idx: int = 0, total_files: int = 1,
    ) -> bool:
        t_total = time.perf_counter()
        self._log(f">> {image_path.name}")

        # Normalize output filename: OpenCV cannot write .jfif → save as .jpg
        ext = image_path.suffix.lower()
        out_name = image_path.stem + (ext if ext in CV2_WRITABLE_EXTENSIONS else ".jpg")
        if out_name != image_path.name:
            self._log(f"  [INFO] Output format: {ext} → .jpg (OpenCV cannot write {ext})")

        cv_image = cv2.imread(str(image_path))
        if cv_image is None:
            self._log("  [ERROR] Could not load image.")
            return False

        # Keep the original for SAM3 text-search passes
        cv_original = cv_image.copy()

        # Stats for end-of-image summary
        sam3_pass1    = 0
        third_count   = 0
        ollama_missed = False

        if not self.sam3_loaded:
            self._log("  [ERROR] Skipping - SAM3 required but not available.")
            return False

        # -- Step 1: SAM3 text-prompt search --
        self._status(f"SAM3 text-search | {image_path.name} ({file_idx+1}/{total_files})")
        self._progress((file_idx + 0.1) / total_files, file_idx, total_files)
        t_sam1 = time.perf_counter()
        masks = self._sam3_text_search(cv_original, [])
        dt_sam1 = time.perf_counter() - t_sam1
        sam3_pass1 = len(masks)

        if not masks:
            self._log(f"  SAM3 pass 1: no objects found ({dt_sam1:.2f}s) – image unchanged.")
            cv2.imwrite(str(output_dir / out_name), cv_image)
            embed_image_metadata(output_dir / out_name)
            dt_total = time.perf_counter() - t_total
            self._log(self._summary(
                image_path.name, dt_total,
                sam3_pass1, 0, 0, False, 0,
            ))
            return True
        self._log(f"  SAM3 pass 1: {sam3_pass1} object(s) found in {dt_sam1:.2f}s")

        # -- Step 1b: Falcon Perception additional search --
        falcon_count = 0
        if self.falcon_loaded:
            self._status(f"Falcon search | {image_path.name} ({file_idx+1}/{total_files})")
            self._progress((file_idx + 0.25) / total_files, file_idx, total_files)
            falcon_masks = self._falcon_search(cv_original, masks)
            falcon_count = len(falcon_masks)
            if falcon_masks:
                masks = masks + falcon_masks
                self._log(f"  Combined: {sam3_pass1} (SAM3) + {falcon_count} (Falcon) = {len(masks)} total")

        # -- Step 2: Apply pixelation (1st pass masks) --
        self._status(f"Pixelating {len(masks)} mask(s) | {image_path.name} ({file_idx+1}/{total_files})")
        self._progress((file_idx + 0.4) / total_files, file_idx, total_files)
        t_blur = time.perf_counter()
        self._log(f"  -> Pixelating {len(masks)} mask(s), {cfg.blur_passes}x blur ...")
        result = cv_image.copy()
        for mask in masks:
            result = blur_region(result, mask)
        self._log(f"  Pixelation (pass 1): {time.perf_counter() - t_blur:.2f}s")

        # -- Step 3: Save intermediate result --
        all_masks = masks
        out_path = output_dir / out_name
        cv2.imwrite(str(out_path), result)
        embed_image_metadata(out_path)

        # -- Step 4: Ollama verification loop (max MAX_OLLAMA_PASSES re-passes) --
        ollama_pass_count = 0
        ollama_aborted    = False

        if not use_ollama:
            self._log("  [INFO] Ollama verification disabled. Skipping.")
            cv2.imwrite(str(out_path), result)
            embed_image_metadata(out_path)
            dt_total = time.perf_counter() - t_total
            self._log(self._summary(
                image_path.name, dt_total,
                sam3_pass1, 0, 0, False, falcon_count,
            ))
            # Write report entry
            report_file = output_dir / "Anonymization_Report.txt"
            with open(report_file, "a", encoding="utf-8") as f:
                total_masks = sam3_pass1 + falcon_count
                model_label = "SAM3+Falcon" if self.falcon_loaded else "SAM3"
                f.write(
                    f"{image_path.name} | {dt_total:.1f}s"
                    f" | {model_label}: {sam3_pass1 + falcon_count} masks"
                    f" | Ollama: disabled"
                    f" | Total masks: {total_masks}\n"
                )
            return True

        while True:
            t_ollama = time.perf_counter()
            check_n  = ollama_pass_count + 1
            self._status(f"Ollama verification {check_n} | {image_path.name} ({file_idx+1}/{total_files})")
            self._progress((file_idx + 0.6 + ollama_pass_count * 0.1) / total_files, file_idx, total_files)
            self._log(f"  -> Ollama verification ({check_n}) ...")
            still_visible = self._verify_with_ollama(result, model)
            self._log(f"  Ollama ({check_n}): {time.perf_counter() - t_ollama:.2f}s")

            if not still_visible:
                self._log("  [OK] Ollama: all clear.")
                break

            # Ollama found something
            ollama_missed = True

            if ollama_pass_count >= cfg.max_ollama_passes:
                # Reached max re-passes – save what we have and move on
                self._log(
                    f"  [WARN] Ollama still detecting after {cfg.max_ollama_passes} "
                    f"re-pass(es) – saving current result and continuing."
                )
                ollama_aborted = True
                break

            # Trigger another SAM3 pass with lower confidence
            ollama_pass_count += 1
            self._log(
                f"  [WARN] Ollama found missed objects – "
                f"SAM3 re-pass {ollama_pass_count}/{cfg.max_ollama_passes} "
                f"(conf={cfg.sam3_confidence_retry}) ..."
            )
            t_refine  = time.perf_counter()
            new_masks = self._sam3_text_search(
                cv_original, all_masks, confidence=cfg.sam3_confidence_retry
            )
            dt_refine = time.perf_counter() - t_refine
            n_new     = len(new_masks)
            third_count += n_new
            if new_masks:
                for mask in new_masks:
                    result = blur_region(result, mask)
                all_masks = all_masks + new_masks
                self._log(f"  Re-pass {ollama_pass_count}: {n_new} new mask(s) ({dt_refine:.2f}s).")
            else:
                self._log(
                    f"  Re-pass {ollama_pass_count}: nothing new found – "
                    f"Ollama false positive ({dt_refine:.2f}s)."
                )
                break  # No new masks found – no point looping further

        # -- Step 6: Save final image --
        cv2.imwrite(str(out_path), result)
        embed_image_metadata(out_path)
        dt_total = time.perf_counter() - t_total

        # -- End-of-image summary --
        self._log(self._summary(
            image_path.name, dt_total,
            sam3_pass1, ollama_pass_count, third_count, ollama_aborted,
            falcon_count,
        ))

        # -- Report file --
        total_masks = sam3_pass1 + falcon_count + third_count
        report_file = output_dir / "Anonymization_Report.txt"
        with open(report_file, "a", encoding="utf-8") as f:
            if not ollama_missed:
                ollama_str = "Ollama: OK"
            elif ollama_aborted:
                ollama_str = f"Ollama: ABORTED after {ollama_pass_count} re-pass(es) (still detecting)"
            else:
                ollama_str = f"Ollama: {ollama_pass_count} re-pass(es) -> +{third_count} new"
            
            model_label = "SAM3+Falcon" if self.falcon_loaded else "SAM3"
            f.write(
                f"{image_path.name} | {dt_total:.1f}s"
                f" | {model_label}: {sam3_pass1 + falcon_count} masks"
                f" | {ollama_str}"
                f" | Total masks: {total_masks}\n"
            )

        return True

    # -- End-of-image summary block -----------------------------------------
    @staticmethod
    def _summary(
        filename: str,
        dt: float,
        sam3_pass1: int,
        ollama_pass_count: int,
        third_count: int,
        ollama_aborted: bool,
        falcon_count: int = 0,
    ) -> str:
        """
        Returns a formatted summary block for the processing log.
        Example:
          ┌─ photo.jpg ──────────────────────── 22.5s ─┐
          │  SAM3:    27 objects (text-search)           │
          │  Falcon:  +5 new objects                     │
          │  Ollama:  1 re-pass -> +5 new objects       │
          │  Total:   32 masked regions saved            │
          └────────────────────────────────────────────┘
        """
        width = 54
        def row(label: str, value: str) -> str:
            content = f"  {label:<10}{value}"
            pad = width - 2 - len(content)
            return f"\u2502{content}{' ' * max(pad, 0)}\u2502"

        total_masks = sam3_pass1 + falcon_count + third_count

        time_str = f"{dt:.1f}s"
        title    = f" {filename} "
        dashes   = width - 2 - len(title) - len(time_str) - 1
        header   = f"\u250c\u2500{title}{'\u2500' * max(dashes, 2)} {time_str} \u2500\u2510"

        lines = [header]
        lines.append(row("SAM3:", f"{sam3_pass1} objects (text-search)"))

        if falcon_count > 0:
            lines.append(row("Falcon:", f"+{falcon_count} new objects"))

        if ollama_pass_count == 0:
            lines.append(row("Ollama:", "all clear"))
        elif ollama_aborted:
            lines.append(row(
                "Ollama:",
                f"{ollama_pass_count} re-pass(es) -> still detecting  ABORTED",
            ))
        else:
            suffix = f"+{third_count} new objects" if third_count > 0 else "nothing new (false positive)"
            passes_str = f"{ollama_pass_count} re-pass" + ("es" if ollama_pass_count > 1 else "")
            lines.append(row("Ollama:", f"{passes_str} -> {suffix}"))

        lines.append(row("Total:", f"{total_masks} masked regions saved"))
        lines.append("\u2514" + "\u2500" * width + "\u2518")
        return "\n".join(lines)

    # -- Main processing loop -----------------------------------------------
    def run(
        self,
        image_paths: list[Path],
        output_dir: Path | None,
        model: str,
        use_ollama: bool,
        video_quality: str = "",
        output_dir_map: dict[Path, Path] | None = None,
    ):
        self._stop_event.clear()
        total = len(image_paths)
        mode = "subfolder mode" if output_dir_map else "single output folder"
        self._log(f"Starting processing: {total} image(s) | {mode} | Verification model: {model}")

        has_images = any(p.suffix.lower() in IMAGE_EXTENSIONS for p in image_paths)
        has_videos = any(p.suffix.lower() in VIDEO_EXTENSIONS for p in image_paths)

        # Load SAM3 (required for both images and video)
        if not self._load_sam3():
            self._log("[ERROR] SAM3 could not be loaded. Aborting.")
            self._log("  -> Please run start.bat to install SAM3 and download the checkpoint.")
            self._done(success=False)
            return

        # Try to load Falcon Perception (non-critical – continues without if unavailable)
        self._load_falcon()

        # Warmup Ollama if enabled (used for images + video spot-check)
        if use_ollama:
            if not self._warmup_ollama(model):
                self._log("[WARN] Ollama warmup failed - verification may be slow.")

        if has_images:
            falcon_str = " + Falcon" if self.falcon_loaded else ""
            if use_ollama:
                self._log(f"Pipeline (images): SAM3 text-search{falcon_str} \u2192 pixelation \u2192 Ollama")
            else:
                self._log(f"Pipeline (images): SAM3 text-search{falcon_str} \u2192 pixelation")
        if has_videos:
            if use_ollama:
                self._log("Pipeline (videos): SAM3 text-prompt \u2192 pixelation \u2192 Ollama spot-check")
            else:
                self._log("Pipeline (videos): SAM3 text-prompt \u2192 pixelation")

        success_count = 0
        for idx, img_path in enumerate(image_paths):
            if self._stop_event.is_set():
                self._log("Processing cancelled by user.")
                break

            # Resolve output dir: per-image map takes priority over global output_dir
            img_output_dir = (
                output_dir_map[img_path]
                if output_dir_map and img_path in output_dir_map
                else output_dir
            )
            # Ensure output dir exists (subfolder mode creates on demand)
            img_output_dir.mkdir(parents=True, exist_ok=True)

            self._progress(idx / total, idx, total)
            # Update status to show which file is being processed
            is_video = img_path.suffix.lower() in VIDEO_EXTENSIONS
            file_type = "[VIDEO]" if is_video else "[IMAGE]"
            self._status(f"{file_type} {img_path.name} ({idx+1}/{total})")

            if is_video:
                ok = self._process_video(
                    img_path, img_output_dir,
                    model=model, use_ollama=use_ollama,
                    file_idx=idx, total_files=total,
                )
            else:
                ok = self._process_image(
                    img_path, img_output_dir, model, use_ollama,
                    file_idx=idx, total_files=total,
                )
            if ok:
                success_count += 1

            # Free VRAM between images
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self._progress(1.0, total, total)
        self._log(f"\nDone: {success_count}/{total} image(s) anonymized successfully.")
        self._done(success=True)



# ──────────────────────────────────────────────────────────────
# GUI class
# ──────────────────────────────────────────────────────────────

class NeuralCensorApp(ctk.CTk):

    MODEL_OPTIONS = ["gemma4:e4b", "gemma4:12b", "gemma4:26b", "gemma4:31b"]

    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.title("NeuralCensor – Image Anonymization")
        self.geometry("860x700")
        self.minsize(760, 600)
        self.configure(fg_color="#0d0d1a")

        self._input_paths: list[Path] = []
        self._output_dir: Path | None = None
        self._proc_thread: threading.Thread | None = None
        self._processor: Processor | None = None
        self._msg_queue: queue.Queue = queue.Queue()
        self._output_dir_map: dict[Path, Path] | None = None  # subfolder mode: maps each image to its output dir
        self._n_images: int = 0   # count of image files in current batch
        self._n_videos: int = 0   # count of video files in current batch

        self._build_ui()
        self._poll_queue()

        # Clean exit when window is closed
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI construction ─────────────────────────────────────
    def _build_ui(self):
        # Header
        header = ctk.CTkFrame(self, fg_color="#0f3460", corner_radius=0)
        header.pack(fill="x", padx=0, pady=0)

        ctk.CTkLabel(
            header,
            text="🛡 NeuralCensor",
            font=ctk.CTkFont(family="Segoe UI", size=28, weight="bold"),
            text_color="#e94560",
        ).pack(side="left", padx=24, pady=16)

        ctk.CTkLabel(
            header,
            text="Automatic Image Anonymization",
            font=ctk.CTkFont(family="Segoe UI", size=13),
            text_color="#a0a0b0",
        ).pack(side="left", padx=0, pady=16)

        # Gear settings button (right side of header)
        ctk.CTkButton(
            header,
            text="\u2699",
            font=ctk.CTkFont(family="Segoe UI", size=20),
            fg_color="transparent",
            hover_color="#1a4a7a",
            text_color="#a0a0b0",
            corner_radius=8,
            width=44, height=44,
            command=self._show_settings_dialog,
        ).pack(side="right", padx=16, pady=10)

        # Main container
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=24, pady=16)

        # Left column: settings
        left = ctk.CTkFrame(main, fg_color="#16213e", corner_radius=12)
        left.pack(side="left", fill="y", padx=(0, 12), pady=0)
        left.pack_propagate(False)
        left.configure(width=280)

        ctk.CTkLabel(
            left,
            text="SETTINGS",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color="#a0a0b0",
        ).pack(anchor="w", padx=16, pady=(16, 4))

        self._build_separator(left)

        # Verification model
        ctk.CTkLabel(left, text="Verification Model (Ollama)",
                     font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", padx=16, pady=(0, 4))

        self.model_var = ctk.StringVar(value="gemma4:e4b")
        ctk.CTkOptionMenu(
            left,
            values=self.MODEL_OPTIONS,
            variable=self.model_var,
            fg_color="#0f3460",
            button_color="#e94560",
            button_hover_color="#c73652",
            dropdown_fg_color="#1a1a2e",
            font=ctk.CTkFont(family="Segoe UI", size=13),
            width=248,
        ).pack(padx=16, pady=(0, 14))

        self.use_ollama_var = ctk.BooleanVar(value=True)
        self._ollama_switch = ctk.CTkSwitch(
            left,
            text="Enable Ollama",
            variable=self.use_ollama_var,
            font=ctk.CTkFont(family="Segoe UI", size=12),
            progress_color="#e94560",
            button_color="#ffffff",
            button_hover_color="#eaeaea"
        )
        self._ollama_switch.pack(anchor="w", padx=16, pady=(0, 10))

        self._frame_skip_panel = None  # removed – SAM3 processes every frame

        self._build_separator(left)

        # Input – single browse button with folder/files choice
        ctk.CTkLabel(left, text="INPUT",
                     font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
                     text_color="#a0a0b0").pack(anchor="w", padx=16, pady=(14, 4))

        self._make_button(left, "📂  Browse Input", self._browse_input_menu)

        self.lbl_input = ctk.CTkLabel(left, text="No path selected",
                                      font=ctk.CTkFont(size=11), text_color="#a0a0b0",
                                      wraplength=240, justify="left")
        self.lbl_input.pack(anchor="w", padx=16, pady=(2, 12))

        self._build_separator(left)

        # Output
        ctk.CTkLabel(left, text="OUTPUT",
                     font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
                     text_color="#a0a0b0").pack(anchor="w", padx=16, pady=(14, 4))

        self._make_button(left, "💾  Select Output Folder", self._choose_output_folder)

        self.lbl_output = ctk.CTkLabel(left, text="No path selected",
                                       font=ctk.CTkFont(size=11), text_color="#a0a0b0",
                                       wraplength=240, justify="left")
        self.lbl_output.pack(anchor="w", padx=16, pady=(2, 12))

        # Right column: log
        right = ctk.CTkFrame(main, fg_color="#16213e", corner_radius=12)
        right.pack(side="left", fill="both", expand=True)

        ctk.CTkLabel(right, text="PROCESSING LOG",
                     font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
                     text_color="#a0a0b0").pack(anchor="w", padx=16, pady=(16, 4))

        self._build_separator(right)

        self.log_box = ctk.CTkTextbox(
            right, fg_color="#0d0d1a", text_color="#c0c0d0",
            font=ctk.CTkFont(family="Consolas", size=12),
            corner_radius=8, border_width=1, border_color="#0f3460",
        )
        self.log_box.pack(fill="both", expand=True, padx=12, pady=8)
        self.log_box.configure(state="disabled")

        # Progress
        prog_frame = ctk.CTkFrame(right, fg_color="transparent")
        prog_frame.pack(fill="x", padx=12, pady=(0, 8))

        self.lbl_status = ctk.CTkLabel(prog_frame, text="Ready",
                                       font=ctk.CTkFont(size=12), text_color="#a0a0b0")
        self.lbl_status.pack(anchor="w")

        self.progress_bar = ctk.CTkProgressBar(prog_frame, fg_color="#0f3460",
                                               progress_color="#e94560", corner_radius=4, height=10)
        self.progress_bar.set(0)
        self.progress_bar.pack(fill="x", pady=(4, 0))

        # Bottom bar
        bar = ctk.CTkFrame(self, fg_color="#0f3460", corner_radius=0)
        bar.pack(fill="x", padx=0, pady=0, side="bottom")

        self.btn_start = ctk.CTkButton(
            bar, text="▶  Start Processing",
            font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
            fg_color="#e94560", hover_color="#c73652",
            corner_radius=8, height=42, command=self._start_processing,
        )
        self.btn_start.pack(side="left", padx=16, pady=10)

        self.btn_stop = ctk.CTkButton(
            bar, text="⏹  Cancel",
            font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
            fg_color="#374151", hover_color="#4b5563",
            corner_radius=8, height=42, command=self._stop_processing, state="disabled",
        )
        self.btn_stop.pack(side="left", padx=(0, 16), pady=10)

        self.lbl_counter = ctk.CTkLabel(bar, text="",
                                        font=ctk.CTkFont(size=12), text_color="#eaeaea")
        self.lbl_counter.pack(side="right", padx=16)

        # Info button
        btn_info = ctk.CTkButton(
            bar, text="ℹ  Info",
            font=ctk.CTkFont(family="Segoe UI", size=13),
            fg_color="#374151", hover_color="#4b5563",
            corner_radius=8, height=42, width=80, command=self._show_info_dialog,
        )
        btn_info.pack(side="right", padx=(0, 8), pady=10)

    # ── UI helpers ──────────────────────────────────────────
    def _make_button(self, parent, text: str, command) -> ctk.CTkButton:
        btn = ctk.CTkButton(
            parent, text=text, font=ctk.CTkFont(family="Segoe UI", size=13),
            fg_color="#0f3460", hover_color="#1a4a7a",
            corner_radius=8, height=36, anchor="w", command=command, width=248,
        )
        btn.pack(padx=16, pady=(0, 6))
        return btn

    def _build_separator(self, parent):
        ctk.CTkFrame(parent, height=1, fg_color="#0f3460").pack(fill="x", padx=16, pady=2)

    def _show_settings_dialog(self):
        """Opens the settings dialog where all pipeline constants can be adjusted."""
        dialog = ctk.CTkToplevel(self)
        dialog.title("NeuralCensor – Pipeline Settings")
        dialog.geometry("620x700")
        dialog.configure(fg_color="#0d0d1a")
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.grab_set()

        dialog.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - 620) // 2
        y = self.winfo_y() + (self.winfo_height() - 700) // 2
        dialog.geometry(f"+{x}+{y}")

        ctk.CTkLabel(
            dialog,
            text="Pipeline Settings",
            font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold"),
            text_color="#e94560",
        ).pack(padx=24, pady=(20, 2))

        ctk.CTkLabel(
            dialog,
            text="Values take effect on the next processing run. Code defaults are shown in grey.",
            font=ctk.CTkFont(size=11),
            text_color="#a0a0b0",
            wraplength=580,
        ).pack(padx=24, pady=(0, 12))

        tabs = ctk.CTkTabview(
            dialog,
            fg_color="#16213e",
            segmented_button_fg_color="#0f3460",
            segmented_button_selected_color="#e94560",
            segmented_button_selected_hover_color="#c73652",
            segmented_button_unselected_hover_color="#1a4a7a",
            text_color="#eaeaea",
        )
        tabs.pack(fill="both", expand=True, padx=16, pady=(0, 8))

        tab_blur   = tabs.add("Blur")
        tab_sam3   = tabs.add("SAM3")
        tab_falcon = tabs.add("Falcon")
        tab_ollama = tabs.add("Ollama")

        entries: dict = {}

        def add_row(parent, key, label, default, description):
            frame = ctk.CTkFrame(parent, fg_color="#1a1a2e", corner_radius=8)
            frame.pack(fill="x", padx=8, pady=4)
            ctk.CTkLabel(frame, text=label,
                font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
                text_color="#eaeaea").pack(anchor="w", padx=12, pady=(8, 0))
            ctk.CTkLabel(frame, text=description,
                font=ctk.CTkFont(size=11), text_color="#a0a0b0").pack(anchor="w", padx=12)
            row2 = ctk.CTkFrame(frame, fg_color="transparent")
            row2.pack(fill="x", padx=12, pady=(2, 8))
            var = ctk.StringVar(value=str(getattr(cfg, key)))
            entries[key] = var
            ctk.CTkEntry(row2, textvariable=var, fg_color="#0d0d1a", text_color="#eaeaea",
                border_color="#0f3460", border_width=1,
                font=ctk.CTkFont(family="Consolas", size=13),
                width=140, height=32).pack(side="left")
            ctk.CTkLabel(row2, text=f"Default: {default}",
                font=ctk.CTkFont(size=11), text_color="#555577").pack(side="left", padx=(12, 0))

        def add_prompt_row(parent, key, label, default_list):
            frame = ctk.CTkFrame(parent, fg_color="#1a1a2e", corner_radius=8)
            frame.pack(fill="x", padx=8, pady=4)
            ctk.CTkLabel(frame, text=label,
                font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
                text_color="#eaeaea").pack(anchor="w", padx=12, pady=(8, 0))
            ctk.CTkLabel(frame, text="Comma-separated list of detection prompts",
                font=ctk.CTkFont(size=11), text_color="#a0a0b0").pack(anchor="w", padx=12)
            row2 = ctk.CTkFrame(frame, fg_color="transparent")
            row2.pack(fill="x", padx=12, pady=(2, 8))
            var = ctk.StringVar(value=", ".join(getattr(cfg, key)))
            entries[key] = var
            ctk.CTkEntry(row2, textvariable=var, fg_color="#0d0d1a", text_color="#eaeaea",
                border_color="#0f3460", border_width=1,
                font=ctk.CTkFont(family="Consolas", size=13),
                width=340, height=32).pack(side="left")
            default_str = ", ".join(default_list)
            ctk.CTkLabel(row2, text=f"Default: {default_str}",
                font=ctk.CTkFont(size=11), text_color="#555577").pack(side="left", padx=(12, 0))

        # Blur tab
        sf_blur = ctk.CTkScrollableFrame(tab_blur, fg_color="transparent")
        sf_blur.pack(fill="both", expand=True)
        add_row(sf_blur, "blur_kernel_base",  "Blur Kernel Base",  BLUR_KERNEL_BASE,
                "Gaussian blur kernel size (odd number, larger = stronger blur)")
        add_row(sf_blur, "blur_passes",        "Blur Passes",       BLUR_PASSES,
                "Number of blur passes per mask region")
        add_row(sf_blur, "quantize_step",      "Quantize Step",     QUANTIZE_STEP,
                "Pixel quantization grid step after blur (prevents frequency reconstruction)")
        add_row(sf_blur, "padding_fraction",   "Padding Fraction",  PADDING_FRACTION,
                "Mask edge padding as fraction of mask size (e.g. 0.04 = 4 percent)")

        # SAM3 tab
        sf_sam3 = ctk.CTkScrollableFrame(tab_sam3, fg_color="transparent")
        sf_sam3.pack(fill="both", expand=True)
        add_row(sf_sam3, "sam3_confidence",       "Confidence",            SAM3_CONFIDENCE,
                "Detection confidence threshold (lower = more detections)")
        add_row(sf_sam3, "sam3_confidence_retry", "Confidence (Retry)",    SAM3_CONFIDENCE_RETRY,
                "Lower confidence used for Ollama-triggered re-passes")
        add_row(sf_sam3, "sam3_min_mask_px",      "Min Mask Pixels",       SAM3_MIN_MASK_PX,
                "Masks smaller than this pixel count are discarded")
        add_row(sf_sam3, "sam3_overlap_iou",      "Overlap IoU Threshold", SAM3_OVERLAP_IOU,
                "Masks overlapping existing ones above this ratio are skipped (0.0-1.0)")
        add_prompt_row(sf_sam3, "sam3_text_prompts",  "Image SAM3 Prompts",  SAM3_TEXT_PROMPTS)
        add_prompt_row(sf_sam3, "video_sam3_prompts", "Video SAM3 Prompts",  VIDEO_SAM3_PROMPTS)

        # Ollama tab
        sf_ollama = ctk.CTkScrollableFrame(tab_ollama, fg_color="transparent")
        sf_ollama.pack(fill="both", expand=True)
        add_row(sf_ollama, "max_ollama_passes",        "Max Re-passes",       MAX_OLLAMA_PASSES,
                "Max Ollama-triggered SAM3 re-passes per image before giving up")
        add_row(sf_ollama, "ollama_max_size",           "Max Send Size (px)",  OLLAMA_MAX_SIZE,
                "Maximum image edge length sent to Ollama (larger = more detail, slower)")
        add_row(sf_ollama, "video_spot_check_frames",   "Spot-Check Frames",   VIDEO_SPOT_CHECK_FRAMES,
                "Frames sampled from rendered video for Ollama verification")

        # Falcon tab
        sf_falcon = ctk.CTkScrollableFrame(tab_falcon, fg_color="transparent")
        sf_falcon.pack(fill="both", expand=True)

        falcon_frame = ctk.CTkFrame(sf_falcon, fg_color="#1a1a2e", corner_radius=8)
        falcon_frame.pack(fill="x", padx=8, pady=4)
        ctk.CTkLabel(falcon_frame, text="Falcon Perception",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color="#eaeaea").pack(anchor="w", padx=12, pady=(8, 0))
        ctk.CTkLabel(falcon_frame, text="Additional AI detector for higher detection coverage (~1 GB VRAM)",
            font=ctk.CTkFont(size=11), text_color="#a0a0b0").pack(anchor="w", padx=12)

        falcon_var = ctk.BooleanVar(value=cfg.falcon_enabled)
        entries["falcon_enabled"] = falcon_var
        falcon_row = ctk.CTkFrame(falcon_frame, fg_color="transparent")
        falcon_row.pack(fill="x", padx=12, pady=(2, 8))
        ctk.CTkSwitch(
            falcon_row,
            text="Enable Falcon Perception",
            variable=falcon_var,
            font=ctk.CTkFont(family="Segoe UI", size=12),
            progress_color="#e94560",
            button_color="#ffffff",
            button_hover_color="#eaeaea",
        ).pack(side="left")

        add_prompt_row(sf_falcon, "falcon_text_prompts", "Falcon Text Prompts", FALCON_TEXT_PROMPTS)
        add_row(sf_falcon, "falcon_max_dimension", "Max Dimension", FALCON_MAX_DIMENSION,
                "Maximum image edge length for Falcon inference (larger = more detail, slower)")

        def _apply():
            errors = []
            for key, var in entries.items():
                raw = var.get().strip()
                if key in ("sam3_text_prompts", "video_sam3_prompts", "falcon_text_prompts"):
                    parsed = [p.strip() for p in raw.split(",") if p.strip()]
                    if not parsed:
                        errors.append(f"{key}: at least one prompt required")
                        continue
                    setattr(cfg, key, parsed)
                    continue
                if key == "falcon_enabled":
                    setattr(cfg, key, var.get())
                    continue
                attr_default = getattr(_Cfg, key)
                try:
                    if isinstance(attr_default, float):
                        setattr(cfg, key, float(raw))
                    else:
                        setattr(cfg, key, int(raw))
                except ValueError:
                    errors.append(f"{key}: '{raw}' is not a valid number")
            if errors:
                err_text.configure(text="Errors:\n" + "\n".join(errors), text_color="#f87171")
            else:
                err_text.configure(text="Settings applied.", text_color="#4ade80")
                dialog.after(700, dialog.destroy)

        def _reset():
            cfg.reset()
            for key, var in entries.items():
                if key in ("sam3_text_prompts", "video_sam3_prompts", "falcon_text_prompts"):
                    var.set(", ".join(getattr(cfg, key)))
                elif key == "falcon_enabled":
                    var.set(getattr(cfg, key))
                else:
                    var.set(str(getattr(cfg, key)))
            err_text.configure(text="Reset to defaults.", text_color="#fbbf24")

        btn_row = ctk.CTkFrame(dialog, fg_color="transparent")
        btn_row.pack(fill="x", padx=16, pady=(0, 4))

        ctk.CTkButton(btn_row, text="Apply", width=120, height=36,
            fg_color="#e94560", hover_color="#c73652",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            command=_apply).pack(side="left", padx=(0, 8))

        ctk.CTkButton(btn_row, text="Reset to Defaults", width=140, height=36,
            fg_color="#374151", hover_color="#4b5563",
            font=ctk.CTkFont(family="Segoe UI", size=13),
            command=_reset).pack(side="left", padx=(0, 8))

        ctk.CTkButton(btn_row, text="Cancel", width=90, height=36,
            fg_color="#374151", hover_color="#4b5563",
            font=ctk.CTkFont(family="Segoe UI", size=13),
            command=dialog.destroy).pack(side="left")

        err_text = ctk.CTkLabel(dialog, text="",
            font=ctk.CTkFont(size=11), text_color="#f87171", wraplength=580)
        err_text.pack(padx=16, pady=(0, 12))

    def _show_info_dialog(self):
        """Shows a dialog listing all technologies used with clickable links."""
        import webbrowser

        dialog = ctk.CTkToplevel(self)
        dialog.title("NeuralCensor – Technologies")
        dialog.geometry("520x480")
        dialog.configure(fg_color="#0d0d1a")
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.grab_set()

        # Center on parent
        dialog.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - 520) // 2
        y = self.winfo_y() + (self.winfo_height() - 480) // 2
        dialog.geometry(f"+{x}+{y}")

        # Header
        ctk.CTkLabel(
            dialog, text="Technologies Used",
            font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold"),
            text_color="#e94560",
        ).pack(padx=24, pady=(20, 4))

        ctk.CTkLabel(
            dialog, text="NeuralCensor is built on the following open-source projects:",
            font=ctk.CTkFont(size=12), text_color="#a0a0b0", wraplength=480,
        ).pack(padx=24, pady=(0, 16))

        # Technology list
        technologies = [
            ("SAM 3 (Segment Anything 3)", "Object Detection and Pixel-precise Mask Generation",
             "https://github.com/facebookresearch/sam3"),
            ("Ollama", "AI Verification of Anonymization Quality",
             "https://ollama.com"),
            ("Gemma 4", "Vision Language Model for Verification",
             "https://ai.google.dev/gemma"),
            ("OpenCV", "Image Processing and Multi-Pass Pixelation",
             "https://opencv.org"),
            ("CustomTkinter", "Modern Desktop User Interface",
             "https://github.com/TomSchimansky/CustomTkinter"),
            ("PyTorch", "Deep Learning Framework (GPU Acceleration)",
             "https://pytorch.org"),
            ("HuggingFace", "Model Hosting and Distribution",
             "https://huggingface.co"),
        ]

        container = ctk.CTkScrollableFrame(
            dialog, fg_color="#16213e", corner_radius=10,
        )
        container.pack(fill="both", expand=True, padx=20, pady=(0, 12))

        for name, desc, url in technologies:
            row = ctk.CTkFrame(container, fg_color="transparent")
            row.pack(fill="x", padx=8, pady=4)

            ctk.CTkLabel(
                row, text=name,
                font=ctk.CTkFont(size=13, weight="bold"), text_color="#eaeaea",
            ).pack(anchor="w")

            ctk.CTkLabel(
                row, text=desc,
                font=ctk.CTkFont(size=11), text_color="#a0a0b0",
            ).pack(anchor="w")

            link = ctk.CTkLabel(
                row, text=url,
                font=ctk.CTkFont(size=11), text_color="#4a9eff",
                cursor="hand2",
            )
            link.pack(anchor="w")
            link.bind("<Button-1>", lambda e, u=url: webbrowser.open(u))

        # Close button
        ctk.CTkButton(
            dialog, text="Close", width=120, height=36,
            fg_color="#e94560", hover_color="#c73652",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            command=dialog.destroy,
        ).pack(pady=(8, 16))

    def _log(self, text: str):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", text + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _clear_log(self):
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

    # ── File/folder selection ───────────────────────────────
    def _browse_input_menu(self):
        """
        Shows a small popup letting the user choose between
        'Select Folder' and 'Select Files'.
        """
        popup = ctk.CTkToplevel(self)
        popup.title("Select Input")
        popup.geometry("300x130")
        popup.configure(fg_color="#16213e")
        popup.resizable(False, False)
        popup.transient(self)
        popup.grab_set()

        # Center on parent
        popup.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - 300) // 2
        y = self.winfo_y() + (self.winfo_height() - 130) // 2
        popup.geometry(f"+{x}+{y}")

        ctk.CTkLabel(
            popup, text="What would you like to select?",
            font=ctk.CTkFont(family="Segoe UI", size=13),
            text_color="#eaeaea",
        ).pack(pady=(16, 12))

        btn_frame = ctk.CTkFrame(popup, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20)

        def pick_folder():
            popup.destroy()
            self._browse_input_folder()

        def pick_files():
            popup.destroy()
            self._browse_input_files()

        ctk.CTkButton(
            btn_frame, text="📂  Folder",
            font=ctk.CTkFont(family="Segoe UI", size=13),
            fg_color="#0f3460", hover_color="#1a4a7a",
            corner_radius=8, height=36, width=120,
            command=pick_folder,
        ).pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            btn_frame, text="📎  Files",
            font=ctk.CTkFont(family="Segoe UI", size=13),
            fg_color="#0f3460", hover_color="#1a4a7a",
            corner_radius=8, height=36, width=120,
            command=pick_files,
        ).pack(side="left")

    def _browse_input_files(self):
        """
        Opens a file dialog to select individual image/video files.
        Output dir defaults to the parent folder of the first file.
        """
        ext_img = " ".join(f"*{e}" for e in sorted(IMAGE_EXTENSIONS))
        ext_vid = " ".join(f"*{e}" for e in sorted(VIDEO_EXTENSIONS))
        ext_all = " ".join(f"*{e}" for e in sorted(SUPPORTED_EXTENSIONS))

        files = filedialog.askopenfilenames(
            title="Select Images and Videos",
            filetypes=[
                ("All Supported", ext_all),
                ("Images", ext_img),
                ("Videos", ext_vid),
            ],
        )
        if not files:
            return

        paths = sorted(Path(f) for f in files)

        # Reset state
        self._output_dir     = None
        self._output_dir_map = None
        self._input_paths    = paths

        n_img = sum(1 for p in paths if p.suffix.lower() in IMAGE_EXTENSIONS)
        n_vid = sum(1 for p in paths if p.suffix.lower() in VIDEO_EXTENSIONS)
        parts = []
        if n_img: parts.append(f"{n_img} image{'s' if n_img != 1 else ''}")
        if n_vid: parts.append(f"{n_vid} video{'s' if n_vid != 1 else ''}")

        # Show first filename + count
        if len(paths) == 1:
            display = paths[0].name
        else:
            display = f"{paths[0].name} + {len(paths)-1} more"
        self.lbl_input.configure(
            text=f"{display} ({', '.join(parts)})",
            text_color="#4ade80",
        )

        # Auto output: parent of first file
        auto_out = paths[0].parent / AUTO_OUTPUT_NAME
        self.lbl_output.configure(
            text=f"Auto: {auto_out.parent.name}/{AUTO_OUTPUT_NAME}/",
            text_color="#fbbf24",
        )

        # Show/hide video quality panel
        has_video = any(p.suffix.lower() in VIDEO_EXTENSIONS for p in self._input_paths)
        self._update_frame_skip_visibility(has_video)

    def _browse_input_folder(self):
        """
        Opens a folder dialog.  Automatically detects the structure:
          • Folder has images directly        → flat mode
          • Folder has subfolders with images  → subfolder mode
          • Both                               → combines both
          • No images anywhere                 → error label
        """
        folder = filedialog.askdirectory(title="Select Folder with Images and Videos")
        if not folder:
            return

        try:
            root = Path(folder)

            # 1. Collect images directly in the selected folder
            root_images: list[Path] = []
            for f in root.iterdir():
                try:
                    if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                        root_images.append(f)
                except (OSError, PermissionError):
                    continue
            root_images.sort()

            # 2. Collect images inside immediate subdirectories
            output_dir_map: dict[Path, Path] = {}
            sub_images: list[Path] = []
            subdirs_found = 0
            for sub in sorted(root.iterdir()):
                try:
                    if not sub.is_dir() or sub.name == AUTO_OUTPUT_NAME:
                        continue
                except (OSError, PermissionError):
                    continue
                images: list[Path] = []
                try:
                    for f in sub.iterdir():
                        try:
                            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                                images.append(f)
                        except (OSError, PermissionError):
                            continue
                except (OSError, PermissionError):
                    continue
                images.sort()
                if not images:
                    continue
                subdirs_found += 1
                out_dir = sub / AUTO_OUTPUT_NAME
                for img in images:
                    output_dir_map[img] = out_dir
                sub_images.extend(images)

            # 3. Determine mode based on what was found
            has_root = len(root_images) > 0
            has_subs = len(sub_images) > 0

            if not has_root and not has_subs:
                self.lbl_input.configure(
                    text=f"{root.name}/ – no images found",
                    text_color="#f87171",
                )
                return

            # Reset state
            self._output_dir     = None
            self._output_dir_map = None
            self._input_paths    = []

            if has_root and not has_subs:
                self._input_paths = root_images
                n_img = sum(1 for p in root_images if p.suffix.lower() in IMAGE_EXTENSIONS)
                n_vid = sum(1 for p in root_images if p.suffix.lower() in VIDEO_EXTENSIONS)
                parts = []
                if n_img: parts.append(f"{n_img} image{'s' if n_img != 1 else ''}")
                if n_vid: parts.append(f"{n_vid} video{'s' if n_vid != 1 else ''}")
                self.lbl_input.configure(
                    text=f"{root.name}/ ({', '.join(parts)})",
                    text_color="#4ade80",
                )
                self.lbl_output.configure(
                    text=f"Auto: {root.name}/{AUTO_OUTPUT_NAME}/",
                    text_color="#fbbf24",
                )

            elif has_subs and not has_root:
                self._input_paths    = sub_images
                self._output_dir_map = output_dir_map
                n_img = sum(1 for p in sub_images if p.suffix.lower() in IMAGE_EXTENSIONS)
                n_vid = sum(1 for p in sub_images if p.suffix.lower() in VIDEO_EXTENSIONS)
                parts = []
                if n_img: parts.append(f"{n_img} image{'s' if n_img != 1 else ''}")
                if n_vid: parts.append(f"{n_vid} video{'s' if n_vid != 1 else ''}")
                self.lbl_input.configure(
                    text=f"{root.name}/ ({subdirs_found} subfolders, {', '.join(parts)})",
                    text_color="#4ade80",
                )
                self.lbl_output.configure(
                    text=f"Auto: each subfolder/{AUTO_OUTPUT_NAME}/",
                    text_color="#4ade80",
                )

            else:
                root_out = root / AUTO_OUTPUT_NAME
                for img in root_images:
                    output_dir_map[img] = root_out
                all_files = root_images + sub_images
                self._input_paths    = all_files
                self._output_dir_map = output_dir_map
                n_img = sum(1 for p in all_files if p.suffix.lower() in IMAGE_EXTENSIONS)
                n_vid = sum(1 for p in all_files if p.suffix.lower() in VIDEO_EXTENSIONS)
                parts = []
                if n_img: parts.append(f"{n_img} image{'s' if n_img != 1 else ''}")
                if n_vid: parts.append(f"{n_vid} video{'s' if n_vid != 1 else ''}")
                self.lbl_input.configure(
                    text=f"{root.name}/ ({len(root_images)} root + {subdirs_found} subfolders, {', '.join(parts)})",
                    text_color="#4ade80",
                )
                self.lbl_output.configure(
                    text=f"Auto: each folder/{AUTO_OUTPUT_NAME}/",
                    text_color="#4ade80",
                )

        except Exception as exc:
            self.lbl_input.configure(
                text=f"Error scanning folder: {exc}",
                text_color="#f87171",
            )

        # Show/hide frame skip panel depending on whether videos were found
        has_video = any(p.suffix.lower() in VIDEO_EXTENSIONS for p in self._input_paths)
        self._update_frame_skip_visibility(has_video)

    def _update_frame_skip_visibility(self, show: bool):
        pass  # Video Quality panel removed – SAM3 processes every frame

    def _choose_output_folder(self):
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self._output_dir = Path(folder)
            self.lbl_output.configure(text=str(self._output_dir), text_color="#4ade80")

    # ── Start / stop processing ─────────────────────────────
    def _start_processing(self):
        if not self._input_paths:
            self._log("⚠ No input path selected.")
            return

        # Subfolder mode: output dirs are determined per image
        if self._output_dir_map:
            # Create all output dirs upfront
            for out_dir in set(self._output_dir_map.values()):
                out_dir.mkdir(parents=True, exist_ok=True)
            effective_output_dir = None
        else:
            # Normal mode: single output dir
            if self._output_dir is None:
                parent = self._input_paths[0].parent
                self._output_dir = parent / AUTO_OUTPUT_NAME
                self._log(f"📁 Auto output folder: {self._output_dir}")
                self.lbl_output.configure(text=str(self._output_dir), text_color="#fbbf24")
            self._output_dir.mkdir(parents=True, exist_ok=True)
            effective_output_dir = self._output_dir

        # Count images vs videos for the progress counter label
        self._n_images = sum(1 for p in self._input_paths if p.suffix.lower() in IMAGE_EXTENSIONS)
        self._n_videos = sum(1 for p in self._input_paths if p.suffix.lower() in VIDEO_EXTENSIONS)

        self._clear_log()
        self.progress_bar.set(0)
        self.lbl_status.configure(text="Processing ...", text_color="#fbbf24")
        self.lbl_counter.configure(text="")
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")

        # video_quality removed – SAM3 processes every frame

        self._processor = Processor(self._msg_queue)
        self._proc_thread = threading.Thread(
            target=self._processor.run,
            args=(self._input_paths, effective_output_dir,
                  self.model_var.get(), self.use_ollama_var.get()),
            kwargs={"output_dir_map": self._output_dir_map},
            daemon=True,
        )
        self._proc_thread.start()

    def _stop_processing(self):
        if self._processor:
            self._processor.stop()
        self.btn_stop.configure(state="disabled")
        self.lbl_status.configure(text="Cancelling ...", text_color="#f87171")

    # ── Queue polling ────────────────────────────────────────
    def _poll_queue(self):
        try:
            while True:
                msg = self._msg_queue.get_nowait()
                self._handle_message(msg)
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _handle_message(self, msg: dict):
        kind = msg["kind"]
        if kind == "log":
            self._log(msg["text"])
        elif kind == "status":
            self.lbl_status.configure(text=msg["text"], text_color="#fbbf24")
        elif kind == "progress":
            val     = msg["value"]
            current = msg.get("current", 0)
            total   = msg.get("total", 0)
            self.progress_bar.set(val)
            if total > 0:
                parts = []
                if self._n_images > 0:
                    parts.append(f"{self._n_images} image{'s' if self._n_images != 1 else ''}")
                if self._n_videos > 0:
                    parts.append(f"{self._n_videos} video{'s' if self._n_videos != 1 else ''}")
                total_str = ", ".join(parts) if parts else f"{total} files"
                self.lbl_counter.configure(text=f"{current} / {total_str}")
        elif kind == "done":
            self.btn_start.configure(state="normal")
            self.btn_stop.configure(state="disabled")
            if msg["success"]:
                self.lbl_status.configure(text="✅ Completed", text_color="#4ade80")
            else:
                self.lbl_status.configure(text="✗ Error occurred", text_color="#f87171")

    # ── Clean shutdown ───────────────────────────────────────
    def _on_close(self):
        """Stop processing, unload all models from VRAM, and exit."""
        # Stop any running processing thread
        if self._processor:
            self._processor.stop()





            # Unload SAM3 from VRAM
            try:
                if self._processor.sam3_proc is not None:
                    del self._processor.sam3_proc
                    self._processor.sam3_proc = None
            except Exception:
                pass

            # Unload second SAM3 instance (GPU 1) if loaded
            try:
                if self._processor.sam3_proc2 is not None:
                    del self._processor.sam3_proc2
                    self._processor.sam3_proc2 = None
            except Exception:
                pass

            # Unload Falcon Perception
            try:
                if self._processor.falcon_model is not None:
                    del self._processor.falcon_model
                    self._processor.falcon_model = None
            except Exception:
                pass

        # Unload Ollama model from VRAM (keep_alive=0)
        try:
            import ollama as ollama_client
            model = self.model_var.get()
            ollama_client.generate(model=model, prompt="", keep_alive=0)
        except Exception:
            pass

        # Clear all CUDA memory
        try:
            import torch
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        self.destroy()

        # Force exit – kills process and CMD window
        os._exit(0)


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = NeuralCensorApp()
    app.mainloop()