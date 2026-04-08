"""
NeuralCensor – Automatic Image Anonymization
Persons & vehicles are detected via YOLO, segmented via SAM3 (precise pixel masks),
pixelated multiple times with OpenCV (non-reconstructable), and verified via Ollama.
If Ollama finds missed objects, SAM3 is used again (not YOLO) to generate precise
masks for those objects – because re-running YOLO would return the same results.
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
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

SAM3_CHECKPOINT_DIR = Path(__file__).parent / "checkpoints" / "sam3"

BLUR_KERNEL_BASE    = 101    # Gaussian-Blur base kernel size (odd, large!)
BLUR_PASSES         = 3      # Number of blur passes (security)
QUANTIZE_STEP       = 8      # Pixel quantization after blur (prevents reconstruction)
PADDING_FRACTION    = 0.04   # Edge padding around each mask
OLLAMA_MAX_SIZE     = 1536   # Maximum edge length for Ollama input
AUTO_OUTPUT_NAME    = "NeuralCensor_Blurry"

YOLO_CLASSES        = [0, 2, 3, 5, 7]   # person, bicycle, car, bus, truck
YOLO_CONF           = 0.15              # lowered for higher sensitivity
SAM3_MIN_MASK_PX    = 100               # min mask pixels before falling back to bbox
SAM3_CONFIDENCE     = 0.15              # lowered for higher sensitivity
SAM3_OVERLAP_IOU    = 0.3               # IoU threshold: new mask vs 1st-pass masks

# Text prompts for SAM3 2nd pass (searched independently on the original image)
SAM3_TEXT_PROMPTS   = ["person", "car", "truck", "bus", "motorcycle", "license plate"]

OLLAMA_VERIFY_PROMPT = (
    "This image has been anonymized. Your job is to check if ANY person or vehicle was MISSED. "
    "Be EXTREMELY strict and paranoid. Check every pixel of the image.\n\n"
    "PERSONS: Even a single visible head, face, hair, arm, hand, leg, foot, "
    "silhouette, or ANY recognizable human body part counts as a missed person. "
    "A person partially hidden behind an object, in a window, in a mirror, "
    "or barely visible in the background is STILL a person. "
    "If you can tell it is human in any way, it counts as missed.\n\n"
    "VEHICLES: Any car, truck, bus, motorcycle, bicycle, or license plate "
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
# Helper functions
# ──────────────────────────────────────────────────────────────

def pad_mask(mask: np.ndarray, pad: float = PADDING_FRACTION) -> np.ndarray:
    """Extends a binary mask by dilating its contour (preserves shape)."""
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
    k = max(BLUR_KERNEL_BASE, int(min(roi_h, roi_w) * 0.4))
    k = k if k % 2 == 1 else k + 1
    k = min(k, 301)  # cap to avoid extreme kernel sizes

    blurred = roi
    for _ in range(BLUR_PASSES):
        blurred = cv2.GaussianBlur(blurred, (k, k), sigmaX=0, sigmaY=0)

    # Pixel quantization: round values to QUANTIZE_STEP grid
    blurred = (blurred // QUANTIZE_STEP * QUANTIZE_STEP).astype(np.uint8)

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


# ──────────────────────────────────────────────────────────────
# Processing class
# ──────────────────────────────────────────────────────────────

class Processor:
    """
    Backend: YOLO detection → SAM3 mask refinement → OpenCV multi-pass pixelation → Ollama verification.
    Runs entirely in its own thread.
    """

    def __init__(self, msg_queue: queue.Queue):
        self.msg_queue   = msg_queue
        self.yolo_model   = None
        self.yolo_loaded  = False
        self.sam3_proc    = None
        self.sam3_loaded  = False
        self.ollama_ready = False
        self._stop_event  = threading.Event()

    # ── Message helpers ──────────────────────────────────────
    def _emit(self, kind: str, **kwargs):
        self.msg_queue.put({"kind": kind, **kwargs})

    def _log(self, text: str):
        self._emit("log", text=text)

    def _progress(self, value: float, current: int = 0, total: int = 0):
        self._emit("progress", value=value, current=current, total=total)

    def _done(self, success: bool, message: str = ""):
        self._emit("done", success=success, message=message)

    def stop(self):
        self._stop_event.set()

    # -- Load YOLO (primary detector) ---------------------------------------
    def _load_yolo(self) -> bool:
        if self.yolo_loaded:
            return True
        try:
            from ultralytics import YOLO
            self._log("[LOAD] Loading YOLO detector ...")
            self.yolo_model  = YOLO("yolov8n.pt")
            self.yolo_loaded = True
            self._log("[OK] YOLO loaded.")
            return True
        except ImportError:
            self._log("[ERROR] ultralytics not installed.")
            return False
        except Exception as exc:
            self._log(f"[ERROR] YOLO error: {exc}")
            return False

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
                confidence_threshold=SAM3_CONFIDENCE,
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

    # -- Refine one YOLO box into a SAM3 mask ------------------------------
    def _sam3_mask_from_box(
        self,
        state,
        box_xyxy: tuple,
        img_h: int,
        img_w: int,
    ) -> np.ndarray | None:
        """
        Given a SAM3 image state and a bounding box, queries SAM3 for a
        precise pixel mask.  Returns None if SAM3 fails or mask is too small.
        """
        import torch

        x1, y1, x2, y2 = box_xyxy

        try:
            # Convert xyxy pixel coords to SAM3 format:
            # [center_x, center_y, width, height] normalized to [0, 1]
            cx = ((x1 + x2) / 2.0) / img_w
            cy = ((y1 + y2) / 2.0) / img_h
            bw = (x2 - x1) / img_w
            bh = (y2 - y1) / img_h

            self.sam3_proc.reset_all_prompts(state)
            s = self.sam3_proc.add_geometric_prompt(
                box=[cx, cy, bw, bh], label=True, state=state
            )

            raw_masks = s.get("masks")
            if raw_masks is None or (hasattr(raw_masks, "__len__") and len(raw_masks) == 0):
                return None

            if hasattr(raw_masks, "cpu"):
                raw_masks = raw_masks.cpu().numpy()

            # Take the mask with the highest coverage inside the box
            best_mask = None
            best_px   = 0
            for m in raw_masks:
                if m.ndim == 3:
                    m = m[0]
                if m.shape != (img_h, img_w):
                    m = cv2.resize(
                        m.astype(np.float32), (img_w, img_h),
                        interpolation=cv2.INTER_LINEAR,
                    )
                binary = (m > 0.5).astype(np.uint8) * 255
                # Count pixels inside the YOLO box
                roi_mask = binary[y1:y2, x1:x2]
                px = int(np.count_nonzero(roi_mask))
                if px > best_px:
                    best_px   = px
                    best_mask = binary

            if best_mask is not None and best_px >= SAM3_MIN_MASK_PX:
                # Clip mask to bbox region with padding to prevent full-image masks
                pad_y = int((y2 - y1) * PADDING_FRACTION * 2)
                pad_x = int((x2 - x1) * PADDING_FRACTION * 2)
                cy1 = max(0, y1 - pad_y)
                cy2 = min(img_h, y2 + pad_y)
                cx1 = max(0, x1 - pad_x)
                cx2 = min(img_w, x2 + pad_x)
                clipped = np.zeros_like(best_mask)
                clipped[cy1:cy2, cx1:cx2] = best_mask[cy1:cy2, cx1:cx2]
                return clipped
            return None

        except Exception as exc:
            self._log(f"    SAM3 refine error: {exc}")
            return None

    # -- Hybrid YOLO->SAM3 mask generation ----------------------------------
    def _generate_masks_hybrid(self, cv_image: np.ndarray) -> list[np.ndarray]:
        """
        1. YOLO detects all persons / vehicles -> bounding boxes
        2. SAM3 refines each box into a precise pixel mask
        Returns a list of uint8 masks.
        """
        h, w = cv_image.shape[:2]
        masks: list[np.ndarray] = []

        # -- Step 1: YOLO detection --
        self._log("  -> Detection (YOLO) ...")
        t_yolo = time.perf_counter()
        try:
            results = self.yolo_model.predict(
                cv_image, classes=YOLO_CLASSES, conf=YOLO_CONF, verbose=False
            )
        except Exception as exc:
            self._log(f"  [ERROR] YOLO predict error: {exc}")
            return masks
        dt_yolo = time.perf_counter() - t_yolo

        boxes: list[tuple] = []
        cls_names = self.yolo_model.names if hasattr(self.yolo_model, "names") else {}

        for r in results:
            for box in r.boxes:
                xyxy  = box.xyxy[0].cpu().numpy()
                conf  = float(box.conf[0].cpu().numpy())
                cls   = int(box.cls[0].cpu().numpy())
                x1, y1, x2, y2 = map(int, xyxy)
                if (x2 - x1) > 5 and (y2 - y1) > 5:
                    label = cls_names.get(cls, str(cls))
                    boxes.append((x1, y1, x2, y2, label, conf))
                    self._log(f"  YOLO: {label} ({x1},{y1})->({x2},{y2}) conf={conf:.2f}")
        self._log(f"  YOLO: {len(boxes)} object(s) in {dt_yolo:.2f}s")

        if not boxes:
            self._log("  No objects detected.")
            return masks

        # -- Step 2: SAM3 mask refinement --
        if not self.sam3_loaded:
            self._log("  [ERROR] SAM3 not available. Cannot generate masks.")
            return []

        self._log(f"  -> Refining {len(boxes)} mask(s) with SAM3 ...")
        rgb     = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        t_sam_img = time.perf_counter()
        try:
            state = self.sam3_proc.set_image(pil_img)
        except Exception as exc:
            self._log(f"  [ERROR] SAM3 set_image error: {exc}")
            return []
        dt_sam_img = time.perf_counter() - t_sam_img
        self._log(f"    set_image: {dt_sam_img:.2f}s")

        t_sam_masks = time.perf_counter()
        for idx, (x1, y1, x2, y2, label, conf) in enumerate(boxes):
            t_m = time.perf_counter()
            refined = self._sam3_mask_from_box(state, (x1, y1, x2, y2), h, w)
            dt_m = time.perf_counter() - t_m
            if refined is not None:
                pixel_count = int(np.count_nonzero(refined))
                self._log(f"    Mask {idx+1}/{len(boxes)}: {pixel_count:,} px [{label}] {dt_m:.2f}s")
                masks.append(pad_mask(refined))
            else:
                self._log(f"    Mask {idx+1}/{len(boxes)}: no mask [{label}] {dt_m:.2f}s - skipped")
        dt_sam_total = time.perf_counter() - t_sam_img
        self._log(f"  SAM3 total: {dt_sam_total:.2f}s ({dt_sam_img:.2f}s encode + {time.perf_counter() - t_sam_masks:.2f}s masks)")

        # Free SAM3 state tensors to release VRAM
        import torch
        del state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return masks

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
        scale   = min(OLLAMA_MAX_SIZE / orig_w, OLLAMA_MAX_SIZE / orig_h, 1.0)
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
        self, cv_image_original: np.ndarray, first_pass_masks: list[np.ndarray]
    ) -> list[np.ndarray]:
        """
        Uses SAM3 text prompts to search for ALL persons/vehicles on the
        ORIGINAL (unblurred) image.  Filters out masks that overlap with
        already-handled 1st-pass masks (IoU check) so only genuinely NEW
        detections are returned.
        """
        import torch

        h, w = cv_image_original.shape[:2]
        new_masks: list[np.ndarray] = []

        self._log(f"  -> SAM3 text-prompt search on original image ({len(SAM3_TEXT_PROMPTS)} prompts) ...")
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

        for prompt in SAM3_TEXT_PROMPTS:
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
                if px < SAM3_MIN_MASK_PX:
                    continue

                # Overlap check: how much of this mask was already covered?
                intersection = int(np.count_nonzero(np.bitwise_and(binary, first_pass_union)))
                union = int(np.count_nonzero(np.bitwise_or(binary, first_pass_union)))
                iou = intersection / union if union > 0 else 0.0

                if iou > SAM3_OVERLAP_IOU:
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

        return new_masks

    # -- Process one image --------------------------------------------------
    def _process_image(self, image_path: Path, output_dir: Path, model: str) -> bool:
        t_total = time.perf_counter()
        self._log(f">> {image_path.name}")

        cv_image = cv2.imread(str(image_path))
        if cv_image is None:
            self._log("  [ERROR] Could not load image.")
            return False

        # Keep the original for potential 2nd pass (SAM3 text search)
        cv_original = cv_image.copy()

        # -- Step 1: Detect & refine masks (YOLO -> SAM3 box refinement) --
        masks = self._generate_masks_hybrid(cv_image)

        if not self.sam3_loaded:
            self._log("  [ERROR] Skipping - SAM3 required but not available.")
            return False

        if not masks:
            self._log("  No objects found by YOLO – running SAM3 text-search on original ...")
            # Even if YOLO found nothing, try SAM3 text-search as a safety net
            t_sam_only = time.perf_counter()
            masks = self._sam3_text_search(cv_original, [])
            dt_sam_only = time.perf_counter() - t_sam_only
            if not masks:
                self._log("  SAM3 text-search also found nothing – image unchanged.")
                cv2.imwrite(str(output_dir / image_path.name), cv_image)
                return True
            self._log(f"  SAM3 text-search found {len(masks)} object(s) in {dt_sam_only:.2f}s (YOLO had missed them)")

        # -- Step 2: Apply pixelation (1st pass masks) --
        t_blur = time.perf_counter()
        self._log(f"  -> Pixelating {len(masks)} mask(s), {BLUR_PASSES}x blur ...")
        result = cv_image.copy()
        for mask in masks:
            result = blur_region(result, mask)
        dt_blur = time.perf_counter() - t_blur
        self._log(f"  Pixelation (pass 1): {dt_blur:.2f}s")

        # -- Step 3: SAM3 text-search safety pass (on ORIGINAL image) --
        # This runs BEFORE Ollama as an extra safety net:
        # SAM3 text prompts search the original for anything YOLO may have missed.
        # Only genuinely new regions (not yet covered by pass-1 masks) are added.
        self._log("  -> SAM3 text-search safety pass (BEFORE Ollama) ...")
        t_sam2 = time.perf_counter()
        safety_masks = self._sam3_text_search(cv_original, masks)
        dt_sam2 = time.perf_counter() - t_sam2
        all_masks = masks  # keep reference to all masks used so far
        if safety_masks:
            self._log(f"  SAM3 safety pass: {len(safety_masks)} new object(s) found in {dt_sam2:.2f}s – applying pixelation ...")
            for mask in safety_masks:
                result = blur_region(result, mask)
            all_masks = masks + safety_masks
        else:
            self._log(f"  SAM3 safety pass: nothing new found ({dt_sam2:.2f}s)")

        # -- Step 4: Save intermediate result --
        out_path = output_dir / image_path.name
        cv2.imwrite(str(out_path), result)

        # -- Step 5: Ollama verification (yes/no only) --
        t_ollama = time.perf_counter()
        self._log("  -> Ollama verification (yes/no) ...")
        still_visible = self._verify_with_ollama(result, model)
        dt_ollama = time.perf_counter() - t_ollama
        self._log(f"  Ollama: {dt_ollama:.2f}s")

        safety_info = f" + SAM3 safety:{len(safety_masks)}" if safety_masks else ""
        report_status = f"1 pass (YOLO:{len(masks)}{safety_info})"

        if still_visible:
            self._log("  [WARN] Ollama found missed objects – SAM3 text-prompt 3rd pass on original ...")
            # SAM3 searches the ORIGINAL image with text prompts
            # to find objects that both YOLO and the safety pass missed.
            # all_masks (pass1 + safety) are used as overlap filter.
            t_refine = time.perf_counter()
            new_masks = self._sam3_text_search(cv_original, all_masks)
            if new_masks:
                for mask in new_masks:
                    result = blur_region(result, mask)
                dt_refine = time.perf_counter() - t_refine
                report_status = (
                    f"3 passes (YOLO:{len(masks)}{safety_info} + Ollama-triggered:{len(new_masks)}) "
                    f"in {dt_refine:.1f}s"
                )
                self._log(f"  3rd pass complete: {len(new_masks)} new mask(s) ({dt_refine:.2f}s).")
            else:
                dt_refine = time.perf_counter() - t_refine
                report_status = (
                    f"2 passes (YOLO:{len(masks)}{safety_info}) – "
                    f"Ollama triggered 3rd pass but SAM3 found 0 new objects (false positive, {dt_refine:.2f}s)"
                )
                self._log(f"  3rd pass: SAM3 found nothing new – Ollama was a false positive ({dt_refine:.2f}s).")
        else:
            self._log("  [OK] Ollama verification passed.")

        # -- Step 6: Save final image --
        cv2.imwrite(str(out_path), result)
        dt_total = time.perf_counter() - t_total
        self._log(f"  [SAVED] {out_path.name}")
        self._log(f"  ── Total: {dt_total:.2f}s ──")

        # -- Report --
        report_file = output_dir / "Anonymization_Report.txt"
        with open(report_file, "a", encoding="utf-8") as f:
            f.write(f"{image_path.name}: {report_status}\n")

        return True

    # -- Main processing loop -----------------------------------------------
    def run(
        self,
        image_paths: list[Path],
        output_dir: Path | None,
        model: str,
        output_dir_map: dict[Path, Path] | None = None,
    ):
        self._stop_event.clear()
        total = len(image_paths)
        mode = "subfolder mode" if output_dir_map else "single output folder"
        self._log(f"Starting processing: {total} image(s) | {mode} | Verification model: {model}")

        # Load YOLO (detection)
        if not self._load_yolo():
            self._log("[ERROR] YOLO could not be loaded. Aborting.")
            self._done(success=False)
            return

        # Load SAM3 (required for mask refinement)
        if not self._load_sam3():
            self._log("[ERROR] SAM3 is required but could not be loaded. Aborting.")
            self._log("  -> Please run start.bat to install SAM3 and download the checkpoint.")
            self._done(success=False)
            return

        # Warmup Ollama (pre-load model into GPU/RAM)
        if not self._warmup_ollama(model):
            self._log("[WARN] Ollama warmup failed - verification may be slow on first image.")

        self._log("Pipeline: YOLO detection -> SAM3 mask refinement -> pixelation -> SAM3 text-search (safety) -> Ollama verification")

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
            ok = self._process_image(img_path, img_output_dir, model)
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

        self._build_separator(left)

        # Input – unified single browse button
        ctk.CTkLabel(left, text="INPUT",
                     font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
                     text_color="#a0a0b0").pack(anchor="w", padx=16, pady=(14, 4))

        self._make_button(left, "📂  Browse Input", self._browse_input)

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
            ("YOLOv8", "Object Detection (Persons, Vehicles)",
             "https://github.com/ultralytics/ultralytics"),
            ("SAM 3 (Segment Anything 3)", "Pixel-precise Mask Generation",
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
    def _browse_input(self):
        """
        Opens a single folder dialog.  Automatically detects the structure:
          • Folder has images directly        → flat mode
          • Folder has subfolders with images  → subfolder mode
          • Both                               → combines both
          • No images anywhere                 → error label
        """
        folder = filedialog.askdirectory(title="Select Folder with Images")
        if not folder:
            return

        root = Path(folder)

        # 1. Collect images directly in the selected folder
        root_images = sorted(
            f for f in root.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        # 2. Collect images inside immediate subdirectories
        output_dir_map: dict[Path, Path] = {}
        sub_images: list[Path] = []
        subdirs_found = 0
        for sub in sorted(root.iterdir()):
            if not sub.is_dir() or sub.name == AUTO_OUTPUT_NAME:
                continue
            images = sorted(
                f for f in sub.iterdir()
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
            )
            if not images:
                continue
            subdirs_found += 1
            out_dir = sub / AUTO_OUTPUT_NAME
            for img in images:
                output_dir_map[img] = out_dir
            sub_images.extend(images)

        # 3. Determine mode based on what was found
        has_root  = len(root_images) > 0
        has_subs  = len(sub_images) > 0

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
            # ── Flat mode: images only in the root folder ──
            self._input_paths = root_images
            self.lbl_input.configure(
                text=f"{root.name}/ ({len(root_images)} images)",
                text_color="#4ade80",
            )
            self.lbl_output.configure(
                text=f"Auto: {root.name}/{AUTO_OUTPUT_NAME}/",
                text_color="#fbbf24",
            )

        elif has_subs and not has_root:
            # ── Subfolder mode: images only in subdirectories ──
            self._input_paths    = sub_images
            self._output_dir_map = output_dir_map
            self.lbl_input.configure(
                text=f"{root.name}/ ({subdirs_found} subfolders, {len(sub_images)} images)",
                text_color="#4ade80",
            )
            self.lbl_output.configure(
                text=f"Auto: each subfolder/{AUTO_OUTPUT_NAME}/",
                text_color="#4ade80",
            )

        else:
            # ── Mixed mode: images in root AND in subfolders ──
            # Root images get their own output dir too
            root_out = root / AUTO_OUTPUT_NAME
            for img in root_images:
                output_dir_map[img] = root_out
            self._input_paths    = root_images + sub_images
            self._output_dir_map = output_dir_map
            total = len(root_images) + len(sub_images)
            self.lbl_input.configure(
                text=f"{root.name}/ ({len(root_images)} root + {subdirs_found} subfolders, {total} images)",
                text_color="#4ade80",
            )
            self.lbl_output.configure(
                text=f"Auto: each folder/{AUTO_OUTPUT_NAME}/",
                text_color="#4ade80",
            )

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

        self._clear_log()
        self.progress_bar.set(0)
        self.lbl_status.configure(text="Processing ...", text_color="#fbbf24")
        self.lbl_counter.configure(text="")
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")

        self._processor = Processor(self._msg_queue)
        self._proc_thread = threading.Thread(
            target=self._processor.run,
            args=(self._input_paths, effective_output_dir, self.model_var.get()),
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
        elif kind == "progress":
            val     = msg["value"]
            current = msg.get("current", 0)
            total   = msg.get("total", 0)
            self.progress_bar.set(val)
            if total > 0:
                self.lbl_counter.configure(text=f"{current} / {total} images")
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

            # Unload YOLO from VRAM
            try:
                if self._processor.yolo_model is not None:
                    del self._processor.yolo_model
                    self._processor.yolo_model = None
            except Exception:
                pass

            # Unload SAM3 from VRAM
            try:
                if self._processor.sam3_proc is not None:
                    del self._processor.sam3_proc
                    self._processor.sam3_proc = None
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