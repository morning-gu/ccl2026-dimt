"""FluxText rendering API server.

Wraps AMAP-ML/FluxText (FLUX-Text) as a standalone FastAPI service.
FluxText performs erase + render in a single diffusion pass.

Prerequisites:
    - FluxText repo cloned and installed (pip install -r requirements.txt)
    - FLUX.1-Fill-dev base model accessible
    - FLUX-Text LoRA weights (GD-ML/FLUX-Text)
    - flash_attn installed
    - pip install fastapi uvicorn pydantic numpy opencv-python Pillow PyYAML safetensors
    - For quantization: pip install bitsandbytes

Environment variables (server-side):
  FLUXTEXT_REPO_PATH       Path to FluxText repo root
  FLUXTEXT_MODEL_PATH      Path to FLUX-Text LoRA weights (.safetensors)
  FLUXTEXT_CONFIG_PATH     Path to FluxText config YAML
  FLUXTEXT_FONT_PATH       Path to font file (optional)
  FLUXTEXT_QUANTIZE        Quantization (direct disk load, bypasses OminiModelFIll):
                            "8bit" (default, 8-bit transformer, ~12GB CPU RAM),
                            "nf4" (4-bit transformer, ~6GB CPU RAM),
                            "none" (full precision via OminiModelFIll, ~34GB RAM),
                            "8bit_all" (8-bit transformer + T5 encoder)
  FLUXTEXT_OFFLOAD         CPU offload: "none", "model" (per-component),
                            "sequential" (per-layer, slowest).
                            Defaults to "model" when quantize != "none".

Usage:
  python fluxtext_server.py --host 0.0.0.0 --port 8002
"""
import argparse
import logging
import math
import os
import sys
from typing import List, Optional

from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException

from _shared import (
    RenderRequest,
    RenderResponse,
    RegionData,
    decode_request,
    encode_image,
)

logger = logging.getLogger("fluxtext_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
# bitsandbytes 8-bit matmul casts bf16->fp16 on every forward pass and
# spams an INFO line each time. bitsandbytes re-sets its logger level on
# import, so setLevel alone is unreliable -- install a root Filter that
# drops only that specific cast notice (warnings/errors still surface).


class _BnbMatmulCastFilter(logging.Filter):
    def filter(self, record):
        if record.name.startswith("bitsandbytes") and record.levelno == logging.INFO:
            msg = record.getMessage()
            if "MatMul8bitLt" in msg and "cast from" in msg:
                return False
        return True


logging.getLogger().addFilter(_BnbMatmulCastFilter())

_pipe = None
_flux_config = None
_repo_path = None
_font = None


def _init_model():
    """Load FLUX-Text pipeline (FLUX.1-Fill-dev + LoRA)."""
    global _pipe, _flux_config, _repo_path, _font

    repo = os.environ.get("FLUXTEXT_REPO_PATH", "")
    if not repo:
        raise RuntimeError(
            "FLUXTEXT_REPO_PATH is not set. Clone the FluxText repo and "
            "set this env var to its root path."
        )
    if repo not in sys.path:
        sys.path.insert(0, repo)

    model_path = os.environ.get("FLUXTEXT_MODEL_PATH", "")
    config_path = os.environ.get("FLUXTEXT_CONFIG_PATH", "")
    if not model_path or not config_path:
        raise RuntimeError(
            "FLUXTEXT_MODEL_PATH and FLUXTEXT_CONFIG_PATH must be set "
            "to the LoRA weights and config YAML respectively."
        )

    quantize = os.environ.get("FLUXTEXT_QUANTIZE", "8bit").lower()
    offload_default = "model" if quantize != "none" else "none"
    offload = os.environ.get("FLUXTEXT_OFFLOAD", offload_default).lower()

    import yaml
    import torch
    from safetensors.torch import load_file

    with open(config_path, "r") as f:
        flux_config = yaml.safe_load(f)

    flux_path = flux_config["flux_path"]
    dtype = getattr(torch, flux_config["dtype"])
    training_config = flux_config["train"]

    # LoRA weights with remapped keys (common to both loading paths)
    lora_state_dict = {
        x.replace("lora_A", "lora_A.default")
         .replace("lora_B", "lora_B.default")
         .replace("transformer.", ""): v
        for x, v in load_file(model_path).items()
    }

    if quantize in ("8bit", "nf4", "8bit_all"):
        # Direct quantized loading: bypass OminiModelFIll to avoid
        # the ~34GB CPU RAM spike from full-precision weight materialization.
        from diffusers import (
            BitsAndBytesConfig,
            FluxFillPipeline,
            FluxTransformer2DModel,
        )
        from peft import LoraConfig

        if quantize == "nf4":
            qconfig = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
            )
            logger.info("Direct load: NF4 4-bit transformer (~6GB CPU RAM)")
        else:
            qconfig = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_has_fp16_weight=False,
            )
            logger.info("Direct load: 8-bit transformer (~12GB CPU RAM)")

        transformer = FluxTransformer2DModel.from_pretrained(
            flux_path,
            subfolder="transformer",
            quantization_config=qconfig,
            torch_dtype=dtype,
        )
        pipeline = FluxFillPipeline.from_pretrained(
            flux_path,
            transformer=transformer,
            torch_dtype=dtype,
        )

        lora_config = training_config.get("lora_config")
        if lora_config:
            pipeline.transformer.add_adapter(LoraConfig(**lora_config))
        pipeline.transformer.load_state_dict(lora_state_dict, strict=False)
        logger.info("LoRA weights loaded from %s", model_path)

        if quantize == "8bit_all":
            t5 = getattr(pipeline, "text_encoder_2", None)
            if t5 is not None:
                n = _quantize_8bit(t5)
                logger.info("Quantized T5: %d Linear layers -> 8-bit", n)

        _pipe = pipeline
    else:
        # Full precision via OminiModelFIll (needs ~34GB CPU RAM)
        load_device = "cpu" if offload != "none" else "cuda"
        from src.train.model import OminiModelFIll

        trainable_model = OminiModelFIll(
            flux_pipe_id=flux_path,
            lora_config=training_config["lora_config"],
            device=load_device,
            dtype=dtype,
            optimizer_config=training_config["optimizer"],
            model_config=flux_config.get("model", {}),
            gradient_checkpointing=training_config.get(
                "gradient_checkpointing", False
            ),
            byt5_encoder_config=training_config.get("byt5_encoder", None),
        )
        trainable_model.transformer.load_state_dict(lora_state_dict, strict=False)
        _pipe = trainable_model.flux_pipe
        logger.info("Loaded via OminiModelFIll (full precision)")

    _flux_config = flux_config
    _repo_path = repo


    # Device placement / CPU offload
    if quantize in ("8bit", "nf4", "8bit_all") and offload in ("model", "sequential"):
        # bitsandbytes quantized transformers cannot be moved to CPU by
        # enable_*_cpu_offload -- quantized weights stay pinned to GPU,
        # causing OOM when T5 is also loaded.  Keep everything on GPU.
        if quantize == "8bit":
            # Also quantize T5 so transformer+T5 fit in 24GB VRAM:
            #   8-bit transformer (~12GB) + 8-bit T5 (~5GB) ~ 17GB
            t5 = getattr(_pipe, "text_encoder_2", None)
            if t5 is not None:
                n = _quantize_8bit(t5)
                logger.info("Auto-quantized T5 to 8-bit (%d layers)", n)
        _pipe = _pipe.to("cuda")
        torch.cuda.empty_cache()
        if quantize == "nf4":
            logger.info("GPU: NF4 + T5 bf16 (~17GB VRAM)")
        else:
            logger.info("GPU: 8-bit transformer + 8-bit T5 (~18GB VRAM)")
    elif offload == "sequential":
        _pipe.enable_sequential_cpu_offload()
        logger.info("Offload: sequential (per-layer)")
    elif offload == "model":
        _pipe.enable_model_cpu_offload()
        logger.info("Offload: model (per-component)")
    else:
        _pipe = _pipe.to("cuda")
        torch.cuda.empty_cache()
        if quantize == "nf4":
            logger.info("GPU: NF4 + T5 bf16 (~17GB VRAM)")
        elif quantize == "8bit":
            logger.info("GPU: 8-bit + T5 bf16 (~23GB VRAM)")
        elif quantize == "8bit_all":
            logger.info("GPU: 8-bit all (~18GB VRAM)")
        else:
            logger.info("GPU: full precision (~34GB VRAM)")
    font_path = os.environ.get(
        "FLUXTEXT_FONT_PATH",
        os.path.join(repo, "font", "Arial_Unicode.ttf"),
    )
    from PIL import ImageFont
    _font = ImageFont.truetype(font_path, size=60)

    logger.info("FLUX-Text model loaded from %s", model_path)


def _quantize_8bit(module):
    """Replace all nn.Linear with 8-bit Linear8bitLt in-place.

    Used for T5 encoder quantization (8bit_all mode).  Call while module
    is on CPU; int8 quantization happens on first CUDA forward pass.
    Returns the number of layers replaced.
    """
    import torch.nn as nn
    import bitsandbytes as bnb

    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, bnb.nn.Linear8bitLt):
            continue
        if isinstance(child, nn.Linear):
            has_bias = child.bias is not None
            new_module = bnb.nn.Linear8bitLt(
                child.in_features,
                child.out_features,
                bias=has_bias,
                has_fp16_weights=False,
                threshold=6.0,
            )
            new_module.weight = bnb.nn.Int8Params(
                child.weight.data.contiguous(),
                requires_grad=False,
                has_fp16_weights=False,
            )
            if has_bias:
                new_module.bias = nn.Parameter(child.bias.data.contiguous())
            setattr(module, name, new_module)
            count += 1
        elif len(list(child.children())) > 0:
            count += _quantize_8bit(child)
    return count


# ------------------------------------------------------------------
# Glyph / mask generation
# ------------------------------------------------------------------

def _insert_spaces(string, n_space):
    if n_space == 0:
        return string
    new_string = ""
    for char in string:
        new_string += char + " " * n_space
    return new_string[:-n_space]


def _draw_glyph_bbox(text, width, height, bbox=None):
    from PIL import Image, ImageDraw
    img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)
    if bbox:
        x1, y1, x2, y2 = bbox
    else:
        x1, y1, x2, y2 = 0, 0, width, height
    box_h = y2 - y1
    font_size = max(10, int(box_h * 0.8))
    font = _font.font_variant(size=font_size)
    draw.text((x1, y1), text, font=font, fill=255)
    return np.array(img)


def _draw_glyph(text, polygon, width, height):
    from PIL import Image, ImageDraw

    if polygon is None or len(polygon) < 3:
        return _draw_glyph_bbox(text, width, height)

    pts = np.array(polygon, dtype=np.float32)
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    w, h = rect[1]
    angle = rect[2]
    if angle < -45:
        angle += 90
    angle = -angle
    if w < h:
        angle += 90

    vert = False
    if abs(angle) % 90 < 10 or abs(90 - abs(angle) % 90) % 90 < 10:
        _w = max(box[:, 0]) - min(box[:, 0])
        _h = max(box[:, 1]) - min(box[:, 1])
        if _h >= _w:
            vert = True
            angle = 0

    img = Image.new("RGB", (width, height), (0, 0, 0))
    font = _font

    image4ratio = Image.new("RGB", img.size, "white")
    draw_ratio = ImageDraw.Draw(image4ratio)
    _, _, _tw, _th = draw_ratio.textbbox(xy=(0, 0), text=text, font=font)
    if _th == 0:
        _th = 1
    text_w = min(w, h) * (_tw / _th)
    if text_w <= max(w, h) and len(text) > 1 and not vert:
        for i in range(1, 100):
            text_space = _insert_spaces(text, i)
            _, _, _tw2, _th2 = draw_ratio.textbbox(
                xy=(0, 0), text=text_space, font=font
            )
            if min(w, h) * (_tw2 / _th2) > max(w, h):
                break
        text = _insert_spaces(text, i - 1)
        font_size = min(w, h) * 0.80
    else:
        shrink = 0.75 if vert else 0.85
        font_size = min(w, h) / (text_w / max(w, h)) * shrink

    new_font = font.font_variant(size=int(font_size))
    left, top, right, bottom = new_font.getbbox(text)
    text_width = right - left
    text_height = bottom - top

    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    if not vert:
        draw.text(
            (rect[0][0] - text_width // 2, rect[0][1] - text_height // 2 - top),
            text,
            font=new_font,
            fill=(255, 255, 255, 255),
        )
    else:
        x_s = min(box[:, 0]) + (max(box[:, 0]) - min(box[:, 0])) // 2 - text_height // 2
        y_s = min(box[:, 1])
        for c in text:
            draw.text((x_s, y_s), c, font=new_font, fill=(255, 255, 255, 255))
            _, _t, _, _b = new_font.getbbox(c)
            y_s += _b

    rotated_layer = layer.rotate(
        angle, expand=1, center=(rect[0][0], rect[0][1])
    )
    x_offset = int((img.width - rotated_layer.width) / 2)
    y_offset = int((img.height - rotated_layer.height) / 2)
    img.paste(rotated_layer, (x_offset, y_offset), rotated_layer)
    return np.array(img.convert("L"))


def _build_mask(regions, height, width, dilate=3):
    mask = np.zeros((height, width), dtype=np.uint8)
    for r in regions:
        poly = getattr(r, "bbox_poly", None)
        if poly and len(poly) >= 3:
            pts = np.array([[int(p[0]), int(p[1])] for p in poly], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
        else:
            x1, y1, x2, y2 = [int(v) for v in r.bbox[:4]]
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    if dilate > 0:
        kernel = np.ones((dilate, dilate), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def _build_glyph_image(regions, height, width):
    glyph = np.zeros((height, width), dtype=np.uint8)
    for r in regions:
        poly = getattr(r, "bbox_poly", None)
        if poly and len(poly) >= 3:
            g = _draw_glyph(r.translated_text, poly, width, height)
        else:
            x1, y1, x2, y2 = [int(v) for v in r.bbox[:4]]
            g = _draw_glyph_bbox(
                r.translated_text, width, height, bbox=(x1, y1, x2, y2)
            )
        glyph = np.maximum(glyph, g)
    return glyph


# ------------------------------------------------------------------
# Prompt generation
# ------------------------------------------------------------------

def _build_prompt(image, regions):
    texts = [r.translated_text for r in regions if r.translated_text]
    if not texts:
        return ""
    description = "product image with text"
    text_str = " , ".join(texts)
    return f"{description}, that reads {text_str} ."


# ------------------------------------------------------------------
# Resolution matching
# ------------------------------------------------------------------

_ASPECT_RATIOS = [
    "2.39:1", "2:1", "16:9", "1.85:1", "9:16", "5:8", "3:2", "4:3", "1:1",
]
_PIXELS = [512 * 512, 768 * 768, 1024 * 1024]


def _match_resolution(height, width):
    num_pixel = min(_PIXELS, key=lambda x: abs(x - width * height))
    D = 16
    aspect = height / width
    best = None
    best_diff = float("inf")
    for ratio in _ASPECT_RATIOS:
        wr, hr = map(float, ratio.split(":"))
        w = int(math.sqrt(num_pixel * (wr / hr)) // D) * D
        h = int((num_pixel / w) // D) * D
        if h * w == 0:
            continue
        diff = abs(h / w - aspect)
        if diff < best_diff:
            best_diff = diff
            best = (h, w)
        h2, w2 = w, h
        diff2 = abs(h2 / w2 - aspect)
        if diff2 < best_diff:
            best_diff = diff2
            best = (h2, w2)
    return best or (512, 512)


# ------------------------------------------------------------------
# Core rendering
# ------------------------------------------------------------------

def _render_fluxtext(image, regions):
    import torch
    from PIL import Image
    from src.flux.condition import Condition
    from src.flux.generate_fill import generate_fill

    h, w = image.shape[:2]
    tgt_h, tgt_w = _match_resolution(h, w)

    glyph_full = _build_glyph_image(regions, h, w)
    mask_full = _build_mask(regions, h, w)

    glyph_resized = cv2.resize(glyph_full, (tgt_w, tgt_h), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask_full, (tgt_w, tgt_h), interpolation=cv2.INTER_NEAREST)
    img_resized = cv2.resize(image, (tgt_w, tgt_h), interpolation=cv2.INTER_LINEAR)

    glyph_rgb = np.stack([glyph_resized] * 3, axis=-1)
    glyph_norm = (255 - glyph_rgb.astype(np.float64)) / 255.0
    mask_norm = np.expand_dims(mask_resized.astype(np.float64) / 255.0, axis=-1)
    condition_img = [glyph_norm, mask_norm, img_resized]

    prompt = _build_prompt(img_resized, regions)

    condition = Condition(
        condition_type="word_fill",
        condition=condition_img,
        position_delta=[0, 0],
    )

    generator = torch.Generator(device="cuda")
    generator.manual_seed(42)

    res = generate_fill(
        _pipe,
        prompt=prompt,
        conditions=[condition],
        height=tgt_h,
        width=tgt_w,
        generator=generator,
        model_config=_flux_config.get("model", {}),
        default_lora=True,
    )

    result_pil = res.images[0]
    result_arr = np.array(result_pil)

    if (tgt_h, tgt_w) != (h, w):
        result_arr = cv2.resize(result_arr, (w, h), interpolation=cv2.INTER_LINEAR)

    return result_arr


# ------------------------------------------------------------------
# FastAPI endpoint
# ------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_model()
    yield


app = FastAPI(title="FluxText Renderer API", lifespan=lifespan)


@app.post("/render", response_model=RenderResponse)
def render(req: RenderRequest):
    image, regions, style_ref = decode_request(req)
    if not regions:
        return RenderResponse(image=encode_image(image))

    result = _render_fluxtext(image, regions)
    return RenderResponse(image=encode_image(result))


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _pipe is not None}


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="FluxText Renderer API Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run(
        "fluxtext_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
