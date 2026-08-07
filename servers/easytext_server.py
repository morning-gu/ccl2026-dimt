"""EasyText rendering API server.

Wraps songyiren725/EasyText (FLUX DiT + LoRA) as a standalone FastAPI service.
The pipeline plugin calls this server via HTTP instead of importing the
EasyText repo directly, achieving full dependency isolation.

Prerequisites:
  - EasyText repo cloned: https://github.com/songyiren725/EasyText
  - FLUX.1-dev base model (HuggingFace: black-forest-labs/FLUX.1-dev)
  - EasyText pretrain LoRA + fine-tune LoRA weights
  - pip install fastapi uvicorn pydantic numpy opencv-python Pillow
  - pip install -r EasyText/requirements.txt  (flash_attn, diffusers, torch, ...)
  - Set EASYTEXT_REPO_PATH, EASYTEXT_FLUX_PATH, EASYTEXT_PRETRAIN_LORA,
    EASYTEXT_FINETUNE_LORA, EASYTEXT_FONT_PATH env vars.
  - Set EASYTEXT_OFFLOAD to control GPU memory strategy:
    "none"       - load everything onto GPU (.to("cuda")), needs ~34GB VRAM
    "model"      - per-component CPU offload (default), needs ~12-16GB VRAM
    "sequential" - per-layer CPU offload, needs ~8-10GB VRAM (slowest)
  - Set EASYTEXT_QUANT to quantize the transformer for lower VRAM:
    "none" - no quantization
    "nf4"  - 4-bit NormalFloat via bitsandbytes, transformer ~6GB (default)
    "fp8"  - 8-bit float via optimum-quanto, transformer ~12GB
    "8bit" - 8-bit int via bitsandbytes, transformer ~12GB
           (direct load from disk, no full-precision CPU RAM needed)

Usage:
  python easytext_server.py --host 0.0.0.0 --port 8001
"""
import argparse
import json
import logging
import math
import os
import sys
import tempfile
from typing import List, Optional, Tuple

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

logger = logging.getLogger("easytext_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

_pipeline = None
_repo_path = None
_font = None


def _init_model():
    """Load FLUX pipeline + EasyText LoRA weights."""
    global _pipeline, _repo_path, _font

    repo = os.environ.get("EASYTEXT_REPO_PATH", "")
    if not repo:
        raise RuntimeError(
            "EASYTEXT_REPO_PATH is not set. Clone the EasyText repo and "
            "set this env var to its root path."
        )
    if repo not in sys.path:
        sys.path.insert(0, repo)

    import torch
    from src.pipeline_pe_clone_multisample import FluxPipeline

    # Monkey-patch: fix hardcoded target_width_ in prepare_img_ids_new
    # EasyText hardcodes target_width_=70 but layout width is dynamic,
    # causing ValueError when layout_width > 70.
    import einops
    import src.pipeline_pe_clone_multisample as _et_pipe
    _orig_prepare_img_ids_new = _et_pipe.prepare_img_ids_new

    def _patched_prepare_img_ids_new(batch_size, packed_latent_height_layout, packed_latent_width_layout, packed_latent_height_img, packed_latent_width_img, position_lists):
        all_img_ids_layout = []
        for position_list in position_lists:
            img_ids = torch.zeros(packed_latent_height_img, packed_latent_width_img, 3)
            img_ids[..., 1] = img_ids[..., 1] + torch.arange(packed_latent_height_img)[:, None]
            img_ids[..., 2] = img_ids[..., 2] + torch.arange(packed_latent_width_img)[None, :]
            img_ids_np = img_ids.numpy().copy()
            target_width_ = packed_latent_width_layout
            pad_width = target_width_ - packed_latent_width_img
            if pad_width > 0:
                img_ids_np = np.pad(img_ids_np, ((0, 0), (0, pad_width), (0, 0)), mode="constant")

            layout_width = packed_latent_width_layout
            layout_height = packed_latent_height_layout
            img_ids_layout = torch.zeros(layout_height, layout_width, 3)
            img_ids_np_layout = img_ids_layout.numpy()
            for src_pts_list, dst_pts_2_list in position_list:
                src_pts = np.array(src_pts_list, dtype=np.float32)
                dst_top_left = np.array(dst_pts_2_list[0], dtype=np.float32)
                dst_bottom_right = np.array(dst_pts_2_list[1], dtype=np.float32)
                src_pts = src_pts // 16
                dst_top_left //= 16
                dst_bottom_right //= 16

                dst_pts_2 = np.array([
                    dst_top_left,
                    [dst_bottom_right[0], dst_top_left[1]],
                    dst_bottom_right,
                    [dst_top_left[0], dst_bottom_right[1]]
                ], dtype=np.float32)

                tps = cv2.createThinPlateSplineShapeTransformer()
                sourceshape = src_pts.reshape(1, -1, 2)
                targetshape = dst_pts_2.reshape(1, -1, 2)
                matches = [cv2.DMatch(i, i, 0) for i in range(len(src_pts))]
                tps.estimateTransformation(targetshape, sourceshape, matches)
                warped_img = tps.warpImage(img_ids_np)

                x_min, y_min = map(int, dst_top_left)
                x_max, y_max = map(int, dst_bottom_right)

                warped_crop = warped_img[y_min:y_max+1, x_min:x_max+1]
                img_ids_np_layout[y_min:y_max+1, x_min:x_max+1] = warped_crop

            img_ids_tensor = torch.from_numpy(img_ids_np_layout).float()
            all_img_ids_layout.append(img_ids_tensor)
        img_ids = einops.repeat(img_ids, "h w c -> b (h w) c", b=batch_size)
        img_ids_batch = torch.stack(all_img_ids_layout, dim=0)
        img_ids_batch = img_ids_batch.view(batch_size, -1, 3)
        img_ids = torch.cat((img_ids, img_ids_batch), dim=1)
        img_ids = img_ids.squeeze(0)
        return img_ids

    _et_pipe.prepare_img_ids_new = _patched_prepare_img_ids_new
    logger.info("Applied monkey-patch: fix prepare_img_ids_new target_width_")

    flux_path = os.environ.get("EASYTEXT_FLUX_PATH", "black-forest-labs/FLUX.1-dev")
    pretrain_lora = os.environ.get("EASYTEXT_PRETRAIN_LORA", "")
    finetune_lora = os.environ.get("EASYTEXT_FINETUNE_LORA", "")

    quant = os.environ.get("EASYTEXT_QUANT", "nf4").lower()

    if quant in ("nf4", "fp8", "8bit"):
        from diffusers import FluxTransformer2DModel

        if quant == "nf4":
            from diffusers import BitsAndBytesConfig
            qconfig = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            logger.info("Quantization: NF4 (4-bit, needs bitsandbytes)")
        else:
            if quant == "fp8":
                from diffusers import QuantoConfig
                qconfig = QuantoConfig(weights_dtype="float8")
                logger.info("Quantization: FP8 (8-bit float, needs optimum-quanto)")
            else:  # 8bit
                from diffusers import BitsAndBytesConfig
                qconfig = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_has_fp16_weight=False,
                )
                logger.info("Quantization: 8-bit int (needs bitsandbytes)")

        transformer = FluxTransformer2DModel.from_pretrained(
            flux_path,
            subfolder="transformer",
            quantization_config=qconfig,
            torch_dtype=torch.bfloat16,
        )
        pipeline = FluxPipeline.from_pretrained(
            flux_path,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )
    else:
        pipeline = FluxPipeline.from_pretrained(
            flux_path,
            torch_dtype=torch.bfloat16,
        )

    offload = os.environ.get("EASYTEXT_OFFLOAD", "model").lower()
    if offload == "none":
        pipeline = pipeline.to("cuda")
        logger.info("GPU offload: none (full GPU residency, ~34GB VRAM)")
    elif offload == "sequential":
        pipeline.enable_sequential_cpu_offload()
        logger.info("GPU offload: sequential (per-layer, ~8-10GB VRAM)")
    else:
        pipeline.enable_model_cpu_offload()
        logger.info("GPU offload: model (per-component, ~12-16GB VRAM)")

    if pretrain_lora:
        pipeline.load_lora_weights(pretrain_lora)
        if quant == "none":
            pipeline.fuse_lora()
            pipeline.unload_lora_weights()
            logger.info("EasyText pretrain LoRA fused from %s", pretrain_lora)
        else:
            logger.info("EasyText pretrain LoRA loaded (not fused, quantized model) from %s", pretrain_lora)

    if finetune_lora:
        pipeline.load_lora_weights(finetune_lora)
        logger.info("EasyText fine-tune LoRA loaded from %s", finetune_lora)

    _pipeline = pipeline
    _repo_path = repo


    font_path = os.environ.get(
        "EASYTEXT_FONT_PATH",
        os.path.join(repo, "font", "arial.ttf"),
    )
    from PIL import ImageFont
    _font = ImageFont.truetype(font_path, size=55)

    logger.info("EasyText model loaded (FLUX + LoRA)")


# ------------------------------------------------------------------
# Condition image generation (EasyFont representation)
# ------------------------------------------------------------------

def _build_condition_image(
    regions: List[RegionData],
) -> Tuple["object", List[Tuple[Tuple, Tuple]]]:
    """Render target text in standard font -> condition image + source positions."""
    from PIL import Image, ImageDraw

    cell_h = 64
    margin = 3
    line_images = []
    src_positions = []

    for r in regions:
        text = r.translated_text
        if not text:
            continue

        font_size = 55
        font = _font.font_variant(size=font_size)

        tmp = Image.new("RGB", (1, 1), "white")
        tmp_draw = ImageDraw.Draw(tmp)
        bbox = tmp_draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        while th > cell_h - margin and font_size > 10:
            font_size -= 2
            font = _font.font_variant(size=font_size)
            bbox = tmp_draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]

        line_w = tw - tw % 16 + 32
        line_img = Image.new("RGB", (line_w, cell_h), "white")
        draw = ImageDraw.Draw(line_img)

        x = (line_w - tw) // 2
        y = (cell_h - th) // 2 - bbox[1]
        draw.text((x, y), text, font=font, fill="black")

        line_images.append(line_img)

    if not line_images:
        return None, []

    total_w = max(img.width for img in line_images)
    total_h = len(line_images) * cell_h
    condition = Image.new("RGB", (total_w, total_h), "white")

    current_y = 0
    for i, img in enumerate(line_images):
        condition.paste(img, (0, current_y))
        src_positions.append(((0, current_y), (img.width - 1, current_y + cell_h - 1)))
        current_y += cell_h

    return condition, src_positions


# ------------------------------------------------------------------
# Position file generation (ICPA mapping)
# ------------------------------------------------------------------

def _build_position_data(
    regions: List[RegionData],
    src_positions: List[Tuple],
    orig_h: int,
    orig_w: int,
    tgt_h: int,
    tgt_w: int,
) -> list:
    """Build ICPA position mapping ([dst_quad, src_rect], ...).

    The destination quad comes from the source image text bbox in original
    pixel space (orig_h x orig_w). The FLUX pipeline generates at (tgt_h,
    tgt_w) and ``prepare_img_ids_new`` interprets destination coordinates in
    that generation canvas, so the quad is rescaled and clamped to the canvas
    before being written to the position file.
    """
    sx = tgt_w / orig_w if orig_w else 1.0
    sy = tgt_h / orig_h if orig_h else 1.0

    position_data = []

    for i, r in enumerate(regions):
        if i >= len(src_positions):
            break
        if not r.translated_text:
            continue

        src_rect = [list(src_positions[i][0]), list(src_positions[i][1])]

        if r.bbox_poly and len(r.bbox_poly) >= 4:
            dst_quad = [[float(p[0]) * sx, float(p[1]) * sy] for p in r.bbox_poly[:4]]
        else:
            x1, y1, x2, y2 = [float(v) for v in r.bbox[:4]]
            dst_quad = [
                [x1 * sx, y1 * sy],
                [x2 * sx, y1 * sy],
                [x2 * sx, y2 * sy],
                [x1 * sx, y2 * sy],
            ]

        dst_quad = [
            [min(max(px, 0.0), float(tgt_w)), min(max(py, 0.0), float(tgt_h))]
            for px, py in dst_quad
        ]

        position_data.append([dst_quad, src_rect])

    return position_data


# ------------------------------------------------------------------
# Prompt generation
# ------------------------------------------------------------------

def _color_to_name(bgr: Tuple) -> str:
    b, g, r = int(bgr[0]), int(bgr[1]), int(bgr[2])
    if r > 200 and g > 200 and b > 200:
        return "white"
    if r < 60 and g < 60 and b < 60:
        return "black"
    if r > 180 and g < 100 and b < 100:
        return "red"
    if r < 100 and g > 180 and b < 100:
        return "green"
    if r < 100 and g < 100 and b > 180:
        return "blue"
    if r > 180 and g > 180 and b < 100:
        return "yellow"
    if r > 180 and g < 150 and b > 100:
        return "orange"
    if r < 100 and g > 150 and b > 150:
        return "cyan"
    if r > 150 and g < 100 and b > 150:
        return "magenta"
    return "colored"


def _build_prompt(
    regions: List[RegionData],
    image_context: Optional[str] = None,
) -> str:
    triggers = []
    descriptions = []
    for i, r in enumerate(regions):
        if not r.translated_text:
            continue
        trigger = f"<sks{i+1}>"
        triggers.append(trigger)

        style = r.style_info or {}
        parts = []

        color = style.get("color")
        if isinstance(color, (tuple, list)) and len(color) >= 3:
            parts.append(_color_to_name(color))

        weight = style.get("font_weight", "normal")
        if weight == "bold":
            parts.append("bold")

        if parts:
            descriptions.append(f"The {trigger} is {' '.join(parts)}, clearly rendered")
        else:
            descriptions.append(f"The {trigger} is clearly rendered")

    trigger_str = " ".join(triggers)
    style_str = ". ".join(descriptions)

    if image_context:
        base = image_context
    else:
        base = "a product image with text"

    prompt = f"{base}. {style_str}."
    return prompt


# ------------------------------------------------------------------
# Resolution matching
# ------------------------------------------------------------------

def _match_resolution(height: int, width: int) -> Tuple[int, int]:
    target_pixels = 1024 * 1024
    aspect = width / height

    tgt_w = int(math.sqrt(target_pixels * aspect) // 16) * 16
    tgt_h = int(target_pixels / tgt_w // 16) * 16

    # EasyText prepare_img_ids_new hardcodes a max latent width (target_width_=70),
    # so the generation width must satisfy tgt_w // 16 <= 70. Cap wide canvases
    # while preserving the source aspect ratio to avoid a negative np.pad.
    max_w = 70 * 16
    if tgt_w > max_w:
        tgt_w = max_w
        tgt_h = int(tgt_w / aspect // 16) * 16

    tgt_w = max(tgt_w, 256)
    tgt_h = max(tgt_h, 256)

    return tgt_h, tgt_w


# ------------------------------------------------------------------
# Mask-based compositing
# ------------------------------------------------------------------

def _composite_result(
    erased_image: np.ndarray,
    generated_image: np.ndarray,
    regions: List[RegionData],
    dilate: int = 3,
) -> np.ndarray:
    h, w = erased_image.shape[:2]
    gen_resized = cv2.resize(generated_image, (w, h), interpolation=cv2.INTER_LINEAR)

    mask = np.zeros((h, w), dtype=np.uint8)
    for r in regions:
        if not r.translated_text:
            continue
        if r.bbox_poly and len(r.bbox_poly) >= 3:
            pts = np.array([[int(p[0]), int(p[1])] for p in r.bbox_poly], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
        else:
            x1, y1, x2, y2 = [int(v) for v in r.bbox[:4]]
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    if dilate > 0:
        kernel = np.ones((dilate, dilate), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

    mask_f32 = mask.astype(np.float32) / 255.0
    mask_blurred = cv2.GaussianBlur(mask_f32, (7, 7), 0)

    mask_3c = np.stack([mask_blurred] * 3, axis=-1)
    result = (
        gen_resized.astype(np.float32) * mask_3c
        + erased_image.astype(np.float32) * (1.0 - mask_3c)
    )
    return result.astype(np.uint8)


# ------------------------------------------------------------------
# Core rendering
# ------------------------------------------------------------------

def _render_easytext(
    erased_image: np.ndarray,
    regions: List[RegionData],
    style_reference: Optional[np.ndarray],
) -> np.ndarray:
    import torch
    from PIL import Image

    h, w = erased_image.shape[:2]
    tgt_h, tgt_w = _match_resolution(h, w)

    condition_image, src_positions = _build_condition_image(regions)
    if condition_image is None:
        return erased_image

    position_data = _build_position_data(regions, src_positions, h, w, tgt_h, tgt_w)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix="-position.txt", delete=False, encoding="utf-8"
    ) as f:
        json.dump(position_data, f, ensure_ascii=False)
        position_file = f.name

    try:
        image_context = None
        prompt = _build_prompt(regions, image_context)

        generator = torch.Generator(device="cuda")
        generator.manual_seed(42)

        result = _pipeline(
            prompt=prompt,
            condition_image=condition_image,
            height=tgt_h,
            width=tgt_w,
            guidance_scale=3.5,
            num_inference_steps=20,
            position_file=position_file,
            max_sequence_length=512,
            generator=generator,
        ).images[0]

        generated = np.array(result)
        final = _composite_result(erased_image, generated, regions, dilate=3)
        return final

    finally:
        os.unlink(position_file)


# ------------------------------------------------------------------
# FastAPI endpoint
# ------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_model()
    yield


app = FastAPI(title="EasyText Renderer API", lifespan=lifespan)


@app.post("/render", response_model=RenderResponse)
def render(req: RenderRequest):
    image, regions, style_ref = decode_request(req)
    if not regions:
        return RenderResponse(image=encode_image(image))

    result = _render_easytext(image, regions, style_ref)
    return RenderResponse(image=encode_image(result))


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _pipeline is not None}


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="EasyText Renderer API Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run(
        "easytext_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
