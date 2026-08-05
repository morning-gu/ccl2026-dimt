"""
FluxText rendering plugin (Solution E).

Wraps AMAP-ML/FluxText (FLUX-Text) for end-to-end scene text editing.
FluxText performs erase + render in a single diffusion pass, so the pipeline
eraser stage should be set to ``noop`` when this renderer is active.

Prerequisites:
    - FluxText repo cloned and installed (pip install -r requirements.txt)
    - FLUX.1-Fill-dev base model accessible (HuggingFace cache or local path)
    - FLUX-Text LoRA weights downloaded from HuggingFace (GD-ML/FLUX-Text)
    - flash_attn installed

Environment variables:
    FLUXTEXT_REPO_PATH   Path to FluxText repo root (containing src/, train/)
    FLUXTEXT_MODEL_PATH  Path to .safetensors LoRA weights
    FLUXTEXT_CONFIG_PATH Path to config YAML (e.g. train/config/word_multi_size.yaml)
    FLUXTEXT_FONT_PATH   Path to Unicode font for glyph rendering (default: repo font/Arial_Unicode.ttf)
"""
import logging
import os
import sys
import math
import numpy as np
from typing import List, Optional

from interfaces.base import StageType
from interfaces.renderer import IRendererPlugin
from plugins.registry import register_plugin
from common.config import PipelineConfig
from common.selective_translator import TextRegion

logger = logging.getLogger(__name__)


@register_plugin(StageType.RENDERER, "fluxtext")
class FluxTextRendererPlugin(IRendererPlugin):
    """FLUX-Text scene text editing renderer.

    Unlike PIL/AnyText2 which render onto a pre-erased image, FluxText takes
    the *original* image and performs inpainting-based text replacement in one
    pass.  When this plugin is active, set the pipeline eraser to ``noop``
    so the original image is passed through unchanged.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._pipe = None
        self._flux_config = None
        self._repo_path = None
        self._font = None

    # ------------------------------------------------------------------
    # Lazy model initialisation
    # ------------------------------------------------------------------

    def _init_model(self):
        """Load FLUX-Text pipeline (FLUX.1-Fill-dev + LoRA)."""
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

        import yaml
        import torch
        from safetensors.torch import load_file
        from src.train.model import OminiModelFIll

        with open(config_path, "r") as f:
            flux_config = yaml.safe_load(f)

        training_config = flux_config["train"]
        trainable_model = OminiModelFIll(
            flux_pipe_id=flux_config["flux_path"],
            lora_config=training_config["lora_config"],
            device="cuda",
            dtype=getattr(torch, flux_config["dtype"]),
            optimizer_config=training_config["optimizer"],
            model_config=flux_config.get("model", {}),
            gradient_checkpointing=training_config.get(
                "gradient_checkpointing", False
            ),
            byt5_encoder_config=training_config.get("byt5_encoder", None),
        )

        state_dict = load_file(model_path)
        state_dict_new = {
            x.replace("lora_A", "lora_A.default")
             .replace("lora_B", "lora_B.default")
             .replace("transformer.", ""): v
            for x, v in state_dict.items()
        }
        trainable_model.transformer.load_state_dict(state_dict_new, strict=False)

        self._pipe = trainable_model.flux_pipe
        self._flux_config = flux_config
        self._repo_path = repo

        font_path = os.environ.get(
            "FLUXTEXT_FONT_PATH",
            os.path.join(repo, "font", "Arial_Unicode.ttf"),
        )
        from PIL import ImageFont
        self._font = ImageFont.truetype(font_path, size=60)

        logger.info("FLUX-Text model loaded from %s", model_path)

    # ------------------------------------------------------------------
    # Glyph / mask generation
    # ------------------------------------------------------------------

    def _draw_glyph(self, text, polygon, width, height):
        """Render target text into a binary glyph image at the target position.

        Simplified version of FluxText's ``draw_glyph2``: uses the OCR
        polygon to place and orient the text.
        """
        import cv2
        from PIL import Image, ImageDraw

        if polygon is None or len(polygon) < 3:
            return self._draw_glyph_bbox(text, width, height)

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
        font = self._font

        image4ratio = Image.new("RGB", img.size, "white")
        draw_ratio = ImageDraw.Draw(image4ratio)
        _, _, _tw, _th = draw_ratio.textbbox(xy=(0, 0), text=text, font=font)
        if _th == 0:
            _th = 1
        text_w = min(w, h) * (_tw / _th)
        if text_w <= max(w, h) and len(text) > 1 and not vert:
            for i in range(1, 100):
                text_space = self._insert_spaces(text, i)
                _, _, _tw2, _th2 = draw_ratio.textbbox(
                    xy=(0, 0), text=text_space, font=font
                )
                if min(w, h) * (_tw2 / _th2) > max(w, h):
                    break
            text = self._insert_spaces(text, i - 1)
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

    def _draw_glyph_bbox(self, text, width, height, bbox=None):
        """Simple axis-aligned glyph rendering (fallback)."""
        from PIL import Image, ImageDraw
        img = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(img)
        if bbox:
            x1, y1, x2, y2 = bbox
        else:
            x1, y1, x2, y2 = 0, 0, width, height
        box_h = y2 - y1
        font_size = max(10, int(box_h * 0.8))
        font = self._font.font_variant(size=font_size)
        draw.text((x1, y1), text, font=font, fill=255)
        return np.array(img)

    @staticmethod
    def _insert_spaces(string, n_space):
        if n_space == 0:
            return string
        new_string = ""
        for char in string:
            new_string += char + " " * n_space
        return new_string[:-n_space]

    def _build_mask(self, regions, height, width, dilate=3):
        """Build a binary mask covering all text regions."""
        import cv2
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

    def _build_glyph_image(self, regions, height, width):
        """Combine all region glyphs into a single glyph image."""
        glyph = np.zeros((height, width), dtype=np.uint8)
        for r in regions:
            poly = getattr(r, "bbox_poly", None)
            if poly and len(poly) >= 3:
                g = self._draw_glyph(r.translated_text, poly, width, height)
            else:
                x1, y1, x2, y2 = [int(v) for v in r.bbox[:4]]
                g = self._draw_glyph_bbox(
                    r.translated_text, width, height, bbox=(x1, y1, x2, y2)
                )
            glyph = np.maximum(glyph, g)
        return glyph

    # ------------------------------------------------------------------
    # Prompt generation
    # ------------------------------------------------------------------

    def _build_prompt(self, image, regions):
        """Build a text prompt for FluxText.

        Format: "<image description>, that reads <text1> , <text2> ."
        The image description is kept simple; a VLM caption can be injected
        via the ``image_context`` field on regions if available.
        """
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

    def _match_resolution(self, height, width):
        """Find the closest supported resolution for FluxText inference."""
        num_pixel = min(self._PIXELS, key=lambda x: abs(x - width * height))
        D = 16  # VAE spatial compression
        aspect = height / width
        best = None
        best_diff = float("inf")
        for ratio in self._ASPECT_RATIOS:
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
    # Main render entry point
    # ------------------------------------------------------------------

    def render(
        self,
        image: np.ndarray,
        regions: List[TextRegion],
        style_reference: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Render translated text using FLUX-Text scene text editing.

        ``image`` should be the *original* (or erased) image.  When the
        pipeline eraser is ``noop``, this receives the original image and
        FluxText handles erasure internally via inpainting.
        """
        if not regions:
            return image
        render_regions = [r for r in regions if r.translated_text]
        if not render_regions:
            return image

        if self._pipe is None:
            self._init_model()

        return self._render_fluxtext(image, render_regions)

    def _render_fluxtext(self, image, regions):
        import torch
        import cv2
        from PIL import Image
        from src.flux.condition import Condition
        from src.flux.generate_fill import generate_fill

        h, w = image.shape[:2]
        tgt_h, tgt_w = self._match_resolution(h, w)

        # Build glyph and mask at original resolution
        glyph_full = self._build_glyph_image(regions, h, w)
        mask_full = self._build_mask(regions, h, w)

        # Resize to target resolution
        glyph_resized = cv2.resize(glyph_full, (tgt_w, tgt_h), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask_full, (tgt_w, tgt_h), interpolation=cv2.INTER_NEAREST)
        img_resized = cv2.resize(image, (tgt_w, tgt_h), interpolation=cv2.INTER_LINEAR)

        # Prepare condition inputs for FluxText:
        # condition_img = [glyph_normalized, mask_normalized, original_image]
        glyph_rgb = np.stack([glyph_resized] * 3, axis=-1)
        glyph_norm = (255 - glyph_rgb.astype(np.float64)) / 255.0
        mask_norm = mask_resized.astype(np.float64) / 255.0
        condition_img = [glyph_norm, mask_norm, img_resized]

        prompt = self._build_prompt(img_resized, regions)

        condition = Condition(
            condition_type="word_fill",
            condition=condition_img,
            position_delta=[0, 0],
        )

        generator = torch.Generator(device="cuda")
        generator.manual_seed(42)

        res = generate_fill(
            self._pipe,
            prompt=prompt,
            conditions=[condition],
            height=tgt_h,
            width=tgt_w,
            generator=generator,
            model_config=self._flux_config.get("model", {}),
            default_lora=True,
        )

        result_pil = res.images[0]
        result_arr = np.array(result_pil)

        # Resize back to original resolution if needed
        if (tgt_h, tgt_w) != (h, w):
            result_arr = cv2.resize(result_arr, (w, h), interpolation=cv2.INTER_LINEAR)

        return result_arr
