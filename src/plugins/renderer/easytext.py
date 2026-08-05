"""
EasyText rendering plugin.

Wraps songyiren725/EasyText (FLUX DiT + LoRA) for multilingual text rendering.
EasyText is a generation-only model: it generates a new image from a prompt +
condition image (standard-font glyph) + position file (ICPA mapping).  It does
NOT have a native text-editing/inpainting mode like AnyText2.

Integration strategy for rendering backfill:
  1. The pipeline eraser has already removed original text -> erased_image.
  2. We use the erased_image as the base and build an EasyText condition:
     - condition_image: target text rendered in a standard font (white bg, black text)
     - position_file:   maps condition-image coords -> target bbox/polygon in the output
     - prompt:          image description with <sks1> triggers for each text line
  3. EasyText generates a full image.  We then composite only the text regions
     back onto the erased_image using the mask, preserving the original background
     outside the text areas.

  This "generate + composite" approach bridges the generation/editing gap.

Prerequisites:
  - EasyText repo cloned: https://github.com/songyiren725/EasyText
  - FLUX.1-dev base model (HuggingFace: black-forest-labs/FLUX.1-dev)
  - EasyText pretrain LoRA + fine-tune LoRA weights
  - pip install -r EasyText/requirements.txt
  - flash_attn installed (for FLUX transformer attention)

Environment variables:
  EASYTEXT_REPO_PATH     Path to EasyText repo root (containing src/, font/)
  EASYTEXT_FLUX_PATH     Path to FLUX.1-dev model (or HF repo id)
  EASYTEXT_PRETRAIN_LORA Path to pretrain LoRA weights (.safetensors)
  EASYTEXT_FINETUNE_LORA Path to fine-tune LoRA weights (.safetensors)
  EASYTEXT_FONT_PATH     Path to Unicode font for condition rendering (default: repo font/arial.ttf)
"""
import logging
import os
import sys
import json
import math
import tempfile
import numpy as np
from typing import List, Optional, Tuple

from interfaces.base import StageType
from interfaces.renderer import IRendererPlugin
from plugins.registry import register_plugin
from common.config import PipelineConfig
from common.selective_translator import TextRegion

logger = logging.getLogger(__name__)


@register_plugin(StageType.RENDERER, "easytext")
class EasyTextRendererPlugin(IRendererPlugin):
    """EasyText multilingual text rendering via FLUX DiT + LoRA.

    Unlike AnyText2 which natively supports text editing (masked image input),
    EasyText is generation-only.  This plugin uses a generate-then-composite
    strategy: generate a full image with EasyText, then blend only the text
    regions back onto the pre-erased image to preserve the original background.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._pipeline = None
        self._repo_path = None
        self._font = None

    # ------------------------------------------------------------------
    # Lazy model initialisation
    # ------------------------------------------------------------------

    def _init_model(self):
        """Load FLUX pipeline + EasyText LoRA weights."""
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

        flux_path = os.environ.get("EASYTEXT_FLUX_PATH", "black-forest-labs/FLUX.1-dev")
        pretrain_lora = os.environ.get("EASYTEXT_PRETRAIN_LORA", "")
        finetune_lora = os.environ.get("EASYTEXT_FINETUNE_LORA", "")

        pipeline = FluxPipeline.from_pretrained(
            flux_path,
            torch_dtype=torch.bfloat16,
        ).to("cuda")

        # Stage 1: load + fuse pretrain LoRA (glyph generation capability)
        if pretrain_lora:
            pipeline.load_lora_weights(pretrain_lora)
            pipeline.fuse_lora()
            pipeline.unload_lora_weights()
            logger.info("EasyText pretrain LoRA fused from %s", pretrain_lora)

        # Stage 2: load fine-tune LoRA (text-image blending quality)
        if finetune_lora:
            pipeline.load_lora_weights(finetune_lora)
            logger.info("EasyText fine-tune LoRA loaded from %s", finetune_lora)

        self._pipeline = pipeline
        self._repo_path = repo

        # Font for condition image rendering
        font_path = os.environ.get(
            "EASYTEXT_FONT_PATH",
            os.path.join(repo, "font", "arial.ttf"),
        )
        from PIL import ImageFont
        self._font = ImageFont.truetype(font_path, size=55)

        logger.info("EasyText model loaded (FLUX + LoRA)")

    # ------------------------------------------------------------------
    # Condition image generation (EasyFont representation)
    # ------------------------------------------------------------------

    def _build_condition_image(
        self, regions: List[TextRegion]
    ) -> Tuple["Image.Image", List[Tuple[Tuple, Tuple]]]:
        """Render target text in standard font -> condition image + source positions.

        Each text line is rendered at 64px height on a white background.
        Returns the condition PIL image and a list of (top_left, bottom_right)
        positions for each line in the condition image.
        """
        from PIL import Image, ImageDraw

        cell_h = 64
        margin = 3
        line_images = []
        src_positions = []

        for r in regions:
            text = r.translated_text
            if not text:
                continue

            # Render text with standard font at 55px, auto-shrink to fit 64px height
            font_size = 55
            font = self._font.font_variant(size=font_size)

            # Measure text
            tmp = Image.new("RGB", (1, 1), "white")
            tmp_draw = ImageDraw.Draw(tmp)
            bbox = tmp_draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]

            # Shrink font if text height exceeds cell
            while th > cell_h - margin and font_size > 10:
                font_size -= 2
                font = self._font.font_variant(size=font_size)
                bbox = tmp_draw.textbbox((0, 0), text, font=font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]

            # Width: align to 16px boundary + 32px padding (matching EasyText convention)
            line_w = tw - tw % 16 + 32
            line_img = Image.new("RGB", (line_w, cell_h), "white")
            draw = ImageDraw.Draw(line_img)

            # Center text in the cell
            x = (line_w - tw) // 2
            y = (cell_h - th) // 2 - bbox[1]
            draw.text((x, y), text, font=font, fill="black")

            line_images.append(line_img)

        if not line_images:
            return None, []

        # Stack all lines vertically
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
        self,
        regions: List[TextRegion],
        src_positions: List[Tuple],
        height: int,
        width: int,
    ) -> list:
        """Build EasyText position file data (ICPA source->target mapping).

        Format: [[dst_quad, src_rect], ...]
        - dst_quad: 4-point polygon in target image (supports irregular regions via TPS)
        - src_rect: [top_left, bottom_right] in condition image

        For regular (axis-aligned) boxes, dst_quad is a simple rectangle.
        For rotated/irregular regions (bbox_poly available), use the 4-point polygon.
        """
        position_data = []

        for i, r in enumerate(regions):
            if i >= len(src_positions):
                break
            if not r.translated_text:
                continue

            src_rect = [list(src_positions[i][0]), list(src_positions[i][1])]

            # Determine target quadrilateral
            if r.bbox_poly and len(r.bbox_poly) >= 4:
                # Use OCR polygon for irregular/rotated text (TPS alignment)
                dst_quad = [list(p) for p in r.bbox_poly[:4]]
            else:
                # Axis-aligned bounding box
                x1, y1, x2, y2 = [int(v) for v in r.bbox[:4]]
                dst_quad = [
                    [x1, y1],
                    [x2, y1],
                    [x2, y2],
                    [x1, y2],
                ]

            position_data.append([dst_quad, src_rect])

        return position_data

    # ------------------------------------------------------------------
    # Prompt generation
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        regions: List[TextRegion],
        image_context: Optional[str] = None,
    ) -> str:
        """Build a FLUX prompt with <sksN> triggers for each text line.

        EasyText uses <sks1>, <sks2>, ... as placeholder triggers in the
        prompt.  The actual text content is provided via the condition image.
        The prompt describes the image scene and text styling.
        """
        triggers = []
        descriptions = []
        for i, r in enumerate(regions):
            if not r.translated_text:
                continue
            trigger = f"<sks{i+1}>"
            triggers.append(trigger)

            # Build per-region style description from extracted style_info
            style = r.style_info or {}
            parts = []

            color = style.get("color")
            if isinstance(color, (tuple, list)) and len(color) >= 3:
                # Convert BGR to color name approximation
                parts.append(self._color_to_name(color))

            weight = style.get("font_weight", "normal")
            if weight == "bold":
                parts.append("bold")

            if parts:
                descriptions.append(
                    f"The {trigger} is {' '.join(parts)}, clearly rendered"
                )
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

    @staticmethod
    def _color_to_name(bgr: Tuple) -> str:
        """Approximate BGR tuple to a color name for prompt."""
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

    # ------------------------------------------------------------------
    # Resolution matching (FLUX supports arbitrary resolutions)
    # ------------------------------------------------------------------

    def _match_resolution(self, height: int, width: int) -> Tuple[int, int]:
        """Match to nearest FLUX-supported resolution (multiple of 16).

        FLUX works best at 1024x1024 or similar total pixel counts.
        We preserve aspect ratio while targeting ~1024x1024 total pixels.
        """
        target_pixels = 1024 * 1024
        aspect = width / height

        tgt_w = int(math.sqrt(target_pixels * aspect) // 16) * 16
        tgt_h = int(target_pixels / tgt_w // 16) * 16

        # Ensure minimum dimensions
        tgt_w = max(tgt_w, 256)
        tgt_h = max(tgt_h, 256)

        return tgt_h, tgt_w

    # ------------------------------------------------------------------
    # Mask-based compositing (bridge generation -> editing)
    # ------------------------------------------------------------------

    def _composite_result(
        self,
        erased_image: np.ndarray,
        generated_image: np.ndarray,
        regions: List[TextRegion],
        dilate: int = 3,
    ) -> np.ndarray:
        """Blend generated text regions back onto the erased image.

        EasyText generates a full image.  We keep the generated content only
        within the text region masks and preserve the erased background outside.
        This bridges the generation/editing gap.
        """
        import cv2

        h, w = erased_image.shape[:2]
        gen_resized = cv2.resize(generated_image, (w, h), interpolation=cv2.INTER_LINEAR)

        # Build mask from region polygons/bboxes
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

        # Feather mask edges for smooth blending
        mask_f32 = mask.astype(np.float32) / 255.0
        mask_blurred = cv2.GaussianBlur(mask_f32, (7, 7), 0)

        # Alpha blend: result = gen * mask + erased * (1 - mask)
        mask_3c = np.stack([mask_blurred] * 3, axis=-1)
        result = (
            gen_resized.astype(np.float32) * mask_3c
            + erased_image.astype(np.float32) * (1.0 - mask_3c)
        )
        return result.astype(np.uint8)

    # ------------------------------------------------------------------
    # Main render entry point
    # ------------------------------------------------------------------

    def render(
        self,
        image: np.ndarray,
        regions: List[TextRegion],
        style_reference: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Render translated text using EasyText.

        Args:
            image: Pre-erased image (original text removed by eraser stage).
            regions: Text regions with translated_text populated.
            style_reference: Original image (for style context, optional).

        Returns:
            Image with translated text rendered.
        """
        if not regions:
            return image

        render_regions = [r for r in regions if r.translated_text]
        if not render_regions:
            return image

        if self._pipeline is None:
            self._init_model()

        return self._render_easytext(image, render_regions, style_reference)

    def _render_easytext(
        self,
        erased_image: np.ndarray,
        regions: List[TextRegion],
        style_reference: Optional[np.ndarray],
    ) -> np.ndarray:
        """Core EasyText rendering logic."""
        import torch
        import cv2
        from PIL import Image

        h, w = erased_image.shape[:2]
        tgt_h, tgt_w = self._match_resolution(h, w)

        # 1. Build condition image (standard-font glyph)
        condition_image, src_positions = self._build_condition_image(regions)
        if condition_image is None:
            return erased_image

        # 2. Build position file (ICPA mapping: condition -> target)
        position_data = self._build_position_data(
            regions, src_positions, tgt_h, tgt_w
        )

        # Write position file to temp
        with tempfile.NamedTemporaryFile(
            mode="w", suffix="-position.txt", delete=False, encoding="utf-8"
        ) as f:
            json.dump(position_data, f, ensure_ascii=False)
            position_file = f.name

        try:
            # 3. Build prompt with <sksN> triggers
            # Use image context from style_reference if available
            image_context = None
            prompt = self._build_prompt(regions, image_context)

            # 4. Run EasyText inference
            generator = torch.Generator(device="cuda")
            generator.manual_seed(42)

            result = self._pipeline(
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

            # 5. Composite: keep generated text regions, preserve erased background
            final = self._composite_result(erased_image, generated, regions, dilate=3)

            return final

        finally:
            os.unlink(position_file)
