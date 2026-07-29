"""Text erasure and rendering module with multiple backends.

Supports:
- LaMA (Large Mask Inpainting) for high-quality text erasure
- OpenCV inpainting as fallback
- AnyText2 for style-preserving text rendering
- Stable Diffusion inpainting for text fusion
- PIL/Pillow as basic fallback renderer
"""
import logging
from typing import List, Optional, Tuple
import numpy as np

from .config import PipelineConfig
from .selective_translator import TextRegion

logger = logging.getLogger(__name__)


class TextEraser:
    """Erase text from images while preserving background.

    Backends: lama > opencv
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._lama_model = None
        self._backend = config.erasure_model

    def _init_lama(self):
        """Initialize LaMA inpainting model."""
        try:
            # LaMA from simple-lama-inpainting or lama-cleaner
            from simple_lama_inpainting import SimpleLAMA
            self._lama_model = SimpleLAMA()
            self._backend = "lama"
            logger.info("LaMA erasure model initialized")
        except ImportError:
            try:
                # Alternative: lama-cleaner
                logger.info("simple-lama not available, trying lama-cleaner")
                self._backend = "lama_server"
            except Exception:
                logger.warning("LaMA not available, falling back to OpenCV inpainting")
                self._backend = "opencv"

    def erase(
        self,
        image: np.ndarray,
        regions: List[TextRegion],
        dilate_pixels: int = 0,
    ) -> np.ndarray:
        """Erase text regions from image.

        Args:
            image: Input image (H, W, 3) uint8.
            regions: Text regions to erase.
            dilate_pixels: Extra dilation around text bbox for clean erasure.

        Returns:
            Image with text erased (background inpainted).
        """
        if not regions:
            return image

        if self._backend == "lama" and self._lama_model is None:
            self._init_lama()

        # Build mask from regions
        mask = self._build_mask(image.shape[:2], regions, dilate_pixels or self.config.erasure_dilate_pixels)

        if self._backend == "lama" and self._lama_model is not None:
            return self._erase_lama(image, mask)
        else:
            return self._erase_opencv(image, mask)

    def _build_mask(
        self,
        shape: Tuple[int, int],
        regions: List[TextRegion],
        dilate: int,
    ) -> np.ndarray:
        """Build binary mask from region bboxes."""
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)
        for region in regions:
            x1 = max(0, int(region.bbox[0]) - dilate)
            y1 = max(0, int(region.bbox[1]) - dilate)
            x2 = min(w, int(region.bbox[2]) + dilate)
            y2 = min(h, int(region.bbox[3]) + dilate)
            mask[y1:y2, x1:x2] = 255
        return mask

    def _erase_lama(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Erase using LaMA inpainting."""
        from PIL import Image
        img_pil = Image.fromarray(image)
        mask_pil = Image.fromarray(mask)
        result = self._lama_model(img_pil, mask_pil)
        return np.array(result)

    def _erase_opencv(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Erase using OpenCV inpainting (Telea / NS)."""
        try:
            import cv2
            result = cv2.inpaint(image, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
            return result
        except ImportError:
            # Fallback: simple blur-based inpainting
            logger.warning("OpenCV not available, using blur-based erasure")
            from PIL import Image, ImageFilter
            img_pil = Image.fromarray(image)
            mask_pil = Image.fromarray(mask).convert('L')
            # Blur the masked region
            blurred = img_pil.filter(ImageFilter.GaussianBlur(radius=10))
            result = Image.composite(blurred, img_pil, mask_pil)
            return np.array(result)


class TextRenderer:
    """Render translated text back into images.

    Backends: anytext2 > sd_inpaint > pil
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._anytext2 = None
        self._sd_pipeline = None
        self._backend = config.render_model

    def _init_anytext2(self):
        """Initialize AnyText2 model for style-preserving rendering."""
        try:
            # AnyText2 from: https://github.com/tyx-ch/AnyText2
            # Expected import pattern (adjust based on actual repo structure)
            logger.info("Attempting to load AnyText2 model...")
            # from anytext2 import AnyText2Pipeline
            # self._anytext2 = AnyText2Pipeline.from_pretrained(
            #     "model_path", device=self.config.device
            # )
            # For now, mark as unavailable until repo is cloned
            raise ImportError("AnyText2 not yet installed - clone from GitHub first")
        except ImportError as e:
            logger.warning("AnyText2 not available: %s", e)
            self._backend = "sd_inpaint"

    def _init_sd_inpaint(self):
        """Initialize Stable Diffusion inpainting pipeline."""
        try:
            from diffusers import StableDiffusionInpaintPipeline
            import torch
            model_id = "runwayml/stable-diffusion-inpainting"
            self._sd_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
            )
            if self.config.device == "cuda":
                self._sd_pipeline = self._sd_pipeline.to("cuda")
            self._backend = "sd_inpaint"
            logger.info("SD inpainting pipeline initialized")
        except ImportError:
            logger.warning("diffusers not available, falling back to PIL rendering")
            self._backend = "pil"

    def render(
        self,
        image: np.ndarray,
        regions: List[TextRegion],
        style_reference: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Render translated text onto the image.

        Args:
            image: Image with text erased (background only).
            regions: Text regions with translated_text filled.
            style_reference: Optional original image for style reference.

        Returns:
            Image with translated text rendered.
        """
        if not regions:
            return image

        # Only render regions that have translated text
        render_regions = [r for r in regions if r.translated_text]
        if not render_regions:
            return image

        if self._backend == "anytext2":
            if self._anytext2 is None:
                self._init_anytext2()
            if self._anytext2 is not None:
                return self._render_anytext2(image, render_regions, style_reference)
            # Fall through if init failed

        if self._backend in ("sd_inpaint", "anytext2"):
            if self._sd_pipeline is None:
                self._init_sd_inpaint()
            if self._sd_pipeline is not None:
                return self._render_sd_inpaint(image, render_regions)
            # Fall through if init failed

        return self._render_pil(image, render_regions)

    def _render_anytext2(
        self,
        image: np.ndarray,
        regions: List[TextRegion],
        style_reference: Optional[np.ndarray],
    ) -> np.ndarray:
        """Render using AnyText2 - best quality, preserves style."""
        # Build prompts for AnyText2
        # AnyText2 takes: image, text prompts with position hints
        prompts = []
        for region in regions:
            prompts.append({
                "text": region.translated_text,
                "bbox": region.bbox,
                "style": region.style_info,
            })

        # result = self._anytext2(
        #     image=image,
        #     prompts=prompts,
        #     style_reference=style_reference,
        # )
        # return result
        logger.warning("AnyText2 render not yet implemented - falling back to PIL")
        return self._render_pil(image, regions)

    def _render_sd_inpaint(self, image: np.ndarray, regions: List[TextRegion]) -> np.ndarray:
        """Render using SD inpainting - good quality for single regions."""
        from PIL import Image
        import torch

        result = image.copy()
        for region in regions:
            # Create mask for this region
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            x1, y1, x2, y2 = [int(v) for v in region.bbox[:4]]
            mask[y1:y2, x1:x2] = 255

            img_pil = Image.fromarray(result).resize((512, 512))
            mask_pil = Image.fromarray(mask).resize((512, 512))

            prompt = f"Text saying '{region.translated_text}', product label, clear typography"
            output = self._sd_pipeline(
                prompt=prompt,
                image=img_pil,
                mask_image=mask_pil,
                num_inference_steps=20,
            ).images[0]

            result = np.array(output.resize((w, h)))

        return result

    def _render_pil(self, image: np.ndarray, regions: List[TextRegion]) -> np.ndarray:
        """Render using PIL - basic but reliable fallback.

        Attempts to match original font size and approximate style.
        """
        from PIL import Image, ImageDraw, ImageFont

        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)

        for region in regions:
            x1, y1, x2, y2 = [int(v) for v in region.bbox[:4]]
            box_w = x2 - x1
            box_h = y2 - y1
            text = region.translated_text

            # Estimate font size from bbox height
            font_size = max(10, int(box_h * 0.8))

            # Try to load a suitable font
            font = self._load_font(font_size, text)

            # Get text color from style_info or default to dark
            fill_color = region.style_info.get("color", (0, 0, 0))
            if isinstance(fill_color, str):
                fill_color = (0, 0, 0)

            # Center text in bbox
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]

            # Scale font if text doesn't fit
            if text_w > box_w and font_size > 10:
                scale = box_w / text_w
                new_size = max(10, int(font_size * scale))
                font = self._load_font(new_size, text)
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_w = text_bbox[2] - text_bbox[0]
                text_h = text_bbox[3] - text_bbox[1]

            tx = x1 + (box_w - text_w) // 2
            ty = y1 + (box_h - text_h) // 2

            draw.text((tx, ty), text, font=font, fill=fill_color)

        return np.array(img_pil)

    def _load_font(self, size: int, text: str) -> object:
        """Load an appropriate font for the text content."""
        from PIL import ImageFont
        import os

        # Check if text contains CJK characters
        has_cjk = any('\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff' for c in text)

        # Try system fonts
        font_paths = []
        if has_cjk or any(c >= '\u3040' for c in text):
            # Japanese
            font_paths.append("C:/Windows/Fonts/meiryo.ttc")
            font_paths.append("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc")
        if has_cjk:
            # Chinese
            font_paths.append("C:/Windows/Fonts/msyh.ttc")
            font_paths.append("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc")
        # Latin fallback
        font_paths.append("C:/Windows/Fonts/arial.ttf")
        font_paths.append("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")

        for path in font_paths:
            if os.path.exists(path):
                try:
                    return ImageFont.truetype(path, size)
                except Exception:
                    continue

        return ImageFont.load_default()
