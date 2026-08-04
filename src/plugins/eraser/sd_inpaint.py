"""SD inpainting erasure plugin (Solution A default)."""
import logging
import numpy as np
from typing import List
from interfaces.base import StageType
from interfaces.eraser import IEraserPlugin
from plugins.registry import register_plugin
from common.config import PipelineConfig
from common.selective_translator import TextRegion
from ._common import build_mask

logger = logging.getLogger(__name__)


@register_plugin(StageType.ERASER, "sd_inpaint")
class SDInpaintEraserPlugin(IEraserPlugin):
    """SD inpainting, migrated from TextEraser._erase_sd."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._pipeline = None

    def _init_sd(self):
        from diffusers import StableDiffusionInpaintPipeline
        import torch
        model_id = "runwayml/stable-diffusion-inpainting"
        self._pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
        )
        if self.config.device == "cuda":
            self._pipeline = self._pipeline.to("cuda")
        logger.info("SD inpainting erasure backend initialized")

    def erase(self, image: np.ndarray, regions: List[TextRegion], dilate_pixels: int = 0) -> np.ndarray:
        if not regions:
            return image
        if self._pipeline is None:
            self._init_sd()
        mask = build_mask(
            image.shape[:2], regions,
            dilate_pixels or self.config.erasure_dilate_pixels,
            image=image,
        )
        import torch
        from PIL import Image
        h, w = image.shape[:2]
        img_pil = Image.fromarray(image).convert("RGB").resize((512, 512))
        mask_pil = Image.fromarray(mask).convert("L").resize((512, 512))
        with torch.no_grad():
            output = self._pipeline(
                prompt="", image=img_pil, mask_image=mask_pil,
                num_inference_steps=25, guidance_scale=1.0,
            ).images[0]
        result = np.array(output.resize((w, h)))
        keep = mask == 0
        if image.ndim == result.ndim and image.shape[:2] == result.shape[:2]:
            result[keep] = image[keep]
        return result
