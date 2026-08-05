"""Self-supervised Text Erasing plugin (STRnet2).

Repo: https://github.com/alimama-creative/Self-supervised-Text-Erasing
Checkpoints: https://huggingface.co/alimama-creative/Self-Supervised-Text-Erasing
License: Apache-2.0
"""
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


@register_plugin(StageType.ERASER, "sser")
class SSEREraserPlugin(IEraserPlugin):
    """Self-supervised Text Erasing via STRnet2 (alimama-creative).

    A GAN-based text removal network trained on PosterErase (210K poster
    images) and SCUT-EnsText. Unlike mask-driven inpainters (LaMA, SD),
    STRnet2 predicts both the erased image and a text mask in one forward
    pass, so it does not need an explicit mask as input -- only the image.
    The OCR-derived mask is still used for the final composite to preserve
    the original background outside text regions.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._model = None

    def _init_model(self):
        import os
        import sys
        import torch
        repo = os.environ.get("SSER_REPO", "")
        if not repo:
            raise RuntimeError(
                "SSER_REPO not set. Clone "
                "https://github.com/alimama-creative/Self-supervised-Text-Erasing "
                "and set SSER_REPO to the repo root."
            )
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from models.src.erase.sa_gan import STRnet2  # type: ignore

        ckpt = os.environ.get("SSER_CKPT", "")
        if not ckpt:
            raise RuntimeError(
                "SSER_CKPT not set. Download best_net_G.pth from "
                "https://huggingface.co/alimama-creative/Self-Supervised-Text-Erasing "
                "and set SSER_CKPT to the file path."
            )

        self._model = STRnet2(3)
        state_dict = torch.load(ckpt, map_location="cpu")
        # Strip DataParallel "module." prefix from training checkpoints.
        cleaned = {}
        for k, v in state_dict.items():
            cleaned[k[7:] if k.startswith("module.") else k] = v
        missing, unexpected = self._model.load_state_dict(cleaned, strict=False)
        if missing or unexpected:
            logger.warning(
                "SSER state_dict mismatch -- missing %d keys, unexpected %d keys",
                len(missing), len(unexpected),
            )
        self._model.eval()
        if self.config.device == "cuda":
            self._model = self._model.cuda()
        logger.info("SSER (STRnet2) initialized from %s", ckpt)

    def erase(
        self,
        image: np.ndarray,
        regions: List[TextRegion],
        dilate_pixels: int = 0,
    ) -> np.ndarray:
        if not regions:
            return image
        if self._model is None:
            self._init_model()
        import cv2
        import torch

        dilate = dilate_pixels or self.config.erasure_dilate_pixels
        mask = build_mask(image.shape[:2], regions, dilate, image=image)

        h, w = image.shape[:2]
        # STRnet2 is fully convolutional with /32 downsampling; pad to /32.
        h_pad = h + (32 - h % 32) % 32
        w_pad = w + (32 - w % 32) % 32
        # BGR -> RGB for the model (PIL/torchvision convention).
        img_rgb = image[..., ::-1].copy()
        if h_pad != h or w_pad != w:
            img_rgb = cv2.resize(img_rgb, (w_pad, h_pad))

        img_t = torch.from_numpy(img_rgb.astype("float32") / 255.0)
        img_t = img_t.permute(2, 0, 1).unsqueeze(0)
        dev = next(self._model.parameters()).device
        img_t = img_t.to(dev)

        with torch.no_grad():
            outputs = self._model(img_t)
            # STRnet2.forward returns 7 tensors; index 4 is fake_B (erased image).
            fake_b = outputs[4]

        out = fake_b[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
        out = (out * 255).astype(np.uint8)
        if h_pad != h or w_pad != w:
            out = cv2.resize(out, (w, h))
        # RGB -> BGR for the pipeline.
        out = out[..., ::-1]

        # Composite: keep original where there's no text (mask == 0).
        keep = mask == 0
        if image.ndim == out.ndim and image.shape[:2] == out.shape[:2]:
            out[keep] = image[keep]
        return out
