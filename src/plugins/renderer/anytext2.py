"""AnyText2 rendering plugin (Solution A/B)."""
import logging
import numpy as np
from typing import List, Optional
from interfaces.base import StageType
from interfaces.renderer import IRendererPlugin
from plugins.registry import register_plugin
from common.config import PipelineConfig
from common.selective_translator import TextRegion

logger = logging.getLogger(__name__)


@register_plugin(StageType.RENDERER, "anytext2")
class AnyText2RendererPlugin(IRendererPlugin):
    """AnyText2 rendering, migrated from common/renderer.py TextRenderer."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._anytext2 = None

    def _init_anytext2(self):
        import os, sys
        repo = os.environ.get(
            "ANYTEXT2_MODEL_PATH",
            getattr(self.config, "anytext2_model_path", "") or "AnyText2",
        )
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from ms_wrapper import AnyText2Model  # type: ignore
        infer_params = {
            "use_fp16": self.config.device == "cuda",
            "use_translator": False,
            "font_path": os.path.join(repo, "font", "Arial_Unicode.ttf"),
        }
        ckpt = os.environ.get("ANYTEXT2_CKPT", "")
        if ckpt:
            infer_params["model_path"] = ckpt
        self._anytext2 = AnyText2Model(model_dir=os.path.join(repo, "models"), **infer_params)
        if self.config.device == "cuda":
            self._anytext2 = self._anytext2.cuda(0)
        self._anytext2_repo = repo
        logger.info("AnyText2 model initialized from %s", repo)

    def render(self, image, regions, style_reference=None):
        if not regions:
            return image
        render_regions = [r for r in regions if r.translated_text]
        if not render_regions:
            return image
        if self._anytext2 is None:
            self._init_anytext2()
        return self._render_anytext2(image, render_regions, style_reference)

    def _render_anytext2(self, image, regions, style_reference):
        import os, cv2
        h, w = image.shape[:2]
        texts = [r.translated_text for r in regions]
        text_prompt = "#".join(texts)
        pos = np.zeros((h, w), dtype=np.uint8)
        for r in regions:
            x1, y1, x2, y2 = [int(v) for v in r.bbox[:4]]
            pos[y1:y2, x1:x2] = 255
        color_parts = []
        for r in regions:
            c = r.style_info.get("color")
            if isinstance(c, (tuple, list)) and len(c) >= 3:
                color_parts.append(",".join(str(int(v)) for v in c[:3]))
            else:
                color_parts.append("500,500,500")
        text_colors = " ".join(color_parts)
        params = {
            "mode": "edit",
            "sort_priority": "↕↓→",
            "show_debug": False,
            "revise_pos": False,
            "image_count": 1,
            "ddim_steps": 20,
            "image_width": w,
            "image_height": h,
            "strength": 1.0,
            "attnx_scale": 1.0,
            "font_hollow": None,
            "cfg_scale": 9.0,
            "eta": 0.0,
            "a_prompt": "best quality, extremely detailed,4k, HD, supper legible text, clear text edges, clear strokes, neat writing, no watermarks",
            "n_prompt": "low-res, bad anatomy, extra digit, fewer digits, cropped, worst quality, low quality, watermark, unreadable text, messy words, distorted text, disorganized writing, advertising picture",
            "base_model_path": "",
            "lora_path_ratio": "",
            "glyline_font_path": ["None"] * len(regions),
            "font_hint_image": [None] * len(regions),
            "font_hint_mask": [None] * len(regions),
            "text_colors": text_colors,
        }
        input_data = {
            "img_prompt": image[..., ::-1].copy(),
            "text_prompt": text_prompt,
            "seed": 1,
            "draw_pos": pos,
            "ori_image": image[..., ::-1].copy(),
        }
        results, rtn_code, rtn_warning, debug_info = self._anytext2(input_data, **params)
        if rtn_code < 0 or not results:
            raise RuntimeError(f"AnyText2 rendering failed (rtn_code={rtn_code}): {rtn_warning}")
        return np.array(results[0])[..., ::-1]
