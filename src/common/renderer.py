"""Text erasure and rendering, single-backend per solution (no degradation).
Each solution pins one erasure backend and one render backend via config:
    Solution A (AnyTrans):  erasure=sd_inpaint  render=anytext2
    Solution B (AnyText2):  erasure=lama        render=anytext2
    Solution C (e-commerce):erasure=opencv     render=pil
A missing dependency raises immediately instead of silently falling back to a
weaker method. PIL rendering is Solution C's own method (not a fallback).
"""
import logging
from typing import List, Optional, Tuple
import numpy as np
from .config import PipelineConfig
from .selective_translator import TextRegion
logger = logging.getLogger(__name__)
_font_manager = None  # lazy global so the FontManager is created only once
def _get_font_manager():
    global _font_manager
    if _font_manager is None:
        from .font_manager import FontManager
        _font_manager = FontManager()
    return _font_manager
class TextEraser:
    """Erase text from images while preserving background. Single backend, no fallback.
    Backends (by config.erasure_model):
        "pert"     : PERT scene text removal (wangyuxin87/PERT, TIP 2023).
                    Takes the full image and iteratively erases all text
                    strokes (3-stage cascade). Pretrained weights available
                    on Google Drive. This is the closest open-source
                    approximation of the stroke-level text erasure
                    (Li et al. 2023) used in the AnyTrans paper. After
                    erasure, preserved (brand/logo) regions are restored
                    from the original image.
        "lama"      : simple-lama-inpainting (Solution B). AnyTrans uses
                    stroke-level erasure (Li et al. 2023), which is not
                    open-sourced; LaMA is the documented eraser here.
        "sd_inpaint": diffusers StableDiffusionInpaintPipeline (Solution A),
                    used for background restoration only.
        "opencv"    : cv2.inpaint Telea (Solution C, e-commerce fast path).
        "strokenet" : the cascaded stroke-level erasure used in the AnyTrans
                    paper (STRNet from ZeroAct/SceneTextRemover-pytorch,
                    a reimplementation of the cited stroke-level text erasure).
    """
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._lama_model = None
        self._sd_pipeline = None
        self._strokenet = None
        self._pert_model = None
        self._backend = config.erasure_model
    def _init_lama(self):
        """Initialize LaMA. Raises if simple-lama-inpainting is not installed."""
        from simple_lama_inpainting import SimpleLAMA
        self._lama_model = SimpleLAMA()
        logger.info("LaMA erasure model initialized")
    def _init_sd_inpaint(self):
        """Initialize SD inpainting for background restoration. Raises if
        diffusers/torch are missing or the model cannot be loaded."""
        from diffusers import StableDiffusionInpaintPipeline
        import torch
        model_id = "runwayml/stable-diffusion-inpainting"
        self._sd_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
        )
        if self.config.device == "cuda":
            self._sd_pipeline = self._sd_pipeline.to("cuda")
        logger.info("SD inpainting erasure backend initialized")
    def _init_strokenet(self):
        """Initialize the cascaded stroke-level erasure model (STRNet).
        Clone https://github.com/ZeroAct/SceneTextRemover-pytorch and set
        STROKENET_REPO=/path/to/that/repo and STROKENET_CKPT=/path/to/weights.pth
        (the repo ships only training code; you must train or supply a
        checkpoint). Raises otherwise.
        """
        import os, sys
        import torch
        repo = os.environ.get("STROKENET_REPO", "SceneTextRemover-pytorch")
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from network import STRNet  # type: ignore
        self._strokenet = STRNet()
        ckpt = os.environ.get("STROKENET_CKPT", "")
        if not ckpt:
            raise RuntimeError(
                "STROKENET_CKPT not set. SceneTextRemover-pytorch ships no "
                "pretrained weights; train one (python train.py) or supply a "
                "checkpoint path, then set STROKENET_CKPT=/path/to/weights.pth"
            )
        self._strokenet.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self._strokenet.eval()
        if self.config.device == "cuda":
            self._strokenet = self._strokenet.cuda()
        logger.info("STRNet (stroke-level erasure) initialized from %s", ckpt)
    def _init_pert(self):
        """Initialize PERT scene text removal model.
        Clone https://github.com/wangyuxin87/PERT and download the pretrained
        model from Google Drive:
            git clone https://github.com/wangyuxin87/PERT /path/PERT
            # Download net_epoch_200.pth from the Google Drive link in README
        Then set:
            PERT_REPO=/path/PERT
            PERT_CKPT=/path/PERT/net_epoch_200.pth
        Raises if either is missing.
        """
        import os, sys
        import torch
        repo = os.environ.get("PERT_REPO", "")
        if not repo:
            raise RuntimeError(
                "PERT_REPO not set. Clone https://github.com/wangyuxin87/PERT "
                "and set PERT_REPO=/path/to/PERT to use stroke-level erasure."
            )
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from networks_sfnet import Pert  # type: ignore
        self._pert_model = Pert(use_GPU=(self.config.device == "cuda"))
        ckpt = os.environ.get("PERT_CKPT", "")
        if not ckpt:
            raise RuntimeError(
                "PERT_CKPT not set. Download pretrained weights from "
                "https://drive.google.com/file/d/1uU8lGUIp62W5HkwyjzY3Mc15O0-_jkKP "
                "and set PERT_CKPT=/path/to/net_epoch_200.pth"
            )
        state_dict = torch.load(ckpt, map_location="cpu")
        self._pert_model.load_state_dict(state_dict, strict=True)
        self._pert_model.eval()
        if self.config.device == "cuda":
            self._pert_model = self._pert_model.cuda()
        logger.info("PERT (scene text removal) initialized from %s", ckpt)
    def erase(self, image, regions, dilate_pixels=0):
        if not regions:
            return image
        # Lazily init the configured backend. No fallback: missing dep raises.
        if self._backend == "lama" and self._lama_model is None:
            self._init_lama()
        elif self._backend == "sd_inpaint" and self._sd_pipeline is None:
            self._init_sd_inpaint()
        elif self._backend == "strokenet" and self._strokenet is None:
            self._init_strokenet()
        elif self._backend == "pert" and self._pert_model is None:
            self._init_pert()
        mask = self._build_mask(image.shape[:2], regions, dilate_pixels or self.config.erasure_dilate_pixels)
        if self._backend == "lama":
            return self._erase_lama(image, mask)
        if self._backend == "sd_inpaint":
            return self._erase_sd(image, mask)
        if self._backend == "opencv":
            return self._erase_opencv(image, mask)
        if self._backend == "strokenet":
            return self._erase_strokenet(image, mask)
        if self._backend == "pert":
            return self._erase_pert(image, mask)
        raise ValueError(f"Unknown erasure_model: {self._backend!r}")
    def _build_mask(self, shape, regions, dilate):
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)
        for region in regions:
            x1 = max(0, int(region.bbox[0]) - dilate)
            y1 = max(0, int(region.bbox[1]) - dilate)
            x2 = min(w, int(region.bbox[2]) + dilate)
            y2 = min(h, int(region.bbox[3]) + dilate)
            mask[y1:y2, x1:x2] = 255
        return mask
    def _erase_lama(self, image, mask):
        from PIL import Image
        result = self._lama_model(Image.fromarray(image), Image.fromarray(mask))
        return np.array(result)
    def _erase_sd(self, image, mask):
        import torch
        from PIL import Image
        h, w = image.shape[:2]
        img_pil = Image.fromarray(image).convert("RGB").resize((512, 512))
        mask_pil = Image.fromarray(mask).convert("L").resize((512, 512))
        with torch.no_grad():
            output = self._sd_pipeline(
                prompt="", image=img_pil, mask_image=mask_pil,
                num_inference_steps=25, guidance_scale=1.0,
            ).images[0]
        result = np.array(output.resize((w, h)))
        keep = mask == 0
        if image.ndim == result.ndim and image.shape[:2] == result.shape[:2]:
            result[keep] = image[keep]
        return result
    def _erase_opencv(self, image, mask):
        import cv2
        return cv2.inpaint(image, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    def _erase_strokenet(self, image, mask):
        """Erase via the cascaded stroke-level STRNet.
        STRNet.forward(I, Mm) -> (Ms, Ite, Ms_, Ite_). Ite_ (the output of the
        final G'r stage) is the erased image. Inputs are the image tensor
        (B,3,H,W) in [0,1] and the text-region mask (B,1,H,W) in {0,1}.
        """
        import torch
        import torch.nn.functional as F
        h, w = image.shape[:2]
        dev = next(self._strokenet.parameters()).device
        # BGR uint8 -> RGB float [0,1], NCHW
        img_t = torch.from_numpy(image[..., ::-1].copy()).float().div(255.0).permute(2, 0, 1).unsqueeze(0).to(dev)
        m_t = torch.from_numpy((mask > 0).astype("float32")).float().unsqueeze(0).unsqueeze(0).to(dev)
        with torch.no_grad():
            _, _, _, ite_ = self._strokenet(img_t, m_t)
        out = ite_[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        out = (out * 255).astype("uint8")[..., ::-1]  # RGB -> BGR
        # STRNet replaces the whole image; keep non-text pixels identical
        # so untouched regions are bit-exact to the source.
        keep = mask == 0
        if image.ndim == out.ndim and image.shape[:2] == out.shape[:2]:
            out[keep] = image[keep]
        return out
    def _erase_pert(self, image, mask):
        """Erase text via PERT (scene text removal, TIP 2023).
        PERT takes the full image and iteratively erases all text strokes
        (3-stage cascade). Unlike STRNet/SD-inpaint, it does not accept a
        mask -- it detects text itself. To support selective erasure, we
        run PERT on the full image, then restore non-translatable regions
        from the original using the caller's mask.
        """
        import torch
        from torch.autograd import Variable
        h, w = image.shape[:2]
        dev = next(self._pert_model.parameters()).device
        # PERT expects RGB float [0,1], NCHW, dimensions multiple of 32
        h_pad = h + (32 - h % 32) % 32
        w_pad = w + (32 - w % 32) % 32
        img_rgb = image[..., ::-1].copy()  # BGR -> RGB
        if h_pad != h or w_pad != w:
            import cv2
            img_rgb = cv2.resize(img_rgb, (w_pad, h_pad))
        img_t = torch.from_numpy(img_rgb.astype("float32") / 255.0)
        img_t = img_t.permute(2, 0, 1).unsqueeze(0).to(dev)
        with torch.no_grad():
            out, _, _, _, _ = self._pert_model(img_t)
        out = out[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        out = (out * 255).astype("uint8")
        if h_pad != h or w_pad != w:
            import cv2
            out = cv2.resize(out, (w, h))
        out = out[..., ::-1]  # RGB -> BGR
        # Restore non-translatable regions (brand/logo/spec text) from
        # the original so preserved text is bit-exact.
        keep = mask == 0
        if image.ndim == out.ndim and image.shape[:2] == out.shape[:2]:
            out[keep] = image[keep]
        return out
class TextRenderer:
    """Render translated text. Single backend per solution, no degradation.
    Backends (by config.render_model):
        "anytext2": the AnyText2 pipeline (tyxsspa/AnyText2). Required for
                    Solutions A and B. Raises if AnyText2 is not installed.
        "pil"     : Pillow-based rendering (Solution C's own method). Style
                    info (color/font/weight/alignment) is preserved and, when
                    a style_reference is given, extracted on the fly.
    """
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._anytext2 = None
        self._backend = config.render_model
    def _init_anytext2(self):
        """Initialize AnyText2.
        Clone https://github.com/tyxsspa/AnyText2 and install its deps first:
            git clone https://github.com/tyxsspa/AnyText2 /path/AnyText2
            pip install -r /path/AnyText2/requirements.txt
        Then set ANYTEXT2_MODEL_PATH=/path/AnyText2 (repo root contains
        ms_wrapper.py and the models/ checkpoint dir). Raises otherwise.
        """
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
            "use_translator": False,  # we supply already-translated text
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
        if self._backend == "anytext2":
            if self._anytext2 is None:
                self._init_anytext2()
            return self._render_anytext2(image, render_regions, style_reference)
        if self._backend == "pil":
            return self._render_pil(image, render_regions, style_reference)
        raise ValueError(f"Unknown render_model: {self._backend!r}")
    def _render_anytext2(self, image, regions, style_reference):
        """Call the real AnyText2 pipeline in edit mode.
        AnyText2 expects: an image, the translated text (one string per line,
        separated by '#'), a draw_pos position mask marking where text goes,
        and a params dict. We build all of these from the regions. Any failure
        raises (no silent PIL fallback) so broken rendering is visible.
        """
        import os, cv2
        h, w = image.shape[:2]
        # AnyText2 text prompt: lines joined by '#'.
        texts = [r.translated_text for r in regions]
        text_prompt = "#".join(texts)
        # draw_pos: white rectangle per region on a black canvas (255 = where
        # text should appear). This is the position mask AnyText2 expects.
        pos = np.zeros((h, w), dtype=np.uint8)
        for r in regions:
            x1, y1, x2, y2 = [int(v) for v in r.bbox[:4]]
            pos[y1:y2, x1:x2] = 255
        # Per-region text colors "r,g,b r,g,b ..." (AnyText2 text_colors fmt).
        color_parts = []
        for r in regions:
            c = r.style_info.get("color")
            if isinstance(c, (tuple, list)) and len(c) >= 3:
                color_parts.append(",".join(str(int(v)) for v in c[:3]))
            else:
                color_parts.append("500,500,500")  # AnyText2 "default" sentinel
        text_colors = " ".join(color_parts)
        params = {
            "mode": "edit",
            "sort_priority": "↔↕",  # AnyText2 valid: ↔↕ (horizontal-first) or ↕↔ (vertical-first),
            "show_debug": False,
            "revise_pos": False,
            "image_count": 1,
            "ddim_steps": 20,
            "image_width": w,
            "image_height": h,
            "strength": 1.0,
            "attnx_scale": 1.0,
            "font_hollow": None,  # AnyText2 API expects None, not False
            "cfg_scale": 9.0,  # aligned to AnyText2 default (was 7.5)
            "eta": 0.0,
            "a_prompt": "best quality, extremely detailed,4k, HD, supper legible text,  clear text edges,  clear strokes, neat writing, no watermarks",  # aligned to AnyText2 ms_wrapper default
            "n_prompt": "low-res, bad anatomy, extra digit, fewer digits, cropped, worst quality, low quality, watermark, unreadable text, messy words, distorted text, disorganized writing, advertising picture",  # aligned to AnyText2 ms_wrapper default
            "base_model_path": "",
            "lora_path_ratio": "",  # AnyText2 expects str, not list
            "glyline_font_path": ["None"] * len(regions),
            "font_hint_image": [None] * len(regions),
            "font_hint_mask": [None] * len(regions),
            "text_colors": text_colors,
        }
        input_data = {
            "img_prompt": image[..., ::-1].copy(),  # BGR->RGB for AnyText2
            "text_prompt": text_prompt,
            "seed": 1,
            "draw_pos": pos,
            "ori_image": image[..., ::-1].copy(),
        }
        results, rtn_code, rtn_warning, debug_info = self._anytext2(input_data, **params)
        if rtn_code < 0 or not results:
            raise RuntimeError(f"AnyText2 rendering failed (rtn_code={rtn_code}): {rtn_warning}")
        return np.array(results[0])[..., ::-1]  # RGB->BGR
    # ---- PIL renderer (Solution C's method) ----
    def _render_pil(self, image, regions, style_reference=None):
        from PIL import Image, ImageDraw
        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)
        for region in regions:
            if (not region.style_info.get("color")) and style_reference is not None:
                region.style_info = _PilStyleHelper.enrich(region.style_info, style_reference, region)
            self._draw_single(draw, img_pil, region)
        return np.array(img_pil)
    def _draw_single(self, draw, img_pil, region):
        from PIL import ImageFont
        x1, y1, x2, y2 = [int(v) for v in region.bbox[:4]]
        box_w = x2 - x1
        box_h = y2 - y1
        text = region.translated_text
        if not text or box_w <= 0 or box_h <= 0:
            return
        weight = region.style_info.get("font_weight")
        font_size = int(region.style_info.get("font_size") or max(10, int(box_h * 0.8)))
        font_size = self._fit_font_size(draw, text, font_size, box_w, box_h)
        font = self._load_font(font_size, text, weight)
        fill_color = region.style_info.get("color", (0, 0, 0))
        if not isinstance(fill_color, (tuple, list)) or len(fill_color) < 3:
            fill_color = (0, 0, 0)
        fill_color = tuple(int(c) for c in fill_color[:3])
        lines = self._wrap_text(text, font, box_w)
        line_h = draw.textbbox((0, 0), "Ag", font=font)[3]
        total_h = line_h * len(lines)
        alignment = region.style_info.get("alignment", "center")
        ty = y1 + max(0, (box_h - total_h) // 2)
        for ln in lines:
            ln_w = draw.textbbox((0, 0), ln, font=font)[2]
            if alignment == "left":
                tx = x1 + 2
            elif alignment == "right":
                tx = x1 + max(0, box_w - ln_w - 2)
            else:
                tx = x1 + max(0, (box_w - ln_w) // 2)
            draw.text((tx, ty), ln, font=font, fill=fill_color)
            ty += line_h
    def _fit_font_size(self, draw, text, size, box_w, box_h):
        size = max(10, size)
        for _ in range(12):
            font = self._load_font(size, text)
            lines = self._wrap_text(text, font, box_w)
            line_h = draw.textbbox((0, 0), "Ag", font=font)[3]
            total_h = line_h * len(lines)
            max_w = max((draw.textbbox((0, 0), ln, font=font)[2] for ln in lines), default=0)
            if max_w <= box_w and total_h <= box_h:
                return size
            if size <= 10:
                return 10
            size = max(10, int(size * 0.9))
        return size
    def _wrap_text(self, text, font, max_width):
        from PIL import Image, ImageDraw
        tmp = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        if max_width <= 0:
            return [text]
        words = text.split(" ")
        lines, cur = [], ""
        for w in words:
            trial = w if not cur else cur + " " + w
            if tmp.textbbox((0, 0), trial, font=font)[2] <= max_width or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        final = []
        for ln in lines:
            if tmp.textbbox((0, 0), ln, font=font)[2] <= max_width:
                final.append(ln)
                continue
            cur = ""
            for ch in ln:
                trial = cur + ch
                if tmp.textbbox((0, 0), trial, font=font)[2] <= max_width:
                    cur = trial
                else:
                    if cur:
                        final.append(cur)
                    cur = ch
            if cur:
                final.append(cur)
        return final or [text]
    def _load_font(self, size, text, weight=None):
        from PIL import ImageFont
        bold = (weight == "bold")
        return _get_font_manager().load_font(size, text, bold)
class _PilStyleHelper:
    """On-the-fly style extraction for the PIL renderer (Solution C)."""
    @staticmethod
    def enrich(style_info, image, region):
        style = dict(style_info or {})
        try:
            x1, y1, x2, y2 = [max(0, int(v)) for v in region.bbox[:4]]
            h, w = image.shape[:2]
            x2 = min(w, x2); y2 = min(h, y2)
            if x2 <= x1 or y2 <= y1:
                return style
            roi = image[y1:y2, x1:x2]
            if roi.size == 0:
                return style
            if not style.get("color"):
                style["color"] = _PilStyleHelper._text_color(roi)
            if not style.get("font_size"):
                style["font_size"] = max(10, int((y2 - y1) * 0.75))
        except Exception:
            pass
        return style
    @staticmethod
    def _text_color(roi):
        try:
            pixels = roi.reshape(-1, roi.shape[-1] if roi.ndim == 3 else 1).astype(np.int16)
            if pixels.shape[0] == 0:
                return (0, 0, 0)
            mean = pixels.mean(axis=0)
            bright = pixels.mean(axis=1)
            thresh = mean.mean()
            dark_mask = bright < thresh
            dark_mean = pixels[dark_mask].mean(axis=0) if dark_mask.any() else mean
            light_mean = pixels[~dark_mask].mean(axis=0) if (~dark_mask).any() else dark_mean
            color = dark_mean if dark_mask.sum() <= (~dark_mask).sum() else light_mean
            return tuple(int(c) for c in np.clip(color, 0, 255))
        except Exception:
            return (0, 0, 0)
