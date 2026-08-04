"""VLM image context analysis plugin (Solution B)."""
import logging
import numpy as np
from interfaces.base import StageType
from interfaces.context_analyzer import IContextAnalyzerPlugin
from plugins.registry import register_plugin
from common.config import PipelineConfig

logger = logging.getLogger(__name__)


@register_plugin(StageType.CONTEXT_ANALYZER, "vlm")
class VLMContextAnalyzerPlugin(IContextAnalyzerPlugin):
    """VLM image context analysis, migrated from solution_b ImageContextAnalyzer."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._client = None

    def analyze(self, image: np.ndarray) -> str:
        self._ensure_client()
        try:
            import base64
            from PIL import Image
            import io
            img_pil = Image.fromarray(image)
            buf = io.BytesIO()
            img_pil.save(buf, format="JPEG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            vlm_model = getattr(self.config, "vlm_model", "") or self.config.translation_model
            response = self._client.chat.completions.create(
                model=vlm_model,
                messages=[
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": (
                            "Describe this e-commerce product image briefly. "
                            "Focus on: product type, key features visible, "
                            "and the overall marketing message. "
                            "Be concise (2-3 sentences)."
                        )},
                    ]},
                ],
                max_tokens=256,
                temperature=0.1,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning("Image context analysis failed: %s", e)
            return ""

    def _ensure_client(self):
        if self._client is not None:
            return
        try:
            from openai import OpenAI
            api_base = getattr(self.config, "vlm_api_base", "") or self.config.translation_api_base
            api_key = getattr(self.config, "vlm_api_key", "") or self.config.translation_api_key
            self._client = OpenAI(api_key=api_key, base_url=api_base)
            logger.info("VLM client initialized (base: %s)", api_base)
        except ImportError:
            raise ImportError(
                "openai package not installed; VLM requires it. No stub fallback."
            )
