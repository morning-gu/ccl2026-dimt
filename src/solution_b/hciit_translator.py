"""HCIIT Stage 1: MMLLM-based Text-Image Translation with 4-step CoT.

Implements the translation strategy from the HCIIT paper
(Fu et al., "Ensuring Consistency for In-Image Translation"):

  Stage 1 uses a Multimodal Multilingual Large Language Model (MMLLM)
  with a 4-step Chain-of-Thought to ensure Translation Consistency:

    Step 1: Recognize text in the image AND translate it (MMLLM reads image directly)
    Step 2: Provide a detailed description of the image
    Step 3: Correct recognition errors using image context
    Step 4: Disambiguate translation using image description (e.g., "Bank" -> "河岸" not "银行")

  This is fundamentally different from AnyTrans (Solution A) which uses
  a separate OCR model + text-only LLM. HCIIT's MMLLM directly sees the
  image, enabling true multimodal translation consistency.

  The MMLLM must support image inputs (e.g., Qwen-VL-Chat, GLM-4V, GPT-4o).
  If only a text LLM is available, falls back to OCR + text-LLM with
  image-context injection (degraded but functional).
"""
import base64
import io
import logging
import re
from typing import List, Optional, Tuple

from common.config import PipelineConfig, TARGET_LANGUAGES
from common.selective_translator import TextRegion

logger = logging.getLogger(__name__)


# Language name mapping for prompts
_LANG_NAMES = {
    "en": ("English", "英语"),
    "es": ("Spanish", "西班牙语"),
    "pt": ("Portuguese", "葡萄牙语"),
    "ja": ("Japanese", "日语"),
    "fr": ("French", "法语"),
    "zh": ("Chinese", "中文"),
}


def _encode_image_b64(image) -> str:
    """Encode a numpy BGR image to base64 JPEG string."""
    from PIL import Image as PILImage
    img_pil = PILImage.fromarray(image)
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


class HCIITTranslator:
    """HCIIT Stage 1: MMLLM-based translation with 4-step CoT.

    Two modes:
      - mmlm_mode (preferred): MMLLM directly reads the image for all 4 steps.
        Faithful to the HCIIT paper.
      - ocr_fallback_mode: When no MMLLM is available, uses OCR results +
        text LLM with image-context injection. Degraded but functional.

    The mode is auto-detected based on whether the configured model supports
    image inputs (checked via VLM config or model name heuristics).
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._client = None
        self._vlm_client = None
        self._lang_map = {lc.code: lc for lc in TARGET_LANGUAGES}
        self._mmlm_mode = self._detect_mmlm_capability()

    def _detect_mmlm_capability(self) -> bool:
        """Detect whether the configured model supports image inputs.

        Checks VLM-specific config first, then model name heuristics.
        """
        # If VLM model is explicitly configured, use it
        vlm_model = getattr(self.config, "vlm_model", "")
        if vlm_model:
            return True
        # Heuristic: model names containing "vl", "vision", "4v", "gpt-4o"
        model = self.config.translation_model.lower()
        vision_keywords = ["vl", "vision", "4v", "gpt-4o", "glm-4v", "qwen-vl", "qvq"]
        if any(kw in model for kw in vision_keywords):
            return True
        return False

    def _ensure_client(self):
        """Lazy-initialize the text LLM API client."""
        if self._client is not None:
            return
        from openai import OpenAI
        api_base = self.config.translation_api_base or "http://127.0.0.1:8082/v1"
        api_key = self.config.translation_api_key or ""
        http_kwargs = {}
        if not getattr(self.config, "translation_verify_ssl", True):
            import httpx
            http_kwargs["http_client"] = httpx.Client(verify=False, timeout=120.0)
        self._client = OpenAI(api_key=api_key, base_url=api_base, **http_kwargs)
        logger.info("Translation API client initialized (base: %s)", api_base)

    def _ensure_vlm_client(self):
        """Lazy-initialize the VLM/MMLLM API client."""
        if self._vlm_client is not None:
            return
        from openai import OpenAI
        api_base = getattr(self.config, "vlm_api_base", "") or self.config.translation_api_base
        api_key = getattr(self.config, "vlm_api_key", "") or self.config.translation_api_key
        http_kwargs = {}
        if not getattr(self.config, "translation_verify_ssl", True):
            import httpx
            http_kwargs["http_client"] = httpx.Client(verify=False, timeout=120.0)
        self._vlm_client = OpenAI(api_key=api_key, base_url=api_base, **http_kwargs)
        logger.info("VLM/MMLLM client initialized (base: %s)", api_base)

    def _call_mmlm(self, image, text_prompt: str, max_tokens: int = 1024,
                    temperature: float = 0.3) -> str:
        """Call MMLLM with image + text prompt (multimodal)."""
        self._ensure_vlm_client()
        vlm_model = getattr(self.config, "vlm_model", "") or self.config.translation_model
        b64 = _encode_image_b64(image)
        response = self._vlm_client.chat.completions.create(
            model=vlm_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": text_prompt},
                ],
            }],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    def _call_llm(self, system_prompt: str, user_prompt: str,
                  max_tokens: int = 2048, temperature: float = 0.3) -> str:
        """Call text-only LLM."""
        self._ensure_client()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        for attempt in range(4):
            try:
                stream = self._client.chat.completions.create(
                    model=self.config.translation_model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True,
                )
                parts = []
                for chunk in stream:
                    if not chunk.choices:
                        continue
                    content = getattr(chunk.choices[0].delta, "content", None)
                    if content:
                        parts.append(content)
                return "".join(parts).strip()
            except Exception as e:
                is_retry = "429" in str(e) or any(c in str(e) for c in ("500", "502", "503"))
                if attempt < 3 and is_retry:
                    import time
                    time.sleep(2 ** attempt)
                    continue
                raise
        return ""

    # ----------------------------------------------------------------
    # HCIIT 4-step CoT (MMLLM mode) - faithful to the paper
    # ----------------------------------------------------------------

    def _step1_recognize_and_translate(self, image, src_lang: str,
                                        tgt_lang: str) -> str:
        """Step 1: MMLLM recognizes text in image and translates it.

        Paper prompt: "请识别图片中的[SRC]文，并且将其翻译成[TGT]"
        """
        src_en, src_cn = _LANG_NAMES.get(src_lang, (src_lang, src_lang))
        tgt_en, tgt_cn = _LANG_NAMES.get(tgt_lang, (tgt_lang, tgt_lang))
        prompt = (
            f"请识别图片中的{src_cn}文，并且将其翻译成{tgt_cn}。\n"
            f"Please recognize the {src_en} text within the image "
            f"and translate it into {tgt_en}.\n"
            f"Output format: list each recognized text and its translation."
        )
        return self._call_mmlm(image, prompt, max_tokens=1024, temperature=0.1)

    def _step2_image_description(self, image) -> str:
        """Step 2: MMLLM provides a detailed description of the image.

        Paper prompt: "请给出这张图片的详细描述"
        """
        prompt = (
            "请给出这张图片的详细描述。\n"
            "Please provide a detailed description of the information in the image."
        )
        return self._call_mmlm(image, prompt, max_tokens=512, temperature=0.1)

    def _step3_correct_recognition(self, image, recognized_text: str) -> str:
        """Step 3: Correct recognition errors using image context.

        Paper prompt: "请结合图片信息判断识别的文本是否有拼写问题..."
        """
        prompt = (
            f"之前识别的文本如下：\n{recognized_text}\n\n"
            "请结合图片信息判断识别的文本是否有拼写问题或者和图片含义不同的问题，"
            "如果有的话，请结合图片信息对其进行修改。\n"
            "Determine if there are any spelling errors or semantic inconsistencies "
            "in the identified text. If present, please rectify or remove them "
            "based on the image information."
        )
        return self._call_mmlm(image, prompt, max_tokens=1024, temperature=0.1)

    def _step4_disambiguate_translation(self, image, corrected_text: str,
                                         image_desc: str,
                                         src_lang: str, tgt_lang: str) -> str:
        """Step 4: Disambiguate translation using image description.

        Paper prompt: "请考虑多种翻译结果，结合图片描述消歧..."
        """
        _, src_cn = _LANG_NAMES.get(src_lang, (src_lang, src_lang))
        tgt_en, tgt_cn = _LANG_NAMES.get(tgt_lang, (tgt_lang, tgt_lang))
        prompt = (
            f"图片描述：{image_desc}\n\n"
            f"纠正后的识别文本：{corrected_text}\n\n"
            f"请考虑{src_cn}识别结果的多种{tgt_cn}翻译结果，结合图片的描述和图片中的"
            f"实体信息选择最合适最贴切的翻译（并直接给出最符合图片描述的翻译结果）。\n"
            f"Pay attention to any instances of polysemy in the translated text. "
            f"If there are potential issues with polysemy, disambiguate based on "
            f"the information provided in the image. Avoid excessive explanations "
            f"and provide the most likely translation results directly."
        )
        return self._call_mmlm(image, prompt, max_tokens=1024, temperature=0.3)

    def translate_with_cot_mmlm(self, image, regions: List[TextRegion],
                                 src_lang: str, tgt_lang: str) -> List[TextRegion]:
        """Full HCIIT 4-step CoT using MMLLM (paper-faithful).

        The MMLLM sees the image directly in all 4 steps, ensuring
        Translation Consistency as defined in the HCIIT paper.
        """
        translatable = [r for r in regions if r.is_translatable]
        if not translatable:
            for r in regions:
                r.translated_text = r.text
            return regions

        logger.info("  HCIIT CoT Step 1: Recognize & Translate (MMLLM)")
        step1_result = self._step1_recognize_and_translate(image, src_lang, tgt_lang)
        logger.debug("  Step 1 result: %s", step1_result[:200])

        logger.info("  HCIIT CoT Step 2: Image Description")
        image_desc = self._step2_image_description(image)
        logger.debug("  Step 2 result: %s", image_desc[:200])

        logger.info("  HCIIT CoT Step 3: Correct Recognition")
        step3_result = self._step3_correct_recognition(image, step1_result)
        logger.debug("  Step 3 result: %s", step3_result[:200])

        logger.info("  HCIIT CoT Step 4: Disambiguate Translation")
        step4_result = self._step4_disambiguate_translation(
            image, step3_result, image_desc, src_lang, tgt_lang
        )
        logger.debug("  Step 4 result: %s", step4_result[:200])

        # Parse the final translation result back into regions.
        # The MMLLM may output in various formats; we try structured parsing
        # first, then fall back to line-by-line assignment.
        self._assign_translations(translatable, step4_result, tgt_lang)

        for r in regions:
            if not r.is_translatable:
                r.translated_text = r.text
        return regions

    # ----------------------------------------------------------------
    # Fallback: OCR + text-LLM with image-context injection
    # ----------------------------------------------------------------

    def translate_with_cot_fallback(self, image, regions: List[TextRegion],
                                     src_lang: str, tgt_lang: str,
                                     image_context: str = "") -> List[TextRegion]:
        """Degraded CoT when no MMLLM is available.

        Uses OCR results (already in regions) + text LLM with VLM-generated
        image context. Not paper-faithful but functional.

        Simulates the 4-step CoT logic:
          Step 1: OCR already done; LLM translates with <boxN> tags
          Step 2: VLM image description (pre-computed)
          Step 3: LLM checks for recognition errors
          Step 4: LLM disambiguates translation using image context
        """
        translatable = [r for r in regions if r.is_translatable]
        if not translatable:
            for r in regions:
                r.translated_text = r.text
            return regions

        tgt_en, tgt_cn = _LANG_NAMES.get(tgt_lang, (tgt_lang, tgt_lang))

        # Step 1+4 combined: translate with context-aware CoT prompt
        src_tagged = "".join(
            f"<box{i+1}>{r.text}</box{i+1}>" for i, r in enumerate(translatable)
        )

        system_prompt = (
            f"You are a professional in-image translator. Translate {src_lang} text "
            f"to {tgt_en}.\n\n"
            f"Follow these steps (Chain-of-Thought):\n"
            f"1. ANALYZE: Identify the text content and any potential ambiguities "
            f"(e.g., polysemous words like 'Bank' which could mean '银行' or '河岸').\n"
            f"2. CONTEXTUALIZE: Use the image description to resolve ambiguities. "
            f"Choose the translation that best fits the visual context.\n"
            f"3. TRANSLATE: Produce the final translation preserving <boxN></boxN> tags.\n"
            f"4. OUTPUT: Only the tagged translation result.\n\n"
            f"Rules:\n"
            f"- Preserve brand names, product codes, specifications as-is\n"
            f"- Adapt marketing tone to be natural in {tgt_en}\n"
            f"- Keep <boxN></boxN> tag format in output"
        )

        parts = [f"Text to translate: {src_tagged}"]
        if image_context:
            parts.append(f"\nImage description: {image_context}")
        parts.append(f"\nTranslate to {tgt_en}:")
        user_prompt = "\n".join(parts)

        raw = self._call_llm(system_prompt, user_prompt)

        # Parse results
        results_map = {}
        if raw:
            # Try to extract just the tagged part if CoT produced extra text
            tagged_match = re.search(r'(<box1>.*</box\d+>)', raw, re.DOTALL)
            tagged_text = tagged_match.group(1) if tagged_match else raw
            for m in re.finditer(r'<box(\d+)>(.*?)</box\1>', tagged_text, re.DOTALL):
                results_map[int(m.group(1))] = m.group(2).strip()

        # Fallback: numbered format
        if not results_map and raw:
            for m in re.finditer(r'\[(\d+)\]\s*(.+?)(?=\n\[|\Z)', raw, re.DOTALL):
                results_map[int(m.group(1))] = m.group(2).strip()

        for i, r in enumerate(translatable):
            translated = results_map.get(i + 1, "")
            r.translated_text = translated if translated else r.text

        for r in regions:
            if not r.is_translatable:
                r.translated_text = r.text
        return regions

    # ----------------------------------------------------------------
    # Translation assignment helpers
    # ----------------------------------------------------------------

    def _assign_translations(self, regions: List[TextRegion],
                             translation_text: str, tgt_lang: str):
        """Assign translation results from MMLLM output to regions.

        The MMLLM output may be in various formats:
          - "原文 -> 译文" pairs
          - Numbered lists
          - Just the translated text lines
        We try multiple parsing strategies.
        """
        if not translation_text:
            for r in regions:
                r.translated_text = r.text
            return

        # Strategy 1: "原文 -> 译文" or "原文：译文" pairs
        pair_pattern = re.compile(r'(.+?)\s*[->：:]\s*(.+)')
        pairs = []
        for line in translation_text.strip().split("\n"):
            line = line.strip().strip("-").strip()
            if not line:
                continue
            m = pair_pattern.match(line)
            if m:
                pairs.append((m.group(1).strip(), m.group(2).strip()))

        if len(pairs) == len(regions):
            for r, (_, tgt) in zip(regions, pairs):
                r.translated_text = tgt
            return

        # Strategy 2: Numbered list "1. translation"
        numbered = re.findall(r'\d+[\.\)]\s*(.+)', translation_text)
        if len(numbered) >= len(regions):
            for r, tgt in zip(regions, numbered[:len(regions)]):
                r.translated_text = tgt.strip()
            return

        # Strategy 3: Line-by-line (one translation per line)
        lines = [l.strip() for l in translation_text.strip().split("\n") if l.strip()]
        # Filter out lines that look like source text (contain Chinese for zh->en)
        if tgt_lang != "zh":
            tgt_lines = [l for l in lines if not re.match(r'^[\u4e00-\u9fff]+$', l)]
        else:
            tgt_lines = lines
        if len(tgt_lines) >= len(regions):
            for r, tgt in zip(regions, tgt_lines[:len(regions)]):
                r.translated_text = tgt
            return

        # Strategy 4: Extract just translations from mixed content
        # Use the last N non-empty lines as translations
        if lines:
            # Take from the end (translations usually come after source text)
            candidates = lines[-len(regions):] if len(lines) >= len(regions) else lines
            for i, r in enumerate(regions):
                r.translated_text = candidates[i] if i < len(candidates) else r.text
            return

        # Fallback: keep original text
        for r in regions:
            r.translated_text = r.text

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    def translate_regions(self, image, regions: List[TextRegion],
                          target_lang: str,
                          image_context: str = "") -> List[TextRegion]:
        """Translate regions using HCIIT 4-step CoT.

        Args:
            image: Source image (numpy BGR array), used by MMLLM.
            regions: Text regions with OCR results.
            target_lang: Target language code.
            image_context: Pre-computed VLM image description (fallback mode only).

        Returns:
            Regions with translated_text populated.
        """
        src_lang = self.config.source_lang

        if self._mmlm_mode:
            logger.info("  Using HCIIT 4-step CoT (MMLLM mode)")
            return self.translate_with_cot_mmlm(image, regions, src_lang, target_lang)
        else:
            logger.info("  Using HCIIT CoT (OCR+LLM fallback mode)")
            return self.translate_with_cot_fallback(
                image, regions, src_lang, target_lang, image_context
            )
