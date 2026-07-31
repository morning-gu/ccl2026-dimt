"""Context-aware translation module with CoT support.

Uses an OpenAI-compatible LLM (configured by translation_model, e.g.
Qwen2.5 / GLM) for high-quality translation with Chain-of-Thought
reasoning for e-commerce context.

Image context for translation is NOT done here as a multimodal call.
Instead, a VLM-derived image description can be passed in via the
`image_context` argument (Solution B produces it via ImageContextAnalyzer)
and is injected into the CoT prompt's "Image description" field.
"""
import json
import logging
from typing import List, Optional, Dict

from .config import PipelineConfig, TARGET_LANGUAGES
from .selective_translator import TextRegion

logger = logging.getLogger(__name__)


class ContextAwareTranslator:
    """Translate text regions with e-commerce context awareness.

    Supports:
    - Direct LLM translation
    - Chain-of-Thought (CoT) translation for complex marketing text
    - Batch translation of all regions in one API call (context learning)
    - Optional VLM image-context string injected into the CoT prompt
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._client = None
        self._lang_map = {lc.code: lc for lc in TARGET_LANGUAGES}

    def _ensure_client(self):
        """Lazy-initialize the API client."""
        if self._client is not None:
            return
        try:
            from openai import OpenAI
            api_base = self.config.translation_api_base or "http://127.0.0.1:8082/v1"
            api_key = self.config.translation_api_key or ""
            http_kwargs = {}
            if not getattr(self.config, "translation_verify_ssl", True):
                import httpx
                http_kwargs["http_client"] = httpx.Client(verify=False, timeout=120.0)
            self._client = OpenAI(
                api_key=api_key,
                base_url=api_base,
                **http_kwargs,
            )
            logger.info("Translation API client initialized (base: %s)", api_base)
        except ImportError:
            raise ImportError(
                "openai package not installed; translation requires it "
                "(pip install openai). No stub fallback."
            )

    def translate_text(
        self,
        text: str,
        target_lang: str,
        context: str = "",
        image_context: str = "",
    ) -> str:
        """Translate a single text string.

        Args:
            text: Source text to translate.
            target_lang: Target language code (en, es, pt, ja, fr).
            context: Additional text context from the same image.
            image_context: Description of the image content (from VLM).

        Returns:
            Translated text.
        """
        self._ensure_client()
        lang_info = self._lang_map.get(target_lang)
        is_mt_model = "mt" in self.config.translation_model.lower()
        # MT models use Chinese prompts, so use Chinese lang name
        if is_mt_model and lang_info:
            target_lang_name = lang_info.name_cn
        else:
            target_lang_name = lang_info.qwen_lang if lang_info else target_lang

        if self.config.translation_use_cot:
            return self._translate_with_cot(text, target_lang_name, context, image_context)
        else:
            return self._translate_direct(text, target_lang_name, context)

    def _translate_direct(self, text: str, target_lang: str, context: str) -> str:
        """Direct translation without CoT."""
        # qwen-mt models only support user/assistant roles, not system
        is_mt_model = "mt" in self.config.translation_model.lower()
        
        if is_mt_model:
            user_prompt = f"将以下中文翻译成{target_lang}：{text}"
            if context:
                user_prompt = f"上下文：{context}\n\n将以下中文翻译成{target_lang}：{text}"
            return self._call_llm("", user_prompt)
        
        system_prompt = (
            f"You are a professional e-commerce translator. "
            f"Translate the following Chinese text to {target_lang}. "
            f"Keep the tone suitable for product marketing. "
            f"Do not translate brand names, product codes, or specifications. "
            f"Output ONLY the translation, nothing else."
        )
        user_prompt = text
        if context:
            user_prompt = f"Context from the same product image: {context}\n\nText to translate: {text}"

        return self._call_llm(system_prompt, user_prompt)

    def _translate_with_cot(
        self, text: str, target_lang: str, context: str, image_context: str
    ) -> str:
        """Chain-of-Thought translation for better quality.

        Step 1: Analyze the text (identify key terms, marketing language, etc.)
        Step 2: Translate with awareness of analysis
        """
        # qwen-mt models only support user/assistant roles, not system
        is_mt_model = "mt" in self.config.translation_model.lower()
        
        if is_mt_model:
            # For MT models, use direct translation with context
            user_prompt = f"将以下中文翻译成{target_lang}：{text}"
            if context:
                user_prompt = f"上下文（同一商品图中的其他文本）：{context}\n\n{user_prompt}"
            if image_context:
                user_prompt = f"图片描述：{image_context}\n\n{user_prompt}"
            return self._call_llm("", user_prompt)
        
        system_prompt = (
            f"You are a professional e-commerce translator specializing in cross-border commerce. "
            f"Your task is to translate Chinese product text to {target_lang}.\n\n"
            f"Follow these steps:\n"
            f"1. ANALYZE: Identify brand names, product specifications, marketing phrases, "
            f"and technical terms. Note which parts should NOT be translated.\n"
            f"2. TRANSLATE: Translate the translatable parts, preserving brand names, "
            f"specs, and codes as-is. Adapt marketing language to be natural in {target_lang}.\n"
            f"3. OUTPUT: Provide ONLY the final translation.\n\n"
            f"Rules:\n"
            f"- Brand names stay in original form\n"
            f"- Product codes/model numbers stay as-is\n"
            f"- Specifications (e.g., '256GB') stay as-is\n"
            f"- Marketing tone should be natural and appealing in {target_lang}\n"
            f"- Slogans should be culturally adapted, not literally translated"
        )

        parts = [f"Text to translate: {text}"]
        if context:
            parts.append(f"\nOther text from the same image (for context): {context}")
        if image_context:
            parts.append(f"\nImage description: {image_context}")
        user_prompt = "\n".join(parts)

        return self._call_llm(system_prompt, user_prompt)

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call the LLM API with retry on rate-limit errors."""
        import time
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                # stream=True: some OpenAI-compatible gateways (e.g. MAAS)
                # always return SSE chunks even for non-stream requests, which
                # breaks the non-stream response object. Streaming is a safe
                # superset that works for both.
                stream = self._client.chat.completions.create(
                    model=self.config.translation_model,
                    messages=messages,
                    max_tokens=self.config.translation_max_tokens,
                    temperature=self.config.translation_temperature,
                    stream=True,
                )
                parts = []
                for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    content = getattr(delta, "content", None)
                    if content:
                        parts.append(content)
                result = "".join(parts).strip()
                # Extract just the translation if CoT produced extra text
                if "OUTPUT:" in result:
                    result = result.split("OUTPUT:")[-1].strip()
                return result
            except Exception as e:
                # Retry on rate-limit (429) or server errors (5xx)
                is_rate_limit = "429" in str(e) or "rate" in str(e).lower()
                is_server_error = any(
                    code in str(e) for code in ("500", "502", "503", "504")
                )
                if attempt < max_retries and (is_rate_limit or is_server_error):
                    wait = 2 ** attempt  # 1s, 2s, 4s
                    logger.warning(
                        "LLM call failed (attempt %d/%d), retrying in %ds: %s",
                        attempt + 1, max_retries, wait, e,
                    )
                    time.sleep(wait)
                    continue
                logger.error("LLM call failed after %d attempts: %s", attempt + 1, e)
                raise

    def translate_regions(
        self,
        regions: List[TextRegion],
        target_lang: str,
        image_context: str = "",
    ) -> List[TextRegion]:
        """Translate all translatable regions in one batched API call.

        Uses the AnyTrans <boxidx></boxidx> tag format with few-shot
        demonstrations to preserve positional information and enable
        context-aware translation (AnyTrans Section 3.2).

        Preserved regions are left unchanged.
        """
        # Build context from all text in the image
        all_text = " | ".join(r.text for r in regions if r.text)

        translatable = [r for r in regions if r.is_translatable]
        if not translatable:
            for r in regions:
                r.translated_text = r.text
            return regions

        # Batch: send all texts in one API call
        self._ensure_client()
        lang_info = self._lang_map.get(target_lang)
        target_lang_name = lang_info.qwen_lang if lang_info else target_lang

        # --- AnyTrans <boxidx></boxidx> tag format (Section 3.2) ---
        # Concatenate all translatable texts using HTML-style box tags to
        # preserve positional info. The LLM translates the whole sequence
        # and outputs in the same tag format, maintaining word order.
        src_tagged = "".join(
            f"<box{i+1}>{r.text}</box{i+1}>" for i, r in enumerate(translatable)
        )

        # Few-shot demonstrations (5-shot, AnyTrans uses 5-shot per language pair)
        few_shot_examples = self._build_few_shot_examples(target_lang_name)

        system_prompt = (
            f"You are a professional e-commerce translator. "
            f"Translate Chinese text to {target_lang_name}.\n"
            f"Rules:\n"
            f"- Preserve brand names, product codes, specifications as-is\n"
            f"- Adapt marketing tone to be natural in {target_lang_name}\n"
            f"- Keep the <boxN></boxN> tag format in the output, with each "
            f"box containing the translated text for that region\n"
            f"- Output ONLY the tagged translation, nothing else"
        )

        parts = [few_shot_examples, f"\nTranslate the following to {target_lang_name}:"]
        parts.append(f"Chinese: {src_tagged}")
        parts.append(f"{target_lang_name}:")
        if all_text:
            parts.append(f"\n(Full image context: {all_text[:500]})")
        if image_context:
            parts.append(f"\n(Image description: {image_context})")
        user_prompt = "\n".join(parts)

        raw = self._call_llm(system_prompt, user_prompt)

        # Parse <boxN>...</boxN> results
        results_map = {}
        if raw:
            import re
            for m in re.finditer(r'<box(\d+)>(.*?)</box\1>', raw, re.DOTALL):
                idx = int(m.group(1))
                text = m.group(2).strip()
                results_map[idx] = text

        # Fallback: try numbered [N] format if box tags failed
        if not results_map and raw:
            import re
            for m in re.finditer(r'\[(\d+)\]\s*(.+?)(?=\n\[|\Z)', raw, re.DOTALL):
                idx = int(m.group(1))
                text = m.group(2).strip()
                results_map[idx] = text

        # Apply translations
        for i, r in enumerate(translatable):
            translated = results_map.get(i + 1, "")
            r.translated_text = translated if translated else r.text

        for r in regions:
            if not r.is_translatable:
                r.translated_text = r.text

        return regions

    def _build_few_shot_examples(self, target_lang_name: str) -> str:
        """Build 5-shot demonstrations for the AnyTrans translation prompt.

        Uses generic e-commerce examples that show the <boxN></boxN> format.
        The examples teach the LLM to:
        1. Keep the tag format
        2. Translate contextually (not box-by-box isolation)
        3. Reorder words when the target language requires it
        """
        # Language-specific examples (zh->en shown; others use the same
        # structure with target language name substituted)
        examples = [
            # Example 1: product name + slogan (context helps disambiguate)
            ('<box1>限时</box1><box2>特惠</box2>',
             '<box1>Limited</box1><box2>Time Offer</box2>'),
            # Example 2: brand preservation
            ('<box1>华为</box1><box2>手机</box2>',
             '<box1>Huawei</box1><box2>Phone</box2>'),
            # Example 3: spec preservation
            ('<box1>256GB</box1><box2>存储</box2>',
             '<box1>256GB</box1><box2>Storage</box2>'),
            # Example 4: word reordering (Chinese -> English)
            ('<box1>无线</box1><box2>蓝牙耳机</box2>',
             '<box1>Wireless</box1><box2>Bluetooth Earphone</box2>'),
            # Example 5: marketing text cultural adaptation
            ('<box1>新品上市</box1><box2>买一送一</box2>',
             '<box1>New Arrival</box1><box2>Buy One Get One Free</box2>'),
        ]

        # For non-English targets, we still show the format with English
        # as a structural template (the LLM generalizes to the target language)
        lines = ["Here are some examples of the translation format:"]
        for i, (src, tgt) in enumerate(examples):
            lines.append(f"Chinese: {src}")
            lines.append(f"English: {tgt}")
            lines.append("")

        lines.append(f"Now translate to {target_lang_name} using the same <boxN></boxN> format.")
        return "\n".join(lines)

    def translate_regions_batch(
        self,
        regions: List[TextRegion],
        target_langs: List[str],
        image_context: str = "",
    ) -> Dict[str, List[TextRegion]]:
        """Translate regions into multiple target languages.

        Returns:
            Dict mapping language code to list of translated regions.
        """
        results = {}
        for lang in target_langs:
            results[lang] = self.translate_regions(regions, lang, image_context)
        return results
