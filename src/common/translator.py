"""Context-aware translation module with CoT and VLM support.

Uses Qwen2.5 (text) or Qwen-VL (vision-language) for high-quality
translation with Chain-of-Thought reasoning for e-commerce context.
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
    - Standard LLM translation (Qwen2.5)
    - Vision-Language Model translation (Qwen-VL) with image context
    - Chain-of-Thought (CoT) translation for complex marketing text
    - Batch translation for efficiency
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
            api_key = self.config.translation_api_key or "sk-12345679"
            self._client = OpenAI(
                api_key=api_key,
                base_url=api_base,
            )
            logger.info("Translation API client initialized (base: %s)", api_base)
        except ImportError:
            logger.warning("openai package not available, translation will use stub")
            self._client = None

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

        if self._client is None:
            return self._stub_translate(text, target_lang_name)

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

    def _translate_with_vlm(
        self,
        text: str,
        target_lang: str,
        image_base64: str,
        context: str = "",
    ) -> str:
        """Translate using Vision-Language Model with image context."""
        system_prompt = (
            f"You are a professional e-commerce translator. "
            f"Translate the following Chinese text to {target_lang}, "
            f"considering the visual context of the product image. "
            f"Preserve brand names and specifications. "
            f"Output ONLY the translation."
        )

        try:
            response = self._client.chat.completions.create(
                model=self.config.translation_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                        {"type": "text", "text": f"Translate this text to {target_lang}: {text}"},
                    ]},
                ],
                max_tokens=self.config.translation_max_tokens,
                temperature=self.config.translation_temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error("VLM translation failed: %s", e)
            return self._translate_with_cot(text, target_lang, context, "")

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call the LLM API with rate limiting."""
        import time
        try:
            # Rate limit: small delay between calls to avoid 429 errors
            time.sleep(0.8)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            response = self._client.chat.completions.create(
                model=self.config.translation_model,
                messages=messages,
                max_tokens=self.config.translation_max_tokens,
                temperature=self.config.translation_temperature,
            )
            result = response.choices[0].message.content.strip()
            # Extract just the translation if CoT produced extra text
            if "OUTPUT:" in result:
                result = result.split("OUTPUT:")[-1].strip()
            return result
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            return ""

    def _stub_translate(self, text: str, target_lang: str) -> str:
        """Stub translation for testing without API."""
        return f"[{target_lang}]{text}"

    def translate_regions(
        self,
        regions: List[TextRegion],
        target_lang: str,
        image_context: str = "",
    ) -> List[TextRegion]:
        """Translate all translatable regions in one batched API call.

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

        if self._client is None:
            # Stub fallback
            for r in translatable:
                r.translated_text = f"[{target_lang_name}]{r.text}"
            for r in regions:
                if not r.is_translatable:
                    r.translated_text = r.text
            return regions

        # Build numbered text list for the prompt
        lines = []
        for i, r in enumerate(translatable):
            lines.append(f"[{i+1}] {r.text}")
        numbered_text = "\n".join(lines)

        system_prompt = (
            f"You are a professional e-commerce translator. "
            f"Translate the following numbered Chinese text lines to {target_lang_name}.\n"
            f"Rules:\n"
            f"- Preserve brand names, product codes, specifications as-is\n"
            f"- Adapt marketing tone to be natural in {target_lang_name}\n"
            f"- Output EXACTLY the same number of lines, each prefixed with the same [number]\n"
            f"- Output format: [1] translation1\\n[2] translation2\\n..."
        )
        parts = [f"Texts to translate:\n{numbered_text}"]
        if all_text:
            parts.append(f"\nFull image context: {all_text[:500]}")
        if image_context:
            parts.append(f"\nImage description: {image_context}")
        user_prompt = "\n".join(parts)

        raw = self._call_llm(system_prompt, user_prompt)

        # Parse numbered results
        results_map = {}
        if raw:
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
