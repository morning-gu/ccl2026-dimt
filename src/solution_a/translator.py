"""AnyTrans-faithful translator for Solution A (pure few-shot, no CoT).

Implements the AnyTrans translation strategy (Section 3.2 of the AnyTrans
paper, Qian et al. 2024, arXiv:2406.11432):

  - Concatenate all text regions using <boxidx></boxidx> HTML-style tags to
    retain positional information across the image (Section 3.2).
  - 5-shot demonstrations PER LANGUAGE PAIR, exactly as the paper specifies
    ("we use five-shot demonstrations for each language pair"), rather than
    a single universal zh->en template reused for every target.
  - Pure few-shot instruction prompt. No Chain-of-Thought reasoning, no
    e-commerce system prompt. CoT is an HCIIT technique (Fu et al. 2024) and
    is intentionally excluded so Solution A stays faithful to the paper.
  - LLM-only configuration (no VLM image context). The paper's VLM branch is
    an optional enhancement; Solution A matches the paper's LLM-only mode and
    does not introduce VLM here.

This module is Solution-A-specific and independent from
common/translator.py, which serves Solutions B and C with their own
CoT / VLM strategies.
"""
import logging
import re
from typing import List

from common.config import PipelineConfig, TARGET_LANGUAGES
from common.selective_translator import TextRegion

logger = logging.getLogger(__name__)


# Per-language-pair 5-shot demonstrations (source zh -> target).
# Each pair teaches the LLM to:
#   1. Preserve the <boxN></boxN> tag format in the output
#   2. Translate contextually rather than box-by-box in isolation
#   3. Reorder words when the target language requires it
#   4. Preserve specs/codes (e.g. 256GB) as-is
# Adding a new target language? Add a 5-entry list here so the prompt stays
# per-language-pair as the paper requires.
FEW_SHOT_EXAMPLES = {
    "en": [
        ("<box1>新款</box1><box2>上市</box2>",
         "<box1>New</box1><box2>Arrival</box2>"),
        ("<box1>无线</box1><box2>蓝牙耳机</box2>",
         "<box1>Wireless</box1><box2>Bluetooth Earphone</box2>"),
        ("<box1>买一</box1><box2>送一</box2>",
         "<box1>Buy One</box1><box2>Get One Free</box2>"),
        ("<box1>256GB</box1><box2>存储</box2>",
         "<box1>256GB</box1><box2>Storage</box2>"),
        ("<box1>品质</box1><box2>保证</box2>",
         "<box1>Quality</box1><box2>Guaranteed</box2>"),
    ],
    "es": [
        ("<box1>新款</box1><box2>上市</box2>",
         "<box1>Nuevo</box1><box2>Lanzamiento</box2>"),
        ("<box1>无线</box1><box2>蓝牙耳机</box2>",
         "<box1>Inalámbrico</box1><box2>Auriculares Bluetooth</box2>"),
        ("<box1>买一</box1><box2>送一</box2>",
         "<box1>Compra Uno</box1><box2>Lleva Uno Gratis</box2>"),
        ("<box1>256GB</box1><box2>存储</box2>",
         "<box1>256GB</box1><box2>Almacenamiento</box2>"),
        ("<box1>品质</box1><box2>保证</box2>",
         "<box1>Calidad</box1><box2>Garantizada</box2>"),
    ],
    "pt": [
        ("<box1>新款</box1><box2>上市</box2>",
         "<box1>Novo</box1><box2>Lançamento</box2>"),
        ("<box1>无线</box1><box2>蓝牙耳机</box2>",
         "<box1>Sem Fio</box1><box2>Fone Bluetooth</box2>"),
        ("<box1>买一</box1><box2>送一</box2>",
         "<box1>Compre Um</box1><box2>Leve Outro Grátis</box2>"),
        ("<box1>256GB</box1><box2>存储</box2>",
         "<box1>256GB</box1><box2>Armazenamento</box2>"),
        ("<box1>品质</box1><box2>保证</box2>",
         "<box1>Qualidade</box1><box2>Garantida</box2>"),
    ],
    "ja": [
        ("<box1>新款</box1><box2>上市</box2>",
         "<box1>新製品</box1><box2>発売中</box2>"),
        ("<box1>无线</box1><box2>蓝牙耳机</box2>",
         "<box1>ワイヤレス</box1><box2>Bluetoothイヤホン</box2>"),
        ("<box1>买一</box1><box2>送一</box2>",
         "<box1>1つ買うと</box1><box2>1つ無料</box2>"),
        ("<box1>256GB</box1><box2>存储</box2>",
         "<box1>256GB</box1><box2>ストレージ</box2>"),
        ("<box1>品质</box1><box2>保证</box2>",
         "<box1>品質</box1><box2>保証</box2>"),
    ],
    "fr": [
        ("<box1>新款</box1><box2>上市</box2>",
         "<box1>Nouveau</box1><box2>Lancement</box2>"),
        ("<box1>无线</box1><box2>蓝牙耳机</box2>",
         "<box1>Sans Fil</box1><box2>Écouteurs Bluetooth</box2>"),
        ("<box1>买一</box1><box2>送一</box2>",
         "<box1>Achetez Un</box1><box2>Un Offert</box2>"),
        ("<box1>256GB</box1><box2>存储</box2>",
         "<box1>256GB</box1><box2>Stockage</box2>"),
        ("<box1>品质</box1><box2>保证</box2>",
         "<box1>Qualité</box1><box2>Garantie</box2>"),
    ],
}

# Fallback demonstration set for a target language not in the map above.
# Per the paper each pair needs its own examples; this fallback prevents a
# crash if a new target is added without examples, and logs a warning so the
# gap stays visible.
_FALLBACK_LANG = "en"


class AnyTransTranslator:
    """AnyTrans-faithful few-shot translator (Solution A only).

    Translates all translatable regions in a single API call using the
    <boxidx></boxidx> tag format with per-language-pair 5-shot
    demonstrations, exactly as described in AnyTrans Section 3.2.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._client = None
        self._lang_map = {lc.code: lc for lc in TARGET_LANGUAGES}

    def _ensure_client(self):
        """Lazy-initialize the OpenAI-compatible API client."""
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
            self._client = OpenAI(api_key=api_key, base_url=api_base, **http_kwargs)
            logger.info("AnyTrans translator API client initialized (base: %s)", api_base)
        except ImportError:
            raise ImportError(
                "openai package not installed; translation requires it "
                "(pip install openai). No stub fallback."
            )

    def _build_prompt(self, translatable, target_lang_name, target_lang_code):
        """Build the AnyTrans few-shot prompt (Section 3.2).

        Returns (system_prompt, user_prompt) following the paper's format:
        a brief instruction, 5 per-language-pair demonstrations, then the
        actual query with source texts concatenated in <boxN> tags.
        """
        examples = FEW_SHOT_EXAMPLES.get(target_lang_code)
        if examples is None:
            logger.warning(
                "No per-language-pair examples for zh->%s; falling back to "
                "zh->%s structure. Add a 5-entry list to FEW_SHOT_EXAMPLES "
                "for full AnyTrans fidelity.",
                target_lang_code, _FALLBACK_LANG,
            )
            examples = FEW_SHOT_EXAMPLES[_FALLBACK_LANG]

        # Concatenate source texts with <boxN></boxN> tags (AnyTrans 3.2)
        src_tagged = "".join(
            f"<box{i+1}>{r.text}</box{i+1}>" for i, r in enumerate(translatable)
        )

        # Build the 5-shot demonstration block
        demo_lines = []
        for src, tgt in examples:
            demo_lines.append(f"Chinese: {src}")
            demo_lines.append(f"{target_lang_name}: {tgt}")
            demo_lines.append("")

        # Minimal instruction-only system prompt. No e-commerce framing, no
        # Chain-of-Thought steps -- pure AnyTrans few-shot prompting.
        system_prompt = (
            f"Translate text from Chinese to {target_lang_name}. "
            f"Preserve the <boxN></boxN> tag format in the output, with each "
            f"box holding the translated text for that region. Adjust word "
            f"order as natural for {target_lang_name}. "
            f"Output ONLY the tagged translation, nothing else."
        )

        user_parts = [
            f"Translate the following text from Chinese to {target_lang_name}. "
            f"Keep the <boxN></boxN> tags.\n",
            "Examples:\n",
            "\n".join(demo_lines),
            f"Now translate:\nChinese: {src_tagged}\n{target_lang_name}:",
        ]
        user_prompt = "\n".join(user_parts)
        return system_prompt, user_prompt

    def _call_llm(self, system_prompt, user_prompt):
        """Call the LLM API with retry on rate-limit / server errors."""
        import time
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                # stream=True: some OpenAI-compatible gateways (e.g. MAAS)
                # always return SSE chunks even for non-stream requests.
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
                return "".join(parts).strip()
            except Exception as e:
                is_rate_limit = "429" in str(e) or "rate" in str(e).lower()
                is_server_error = any(
                    code in str(e) for code in ("500", "502", "503", "504")
                )
                if attempt < max_retries and (is_rate_limit or is_server_error):
                    wait = 2 ** attempt
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
    ) -> List[TextRegion]:
        """Translate all translatable regions in one batched API call.

        Uses the AnyTrans <boxidx></boxidx> tag format with per-language-pair
        5-shot demonstrations (AnyTrans Section 3.2). Preserved (brand, logo,
        spec, ...) regions are left unchanged -- this selective filtering is a
        competition-specific extension, not part of the AnyTrans paper.
        """
        translatable = [r for r in regions if r.is_translatable]
        if not translatable:
            for r in regions:
                r.translated_text = r.text
            return regions

        self._ensure_client()
        lang_info = self._lang_map.get(target_lang)
        target_lang_name = lang_info.qwen_lang if lang_info else target_lang

        system_prompt, user_prompt = self._build_prompt(
            translatable, target_lang_name, target_lang
        )
        raw = self._call_llm(system_prompt, user_prompt)

        # Parse <boxN>...</boxN> results
        results_map = {}
        if raw:
            for m in re.finditer(r'<box(\d+)>(.*?)</box\1>', raw, re.DOTALL):
                idx = int(m.group(1))
                results_map[idx] = m.group(2).strip()

        # Fallback: try numbered [N] format if box tags failed
        if not results_map and raw:
            for m in re.finditer(r'\[(\d+)\]\s*(.+?)(?=\n\[|\Z)', raw, re.DOTALL):
                idx = int(m.group(1))
                results_map[idx] = m.group(2).strip()

        for i, r in enumerate(translatable):
            translated = results_map.get(i + 1, "")
            r.translated_text = translated if translated else r.text

        for r in regions:
            if not r.is_translatable:
                r.translated_text = r.text

        return regions
