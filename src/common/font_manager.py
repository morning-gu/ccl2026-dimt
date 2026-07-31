"""Cross-platform multilingual font manager for the PIL renderer.

Resolves fonts for the 6 competition languages (zh, en, es, pt, ja, fr)
on Windows, Linux, and macOS. Bundled Noto fonts in ``fonts/`` take
priority so the pipeline works out-of-the-box on any OS; system fonts
are searched as a secondary source.
"""
from __future__ import annotations

import logging
import os
import platform
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_BUNDLED_FONTS_DIR = os.path.join(_PROJECT_ROOT, "fonts")


def _detect_script(text: str) -> str:
    if not text:
        return "latin"
    cjk_count = 0
    kana_count = 0
    hangul_count = 0
    for ch in text:
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0xF900 <= cp <= 0xFAFF:
            cjk_count += 1
        elif 0x3040 <= cp <= 0x309F or 0x30A0 <= cp <= 0x30FF:
            kana_count += 1
        elif 0xAC00 <= cp <= 0xD7AF or 0x1100 <= cp <= 0x11FF:
            hangul_count += 1
    if cjk_count + kana_count > 0:
        return "cjk"
    if hangul_count > 0:
        return "korean"
    return "latin"


def _system_font_dirs() -> List[str]:
    dirs: List[str] = []
    system = platform.system()
    if system == "Windows":
        win_fonts = os.environ.get("WINDIR", r"C:\Windows")
        dirs.append(os.path.join(win_fonts, "Fonts"))
        local = os.environ.get("LOCALAPPDATA", "")
        if local:
            dirs.append(os.path.join(local, "Microsoft", "Windows", "Fonts"))
    elif system == "Darwin":
        dirs.extend(["/System/Library/Fonts", "/Library/Fonts", os.path.expanduser("~/Library/Fonts")])
    else:
        dirs.extend(["/usr/share/fonts", "/usr/local/share/fonts",
                      os.path.expanduser("~/.fonts"), os.path.expanduser("~/.local/share/fonts")])
    return [d for d in dirs if os.path.isdir(d)]


def _glob_fonts(directory: str) -> List[str]:
    results: List[str] = []
    try:
        for entry in os.scandir(directory):
            if entry.is_dir():
                results.extend(_glob_fonts(entry.path))
            elif entry.is_file():
                low = entry.name.lower()
                if low.endswith((".ttf", ".otf", ".ttc")):
                    results.append(entry.path)
    except (OSError, PermissionError):
        pass
    return results


class FontManager:
    """Cross-platform, script-aware font resolver.

    Search order: bundled fonts -> system fonts -> PIL default.
    """

    _BUNDLED = {
        "latin_regular": "NotoSans-Regular.ttf",
        "cjk_regular": "NotoSansCJKsc-Regular.otf",
        "cjk_bold": "NotoSansCJKsc-Bold.otf",
    }

    _FAMILY_PATTERNS: Dict[str, List[str]] = {
        "cjk_regular": [
            "notosanscjk", "notosanssc", "notosansjp",
            "msyh", "pingfang", "hiragino", "stheiti",
            "simhei", "simsun", "wenquanyi", "wqy",
            "sourcehansans", "sourcehanserif",
        ],
        "cjk_bold": [
            "notosanscjkbold", "notosansscbold", "notosansjpbold",
            "msyhbd", "pingfangsemibold", "pingfangheavy",
            "sourcehansansbold",
        ],
        "latin_regular": [
            "notosans", "arial", "helvetica",
            "dejavusans", "liberationsans", "roboto", "ubuntu",
        ],
        "latin_bold": [
            "notosansbold", "arialbd", "helveticabold",
            "dejavusansbold", "liberationsansbold", "robotobold",
        ],
    }

    def __init__(self):
        self._bundled_index: Dict[str, str] = {}
        self._system_index: Optional[List[str]] = None
        self._build_bundled_index()

    def _build_bundled_index(self) -> None:
        for key, filename in self._BUNDLED.items():
            path = os.path.join(_BUNDLED_FONTS_DIR, filename)
            if os.path.isfile(path):
                self._bundled_index[key] = path
        if self._bundled_index:
            logger.info("FontManager: %d bundled Noto fonts found in %s",
                        len(self._bundled_index), _BUNDLED_FONTS_DIR)
        else:
            logger.warning("FontManager: no bundled fonts in %s; relying on system fonts",
                            _BUNDLED_FONTS_DIR)

    def _ensure_system_index(self) -> List[str]:
        if self._system_index is not None:
            return self._system_index
        fonts: List[str] = []
        for d in _system_font_dirs():
            fonts.extend(_glob_fonts(d))
        self._system_index = fonts
        return fonts

    def load_font(self, size: int, text: str = "", bold: bool = False):
        # Intentionally uncached: box-height-derived sizes are nearly all
        # unique (near-0 hit rate), and each cached ImageFont owns C
        # glyph-bitmap memory that caused OOM (~image 499). PIL caches the
        # FreeType face handle internally, so recreating a font is cheap.
        return self._resolve(size, text, bold)

    def _resolve(self, size: int, text: str, bold: bool):
        from PIL import ImageFont
        script = _detect_script(text)
        weight = "bold" if bold else "regular"

        bundled_key = f"{script}_{weight}"
        if bundled_key not in self._bundled_index:
            bundled_key = f"{script}_regular"
        if bundled_key in self._bundled_index:
            path = self._bundled_index[bundled_key]
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass

        patterns = self._FAMILY_PATTERNS.get(f"{script}_{weight}",
                     self._FAMILY_PATTERNS.get(f"{script}_regular", []))
        for sys_path in self._ensure_system_index():
            low = os.path.basename(sys_path).lower()
            for pat in patterns:
                if pat in low:
                    try:
                        return ImageFont.truetype(sys_path, size)
                    except Exception:
                        continue

        for path in self._bundled_index.values():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue

        logger.warning("FontManager: no TrueType font found; using PIL default")
        return ImageFont.load_default()
