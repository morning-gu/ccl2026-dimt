# Bundled Noto Fonts

These open-source Noto fonts are bundled so the PIL renderer (Solution C)
works out-of-the-box on Windows, Linux, and macOS without requiring system
font installation.

Font binary files are tracked via **Git LFS** (see `.gitattributes`).
A normal `git clone` + `git lfs pull` fetches them automatically.

## Files

| File | Size | Script Coverage | License |
|------|------|-----------------|---------|
| NotoSans-Regular.ttf | ~39 KB | Latin (en, es, pt, fr) | SIL OFL 1.1 |
| NotoSansCJKsc-Regular.otf | ~14 MB | CJK: Chinese + Japanese | SIL OFL 1.1 |
| NotoSansCJKsc-Bold.otf | ~16 MB | CJK bold variant | SIL OFL 1.1 |

## Font Resolution

`src/common/font_manager.py` resolves fonts in this order:
1. Bundled Noto fonts in this directory (always preferred)
2. System font directories (platform-aware search)
3. PIL default (last resort, bitmap only)

The SC (Simplified Chinese) CJK variant covers Japanese kana and kanji
in addition to Chinese characters, so it serves both zh and ja target
languages.
