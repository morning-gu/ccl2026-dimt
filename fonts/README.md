# Bundled Noto Fonts

These open-source Noto fonts are bundled so the PIL renderer (Solution C)
works out-of-the-box on Windows, Linux, and macOS without requiring system
font installation.

## Files

| File | Size | Script Coverage | License |
|------|------|-----------------|---------|
| NotoSans-Regular.ttf | ~2 MB | Latin (en, es, pt, fr) | SIL OFL 1.1 |
| NotoSansCJKsc-Regular.otf | ~16 MB | CJK: Chinese + Japanese | SIL OFL 1.1 |
| NotoSansCJKsc-Bold.otf | ~17 MB | CJK bold variant | SIL OFL 1.1 |

## Download (if missing)

The setup scripts (`setup_and_run.sh` / `setup_and_run.ps1`) download these
automatically. To download manually:

```bash
curl -sL -o fonts/NotoSans-Regular.ttf \
  "https://github.com/google/fonts/raw/main/ofl/notosans/NotoSans%5Bwdth%2Cwght%5D.ttf"
curl -sL -o fonts/NotoSansCJKsc-Regular.otf \
  "https://github.com/notofonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf"
curl -sL -o fonts/NotoSansCJKsc-Bold.otf \
  "https://github.com/notofonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Bold.otf"
```

## Font Resolution

`src/common/font_manager.py` resolves fonts in this order:
1. Bundled Noto fonts in this directory (always preferred)
2. System font directories (platform-aware search)
3. PIL default (last resort, bitmap only)

The SC (Simplified Chinese) CJK variant covers Japanese kana and kanji
in addition to Chinese characters, so it serves both zh and ja target
languages.