# CCL2026 DIMT - Image In-Image Translation

Tianchi Competition [CCL2026-DIMT](https://tianchi.aliyun.com/competition/entrance/532463)

## Project Structure

`
ccl2026-dimt/
  src/                    # Source code
    common/               # Shared modules
      config.py           # Global config (with debug switches)
      debug_saver.py      # Debug intermediate file saver
      ocr_detector.py     # OCR detection (RapidOCR/PaddleOCR/EasyOCR)
      selective_translator.py  # Selective translation classifier
      translator.py       # Context-aware translation (OpenAI API / GLM-5.1)
      renderer.py         # Text erasure + rendering (LaMA/AnyText2/PIL)
      submission.py       # Submission packaging
    solution_a/           # Solution A: AnyTrans pipeline
    solution_b/           # Solution B: HCIIT two-stage pipeline
    solution_c/           # Solution C: E-commerce optimized batch processing
    run_all_solutions.py  # Run all solutions
  docs/                   # Analysis documents
  papers/                 # Related papers PDF + text extraction
  dataset/                # 500 Chinese source images
  outputs/                # Translation results + submission packages
  debug/                  # Debug intermediate files
`

## Three Solutions

| Feature | Solution A (AnyTrans) | Solution B (HCIIT) | Solution C (E-commerce) |
|---------|----------------------|---------------------|------------------------|
| Erasure | PERT (stroke-level) | LaMA | OpenCV |
| Rendering | AnyText2 | AnyText2 (style-conditioned) | PIL |
| Translation | Qwen2.5 + <boxidx> tags | MMLLM + 4-step CoT | Batch API |
| Consistency | - | Translation + Image Gen | - |
| Classification | Basic selective | +Style latent extraction | +E-commerce rules |
| Speed | Medium | Slow | Fast |
| Quality | High | Highest | Medium |

## Quick Start

`ash
cd src
python run_all_solutions.py --solution all
python run_all_solutions.py --solution solution_a --max_images 5
`

## Debug Intermediate Files

Controlled by PipelineConfig debug_* switches, all enabled by default:

| Switch | Output File | Description |
|--------|------------|-------------|
| debug_original | original.png | Original source image copy |
| debug_ocr | ocr_{lang}.png | OCR detection visualization |
| debug_classification | classification_{lang}.json | Selective classification results |
| debug_mask | mask_{lang}.png | Erasure mask |
| debug_erased | erased_{lang}.png | Image after erasure |
| debug_translation | translation_{lang}.json | Translation mapping |
| debug_style | style_{lang}.json | Style extraction (Solution B) |
| debug_context | context_{lang}.json | VLM context (Solution B) |
| debug_product | product_{lang}.json | Product classification (Solution C) |
| debug_quality | quality_{lang}.json | Quality check (Solution C) |
| debug_render | render_{lang}.png | Rendered result |

Override via environment variables: DEBUG_OCR=0 DEBUG_MASK=0 python run_all_solutions.py

## Translation Model Config

Default: OpenAI-compatible API calling GLM-5.1
- base_url: http://127.0.0.1:8082/v1
- api_key: sk-12345679
- model: GLM-5.1
