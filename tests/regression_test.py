"""Regression test framework for solution_c image translation.

Runs the pipeline on known test images and checks quality metrics
to detect cross-image regressions before they are committed.

Usage:
    cd src
    python -m pytest ../tests/regression_test.py -v          # run all tests
    python -m pytest ../tests/regression_test.py -v -k 010   # run only 010 tests
    python ../tests/regression_test.py                       # standalone run

The test images (010.jpg, 009.jpg, etc.) are read from
dataset/source_images/. Quality metrics are checked against thresholds
defined in EXPECTED_QUALITY below.

To add a new regression test:
    1. Add the image stem to EXPECTED_QUALITY with minimum thresholds.
    2. Run the test once to get a baseline.
    3. Future changes that degrade metrics below thresholds will fail.
"""
import os
import sys
import json
import logging
from pathlib import Path

import pytest

# Ensure src/ is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

logging.basicConfig(level=logging.WARNING)

# --- Configuration ---

# Minimum quality thresholds per image stem.
# These are intentionally lenient - they catch gross regressions,
# not subtle quality differences.
EXPECTED_QUALITY = {
    "010": {
        "bg_preservation": 0.95,   # background should be mostly preserved
        "no_blanks": 0.80,         # most translated regions should have text
    },
    "009": {
        "bg_preservation": 0.95,
        "no_blanks": 0.80,
    },
}

# Images to test (must exist in dataset/source_images/)
TEST_IMAGES = list(EXPECTED_QUALITY.keys())

# Target language for regression tests (use 'en' as the standard)
TEST_LANG = "en"

# Output directory for test results (temp)
TEST_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "regression_test"


# --- Fixtures ---

@pytest.fixture(scope="module")
def pipeline():
    """Create a SolutionCPipeline instance."""
    from solution_c.pipeline import SolutionCPipeline
    return SolutionCPipeline()


def get_image_path(stem):
    """Get the path to a test image."""
    img_path = PROJECT_ROOT / "dataset" / "source_images" / f"{stem}.jpg"
    if not img_path.exists():
        pytest.skip(f"Test image {stem}.jpg not found at {img_path}")
    return str(img_path)


# --- Tests ---

@pytest.mark.parametrize("stem", TEST_IMAGES)
def test_image_quality(pipeline, stem):
    """Run the pipeline on a test image and check quality metrics.

    This test catches regressions where fixing one image breaks another.
    The per-image override mechanism (image_overrides.json) ensures that
    parameter changes for one image do not affect others.
    """
    image_path = get_image_path(stem)
    output_dir = str(TEST_OUTPUT_DIR / stem)
    os.makedirs(output_dir, exist_ok=True)

    # Run the pipeline for one language
    results = pipeline.process_image_all_languages(image_path, output_dir)

    # Check that we got results
    assert TEST_LANG in results, f"No result for {stem} -> {TEST_LANG}"

    output_path, quality = results[TEST_LANG]
    assert os.path.exists(output_path), f"Output file not created: {output_path}"

    # Check quality metrics against thresholds
    expected = EXPECTED_QUALITY[stem]
    for metric, threshold in expected.items():
        actual = quality.get(metric, 0.0)
        assert actual >= threshold, (
            f"{stem}.jpg regression: {metric}={actual:.4f} "
            f"below threshold {threshold:.4f}"
        )


def test_image_overrides_isolation():
    """Verify that overrides for one image do not affect another.

    This is the core test for the per-image isolation mechanism.
    """
    from common.image_config import ImageConfig

    config = ImageConfig.load()

    # 009 should have overrides
    ovr_009 = config.get("009")
    if config.has_overrides("009"):
        assert ovr_009.erasure_threshold_method is not None, \
            "009 should have erasure_threshold_method override"

    # 010 should NOT have overrides (uses defaults)
    ovr_010 = config.get("010")
    assert ovr_010.erasure_threshold_method is None, \
        "010 should not have erasure overrides (must use defaults)"
    assert ovr_010.render_color_method is None, \
        "010 should not have render overrides (must use defaults)"


def test_no_global_parameter_pollution():
    """Verify that the renderer's default behavior is unchanged when
    no image_stem is provided (backward compatibility)."""
    from common.image_config import ImageConfig

    config = ImageConfig.load()

    # An unconfigured image should get empty overrides
    ovr = config.get("999_nonexistent")
    assert not ovr.has_erasure_overrides()
    assert not ovr.has_render_overrides()
    assert not ovr.has_ocr_overrides()


if __name__ == "__main__":
    # Standalone runner (without pytest)
    print("Running regression tests...")
    from solution_c.pipeline import SolutionCPipeline
    pipe = SolutionCPipeline()

    all_passed = True
    for stem in TEST_IMAGES:
        try:
            image_path = get_image_path(stem)
            output_dir = str(TEST_OUTPUT_DIR / stem)
            os.makedirs(output_dir, exist_ok=True)

            results = pipe.process_image_all_languages(image_path, output_dir)
            output_path, quality = results.get(TEST_LANG, (None, {}))

            expected = EXPECTED_QUALITY[stem]
            passed = True
            for metric, threshold in expected.items():
                actual = quality.get(metric, 0.0)
                status = "PASS" if actual >= threshold else "FAIL"
                if actual < threshold:
                    passed = False
                    all_passed = False
                print(f"  {stem} {metric}: {actual:.4f} (>= {threshold:.4f}) [{status}]")

            if passed:
                print(f"  {stem}: ALL PASS")
            else:
                print(f"  {stem}: FAILED")

        except Exception as e:
            print(f"  {stem}: ERROR - {e}")
            all_passed = False

    # Run isolation tests
    try:
        test_image_overrides_isolation()
        print("  overrides_isolation: PASS")
    except AssertionError as e:
        print(f"  overrides_isolation: FAIL - {e}")
        all_passed = False

    try:
        test_no_global_parameter_pollution()
        print("  no_global_pollution: PASS")
    except AssertionError as e:
        print(f"  no_global_pollution: FAIL - {e}")
        all_passed = False

    print()
    print("ALL PASSED" if all_passed else "SOME FAILED")
    sys.exit(0 if all_passed else 1)
