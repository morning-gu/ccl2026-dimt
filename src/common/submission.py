"""Submission packaging module for CCL2026 competition.

Competition requires:
- 500 Chinese source images x 5 target languages = 2500 output images
- Specific directory structure in the submission zip
- Only 2 submissions allowed in preliminaries
"""
import os
import logging
import zipfile
import shutil
from typing import List, Optional
from pathlib import Path

from .config import PipelineConfig, TARGET_LANGUAGES

logger = logging.getLogger(__name__)


class SubmissionPackager:
    """Package output images into the correct submission format."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._lang_map = {lc.code: lc for lc in TARGET_LANGUAGES}

    def validate_outputs(self, output_dir: str) -> dict:
        """Validate that all required output images exist.

        Returns:
            Dict with validation stats: total, found, missing, per_language stats.
        """
        output_path = Path(output_dir)
        stats = {
            "total_expected": 0,
            "total_found": 0,
            "total_missing": 0,
            "languages": {},
        }

        for lang_config in TARGET_LANGUAGES:
            lang_dir = output_path / lang_config.code
            if lang_dir.exists():
                images = list(lang_dir.glob("*"))
                # Filter to supported formats
                images = [img for img in images if img.suffix.lower() in self.config.supported_image_formats]
                count = len(images)
            else:
                count = 0
            stats["languages"][lang_config.code] = {
                "found": count,
                "dir": str(lang_dir),
            }
            stats["total_found"] += count

        return stats

    def package(
        self,
        output_dir: str,
        zip_path: Optional[str] = None,
    ) -> str:
        """Package output images into submission zip.

        Expected structure:
            submission.zip
            ├── en/
            │   ├── image_001.jpg
            │   ├── image_002.jpg
            │   └── ...
            ├── es/
            ├── pt/
            ├── ja/
            └── fr/

        Args:
            output_dir: Directory containing per-language subdirectories.
            zip_path: Output zip file path. Defaults to output_dir/submission.zip.

        Returns:
            Path to the created zip file.
        """
        output_path = Path(output_dir)
        if zip_path is None:
            zip_path = str(output_path / self.config.submission_zip_name)

        # Validate outputs first
        stats = self.validate_outputs(output_dir)
        logger.info("Packaging submission: %s", stats)

        # Create zip
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for lang_config in TARGET_LANGUAGES:
                lang_dir = output_path / lang_config.code
                if not lang_dir.exists():
                    logger.warning("Missing output directory for language: %s", lang_config.code)
                    continue

                for img_file in sorted(lang_dir.iterdir()):
                    if img_file.suffix.lower() not in self.config.supported_image_formats:
                        continue
                    # Store with language prefix in zip
                    arcname = f"{lang_config.code}/{img_file.name}"
                    zf.write(str(img_file), arcname)

        logger.info("Submission zip created: %s (%d files)", zip_path, stats["total_found"])
        return zip_path

    def organize_outputs(
        self,
        results_dir: str,
        output_dir: str,
    ) -> None:
        """Organize raw pipeline outputs into the expected directory structure.

        Args:
            results_dir: Directory with raw pipeline outputs (may be flat).
            output_dir: Target directory with per-language subdirectories.
        """
        output_path = Path(output_dir)
        results_path = Path(results_dir)

        for lang_config in TARGET_LANGUAGES:
            lang_output = output_path / lang_config.code
            lang_output.mkdir(parents=True, exist_ok=True)

            # Look for language-tagged files in results_dir
            # Pattern: {source_name}_{lang}.{ext} or {lang}/{source_name}.{ext}
            lang_results = results_path / lang_config.code
            if lang_results.exists():
                # Already organized
                for f in lang_results.iterdir():
                    if f.is_file():
                        shutil.copy2(str(f), str(lang_output / f.name))
            else:
                # Look for flat files with language suffix
                for f in results_path.iterdir():
                    if f.is_file() and f.suffix.lower() in self.config.supported_image_formats:
                        # Check if filename contains language code
                        stem = f.stem
                        if stem.endswith(f"_{lang_config.code}"):
                            # Rename: image_001_en.jpg -> image_001.jpg
                            new_name = stem[:-len(f'_{lang_config.code}')] + f.suffix
                            shutil.copy2(str(f), str(lang_output / new_name))
                        elif lang_config.code == TARGET_LANGUAGES[0].code:
                            # Default: copy to first language if no tag
                            shutil.copy2(str(f), str(lang_output / f.name))

        logger.info("Outputs organized in: %s", output_dir)
