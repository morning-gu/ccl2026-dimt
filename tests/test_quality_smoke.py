# -*- coding: utf-8 -*-
"""Smoke test for quality_checker plugins: basic (bug-fixed) + competition."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import plugins  # noqa: triggers registration
from interfaces.base import StageType
from common.selective_translator import TextRegion
from plugins.registry import registry


def make_image(h=200, w=400):
    import cv2
    img = np.full((h, w, 3), 220, dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (w, h), (200, 200, 200), -1)
    cv2.rectangle(img, (30, 80), (170, 120), (30, 30, 30), -1)
    cv2.rectangle(img, (220, 80), (360, 120), (30, 30, 30), -1)
    return img


def make_result(img):
    import cv2
    res = img.copy()
    cv2.rectangle(res, (30, 80), (170, 120), (218, 218, 218), -1)
    cv2.rectangle(res, (220, 80), (360, 120), (218, 218, 218), -1)
    cv2.putText(res, "Hello", (40, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2)
    cv2.putText(res, "World", (230, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2)
    return res


def main():
    cfg = type("C", (), {})()
    original = make_image()
    result = make_result(original)
    regions = [
        TextRegion(text="zh1", bbox=[30, 80, 170, 120], is_translatable=True,
                   translated_text="Hello", style_info={"color": (30, 30, 30)}),
        TextRegion(text="zh2", bbox=[220, 80, 360, 120], is_translatable=True,
                   translated_text="World", style_info={"color": (30, 30, 30)}),
    ]

    print("=== basic ===")
    basic = registry.create(StageType.QUALITY_CHECKER, "basic", cfg)
    b = basic.check(original, result, regions)
    print({k: round(v, 3) for k, v in b.items()})

    print("=== competition ===")
    comp = registry.create(StageType.QUALITY_CHECKER, "competition", cfg)
    c = comp.check(original, result, regions)
    for k in sorted(c):
        print("  %-10s = %.3f" % (k, c[k]))
    assert 0.0 <= c["score"] <= 1.0, "score out of range"
    assert c["s_pixel"] > 0.95, "s_pixel too low"
    assert c["t_omiss"] >= 0.5, "t_omiss too low"
    print("")
    print("ALL ASSERTS PASSED")


if __name__ == "__main__":
    main()
