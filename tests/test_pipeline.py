"""Automated checks, run on every push by GitHub Actions."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from detect_opencv import detect, iou, line_kernel  # noqa: E402
from evaluate import evaluate  # noqa: E402
import generate_synthetic as gen  # noqa: E402


def plain_surface():
    return np.full((256, 256), 150, np.uint8)


def test_iou_basic():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-9


def test_line_kernel_shape():
    k = line_kernel(45, 15)
    assert k.shape == (15, 15)
    assert line_kernel(0, 15).sum() == 15  # horizontal line spans the full width
    assert k.sum() >= 10  # diagonal line is about 15 px long -> ~11 pixels


def test_clean_surface_has_no_defects():
    assert detect(plain_surface()) == []


def test_finds_a_pit():
    img = plain_surface()
    cv2.circle(img, (128, 128), 7, 60, -1)
    dets = detect(img)
    assert [c for c, _ in dets] == [1]


def test_finds_a_scratch():
    img = plain_surface()
    cv2.line(img, (40, 40), (200, 180), 70, 2)
    dets = detect(img)
    assert any(c == 0 for c, _ in dets)


def test_end_to_end_quality(tmp_path, monkeypatch):
    """Generate a small dataset and check precision/recall stay above a floor."""
    monkeypatch.setattr(sys, "argv", ["gen", "--n", "20", "--seed", "123", "--out", str(tmp_path)])
    gen.main()
    stats = evaluate(tmp_path)
    for s in stats.values():
        precision = s["tp"] / max(s["tp"] + s["fp"], 1)
        recall = s["tp"] / max(s["tp"] + s["fn"], 1)
        assert precision > 0.8 and recall > 0.8
