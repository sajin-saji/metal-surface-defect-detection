"""Classical OpenCV pipeline for detecting scratches and pits on metal surfaces.

Two detection paths share one pre-processing step:

  Pre-processing   subtract a heavily blurred background -> removes uneven lighting,
                   so dark defects become bright peaks on a flat image
  Path A (blobs)   black-hat morphology + threshold -> contours, classified by shape
                   (compact -> pit, elongated -> scratch)
  Path B (lines)   oriented line filter: morphological opening with a thin line kernel
                   at several angles keeps only straight, line-like structures.
                   Finds faint scratches that path A misses.
                   Near-horizontal angles are skipped because the brushing texture
                   is horizontal (known limitation, see README).

Parameters were tuned on the development set (seed 42) and reported on a
separate test set (seed 7).
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

CLASS_NAMES = {0: "scratch", 1: "pit"}
COLORS = {0: (0, 0, 255), 1: (0, 200, 255)}  # BGR: red = scratch, yellow = pit

# Tuned on the development set
BLOB_MIN_THRESHOLD = 25
LINE_THRESHOLD = 10
LINE_LENGTH = 15
LINE_ANGLES = range(20, 161, 10)  # degrees; skips +-20 deg around horizontal
MIN_AREA = 30


def flatten_illumination(gray: np.ndarray) -> np.ndarray:
    """Background minus image: dark defects -> bright peaks, lighting removed."""
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=25)
    return cv2.GaussianBlur(cv2.subtract(background, gray), (3, 3), 0)


def line_kernel(angle_deg: float, length: int) -> np.ndarray:
    """A 1-pixel-wide line of the given angle, used as structuring element."""
    k = np.zeros((length, length), np.uint8)
    c = length // 2
    dx, dy = np.cos(np.deg2rad(angle_deg)) * c, np.sin(np.deg2rad(angle_deg)) * c
    cv2.line(k, (round(c - dx), round(c - dy)), (round(c + dx), round(c + dy)), 1, 1)
    return k


def iou(a, b) -> float:
    """Intersection over union of two boxes (x0, y0, x1, y1)."""
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def detect_blobs(gray: np.ndarray, flat: np.ndarray):
    """Path A: black-hat + threshold, classify contours by elongation."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    blackhat = cv2.morphologyEx(255 - gray, cv2.MORPH_TOPHAT, kernel)
    response = cv2.addWeighted(flat, 0.5, blackhat, 0.5, 0)

    otsu, _ = cv2.threshold(response, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask = cv2.threshold(response, max(otsu, BLOB_MIN_THRESHOLD), 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    detections = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) < MIN_AREA:
            continue
        (_, _), (w, h), _ = cv2.minAreaRect(c)
        elongation = max(w, h) / max(min(w, h), 1)
        x, y, bw, bh = cv2.boundingRect(c)
        detections.append((0 if elongation > 4 else 1, (x, y, x + bw, y + bh)))
    return detections


def detect_lines(flat: np.ndarray):
    """Path B: oriented line filter for faint scratches."""
    response = np.zeros_like(flat)
    for angle in LINE_ANGLES:
        opened = cv2.morphologyEx(flat, cv2.MORPH_OPEN, line_kernel(angle, LINE_LENGTH))
        response = np.maximum(response, opened)

    _, mask = cv2.threshold(response, LINE_THRESHOLD, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    detections = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        (_, _), (w, h), _ = cv2.minAreaRect(c)
        long_side, short_side = max(w, h), max(min(w, h), 1)
        if long_side < 40 or long_side / short_side < 4:  # must be long and thin
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        detections.append((0, (x, y, x + bw, y + bh)))
    return detections


def detect(gray: np.ndarray):
    """Run both paths and merge; a line detection is dropped if path A already found it."""
    flat = flatten_illumination(gray)
    detections = detect_blobs(gray, flat)
    for cls, box in detect_lines(flat):
        if not any(c == 0 and iou(box, b) > 0.2 for c, b in detections):
            detections.append((cls, box))
    return detections


def draw(img_bgr: np.ndarray, detections) -> np.ndarray:
    out = img_bgr.copy()
    for cls, (x0, y0, x1, y1) in detections:
        cv2.rectangle(out, (x0, y0), (x1, y1), COLORS[cls], 2)
        cv2.putText(out, CLASS_NAMES[cls], (x0, max(y0 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS[cls], 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser(description="Annotate images with detected defects.")
    ap.add_argument("--images", default="data/test/images")
    ap.add_argument("--out", default="results/opencv")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(Path(args.images).glob("*.png"))
    for p in paths:
        img = cv2.imread(str(p))
        cv2.imwrite(str(out_dir / p.name), draw(img, detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))))
    print(f"Annotated {len(paths)} images -> {out_dir}")


if __name__ == "__main__":
    main()
