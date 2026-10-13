"""Evaluate the OpenCV pipeline against ground-truth labels (precision / recall).

A detection is a true positive if it has the correct class and IoU >= 0.3 with a
not-yet-matched ground-truth box. A moderate IoU is used because boxes around
thin diagonal scratches are hard to match pixel-exactly.
"""
import argparse
from pathlib import Path

import cv2

from detect_opencv import CLASS_NAMES, detect, iou


def load_labels(path: Path, w: int, h: int):
    """Read a YOLO label file and return [(class, (x0, y0, x1, y1)), ...] in pixels."""
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        cls, xc, yc, bw, bh = line.split()
        xc, yc, bw, bh = float(xc) * w, float(yc) * h, float(bw) * w, float(bh) * h
        boxes.append((int(cls), (xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2)))
    return boxes


def evaluate(data_dir: Path, iou_threshold: float = 0.3):
    stats = {c: {"tp": 0, "fp": 0, "fn": 0} for c in CLASS_NAMES}
    for img_path in sorted((data_dir / "images").glob("*.png")):
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        h, w = gray.shape
        truths = load_labels(data_dir / "labels" / f"{img_path.stem}.txt", w, h)
        matched = set()
        for cls, box in detect(gray):
            best, best_j = 0.0, None
            for j, (t_cls, t_box) in enumerate(truths):
                if j not in matched and t_cls == cls:
                    v = iou(box, t_box)
                    if v > best:
                        best, best_j = v, j
            if best >= iou_threshold:
                stats[cls]["tp"] += 1
                matched.add(best_j)
            else:
                stats[cls]["fp"] += 1
        for j, (t_cls, _) in enumerate(truths):
            if j not in matched:
                stats[t_cls]["fn"] += 1
    return stats


def main():
    ap = argparse.ArgumentParser(description="Precision / recall of the OpenCV pipeline.")
    ap.add_argument("--data", default="data/test")
    ap.add_argument("--iou", type=float, default=0.3)
    args = ap.parse_args()

    stats = evaluate(Path(args.data), args.iou)
    print(f"{'class':<10}{'precision':>10}{'recall':>10}{'TP':>6}{'FP':>6}{'FN':>6}")
    for c, s in stats.items():
        p = s["tp"] / max(s["tp"] + s["fp"], 1)
        r = s["tp"] / max(s["tp"] + s["fn"], 1)
        print(f"{CLASS_NAMES[c]:<10}{p:>10.2f}{r:>10.2f}{s['tp']:>6}{s['fp']:>6}{s['fn']:>6}")


if __name__ == "__main__":
    main()
