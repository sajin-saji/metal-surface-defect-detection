"""Generate synthetic metal-surface images with scratches and pits.

Each image gets a YOLO-format label file (class x_center y_center width height,
all normalised to 0..1), so the same data can later be used to train YOLO.

Classes: 0 = scratch, 1 = pit
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

IMG_SIZE = 512


def brushed_metal(rng: np.random.Generator) -> np.ndarray:
    """Grey base with horizontal brushing texture and uneven lighting."""
    base = rng.integers(120, 170)
    noise = rng.normal(0, 18, (IMG_SIZE, IMG_SIZE)).astype(np.float32)
    # Blur strongly along x only -> brushed streaks
    streaks = cv2.GaussianBlur(noise, (61, 1), 0) * 2.5
    fine = rng.normal(0, 4, (IMG_SIZE, IMG_SIZE)).astype(np.float32)
    # Uneven illumination: a soft gradient across the image
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE].astype(np.float32)
    angle = rng.uniform(0, 2 * np.pi)
    light = 20 * (np.cos(angle) * xx + np.sin(angle) * yy) / IMG_SIZE
    img = base + streaks + fine + light
    return np.clip(img, 0, 255)


def add_scratch(img, rng):
    """Thin, long, dark line (optionally slightly curved)."""
    length = rng.integers(80, 300)
    angle = rng.uniform(0, np.pi)
    x0, y0 = rng.integers(40, IMG_SIZE - 40, 2)
    x1 = int(np.clip(x0 + length * np.cos(angle), 5, IMG_SIZE - 5))
    y1 = int(np.clip(y0 + length * np.sin(angle), 5, IMG_SIZE - 5))
    depth = rng.uniform(45, 80)
    thickness = int(rng.integers(1, 3))
    mask = np.zeros_like(img)
    cv2.line(mask, (int(x0), int(y0)), (x1, y1), 1.0, thickness, cv2.LINE_AA)
    img -= depth * mask
    xs, ys = sorted([x0, x1]), sorted([y0, y1])
    pad = 4
    return 0, (xs[0] - pad, ys[0] - pad, xs[1] + pad, ys[1] + pad)


def add_pit(img, rng):
    """Small round dark spot with a soft edge."""
    r = int(rng.integers(4, 11))
    cx, cy = rng.integers(20, IMG_SIZE - 20, 2)
    mask = np.zeros_like(img)
    cv2.circle(mask, (int(cx), int(cy)), r, 1.0, -1, cv2.LINE_AA)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    img -= rng.uniform(55, 90) * mask
    pad = 3
    return 1, (cx - r - pad, cy - r - pad, cx + r + pad, cy + r + pad)


def to_yolo(cls, box):
    x0, y0, x1, y1 = [float(np.clip(v, 0, IMG_SIZE)) for v in box]
    return (f"{cls} {(x0 + x1) / 2 / IMG_SIZE:.6f} {(y0 + y1) / 2 / IMG_SIZE:.6f} "
            f"{(x1 - x0) / IMG_SIZE:.6f} {(y1 - y0) / IMG_SIZE:.6f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/dev")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    img_dir, lbl_dir = Path(args.out, "images"), Path(args.out, "labels")
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.n):
        img = brushed_metal(rng)
        labels = []
        for _ in range(rng.integers(0, 3)):
            labels.append(to_yolo(*add_scratch(img, rng)))
        for _ in range(rng.integers(0, 4)):
            labels.append(to_yolo(*add_pit(img, rng)))
        cv2.imwrite(str(img_dir / f"img_{i:04d}.png"), np.clip(img, 0, 255).astype(np.uint8))
        (lbl_dir / f"img_{i:04d}.txt").write_text("\n".join(labels))
    print(f"Wrote {args.n} images to {img_dir}")


if __name__ == "__main__":
    main()
