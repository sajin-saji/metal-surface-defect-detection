"""Train a YOLO detector on the synthetic dataset (next project step).

Usage:
    pip install ultralytics
    python src/train_yolo.py --data data/dev --epochs 50

Splits the data 80/20 into train/val, writes a dataset YAML and trains
a small YOLO model. Results go to runs/detect/.
"""
import argparse
import random
import shutil
from pathlib import Path


def split(data_dir: Path, val_ratio: float = 0.2, seed: int = 0) -> Path:
    images = sorted((data_dir / "images").glob("*.png"))
    random.Random(seed).shuffle(images)
    n_val = int(len(images) * val_ratio)
    out = data_dir.parent / (data_dir.name + "_yolo")
    for subset, files in (("val", images[:n_val]), ("train", images[n_val:])):
        for kind in ("images", "labels"):
            (out / kind / subset).mkdir(parents=True, exist_ok=True)
        for img in files:
            shutil.copy(img, out / "images" / subset / img.name)
            lbl = data_dir / "labels" / (img.stem + ".txt")
            shutil.copy(lbl, out / "labels" / subset / lbl.name)
    yaml = out / "dataset.yaml"
    yaml.write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n"
                    "names:\n  0: scratch\n  1: pit\n")
    return yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dev")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--model", default="yolov8n.pt")
    args = ap.parse_args()

    yaml = split(Path(args.data))
    from ultralytics import YOLO  # imported here so the split works without it

    model = YOLO(args.model)
    model.train(data=str(yaml), epochs=args.epochs, imgsz=512)
    print(model.val())


if __name__ == "__main__":
    main()
