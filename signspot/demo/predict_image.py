"""Run the trained ASL detector on a single static image.

Usage:
    python demo/predict_image.py path/to/image.jpg [--out path/to/annotated.jpg]
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).parent.parent
WEIGHTS_DIR = ROOT / "train" / "weights"


def best_weights_path() -> Path:
    best_file = ROOT / "results" / "best_model.txt"
    if best_file.exists():
        tag = best_file.read_text().strip()
        return WEIGHTS_DIR / f"{tag}.pt"
    return WEIGHTS_DIR / "yolov8n.pt"  # fallback default


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--out", default=None, help="Where to save the annotated image")
    parser.add_argument("--weights", default=None, help="Override weights path")
    args = parser.parse_args()

    weights = Path(args.weights) if args.weights else best_weights_path()
    model = YOLO(str(weights))

    results = model.predict(args.image, verbose=False)
    result = results[0]

    out_path = Path(args.out) if args.out else Path(args.image).with_stem(Path(args.image).stem + "_annotated")
    result.save(filename=str(out_path))

    print(f"Model: {weights.name}")
    for box in result.boxes:
        letter = result.names[int(box.cls)]
        conf = float(box.conf)
        print(f"  {letter}: {conf:.2%}")
    print(f"Annotated image saved to {out_path}")


if __name__ == "__main__":
    main()
