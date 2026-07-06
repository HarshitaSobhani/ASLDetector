"""Train yolov8n and yolov8s on the ASL letters dataset.

Same hyperparameters for both variants (imgsz=640) so the comparison is
apples-to-apples. Epochs/batch scale down by available accelerator tier
since this box has no CUDA GPU -- see README 'Design Decisions'.
"""
import json
import time
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).parent.parent
DATA_YAML = ROOT / "data" / "dataset" / "data.yaml"
RUNS_DIR = ROOT / "train" / "runs"
WEIGHTS_DIR = ROOT / "train" / "weights"

MODELS = ["yolov8n.pt", "yolov8s.pt"]


def pick_device_and_hparams():
    if torch.cuda.is_available():
        return "0", dict(epochs=50, batch=16)
    if torch.backends.mps.is_available():
        return "mps", dict(epochs=30, batch=8)
    return "cpu", dict(epochs=10, batch=4)


def main():
    device, hparams = pick_device_and_hparams()
    print(f"Device: {device}, hyperparameters: {hparams}")

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    log = {"device": device, "hyperparameters": hparams, "runs": {}}

    for model_name in MODELS:
        tag = model_name.replace(".pt", "")
        print(f"\n=== Training {tag} ===")
        model = YOLO(model_name)
        start = time.time()
        model.train(
            data=str(DATA_YAML),
            imgsz=640,
            epochs=hparams["epochs"],
            batch=hparams["batch"],
            device=device,
            project=str(RUNS_DIR),
            name=tag,
            exist_ok=True,
            verbose=False,
        )
        elapsed = time.time() - start

        best = RUNS_DIR / tag / "weights" / "best.pt"
        dest = WEIGHTS_DIR / f"{tag}.pt"
        dest.write_bytes(best.read_bytes())

        log["runs"][tag] = {"train_seconds": round(elapsed, 1), "weights": str(dest)}
        print(f"{tag} done in {elapsed:.1f}s -> {dest}")

    (WEIGHTS_DIR / "train_log.json").write_text(json.dumps(log, indent=2))
    print("\nTraining log written to", WEIGHTS_DIR / "train_log.json")


if __name__ == "__main__":
    main()
