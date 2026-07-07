"""Train yolov8n and yolov8s on the ASL letters dataset.

Same hyperparameters for both variants (imgsz=640) so the comparison is
apples-to-apples. Epochs/batch scale down by available accelerator tier
since this box has no CUDA GPU -- see README 'Design Decisions'.
"""
import json
import sys
import time
from pathlib import Path

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import DATA_YAML, HYPERPARAM_TIERS, IMGSZ, device_tier, pick_device  # noqa: E402

ROOT = Path(__file__).parent.parent
RUNS_DIR = ROOT / "train" / "runs"
WEIGHTS_DIR = ROOT / "train" / "weights"

MODELS = ["yolov8n.pt", "yolov8s.pt"]


def main():
    if not DATA_YAML.exists():
        raise SystemExit(f"No dataset at {DATA_YAML}. Run data/download_dataset.py first.")

    device = pick_device()
    hparams = HYPERPARAM_TIERS[device_tier(device)]
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
            imgsz=IMGSZ,
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

        log["runs"][tag] = {"train_seconds": round(elapsed, 1), "weights": str(dest.relative_to(ROOT))}
        print(f"{tag} done in {elapsed:.1f}s -> {dest}")

    (WEIGHTS_DIR / "train_log.json").write_text(json.dumps(log, indent=2))
    print("\nTraining log written to", WEIGHTS_DIR / "train_log.json")


if __name__ == "__main__":
    main()
