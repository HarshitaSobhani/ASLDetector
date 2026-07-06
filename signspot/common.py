"""Shared config and helpers: device selection and model loading.

Centralized so train/eval/demo scripts agree on which accelerator to use
(previously only train.py picked MPS explicitly; eval/demo silently
defaulted to CPU) and so a missing weights file fails loudly instead of
Ultralytics silently substituting base COCO-pretrained weights for a path
that happens to share a filename with an official checkpoint.
"""
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).parent
WEIGHTS_DIR = ROOT / "train" / "weights"
DATA_YAML = ROOT / "data" / "dataset" / "data.yaml"
BEST_MODEL_FILE = ROOT / "results" / "best_model.txt"

IMGSZ = 640
CONF_THRESHOLD = 0.35  # demo detection confidence cutoff, keeps the live feed readable

# epochs/batch per accelerator tier -- see README "Design Decisions"
HYPERPARAM_TIERS = {
    "cuda": dict(epochs=50, batch=16),
    "mps": dict(epochs=30, batch=8),
    "cpu": dict(epochs=10, batch=4),
}


def pick_device() -> str:
    """Best available Ultralytics device string. Ultralytics only auto-selects
    CUDA by default -- MPS must be requested explicitly, which is why this is
    shared rather than left to each script's default."""
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_tier(device: str) -> str:
    """Map an Ultralytics device string to a HYPERPARAM_TIERS key."""
    return "cuda" if device == "0" else device


def best_weights_path() -> Path:
    """Path to whichever model results/evaluate.py determined performs best,
    falling back to yolov8n if evaluation hasn't been run yet."""
    if BEST_MODEL_FILE.exists():
        tag = BEST_MODEL_FILE.read_text().strip()
        candidate = WEIGHTS_DIR / f"{tag}.pt"
        if candidate.exists():
            return candidate
    return WEIGHTS_DIR / "yolov8n.pt"


def load_trained_model(weights_path: Path | None = None) -> YOLO:
    """Load our fine-tuned weights, failing loudly instead of letting
    Ultralytics silently download generic COCO-pretrained weights for a
    missing path that happens to match an official checkpoint filename
    (e.g. train/weights/yolov8n.pt not existing on a fresh clone)."""
    path = weights_path or best_weights_path()
    if not path.exists():
        raise FileNotFoundError(
            f"No trained weights at {path}. Run train/train.py first. "
            "(Refusing to pass this path to Ultralytics as-is: a missing "
            "path named 'yolov8n.pt'/'yolov8s.pt' would silently download "
            "generic COCO-pretrained weights instead of failing.)"
        )
    return YOLO(str(path))
