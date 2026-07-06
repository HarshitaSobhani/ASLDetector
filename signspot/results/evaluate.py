"""Evaluate yolov8n vs yolov8s on the test split: mAP, latency, file size.

Writes results/comparison.md and results/comparison_chart.png.
"""
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO

ROOT = Path(__file__).parent.parent
DATA_YAML = ROOT / "data" / "dataset" / "data.yaml"
WEIGHTS_DIR = ROOT / "train" / "weights"
RESULTS_DIR = ROOT / "results"

TAGS = ["yolov8n", "yolov8s"]
LATENCY_RUNS = 20


def benchmark_latency(model: YOLO, imgsz=640) -> float:
    import numpy as np

    dummy = (np.random.default_rng(0).integers(0, 255, (imgsz, imgsz, 3))).astype("uint8")
    for _ in range(3):  # warmup
        model.predict(dummy, verbose=False)
    start = time.time()
    for _ in range(LATENCY_RUNS):
        model.predict(dummy, verbose=False)
    return (time.time() - start) / LATENCY_RUNS * 1000  # ms/frame


def main():
    rows = []
    for tag in TAGS:
        weights_path = WEIGHTS_DIR / f"{tag}.pt"
        model = YOLO(str(weights_path))

        metrics = model.val(data=str(DATA_YAML), split="test", verbose=False)
        map50 = metrics.box.map50
        map5095 = metrics.box.map

        latency_ms = benchmark_latency(model)
        size_mb = weights_path.stat().st_size / (1024 * 1024)

        rows.append(
            {
                "model": tag,
                "mAP50": round(float(map50), 4),
                "mAP50-95": round(float(map5095), 4),
                "latency_ms_per_frame": round(latency_ms, 2),
                "fps": round(1000 / latency_ms, 1),
                "size_mb": round(size_mb, 2),
            }
        )

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)

    md = ["# Model Comparison: yolov8n vs yolov8s\n", df.to_markdown(index=False), ""]
    (RESULTS_DIR / "comparison.md").write_text("\n".join(md))
    print(df.to_markdown(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(df["model"], df["mAP50"], color=["#4C72B0", "#DD8452"])
    axes[0].set_title("mAP50 (accuracy)")
    axes[0].set_ylim(0, 1)
    axes[1].bar(df["model"], df["latency_ms_per_frame"], color=["#4C72B0", "#DD8452"])
    axes[1].set_title("Latency (ms/frame, lower is better)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "comparison_chart.png", dpi=150)
    print("Wrote results/comparison.md and results/comparison_chart.png")

    best = df.loc[df["mAP50"].idxmax(), "model"]
    (RESULTS_DIR / "best_model.txt").write_text(best)
    print(f"Best model by mAP50: {best}")


if __name__ == "__main__":
    main()
