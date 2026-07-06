"""Fetch the ASL Letters dataset in YOLOv8 format.

Tries the real Roboflow Universe "American Sign Language Letters" dataset
(public, ~1.5k images, 26 classes) when ROBOFLOW_API_KEY is set. Otherwise
(or on any download failure) falls back to a synthetic placeholder so the
rest of the pipeline is still runnable end to end -- see README's "Dataset"
section for how to swap in the real data later.
"""
import os
import shutil
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent / "dataset"
WORKSPACE = "david-lee-d0rhs"
PROJECT = "american-sign-language-letters"
VERSION = 6


def try_roboflow_download() -> Path | None:
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("No ROBOFLOW_API_KEY set -- skipping real dataset download.")
        return None
    try:
        from roboflow import Roboflow

        rf = Roboflow(api_key=api_key)
        project = rf.workspace(WORKSPACE).project(PROJECT)
        dataset = project.version(VERSION).download("yolov8", location=str(DATA_DIR))
        return Path(dataset.location) / "data.yaml"
    except Exception as e:  # noqa: BLE001 -- any failure means fall back
        print(f"Roboflow download failed ({e}); falling back to synthetic dataset.")
        return None


def main():
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    DATA_DIR.mkdir(parents=True)

    data_yaml = try_roboflow_download()
    if data_yaml is None:
        sys.path.insert(0, str(Path(__file__).parent))
        from generate_synthetic_dataset import generate

        data_yaml = generate(DATA_DIR)
        (DATA_DIR / "SYNTHETIC.flag").write_text(
            "This dataset is synthetic placeholder data, not the real ASL Letters "
            "dataset. See README 'Dataset' section.\n"
        )

    print(f"data.yaml ready at: {data_yaml}")


if __name__ == "__main__":
    main()
