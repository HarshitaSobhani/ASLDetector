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
        # location must NOT already exist -- roboflow's SDK silently skips the
        # actual download (treats it as already-cached) if the dir is present.
        dataset = project.version(VERSION).download("yolov8", location=str(DATA_DIR))
        data_yaml = Path(dataset.location) / "data.yaml"
        _fix_roboflow_yaml_paths(data_yaml)
        return data_yaml
    except Exception as e:  # noqa: BLE001 -- any failure means fall back
        print(f"Roboflow download failed ({e}); falling back to synthetic dataset.")
        return None


def _fix_roboflow_yaml_paths(data_yaml: Path) -> None:
    """Roboflow's yolov8 export writes train/val/test as '../train/images' etc,
    assuming data.yaml sits one folder deeper than it actually does. Strip the
    stray '../' so paths resolve relative to data.yaml's own directory."""
    text = data_yaml.read_text()
    fixed = text.replace("../train/", "train/").replace("../valid/", "valid/").replace("../test/", "test/")
    data_yaml.write_text(fixed)


def main():
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)

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
