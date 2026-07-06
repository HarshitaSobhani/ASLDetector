"""Assert-based checks for the two data-pipeline bugs this project actually hit:
label format/range correctness, and the Roboflow yaml path-rewrite regression.

Run directly: python tests/test_data_pipeline.py
"""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.download_dataset import _fix_roboflow_yaml_paths  # noqa: E402
from data.generate_synthetic_dataset import CLASSES, N_PER_CLASS, generate  # noqa: E402


def test_synthetic_labels_are_valid_yolo_format(tmp_dir: Path):
    data_yaml = generate(tmp_dir)
    assert data_yaml.exists()

    total_images = len(CLASSES) * N_PER_CLASS
    expected_train = int(total_images * 0.8)
    expected_val = int(total_images * 0.1)
    expected_test = total_images - expected_train - expected_val

    for split, expected_count in [("train", expected_train), ("val", expected_val), ("test", expected_test)]:
        images = sorted((tmp_dir / "images" / split).glob("*.jpg"))
        labels = sorted((tmp_dir / "labels" / split).glob("*.txt"))
        assert len(images) == expected_count, f"{split}: expected {expected_count} images, got {len(images)}"
        assert len(images) == len(labels), f"{split}: image/label count mismatch"

        image_stems = {p.stem for p in images}
        label_stems = {p.stem for p in labels}
        assert image_stems == label_stems, f"{split}: image/label filenames don't match 1:1"

        for label_path in labels:
            line = label_path.read_text().strip()
            fields = line.split()
            assert len(fields) == 5, f"{label_path}: expected 5 fields (class x y w h), got {len(fields)}"

            class_id = int(fields[0])
            assert 0 <= class_id < len(CLASSES), f"{label_path}: class id {class_id} out of range"

            x, y, w, h = (float(v) for v in fields[1:])
            for name, value in [("x", x), ("y", y), ("w", w), ("h", h)]:
                assert 0.0 <= value <= 1.0, f"{label_path}: {name}={value} not normalized to [0,1]"

    print("test_synthetic_labels_are_valid_yolo_format: PASS")


def test_fix_roboflow_yaml_paths_strips_stray_parent_refs(tmp_dir: Path):
    # Regression test: Roboflow's yolov8 export writes '../train/images' etc,
    # assuming data.yaml lives one directory deeper than it actually does.
    # We hit this for real (silent wrong paths -> "no labels found" from
    # Ultralytics) before adding _fix_roboflow_yaml_paths.
    fake_yaml = tmp_dir / "data.yaml"
    fake_yaml.write_text(
        "train: ../train/images\n"
        "val: ../valid/images\n"
        "test: ../test/images\n"
        "nc: 26\n"
    )

    _fix_roboflow_yaml_paths(fake_yaml)
    fixed = fake_yaml.read_text()

    assert "../" not in fixed, f"stray '../' still present after fix: {fixed!r}"
    assert "train: train/images" in fixed
    assert "val: valid/images" in fixed
    assert "test: test/images" in fixed

    print("test_fix_roboflow_yaml_paths_strips_stray_parent_refs: PASS")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        test_synthetic_labels_are_valid_yolo_format(Path(tmp) / "synthetic")

    with tempfile.TemporaryDirectory() as tmp:
        test_fix_roboflow_yaml_paths_strips_stray_parent_refs(Path(tmp))

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
