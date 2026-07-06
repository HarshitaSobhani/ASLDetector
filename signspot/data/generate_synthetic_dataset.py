"""Synthetic ASL-letter placeholder dataset.

Used only when the real Roboflow "American Sign Language Letters" dataset
can't be downloaded (no ROBOFLOW_API_KEY). Produces random colored blobs on
noisy backgrounds with YOLO bounding-box labels across 26 classes (A-Z),
purely to prove the train/eval/demo pipeline runs end to end.

ponytail: fake data, real pipeline. Swap for the real dataset (see README)
to get meaningful accuracy numbers.
"""
import random

import cv2
import numpy as np

CLASSES = [chr(ord("A") + i) for i in range(26)]
IMG_SIZE = 640
N_PER_CLASS = 12  # 26 * 12 = 312 images, enough for an 80/10/10 split


def _make_image(rng: random.Random):
    img = (rng.randint(0, 40) + np.random.default_rng(rng.randint(0, 2**31)).integers(
        0, 60, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8
    )).astype(np.uint8)

    w = rng.randint(150, 350)
    h = rng.randint(150, 350)
    x = rng.randint(0, IMG_SIZE - w)
    y = rng.randint(0, IMG_SIZE - h)
    color = (rng.randint(150, 255), rng.randint(120, 200), rng.randint(120, 200))
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness=-1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), thickness=2)

    xc = (x + w / 2) / IMG_SIZE
    yc = (y + h / 2) / IMG_SIZE
    nw = w / IMG_SIZE
    nh = h / IMG_SIZE
    return img, (xc, yc, nw, nh)


def generate(out_dir):
    rng = random.Random(42)
    samples = []
    for class_id, letter in enumerate(CLASSES):
        for i in range(N_PER_CLASS):
            img, box = _make_image(rng)
            samples.append((f"{letter}_{i:03d}", img, class_id, box))
    rng.shuffle(samples)

    n = len(samples)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    splits = {
        "train": samples[:n_train],
        "val": samples[n_train : n_train + n_val],
        "test": samples[n_train + n_val :],
    }

    for split, items in splits.items():
        img_dir = out_dir / "images" / split
        lbl_dir = out_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for name, img, class_id, (xc, yc, w, h) in items:
            cv2.imwrite(str(img_dir / f"{name}.jpg"), img)
            (lbl_dir / f"{name}.txt").write_text(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"nc: {len(CLASSES)}\n"
        f"names: {CLASSES}\n"
    )
    print(f"Synthetic dataset written to {out_dir} "
          f"(train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])})")
    return data_yaml
