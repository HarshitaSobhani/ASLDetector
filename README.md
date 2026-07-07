# ASL Detector

**30-second pitch:** Real-time American Sign Language alphabet detector — fine-tunes YOLOv8 (nano vs small) to draw a bounding box and predicted letter over a webcam feed, with a head-to-head speed/accuracy comparison between model sizes served through a live Gradio demo.

## Problem

Sign language recognition is a well-known real-time object detection use case: given a video frame, localize the signing hand and classify which of the 26 ASL alphabet letters it's forming. It's a good showcase project because it exercises the full pipeline — dataset acquisition, fine-tuning, quantitative model comparison, and a live interactive demo — on a task anyone can understand at a glance.

## Approach

1. **Data**: pull the "American Sign Language Letters" object-detection dataset from Roboflow Universe (26 classes, one per letter, pre-labeled bounding boxes), already in YOLOv8 format with Roboflow's own pre-made 70/20/10 train/val/test split (the synthetic fallback dataset, generated only when no real data is reachable, uses an 80/10/10 split instead — see Dataset section).
2. **Train**: fine-tune both `yolov8n` and `yolov8s` from COCO-pretrained weights, identical hyperparameters (imgsz=640) so the comparison isolates model size, not training recipe.
3. **Evaluate**: run both on the held-out test split, measuring mAP50, mAP50-95, inference latency, and file size.
4. **Demo**: serve the better-performing model through a Gradio webcam interface that draws live bounding boxes + letter + confidence.

## Tradeoffs: nano vs small

- **yolov8n** (~3.2M params): fastest inference, smallest file, lowest ceiling on accuracy. Right choice when the deployment target is CPU-only or a low-power device and "good enough, fast" beats "best possible, slower."
- **yolov8s** (~11.2M params): ~3.5x the parameters, meaningfully better representational capacity, but slower per-frame and a larger download/load footprint.
- On the real dataset (see below), yolov8s wins mAP50 by a real margin (0.966 vs 0.896) while being ~2x slower per frame — a genuine accuracy/speed tradeoff, not just noise from a weak synthetic task.
- **Shipped in the demo:** whichever model wins the test-set mAP50 (see `results/best_model.txt`, chosen automatically by `results/evaluate.py`) — the demo app always loads the current winner rather than hardcoding a size. Currently that's **yolov8s**; if a use case needs the extra ~2x throughput of yolov8n and can tolerate ~7 points lower mAP50, that's a one-line override of `common.py`'s `best_weights_path()`.

## Dataset

Uses the real "American Sign Language Letters" object-detection dataset from Roboflow Universe (`david-lee-d0rhs/american-sign-language-letters`, version 6): 26 classes (A-Z), pre-labeled bounding boxes, split by Roboflow into 504 train / 144 valid / 72 test images.

`data/download_dataset.py` pulls this via the `roboflow` pip package, which requires an API key even for public Universe datasets (the "anonymous" access Roboflow documents is really a pre-filled key tied to that specific public project, still passed as `api_key=`). Set `ROBOFLOW_API_KEY` before running it:
```bash
export ROBOFLOW_API_KEY=your_key_here   # from roboflow.com account settings
python data/download_dataset.py
python train/train.py
python results/evaluate.py
```
If no key is set (or the download fails for any reason), the script automatically falls back to **synthetic placeholder data** instead: random colored rectangles on noisy backgrounds, one random "hand blob" per image, labeled with one of the 26 letters at random (see `data/generate_synthetic_dataset.py`). That fallback exists purely to prove the pipeline runs end-to-end when no real data is reachable — it carries no real signal about ASL classification. A `data/dataset/SYNTHETIC.flag` file is written whenever the fallback is used, so it's obvious which mode produced the current `data/dataset/`.

Two gotchas hit while wiring this up, both fixed in `data/download_dataset.py` (and both regression-tested in `tests/test_data_pipeline.py`): (1) Roboflow's SDK silently no-ops the download if the target directory already exists, even if empty — it treats that as "already cached" with no error; (2) Roboflow's YOLOv8 export writes `data.yaml` paths as `../train/images` etc., assuming a folder nesting that isn't actually there, which the script now corrects after download.

**Honest caveat on what the results below actually mean:** this is a small, single-contributor dataset (720 images total per Roboflow's own dataset card, ~28 per class before splitting) collected in what looks like one or a few photo sessions — one signer, a limited set of backgrounds/lighting conditions. I checked there's no exact-duplicate image leakage between train/valid/test (verified by file hash), but a high mAP50 here almost certainly reflects the model learning that narrow visual domain well, not general robustness to new signers, hand shapes, skin tones, backgrounds, or lighting. Treat the numbers below as "this pipeline correctly trains and measures a real detector," not as "this generalizes to arbitrary webcam users."

## Design Decisions

- **No CUDA GPU available** (Apple M3, no discrete NVIDIA GPU) — training uses PyTorch's MPS (Metal) backend. Since MPS is slower than a real CUDA GPU for this workload, epochs/batch were scaled down from the spec's 50/16 to **30/8** (see `common.py`'s `HYPERPARAM_TIERS`); a plain-CPU fallback further scales to 10/4. Actual device and hyperparameters used for a given run are logged to `train/weights/train_log.json` along with wall-clock training time.
- **Device selection is centralized in `common.py`'s `pick_device()`**, used by every script that touches a model. This exists because of a real bug caught during review: Ultralytics only auto-selects CUDA by default — MPS has to be requested explicitly. `train/train.py` did that; `results/evaluate.py` and the demo scripts didn't, so they were silently running on CPU while training ran on MPS, and the first version of the results table below reported CPU latency numbers without saying so. Fixed by routing every script through the same `pick_device()`.
- **Trained weights are committed** (`train/weights/*.pt`), not gitignored. The alternative (gitignoring them, requiring `train/train.py` before the demo works) has a nasty failure mode: Ultralytics resolves a missing path named `yolov8n.pt`/`yolov8s.pt` by silently downloading the generic 80-class COCO-pretrained checkpoint with that filename, so a fresh clone without weights would run "successfully" while detecting COCO objects instead of ASL letters, with no error. `common.py`'s `load_trained_model()` now also refuses to proceed if the expected path is missing, as a second line of defense.
- **Augmentation**: left at Ultralytics' YOLOv8 defaults (mosaic, HSV jitter, flips) rather than hand-tuning — the real dataset already reached 0.90+/0.96+ mAP50 without any custom recipe.
- **Confidence threshold**: `common.py`'s `CONF_THRESHOLD` (0.35) filters demo detections to keep the live feed readable — pulled into shared config rather than a magic number repeated per script.
- **Best-model selection**: automatic, by test-set mAP50 (see Tradeoffs above), not hardcoded.
- **Latency benchmarking uses random noise input**, not real images (`results/evaluate.py`'s `benchmark_latency`) — this is standard practice for CNN latency measurement: a forward pass's compute cost depends on tensor shape, not pixel content, so a real image would measure identically.

## Results

Trained on the real ASL Letters dataset (504 train / 144 valid / 72 test images, 26 classes), on Apple M3 (MPS backend, no CUDA GPU) — yolov8n took 30.1 min (30 epochs), yolov8s took 59.6 min (30 epochs); full numbers in `train/weights/train_log.json`. Test-set evaluation (`results/evaluate.py`), inference on **MPS** (see Design Decisions — an earlier version of this table accidentally reported CPU latency due to a device-selection bug, since fixed):

| model   | mAP50 | mAP50-95 | latency (ms/frame, MPS) | fps  | size (MB) |
|:--------|------:|---------:|-------------------------:|-----:|----------:|
| yolov8n | 0.896 |    0.856 |                     13.2 | 75.8 |      5.95 |
| yolov8s | 0.966 |    0.935 |                     21.5 | 46.5 |     21.48 |

Both models learned the task well on this dataset (see the Dataset section's caveat about what that does and doesn't tell you about generalization). yolov8s wins mAP50 by ~7 points and is only ~1.6x slower per frame on MPS — both comfortably real-time for a webcam demo, so the accuracy gap is effectively free here. `results/evaluate.py` picked **yolov8s** as the winner by mAP50, so that's what `demo/gradio_app.py` loads. For a deployment target without MPS/CUDA acceleration (plain CPU, or a much lower-power device), yolov8n at 0.896 mAP50 / 5.95MB is the safer default. See [`results/comparison.md`](results/comparison.md) and [`results/comparison_chart.png`](results/comparison_chart.png).

## How to run

Trained weights are committed (`train/weights/*.pt`), so the demo runs immediately after installing dependencies — no need to train first unless you're reproducing the pipeline from scratch or swapping in new data.

```bash
# 1. Set up environment (Python 3.11)
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Run the demo right away (uses the committed, already-trained weights)
python demo/gradio_app.py
# or, on a single image:
python demo/predict_image.py path/to/image.jpg
```

To reproduce the full pipeline from scratch (e.g. after swapping in a different dataset):

```bash
# Get the dataset (needs ROBOFLOW_API_KEY; falls back to synthetic placeholder data otherwise)
export ROBOFLOW_API_KEY=your_key_here
python data/download_dataset.py

# Train both model variants (overwrites train/weights/*.pt)
python train/train.py

# Evaluate + regenerate comparison table/chart
python results/evaluate.py

# Run the test suite (fast, no GPU/dataset required)
python tests/test_data_pipeline.py
```

## Structure

```
common.py       shared device selection, model loading, and config (conf threshold, hyperparameter tiers)
data/           download_dataset.py, generate_synthetic_dataset.py (fallback), dataset/ (data.yaml + images/labels)
train/          train.py, weights/ (best.pt per variant, committed, + train_log.json)
results/        evaluate.py, comparison.md, comparison_chart.png, best_model.txt
demo/           gradio_app.py (live webcam), predict_image.py (single image)
tests/          test_data_pipeline.py -- run with `python tests/test_data_pipeline.py`
.github/workflows/ci.yml   syntax check + test suite on every push/PR (no training -- see below)
```

## Testing & CI

`tests/test_data_pipeline.py` is a plain assert-based script (no test framework) covering the two real bugs this project hit during review: synthetic label format/range correctness (class id, normalized bbox coords), and a regression test for the Roboflow `data.yaml` path-rewrite fix. Run it directly, no pytest required.

CI (`.github/workflows/ci.yml`) runs on every push/PR: installs dependencies, syntax-checks every `.py` file, and runs the test suite. It deliberately does **not** run training or evaluation — that needs a `ROBOFLOW_API_KEY`, takes 30-90 minutes even on Apple Silicon MPS, and GitHub-hosted runners have no GPU/MPS acceleration, making a full retrain in CI both credential-dependent and impractically slow for a per-commit check.
