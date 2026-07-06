# SignSpot

**30-second pitch:** Real-time American Sign Language alphabet detector — fine-tunes YOLOv8 (nano vs small) to draw a bounding box and predicted letter over a webcam feed, with a head-to-head speed/accuracy comparison between model sizes served through a live Gradio demo.

## Problem

Sign language recognition is a well-known real-time object detection use case: given a video frame, localize the signing hand and classify which of the 26 ASL alphabet letters it's forming. It's a good showcase project because it exercises the full pipeline — dataset acquisition, fine-tuning, quantitative model comparison, and a live interactive demo — on a task anyone can understand at a glance.

## Approach

1. **Data**: pull the "American Sign Language Letters" object-detection dataset from Roboflow Universe (26 classes, one per letter, pre-labeled bounding boxes), already in/converted to YOLOv8 format with an 80/10/10 train/val/test split.
2. **Train**: fine-tune both `yolov8n` and `yolov8s` from COCO-pretrained weights, identical hyperparameters (imgsz=640) so the comparison isolates model size, not training recipe.
3. **Evaluate**: run both on the held-out test split, measuring mAP50, mAP50-95, inference latency, and file size.
4. **Demo**: serve the better-performing model through a Gradio webcam interface that draws live bounding boxes + letter + confidence.

## Tradeoffs: nano vs small

- **yolov8n** (~3.2M params): fastest inference, smallest file, lowest ceiling on accuracy. Right choice when the deployment target is CPU-only or a low-power device and "good enough, fast" beats "best possible, slower."
- **yolov8s** (~11.2M params): ~3.5x the parameters, meaningfully better representational capacity, but slower per-frame and a larger download/load footprint.
- On a real, larger ASL dataset the accuracy gap between the two typically widens in `s`'s favor; on this run's dataset (see Dataset caveat below) the gap is a lower bound.
- **Shipped in the demo:** whichever model wins the test-set mAP50 (see `results/best_model.txt`, chosen automatically by `results/evaluate.py`) — the demo app always loads the current winner rather than hardcoding a size, so re-running evaluation on the real dataset naturally promotes the better model without touching demo code.

## Dataset

**IMPORTANT — placeholder data notice:** This run downloads via the `roboflow` pip package, which requires an API key even for public Universe datasets (the "anonymous" access Roboflow documents is really a pre-filled key for that specific public project, still passed as `api_key=`). No key was available in this environment, so `data/download_dataset.py` automatically fell back to **synthetic placeholder data**: random colored rectangles on noisy backgrounds, one random "hand blob" per image, labeled with one of the 26 letters at random. This exists purely to prove the download → train → eval → demo pipeline runs end-to-end; it carries **no real signal** about ASL classification and the mAP numbers in `results/comparison.md` reflect learning an easy synthetic task, not sign language.

**To swap in the real dataset:**
```bash
export ROBOFLOW_API_KEY=your_key_here   # from roboflow.com account settings
python data/download_dataset.py         # re-downloads real data, overwriting the synthetic set
python train/train.py
python results/evaluate.py
```
The rest of the pipeline (training, evaluation, demo) needs no code changes — it's driven entirely by `data/dataset/data.yaml`.

## Design Decisions

- **No CUDA GPU available** (Apple M3, no discrete NVIDIA GPU) — training uses PyTorch's MPS (Metal) backend. Since MPS is slower than a real CUDA GPU for this workload, epochs/batch were scaled down from the spec's 50/16 to **30/8** (see `train/train.py:pick_device_and_hparams`); a plain-CPU fallback further scales to 10/4. Actual device and hyperparameters used for a given run are logged to `train/weights/train_log.json` along with wall-clock training time.
- **Dataset size (synthetic)**: 26 classes × 12 images = 312 images total, split 80/10/10. Small on purpose — this is a pipeline smoke test, not a training run meant to produce a usable classifier.
- **Augmentation**: left at Ultralytics' YOLOv8 defaults (mosaic, HSV jitter, flips) rather than hand-tuning — no evidence a custom recipe would help before the real dataset is in place.
- **Confidence threshold**: demo filters detections below 0.35 confidence to keep the live feed readable.
- **Best-model selection**: automatic, by test-set mAP50 (see Tradeoffs above), not hardcoded.

## Results

Trained on Apple M3 (MPS backend, no CUDA GPU) — yolov8n took 12.9 min (30 epochs), yolov8s took 35.0 min (30 epochs); full numbers in `train/weights/train_log.json`. Test-set evaluation (`results/evaluate.py`), **on the synthetic placeholder dataset** (see caveat above — these are pipeline-sanity numbers, not real ASL accuracy):

| model   | mAP50 | mAP50-95 | latency (ms/frame) | fps  | size (MB) |
|:--------|------:|---------:|--------------------:|-----:|----------:|
| yolov8n | 0.150 |    0.148 |                45.0 | 22.2 |      5.97 |
| yolov8s | 0.208 |    0.208 |               102.4 |  9.8 |     21.49 |

yolov8s scores higher mAP50 but is ~2.3x slower per frame and ~3.6x the file size — the expected accuracy/speed tradeoff, though the absolute numbers are low because the synthetic task (random rectangle position/size, arbitrary label) has weak visual signal by design. `results/evaluate.py` picked **yolov8s** as the winner by mAP50, so that's what `demo/gradio_app.py` loads. On the real dataset this comparison should be re-run — with real hand shapes the accuracy gap and the right size/speed pick could both look different (a real-time webcam demo may end up preferring yolov8n's 2x+ speed if the accuracy gap narrows). See [`results/comparison.md`](results/comparison.md) and [`results/comparison_chart.png`](results/comparison_chart.png).

## How to run

```bash
# 1. Set up environment (Python 3.11)
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Get the dataset (real if ROBOFLOW_API_KEY is set, else synthetic fallback)
python data/download_dataset.py

# 3. Train both model variants
python train/train.py

# 4. Evaluate + generate comparison table/chart
python results/evaluate.py

# 5a. Live webcam demo
python demo/gradio_app.py

# 5b. Or run inference on a single image (no webcam needed)
python demo/predict_image.py path/to/image.jpg
```

## Structure

```
signspot/
  data/     download_dataset.py, generate_synthetic_dataset.py (fallback), dataset/ (data.yaml + images/labels)
  train/    train.py, weights/ (best.pt per variant + train_log.json)
  results/  evaluate.py, comparison.md, comparison_chart.png, best_model.txt
  demo/     gradio_app.py (live webcam), predict_image.py (single image)
```
