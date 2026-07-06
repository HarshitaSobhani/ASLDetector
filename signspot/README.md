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
- On the real dataset (see below), yolov8s wins mAP50 by a real margin (0.966 vs 0.896) while being ~2x slower per frame — a genuine accuracy/speed tradeoff, not just noise from a weak synthetic task.
- **Shipped in the demo:** whichever model wins the test-set mAP50 (see `results/best_model.txt`, chosen automatically by `results/evaluate.py`) — the demo app always loads the current winner rather than hardcoding a size. Currently that's **yolov8s**; if a use case needs the extra ~2x throughput of yolov8n and can tolerate ~7 points lower mAP50, that's a one-line override in `demo/gradio_app.py`'s `best_weights_path()`.

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

Two gotchas hit while wiring this up, both fixed in `data/download_dataset.py`: (1) Roboflow's SDK silently no-ops the download if the target directory already exists, even if empty — it treats that as "already cached" with no error; (2) Roboflow's YOLOv8 export writes `data.yaml` paths as `../train/images` etc., assuming a folder nesting that isn't actually there, which the script now corrects after download.

## Design Decisions

- **No CUDA GPU available** (Apple M3, no discrete NVIDIA GPU) — training uses PyTorch's MPS (Metal) backend. Since MPS is slower than a real CUDA GPU for this workload, epochs/batch were scaled down from the spec's 50/16 to **30/8** (see `train/train.py:pick_device_and_hparams`); a plain-CPU fallback further scales to 10/4. Actual device and hyperparameters used for a given run are logged to `train/weights/train_log.json` along with wall-clock training time.
- **Augmentation**: left at Ultralytics' YOLOv8 defaults (mosaic, HSV jitter, flips) rather than hand-tuning — the real dataset already reached 0.90+/0.96+ mAP50 without any custom recipe.
- **Confidence threshold**: demo filters detections below 0.35 confidence to keep the live feed readable.
- **Best-model selection**: automatic, by test-set mAP50 (see Tradeoffs above), not hardcoded.

## Results

Trained on the real ASL Letters dataset (504 train / 144 valid / 72 test images, 26 classes), on Apple M3 (MPS backend, no CUDA GPU) — yolov8n took 30.1 min (30 epochs), yolov8s took 59.6 min (30 epochs); full numbers in `train/weights/train_log.json`. Test-set evaluation (`results/evaluate.py`):

| model   | mAP50 | mAP50-95 | latency (ms/frame) | fps  | size (MB) |
|:--------|------:|---------:|--------------------:|-----:|----------:|
| yolov8n | 0.896 |    0.856 |                35.0 | 28.6 |      5.95 |
| yolov8s | 0.966 |    0.935 |                73.0 | 13.7 |     21.48 |

Both models learned the task well — real ASL hand shapes turned out to be an easier detection target than the tiny synthetic placeholder set implied. yolov8s wins mAP50 by ~7 points but is ~2.1x slower per frame (CPU inference measured here; both still comfortably real-time for a webcam demo). `results/evaluate.py` picked **yolov8s** as the winner by mAP50, so that's what `demo/gradio_app.py` loads — the accuracy gap here is large enough that the extra latency is worth it for a demo focused on correctness. For a deployment target where frame rate matters more (e.g. a low-power device), yolov8n at 0.896 mAP50 / 28.6 fps is a reasonable second choice. See [`results/comparison.md`](results/comparison.md) and [`results/comparison_chart.png`](results/comparison_chart.png).

## How to run

```bash
# 1. Set up environment (Python 3.11)
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Get the dataset (needs ROBOFLOW_API_KEY; falls back to synthetic placeholder data otherwise)
export ROBOFLOW_API_KEY=your_key_here
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
