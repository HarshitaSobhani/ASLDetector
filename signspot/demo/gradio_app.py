"""SignSpot live demo: webcam -> ASL letter detection, via Gradio.

Loads whichever model results/evaluate.py determined performs best
(see results/best_model.txt), falling back to yolov8n if evaluation
hasn't been run yet.
"""
from pathlib import Path

import gradio as gr
from ultralytics import YOLO

ROOT = Path(__file__).parent.parent
WEIGHTS_DIR = ROOT / "train" / "weights"


def best_weights_path() -> Path:
    best_file = ROOT / "results" / "best_model.txt"
    if best_file.exists():
        tag = best_file.read_text().strip()
        candidate = WEIGHTS_DIR / f"{tag}.pt"
        if candidate.exists():
            return candidate
    return WEIGHTS_DIR / "yolov8n.pt"


MODEL_PATH = best_weights_path()
model = YOLO(str(MODEL_PATH))


def detect(frame):
    if frame is None:
        return None
    bgr_frame = frame[:, :, ::-1]  # Gradio gives RGB; Ultralytics assumes BGR for raw numpy input
    results = model.predict(bgr_frame, verbose=False, conf=0.35)
    annotated = results[0].plot()  # returns BGR
    return annotated[:, :, ::-1]  # back to RGB for Gradio


demo = gr.Interface(
    fn=detect,
    inputs=gr.Image(sources=["webcam"], label="Webcam"),
    outputs=gr.Image(label="Detected ASL Letter"),
    live=True,
    title="SignSpot — Real-Time ASL Alphabet Detection",
    description=(
        f"Model: {MODEL_PATH.name}. Show an ASL letter hand sign to the camera; "
        "bounding box, predicted letter, and confidence are drawn live."
    ),
)

if __name__ == "__main__":
    demo.launch()
