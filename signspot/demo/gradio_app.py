"""SignSpot live demo: webcam -> ASL letter detection, via Gradio.

Loads whichever model results/evaluate.py determined performs best
(see results/best_model.txt), falling back to yolov8n if evaluation
hasn't been run yet.
"""
import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import CONF_THRESHOLD, best_weights_path, load_trained_model, pick_device  # noqa: E402

DEVICE = pick_device()
MODEL_PATH = best_weights_path()
model = load_trained_model(MODEL_PATH)


def detect(frame):
    if frame is None:
        return None
    bgr_frame = frame[:, :, ::-1]  # Gradio gives RGB; Ultralytics assumes BGR for raw numpy input
    results = model.predict(bgr_frame, device=DEVICE, verbose=False, conf=CONF_THRESHOLD)
    annotated = results[0].plot()  # returns BGR
    return annotated[:, :, ::-1]  # back to RGB for Gradio


demo = gr.Interface(
    fn=detect,
    inputs=gr.Image(sources=["webcam"], label="Webcam"),
    outputs=gr.Image(label="Detected ASL Letter"),
    live=True,
    flagging_mode="never",
    title="SignSpot — Real-Time ASL Alphabet Detection",
    description=(
        f"Model: {MODEL_PATH.name} (device: {DEVICE}). Show an ASL letter hand sign to the "
        "camera; bounding box, predicted letter, and confidence are drawn live."
    ),
)

if __name__ == "__main__":
    demo.launch()
