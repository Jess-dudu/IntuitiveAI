"""
YOLO object detection - upload an image, see what the model sees.

YOLO frames detection as a single regression problem: one forward pass
outputs bounding boxes + class probabilities for the whole image at once,
instead of proposing regions first and classifying them later.

Drop at pages/yolo_detect.py, wire in app.py:

    from pages import yolo_detect
    ...
    with demo.route("YOLO Detection", "/yolo"):
        yolo_detect.demo.render()

Needs: ultralytics (pip install ultralytics) + torch + pillow.
Add `ultralytics` to requirements.txt before deploying.
Weights (yolo11n.pt, ~5 MB) download automatically on first run.
The GPU path uses @spaces.GPU (ZeroGPU); falls back to CPU-only
if CUDA is unavailable. YOLO11n is fast enough to run on CPU too.
"""

import time
import urllib.request
from io import BytesIO

import gradio as gr
from PIL import Image

try:
    import spaces

    _gpu = spaces.GPU  # ZeroGPU: runs the decorated fn on GPU hardware
except Exception:  # local dev without the `spaces` package
    def _gpu(fn):
        return fn


MODEL_NAME = "yolo11n.pt"  # nano: smallest YOLO11, COCO 80 classes
SAMPLE_URL = "https://ultralytics.com/images/bus.jpg"

_MODEL = None
_SAMPLE = None


def _get_model():
    """Lazy-load so importing the page stays cheap; weights download once."""
    global _MODEL
    if _MODEL is None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is not installed; add it to requirements.txt"
            ) from exc
        _MODEL = YOLO(MODEL_NAME)
    return _MODEL


def _load_sample():
    global _SAMPLE
    if _SAMPLE is None:
        try:
            with urllib.request.urlopen(SAMPLE_URL, timeout=20) as resp:
                _SAMPLE = Image.open(BytesIO(resp.read())).convert("RGB")
        except Exception as exc:
            raise gr.Error("Sample image download failed; please upload an image") from exc
    return _SAMPLE


@_gpu
def detect(image, conf):
    """Run YOLO on a PIL image; return annotated image, table, info line."""
    if image is None:
        return None, [], "Upload an image first, or load the sample image"
    model = _get_model()
    device = "cuda" if _is_cuda() else "cpu"
    t0 = time.perf_counter()
    results = model.predict(source=image, conf=conf, device=device, verbose=False)[0]
    ms = (time.perf_counter() - t0) * 1000.0

    # plot() returns a BGR numpy array; flip channels to RGB for PIL.
    annotated = Image.fromarray(results.plot()[..., ::-1])
    names = results.names
    rows = []
    for box in results.boxes:
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        rows.append(
            [
                names[int(box.cls[0])],
                round(float(box.conf[0]), 3),
                x1,
                y1,
                x2,
                y2,
            ]
        )
    if rows:
        info = f"{len(rows)} objects detected | device {device} | inference {ms:.0f} ms"
    else:
        info = f"No objects detected (device {device}, {ms:.0f} ms) - try lowering the confidence threshold"
    return annotated, rows, info


def _is_cuda():
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


# ------------------------------------------------------------- UI
with gr.Blocks() as demo:
    gr.Markdown(
        "YOLO treats object detection as a single regression: one forward pass "
        "over the whole image outputs every box and class at once. "
        "Guess how many objects are in the image before you hit Detect, "
        "then drag the confidence threshold and watch the count change."
    )
    with gr.Row():
        with gr.Column():
            img_in = gr.Image(
                type="pil", label="Upload image", sources=["upload", "webcam"]
            )
            sample_btn = gr.Button("Load sample image (street scene)")
            conf = gr.Slider(
                0.05,
                0.95,
                value=0.25,
                step=0.05,
                label="Confidence threshold (lower finds more, but also more false positives)",
            )
            run_btn = gr.Button("Detect", variant="primary")
        with gr.Column():
            img_out = gr.Image(label="Detection result")
            table = gr.Dataframe(
                headers=["Class", "Confidence", "x1", "y1", "x2", "y2"],
                datatype=["str", "number", "number", "number", "number", "number"],
                label="Detected objects",
            )
            info = gr.Textbox(label="Info")

    sample_btn.click(fn=_load_sample, outputs=img_in)
    run_btn.click(fn=detect, inputs=[img_in, conf], outputs=[img_out, table, info])
