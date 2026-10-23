"""
YOLO object detection and pose estimation - upload an image, see what
the model sees.

YOLO frames detection as a single regression problem: one forward pass
outputs bounding boxes + class probabilities for the whole image at once,
instead of proposing regions first and classifying them later. The Pose
mode swaps in a pose model that additionally predicts 17 body keypoints
per person (nose, eyes, ears, shoulders, elbows, wrists, hips, knees,
ankles) and draws them as a skeleton.

Drop at pages/yolo_detect.py, wire in app.py:

    from pages import yolo_detect
    ...
    with demo.route("YOLO Detection", "/yolo"):
        yolo_detect.demo.render()

Needs: ultralytics (pip install ultralytics) + torch + pillow.
Add `ultralytics` to requirements.txt before deploying.
Weights (yolo11n.pt and yolo11n-pose.pt, ~5 MB each) download
automatically on first use of each mode.
The GPU path uses @spaces.GPU (ZeroGPU); falls back to CPU-only
if CUDA is unavailable. The nano models are fast enough on CPU too.
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


MODELS = {
    "Detection": "yolo11n.pt",  # nano detection, COCO 80 classes
    "Pose": "yolo11n-pose.pt",  # nano pose, 17 keypoints per person
}
SAMPLE_URL = "https://ultralytics.com/images/bus.jpg"

_MODEL_CACHE = {}
_SAMPLE = None


def _get_model(mode):
    """Lazy-load per mode so importing the page stays cheap."""
    name = MODELS[mode]
    if name not in _MODEL_CACHE:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is not installed; add it to requirements.txt"
            ) from exc
        _MODEL_CACHE[name] = YOLO(name)  # weights download on first use
    return _MODEL_CACHE[name]


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
def detect(image, conf, mode):
    """Run YOLO on a PIL image; return annotated image, table, info line."""
    if image is None:
        return None, [], "Upload an image first, or load the sample image"
    model = _get_model(mode)
    device = "cuda" if _is_cuda() else "cpu"
    t0 = time.perf_counter()
    results = model.predict(source=image, conf=conf, device=device, verbose=False)[0]
    ms = (time.perf_counter() - t0) * 1000.0

    # plot() returns a BGR numpy array; flip channels to RGB for PIL.
    # In Pose mode it draws boxes plus the skeleton keypoints.
    annotated = Image.fromarray(results.plot()[..., ::-1])
    if mode == "Pose":
        rows, summary = _pose_rows(results)
    else:
        rows, summary = _detect_rows(results)
    info = f"{summary} | device {device} | inference {ms:.0f} ms"
    return annotated, rows, info


def _detect_rows(results):
    names = results.names
    rows = []
    for box in results.boxes:
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        rows.append(
            [
                names[int(box.cls[0])],
                round(float(box.conf[0]), 3),
                "-",
                x1,
                y1,
                x2,
                y2,
            ]
        )
    if rows:
        return rows, f"{len(rows)} objects detected"
    return rows, "No objects detected - try lowering the confidence threshold"


def _pose_rows(results):
    rows = []
    keypoints = results.keypoints
    boxes = results.boxes
    n = 0 if keypoints is None else len(keypoints)
    visible_counts = []
    for i in range(n):
        kpt_conf = keypoints.conf[i]
        visible_counts.append(int((kpt_conf > 0.5).sum().item()))
        mean_kpt = round(float(kpt_conf.mean()), 3)
        box_conf = round(float(boxes.conf[i]), 3)
        x1, y1, x2, y2 = (int(v) for v in boxes.xyxy[i].tolist())
        rows.append([f"person {i + 1}", box_conf, f"{mean_kpt:.3f}", x1, y1, x2, y2])
    if rows:
        detail = ", ".join(f"{v}/17" for v in visible_counts)
        return rows, f"{n} people detected (keypoints visible: {detail})"
    return rows, "No people detected - try lowering the confidence threshold"


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
        "Pose mode swaps in a pose model that also predicts 17 body keypoints "
        "per person and draws them as a skeleton. "
        "In Pose mode the table shows both confidences separately: Box conf is "
        "the number drawn on the image, Keypoint conf is the mean confidence "
        "of that person's 17 keypoints. "
        "Guess how many objects (or people) are in the image before you hit Detect, "
        "then drag the confidence threshold and watch the count change."
    )
    with gr.Row():
        with gr.Column():
            img_in = gr.Image(
                type="pil", label="Upload image", sources=["upload", "webcam"]
            )
            sample_btn = gr.Button("Load sample image (street scene)")
            mode = gr.Radio(
                ["Detection", "Pose"],
                value="Pose",
                label="Mode",
            )
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
                headers=["Class", "Box conf", "Keypoint conf", "x1", "y1", "x2", "y2"],
                datatype=["str", "number", "str", "number", "number", "number", "number"],
                label="Detected objects",
            )
            info = gr.Textbox(label="Info")

    sample_btn.click(fn=_load_sample, outputs=img_in)
    run_btn.click(
        fn=detect, inputs=[img_in, conf, mode], outputs=[img_out, table, info]
    )
