import gradio as gr
import torch

try:
    import spaces

    _gpu = spaces.GPU  # ZeroGPU: runs the decorated fn on GPU hardware
except Exception:  # no `spaces` package (local dev) -> plain call, no error
    def _gpu(fn):
        return fn


def _gpu_info():
    # Must run inside a @_gpu call: at import time, and in a plain
    # function, the app runs on the CPU container where torch.cuda
    # is not available, even on the Space.
    if not torch.cuda.is_available():
        return "GPU: not available - running on CPU"
    props = torch.cuda.get_device_properties(0)
    return (
        "- GPU: available (%d device(s))\n" % torch.cuda.device_count()
        + "- Name: %s\n" % props.name
        + "- Memory: %.1f GB\n" % (props.total_memory / 1024 ** 3)
        + "- Multiprocessors: %d\n" % props.multi_processor_count
        + "- Compute capability: %d.%d" % (props.major, props.minor)
    )


@_gpu
def _load_gpu_info():
    # Fills the hardware box when the page opens. Decorated so it
    # runs on the GPU worker, where torch.cuda really is available.
    return _gpu_info()


@_gpu
def make_tensor(n):
    # On the Space this runs inside @spaces.GPU, so the tensor is
    # created on the GPU; without a GPU it runs plain on CPU.
    n = max(1, int(n))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t = torch.randn(n, n, device=device)
    matrix = [[round(float(v), 2) for v in row] for row in t.cpu().tolist()]
    summary = (
        "Created a %dx%d random tensor on %s "
        "(float32, mean %.2f)." % (n, n, t.device, float(t.mean()))
    )
    return matrix, summary, _gpu_info()


with gr.Blocks(title="GPU Test") as demo:
    gr.Markdown(
        "# GPU Test\n"
        "This page runs a tiny computation on a GPU (if one is available) "
        "and shows you the hardware it found. The hardware box below fills "
        "in automatically when the page opens - no button needed."
    )
    gr.Markdown("## Your hardware")
    gpu_info = gr.Markdown("Checking for a GPU...")
    gr.Markdown(
        "## Try it\n"
        "Pick a size n and press Run. It creates an n x n random tensor "
        "on the device above, and the table shows the values."
    )
    with gr.Row():
        size = gr.Slider(1, 10, value=3, step=1, label="Tensor size n")
        run_btn = gr.Button("Run", variant="primary")
    matrix = gr.Dataframe(label="Random tensor", datatype="number")
    result = gr.Text(label="Result")

    run_btn.click(make_tensor, inputs=size,
                  outputs=[matrix, result, gpu_info])
    demo.load(_load_gpu_info, outputs=gpu_info)
