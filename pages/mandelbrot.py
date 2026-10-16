"""
Mandelbrot on GPU — thousands of pixels, computed all at once.

Every pixel of the Mandelbrot set is an independent computation: the perfect
GPU workload. This page renders the same fractal on GPU vs CPU and shows the
timing gap — the same gap that makes deep learning possible.

Drop at pages/mandelbrot.py, wire in app.py:

    from pages import mandelbrot
    ...
    with demo.route("Mandelbrot GPU", "/mandelbrot"):
        mandelbrot.demo.render()

Needs: torch, matplotlib. The GPU path uses @spaces.GPU (ZeroGPU);
falls back to CPU-only if CUDA is unavailable.
"""

import time

import gradio as gr
import matplotlib

matplotlib.use("Agg")  # headless-safe on Spaces
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    import spaces

    _gpu = spaces.GPU  # ZeroGPU: runs the decorated fn on GPU hardware
except Exception:  # local dev without the `spaces` package
    def _gpu(fn):
        return fn


W, H = 800, 600  # render resolution

PRESETS = {
    "Full set": (-0.5, 0.0, 3.2),
    "Seahorse Valley": (-0.745, 0.112, 0.4),
    "Antenna tip": (-1.768, 0.0, 0.2),
}


# ------------------------------------------------------- core math
def mandelbrot_iters(cx, cy, width, max_iter, device):
    """Iteration counts per pixel, vectorized over the whole grid."""
    hh = width * H / W / 2.0
    xs = torch.linspace(cx - width / 2.0, cx + width / 2.0, W, device=device)
    ys = torch.linspace(cy + hh, cy - hh, H, device=device)  # row 0 = top
    c = (xs.unsqueeze(0) + 1j * ys.unsqueeze(1)).to(torch.complex64)
    z = torch.zeros_like(c)
    iters = torch.zeros(c.shape, dtype=torch.float32, device=device)
    alive = torch.ones(c.shape, dtype=torch.bool, device=device)
    for i in range(1, max_iter + 1):
        z = torch.where(alive, z * z + c, z)
        escaped = z.abs() > 2.0
        newly = escaped & alive
        if bool(newly.any()):
            iters[newly] = float(i)
            alive &= ~escaped
        if not bool(alive.any()):
            break
    iters[alive] = float(max_iter)
    return iters


def _timed_ms(fn):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return out, (time.perf_counter() - t0) * 1000.0


@_gpu
def render_gpu(cx, cy, width, max_iter):
    if not torch.cuda.is_available():
        return None, None
    iters, ms = _timed_ms(
        lambda: mandelbrot_iters(cx, cy, width, max_iter, "cuda"))
    return iters.cpu().numpy(), ms


def render_cpu(cx, cy, width, max_iter):
    iters, ms = _timed_ms(
        lambda: mandelbrot_iters(cx, cy, width, max_iter, "cpu"))
    return iters.numpy(), ms


# ------------------------------------------------------- presentation
def colorize(iters, max_iter):
    t = np.clip(iters / max_iter, 0.0, 1.0)
    rgba = plt.get_cmap("turbo")(t)[..., :3]
    rgba[iters >= max_iter] = (0.0, 0.0, 0.0)  # interior = black
    return (rgba * 255).astype(np.uint8)


def plot_timing(cpu_ms, gpu_ms):
    fig, ax = plt.subplots(figsize=(5, 3.0))
    labels, vals, colors = ["CPU"], [cpu_ms], ["#7f7f7f"]
    title = "render time"
    if gpu_ms is not None:
        labels.append("GPU")
        vals.append(gpu_ms)
        colors.append("#2ca02c")
        title = f"render time — GPU {cpu_ms / gpu_ms:.1f}× faster"
    bars = ax.barh(labels, vals, color=colors)
    ax.set_xscale("log")
    ax.set_xlabel("milliseconds (log scale)")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                f"  {v:.0f} ms", va="center", fontsize=11)
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    return fig


def on_preset(preset):
    if preset in PRESETS:
        cx, cy, width = PRESETS[preset]
        return gr.update(value=cx), gr.update(value=cy), \
            gr.update(value=float(np.log10(width)))
    return gr.update(), gr.update(), gr.update()


def on_render(preset, cx, cy, log_width, max_iter):
    width, max_iter = 10.0 ** float(log_width), int(max_iter)
    iters_cpu, cpu_ms = render_cpu(cx, cy, width, max_iter)
    iters_gpu, gpu_ms = render_gpu(cx, cy, width, max_iter)
    if iters_gpu is None:  # no CUDA here — show CPU render
        img = colorize(iters_cpu, max_iter)
        fig = plot_timing(cpu_ms, None)
        summary = (f"GPU not available in this environment — showing the CPU "
                   f"render ({cpu_ms:.0f} ms). On the Space this runs on "
                   f"ZeroGPU via `@spaces.GPU`.")
    else:
        img = colorize(iters_gpu, max_iter)
        fig = plot_timing(cpu_ms, gpu_ms)
        summary = (
            f"**GPU: {gpu_ms:.0f} ms** · CPU: {cpu_ms:.0f} ms · "
            f"**speedup: {cpu_ms / gpu_ms:.1f}×** — identical math, the only "
            f"difference is how many pixels compute at once. "
            f"({W}×{H} pixels, {max_iter} max iterations)"
        )
    return img, fig, summary


# ------------------------------------------------------- UI
with gr.Blocks(title="Mandelbrot GPU") as demo:
    gr.Markdown(
        "# 🌀 Mandelbrot, in Parallel\n"
        "*Every pixel is its own little computation — thousands at once. "
        "That is the entire GPU story, and it is why deep learning runs on them.*"
    )
    with gr.Row():
        preset = gr.Dropdown(list(PRESETS) + ["Custom"], value="Full set",
                             label="view")
        cx = gr.Number(value=-0.5, label="center x")
        cy = gr.Number(value=0.0, label="center y")
        log_width = gr.Slider(-3, 0.6, step=0.1, value=0.5,
                              label="view width = 10^x")
        max_iter = gr.Slider(50, 1000, step=50, value=256,
                             label="max iterations")
    go = gr.Button("Render ⚡", variant="primary")
    with gr.Row():
        img = gr.Image(label="GPU render")
        timing = gr.Plot(label="timing")
    summary = gr.Markdown()
    gr.Markdown(
        "### 🔬 Things to try\n"
        "1. **Push max iterations to 1000 and zoom in** — the CPU visibly "
        "slows down while the GPU barely notices. That gap is the whole game.\n"
        "2. **Seahorse Valley preset, then shrink the width** — dive into "
        "the seahorses. Each zoom level is 10× more pixels doing 10× more work.\n"
        "3. **Notice the small cases**: at low iterations the gap shrinks — "
        "the GPU wins on *massive* parallelism, which is exactly what "
        "neural-network training looks like."
    )

    preset.change(on_preset, inputs=preset, outputs=[cx, cy, log_width])
    go.click(on_render, inputs=[preset, cx, cy, log_width, max_iter],
             outputs=[img, timing, summary])
