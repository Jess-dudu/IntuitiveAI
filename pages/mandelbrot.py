"""
Mandelbrot in parallel — GPU vs CPU render race.

Every pixel of the Mandelbrot set is an independent computation: the perfect
GPU workload. This page renders it on GPU vs CPU and shows the timing gap —
the same gap that makes deep learning possible.

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
import math
import numpy as np
import torch

try:
    import spaces

    _gpu = spaces.GPU  # ZeroGPU: runs the decorated fn on GPU hardware
except Exception:  # local dev without the `spaces` package
    def _gpu(fn):
        return fn


W, H = 800, 600  # render resolution

VIEW_PRESETS = {
    "Full set": (-0.5, 0.0, 3.2),
    "Seahorse Valley": (-0.745, 0.112, 0.4),
    "Antenna tip": (-1.768, 0.0, 0.2),
}

CMAPS = ["turbo", "nipy_spectral", "hsv", "plasma", "inferno"]


# ------------------------------------------------------- core math
def fractal_iters(cx, cy, width, max_iter, device):
    """Smooth iteration counts per pixel, vectorized over the whole grid."""
    hh = width * H / W / 2.0
    xs = torch.linspace(cx - width / 2.0, cx + width / 2.0, W, device=device)
    ys = torch.linspace(cy + hh, cy - hh, H, device=device)  # row 0 = top
    c = (xs.unsqueeze(0) + 1j * ys.unsqueeze(1)).to(torch.complex64)
    z = torch.zeros_like(c)
    iters = torch.zeros(c.shape, dtype=torch.float32, device=device)
    alive = torch.ones(c.shape, dtype=torch.bool, device=device)
    log_2 = math.log(2.0)
    for i in range(1, max_iter + 1):
        z = torch.where(alive, z * z + c, z)
        az = z.abs()
        escaped = az > 2.0
        newly = escaped & alive
        if bool(newly.any()):
            az_new = torch.clamp(az[newly], max=1e10)
            # continuous iteration count: n + 1 - log(log|z|)/log(2)
            smooth = float(i) + 1.0 - torch.log(torch.log(az_new)) / log_2
            iters[newly] = smooth
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
        lambda: fractal_iters(cx, cy, width, max_iter, "cuda"))
    return iters.cpu().numpy(), ms


def render_cpu(cx, cy, width, max_iter):
    iters, ms = _timed_ms(
        lambda: fractal_iters(cx, cy, width, max_iter, "cpu"))
    return iters.numpy(), ms


# ------------------------------------------------------- presentation
def colorize(iters, max_iter, cmap_name="turbo", cycles=2):
    # Log-normalize: most pixels escape in the first few dozen iterations,
    # so a linear map wastes ~80% of the colormap on iterations almost no
    # pixel reaches. Log scale spreads early escapes across the full map.
    # `cycles` wraps the palette for the classic psychedelic look.
    t = np.log1p(np.clip(iters, 0, max_iter)) / np.log1p(max_iter)
    t = (t * cycles) % 1.0
    rgba = plt.get_cmap(cmap_name)(t)[..., :3]
    rgba[iters >= max_iter] = (0.0, 0.0, 0.0)  # interior = black
    return (rgba * 255).astype(np.uint8)


def plot_timing(cpu_ms, gpu_ms):
    fig, ax = plt.subplots(figsize=(5, 3.0))
    labels, vals, colors = ["CPU"], [cpu_ms], ["#7f7f7f"]
    title = "Mandelbrot render time"
    if gpu_ms is not None:
        labels.append("GPU")
        vals.append(gpu_ms)
        colors.append("#2ca02c")
        title = f"Mandelbrot render time — GPU {cpu_ms / gpu_ms:.1f}× faster"
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
    if preset in VIEW_PRESETS:
        cx, cy, width = VIEW_PRESETS[preset]
        return (gr.update(value=cx), gr.update(value=cy),
                gr.update(value=float(np.log10(width))))
    return gr.update(), gr.update(), gr.update()


def on_render(preset, cx, cy, log_width, max_iter, cmap_name, cycles):
    width = 10.0 ** float(log_width)
    max_iter, cycles = int(max_iter), int(cycles)
    args = (float(cx), float(cy), width, max_iter)
    iters_cpu, cpu_ms = render_cpu(*args)
    iters_gpu, gpu_ms = render_gpu(*args)
    if iters_gpu is None:  # no CUDA here — show CPU render
        img = colorize(iters_cpu, max_iter, cmap_name, cycles)
        fig = plot_timing(cpu_ms, None)
        summary = (f"**Mandelbrot set** — GPU not available in this "
                   f"environment, showing CPU render ({cpu_ms:.0f} ms). On "
                   f"the Space this runs on ZeroGPU via `@spaces.GPU`.")
    else:
        img = colorize(iters_gpu, max_iter, cmap_name, cycles)
        fig = plot_timing(cpu_ms, gpu_ms)
        summary = (
            f"**Mandelbrot set** — **GPU: {gpu_ms:.0f} ms** · "
            f"CPU: {cpu_ms:.0f} ms · **speedup: {cpu_ms / gpu_ms:.1f}×** — "
            f"identical math, the only difference is how many pixels compute "
            f"at once. ({W}×{H} pixels, {max_iter} max iterations)"
        )
    return img, fig, summary


# ------------------------------------------------------- UI
with gr.Blocks(title="Mandelbrot GPU") as demo:
    gr.Markdown(
        "# 🌀 The Mandelbrot Set, in Parallel\n"
        "*Every pixel is its own little computation — thousands at once. "
        "That is the entire GPU story, and it is why deep learning runs on them.*"
    )
    with gr.Row():
        preset = gr.Dropdown(list(VIEW_PRESETS) + ["Custom"], value="Full set",
                             label="view")
        cmap = gr.Dropdown(CMAPS, value="turbo", label="colormap")
        cycles = gr.Slider(1, 5, step=1, value=2, label="color cycles")
    with gr.Row():
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
        "1. **Seahorse Valley, then keep halving the width** — zoom into the "
        "seahorse's tail and watch new seahorses appear, forever.\n"
        "2. **Color cycles up to 4** — the classic psychedelic look.\n"
        "3. **Push max iterations to 1000 and zoom deep** — the CPU visibly "
        "slows while the GPU barely notices. That gap is the whole game."
    )

    preset.change(on_preset, inputs=preset, outputs=[cx, cy, log_width])
    go.click(on_render,
             inputs=[preset, cx, cy, log_width, max_iter, cmap, cycles],
             outputs=[img, timing, summary])
