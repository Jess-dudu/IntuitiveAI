"""
Fractals in parallel — Mandelbrot, Julia, Burning Ship, Multibrot on GPU.

Every pixel of these fractals is an independent computation: the perfect
GPU workload. This page renders them on GPU vs CPU and shows the timing
gap — the same gap that makes deep learning possible.

Drop at pages/mandelbrot.py, wire in app.py:

    from pages import mandelbrot
    ...
    with demo.route("Fractals GPU", "/fractals"):
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

KINDS = ["Mandelbrot", "Julia", "Burning Ship", "Multibrot"]

VIEW_PRESETS = {
    "Full set": (-0.5, 0.0, 3.2),
    "Seahorse Valley": (-0.745, 0.112, 0.4),
    "Antenna tip": (-1.768, 0.0, 0.2),
}

JULIA_PRESETS = {
    "Douady rabbit": (-0.123, 0.745),
    "Dendrite": (-0.7269, 0.1889),
    "Classic": (-0.8, 0.156),
    "Spiral": (0.285, 0.01),
}


# ------------------------------------------------------- core math
def fractal_iters(kind, cx, cy, width, max_iter, power, jcr, jci, device):
    """Iteration counts per pixel, vectorized over the whole grid."""
    hh = width * H / W / 2.0
    xs = torch.linspace(cx - width / 2.0, cx + width / 2.0, W, device=device)
    ys = torch.linspace(cy + hh, cy - hh, H, device=device)  # row 0 = top
    p = (xs.unsqueeze(0) + 1j * ys.unsqueeze(1)).to(torch.complex64)
    if kind == "Julia":
        z = p.clone()          # z0 = pixel, c fixed
        c = torch.full_like(p, complex(jcr, jci))
    else:
        z = torch.zeros_like(p)  # z0 = 0, c = pixel
        c = p
    iters = torch.zeros(p.shape, dtype=torch.float32, device=device)
    alive = torch.ones(p.shape, dtype=torch.bool, device=device)
    for i in range(1, max_iter + 1):
        if kind == "Burning Ship":
            za = torch.complex(z.real.abs(), z.imag.abs())
            z = torch.where(alive, za * za + c, z)
        elif kind == "Multibrot":
            z = torch.where(alive, torch.pow(z, power) + c, z)
        else:
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
def render_gpu(kind, cx, cy, width, max_iter, power, jcr, jci):
    if not torch.cuda.is_available():
        return None, None
    iters, ms = _timed_ms(
        lambda: fractal_iters(kind, cx, cy, width, max_iter, power,
                              jcr, jci, "cuda"))
    return iters.cpu().numpy(), ms


def render_cpu(kind, cx, cy, width, max_iter, power, jcr, jci):
    iters, ms = _timed_ms(
        lambda: fractal_iters(kind, cx, cy, width, max_iter, power,
                              jcr, jci, "cpu"))
    return iters.numpy(), ms


# ------------------------------------------------------- presentation
def colorize(iters, max_iter):
    t = np.clip(iters / max_iter, 0.0, 1.0)
    rgba = plt.get_cmap("turbo")(t)[..., :3]
    rgba[iters >= max_iter] = (0.0, 0.0, 0.0)  # interior = black
    return (rgba * 255).astype(np.uint8)


def plot_timing(kind, cpu_ms, gpu_ms):
    fig, ax = plt.subplots(figsize=(5, 3.0))
    labels, vals, colors = ["CPU"], [cpu_ms], ["#7f7f7f"]
    title = f"{kind} render time"
    if gpu_ms is not None:
        labels.append("GPU")
        vals.append(gpu_ms)
        colors.append("#2ca02c")
        title = f"{kind} render time — GPU {cpu_ms / gpu_ms:.1f}× faster"
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


def on_jpreset(jpreset):
    if jpreset in JULIA_PRESETS:
        cr, ci = JULIA_PRESETS[jpreset]
        return gr.update(value=cr), gr.update(value=ci)
    return gr.update(), gr.update()


def on_kind_change(kind):
    is_julia = kind == "Julia"
    is_multi = kind == "Multibrot"
    out = [gr.update(visible=is_multi),   # power
           gr.update(visible=is_julia),   # jpreset
           gr.update(visible=is_julia),   # jcr
           gr.update(visible=is_julia)]   # jci
    if is_julia:  # Julia sets are viewed centered on the origin
        out += [gr.update(value=0.0), gr.update(value=0.0),
                gr.update(value=float(np.log10(3.2)))]
    else:
        out += [gr.update(), gr.update(), gr.update()]
    return out


def on_render(kind, preset, cx, cy, log_width, max_iter, power, jcr, jci):
    width = 10.0 ** float(log_width)
    max_iter, power = int(max_iter), int(power)
    args = (kind, float(cx), float(cy), width, max_iter, power,
            float(jcr), float(jci))
    iters_cpu, cpu_ms = render_cpu(*args)
    iters_gpu, gpu_ms = render_gpu(*args)
    extra = ""
    if kind == "Julia":
        extra = f" (c = {jcr:.4f} + {jci:.4f}i)"
    elif kind == "Multibrot":
        extra = f" (z^({power}) + c)"
    if iters_gpu is None:  # no CUDA here — show CPU render
        img = colorize(iters_cpu, max_iter)
        fig = plot_timing(kind, cpu_ms, None)
        summary = (f"**{kind}**{extra} — GPU not available in this "
                   f"environment, showing CPU render ({cpu_ms:.0f} ms). On "
                   f"the Space this runs on ZeroGPU via `@spaces.GPU`.")
    else:
        img = colorize(iters_gpu, max_iter)
        fig = plot_timing(kind, cpu_ms, gpu_ms)
        summary = (
            f"**{kind}**{extra} — **GPU: {gpu_ms:.0f} ms** · "
            f"CPU: {cpu_ms:.0f} ms · **speedup: {cpu_ms / gpu_ms:.1f}×** — "
            f"identical math, the only difference is how many pixels compute "
            f"at once. ({W}×{H} pixels, {max_iter} max iterations)"
        )
    return img, fig, summary


# ------------------------------------------------------- UI
with gr.Blocks(title="Fractals GPU") as demo:
    gr.Markdown(
        "# 🌀 Fractals, in Parallel\n"
        "*Every pixel is its own little computation — thousands at once. "
        "That is the entire GPU story, and it is why deep learning runs on them.*"
    )
    with gr.Row():
        kind = gr.Dropdown(KINDS, value="Mandelbrot", label="fractal")
        preset = gr.Dropdown(list(VIEW_PRESETS) + ["Custom"], value="Full set",
                             label="view")
    with gr.Row():
        cx = gr.Number(value=-0.5, label="center x")
        cy = gr.Number(value=0.0, label="center y")
        log_width = gr.Slider(-3, 0.6, step=0.1, value=0.5,
                              label="view width = 10^x")
        max_iter = gr.Slider(50, 1000, step=50, value=256,
                             label="max iterations")
    with gr.Row():
        power = gr.Slider(2, 6, step=1, value=3, visible=False,
                          label="power d in z^d + c (Multibrot only)")
        jpreset = gr.Dropdown(list(JULIA_PRESETS) + ["Custom"],
                              value="Douady rabbit", visible=False,
                              label="Julia constant c")
        jcr = gr.Number(value=-0.123, visible=False, label="Re(c)")
        jci = gr.Number(value=0.745, visible=False, label="Im(c)")
    go = gr.Button("Render ⚡", variant="primary")
    with gr.Row():
        img = gr.Image(label="GPU render")
        timing = gr.Plot(label="timing")
    summary = gr.Markdown()
    gr.Markdown(
        "### 🔬 Things to try\n"
        "1. **Julia + Douady rabbit** — then try the Dendrite. Same loop as "
        "Mandelbrot, wildly different creatures.\n"
        "2. **Multibrot, power 3 → 4 → 5** — watch the symmetry multiply. "
        "Each power is a different universe.\n"
        "3. **Burning Ship, full view** — an armada where the Mandelbrot "
        "cardioid used to be.\n"
        "4. **Push max iterations to 1000 and zoom in** — the CPU visibly "
        "slows while the GPU barely notices. That gap is the whole game."
    )

    preset.change(on_preset, inputs=preset, outputs=[cx, cy, log_width])
    jpreset.change(on_jpreset, inputs=jpreset, outputs=[jcr, jci])
    kind.change(on_kind_change, inputs=kind,
                outputs=[power, jpreset, jcr, jci, cx, cy, log_width])
    go.click(on_render,
             inputs=[kind, preset, cx, cy, log_width, max_iter,
                     power, jcr, jci],
             outputs=[img, timing, summary])
