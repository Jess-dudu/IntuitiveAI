"""
Neural network playground - a TensorFlow Playground-style demo for IntuitiveAI.

Pick a 2D dataset, pick input features, pick a small network, then train it
and watch the decision boundary form. Click Train repeatedly to keep training
from the current weights; Reset starts over from a fresh network.

Drop this file at pages/nn_playground.py, then wire it in app.py:

    from pages import nn_playground
    ...
    with demo.route("Playground", "/playground"):
        nn_playground.demo.render()

Dependencies: gradio, torch, matplotlib, numpy. No model downloads.
Training runs on GPU when the Space provides it (ZeroGPU) and on CPU
otherwise. The device is chosen inside the training function, and the
model state is kept on CPU between clicks, so a session works on either.
"""

import gradio as gr
import matplotlib

matplotlib.use("Agg")  # headless-safe on Spaces
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

try:
    import spaces

    _gpu = spaces.GPU  # ZeroGPU: runs the decorated fn on GPU hardware
except Exception:  # no `spaces` package (local dev) -> plain call, no error
    def _gpu(fn):
        return fn


_FEATURES = ["x", "y", "x^2", "y^2", "x*y", "sin(x)", "sin(y)"]
_DATASETS = ["Circle", "XOR", "Spiral", "Gaussian"]
_ACTIVATIONS = ["tanh", "relu", "sigmoid"]
_ACT = {"tanh": torch.nn.Tanh, "relu": torch.nn.ReLU, "sigmoid": torch.nn.Sigmoid}
_COLORS = ["#1f77b4", "#ff8c00"]


# ------------------------------------------------------------ data
def _generate(dataset, noise, test_ratio):
    """Return (x_train, y_train, x_test, y_test), all in about [-1, 1]."""
    rng = np.random.default_rng(0)
    n = 400
    if dataset == "Circle":
        xy = rng.uniform(-1.0, 1.0, size=(n, 2))
        r = np.sqrt((xy ** 2).sum(axis=1))
        y = (r < 0.6).astype(int)
    elif dataset == "XOR":
        xy = rng.uniform(-1.0, 1.0, size=(n, 2))
        y = (xy[:, 0] * xy[:, 1] > 0).astype(int)
    elif dataset == "Spiral":
        m = n // 2
        t = rng.uniform(0.0, 3.0 * np.pi, size=m)
        r = t / (3.0 * np.pi)
        xs, ys, labels = [], [], []
        for k in (0, 1):
            xs.append(r * np.cos(t + k * np.pi))
            ys.append(r * np.sin(t + k * np.pi))
            labels.append(np.full(m, k))
        xy = np.stack([np.concatenate(xs), np.concatenate(ys)], axis=1)
        y = np.concatenate(labels)
    else:  # Gaussian
        m = n // 2
        xy = np.vstack([
            rng.normal(-0.6, 0.35, size=(m, 2)),
            rng.normal(0.6, 0.35, size=(m, 2)),
        ])
        y = np.array([0] * m + [1] * m)
    if noise > 0:
        xy = xy + rng.normal(0.0, noise, size=xy.shape)
    idx = rng.permutation(n)
    xy, y = xy[idx], y[idx]
    n_test = int(round(n * test_ratio))
    return xy[n_test:], y[n_test:], xy[:n_test], y[:n_test]


def _features_np(xy, features):
    """Expand raw (x, y) points into the selected feature columns."""
    x, y = xy[:, 0], xy[:, 1]
    cols = {
        "x": x,
        "y": y,
        "x^2": x ** 2,
        "y^2": y ** 2,
        "x*y": x * y,
        "sin(x)": np.sin(x),
        "sin(y)": np.sin(y),
    }
    picked = [f for f in _FEATURES if f in features]
    return np.stack([cols[f] for f in picked], axis=1), picked


# ------------------------------------------------------------ model
def _build_model(in_dim, hidden1, hidden2, activation):
    act = _ACT[activation]
    layers = []
    prev = in_dim
    for h in (hidden1, hidden2):
        if h and h > 0:
            layers.append(torch.nn.Linear(prev, h))
            layers.append(act())
            prev = h
    layers.append(torch.nn.Linear(prev, 1))
    return torch.nn.Sequential(*layers)


def _init_model(model, activation):
    """He (kaiming) init for ReLU layers; other activations keep the
    PyTorch default, which already works for them here.

    Default Linear init under-scales ReLU weights, which is part of why
    ReLU layers died on Circle with x^2 / sin features at lr 0.03:
    the second hidden layer went entirely inactive and never recovered.
    """
    if activation != "relu":
        return
    for layer in model:
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.kaiming_uniform_(layer.weight, a=0, mode="fan_in",
                                           nonlinearity="relu")
            torch.nn.init.zeros_(layer.bias)


def _to_cpu(obj):
    """Deep-copy tensors to CPU so saved state never shares live storage."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v) for v in obj)
    return obj


def _predict_prob(model, device, xy, features):
    feats, _ = _features_np(xy, features)
    with torch.no_grad():
        t = torch.tensor(feats, dtype=torch.float32, device=device)
        return torch.sigmoid(model(t)).cpu().numpy().reshape(-1)


# ------------------------------------------------------------ plots
def _message_fig(text):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.text(0.5, 0.5, text, ha="center", va="center")
    ax.set_axis_off()
    return fig


def _scatter(ax, xy, y, train):
    for c in (0, 1):
        pts = xy[y == c]
        if train:
            ax.scatter(pts[:, 0], pts[:, 1], c=_COLORS[c], s=25,
                       edgecolors="white", linewidths=0.5)
        else:
            ax.scatter(pts[:, 0], pts[:, 1], facecolors="none",
                       edgecolors=_COLORS[c], s=55, linewidths=1.2)


def _boundary_fig(model, device, x_train, y_train, x_test, y_test, features):
    fig, ax = plt.subplots(figsize=(5, 5))
    if model is not None:
        gx, gy = np.meshgrid(np.linspace(-1.3, 1.3, 120),
                             np.linspace(-1.3, 1.3, 120))
        grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
        p = _predict_prob(model, device, grid, features).reshape(gx.shape)
        ax.contourf(gx, gy, p, levels=[0, 0.5, 1],
                    colors=[_COLORS[0], _COLORS[1]], alpha=0.25)
    _scatter(ax, x_train, y_train, train=True)
    _scatter(ax, x_test, y_test, train=False)
    legend = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=_COLORS[0], markeredgecolor="white",
               markersize=8, label="Train: class 0"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=_COLORS[1], markeredgecolor="white",
               markersize=8, label="Train: class 1"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor=_COLORS[0], markersize=9,
               label="Test: class 0"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor=_COLORS[1], markersize=9,
               label="Test: class 1"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=8)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_title("Decision boundary (background = model prediction)")
    return fig


def _loss_fig(history):
    fig, ax = plt.subplots(figsize=(5, 5))
    if history:
        ep = [h[0] for h in history]
        ls = [h[1] for h in history]
        ax.plot(ep, ls)
    ax.set_xlabel("Epoch (recorded every 10)")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss")
    return fig


# ------------------------------------------------------------ handlers
@_gpu
def _train(dataset, noise, test_ratio, features, hidden1, hidden2,
           activation, lr, epochs, state):
    features = [f for f in _FEATURES if f in features]
    if not features:
        return (_message_fig("Select at least one feature."),
                _message_fig("No training yet."),
                "Select at least one feature.", state)
    # Device is chosen HERE, inside the decorated call (see gpu_demo.py).
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_train, y_train, x_test, y_test = _generate(dataset, noise, test_ratio)
    f_train, _ = _features_np(x_train, features)
    fingerprint = (dataset, float(noise), float(test_ratio), tuple(features),
                   int(hidden1), int(hidden2), activation)
    state = state or {}
    model = _build_model(f_train.shape[1], int(hidden1), int(hidden2),
                         activation)
    model.to(device)
    if state.get("fingerprint") == fingerprint and state.get("model"):
        model.load_state_dict(state["model"])
        opt = torch.optim.Adam(model.parameters(), lr=float(lr))
        opt.load_state_dict(state["opt"])
        history = list(state["history"])
        epoch_count = int(state["epochs"])
    else:
        torch.manual_seed(42)
        model = _build_model(f_train.shape[1], int(hidden1), int(hidden2),
                             activation)
        model.to(device)
        _init_model(model, activation)
        opt = torch.optim.Adam(model.parameters(), lr=float(lr))
        history = []
        epoch_count = 0
    for g in opt.param_groups:
        g["lr"] = float(lr)
    xt = torch.tensor(f_train, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(1)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    epochs = int(epochs)
    model.train()
    for i in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(xt), yt)
        loss.backward()
        opt.step()
        if (i + 1) % 10 == 0 or i == epochs - 1:
            history.append([epoch_count + i + 1, float(loss.item())])
    epoch_count += epochs
    model.eval()
    p_train = _predict_prob(model, device, x_train, features)
    p_test = _predict_prob(model, device, x_test, features)
    acc_train = float(((p_train > 0.5).astype(int) == y_train).mean())
    acc_test = float(((p_test > 0.5).astype(int) == y_test).mean())
    new_state = {
        "fingerprint": fingerprint,
        "model": _to_cpu(model.state_dict()),
        "opt": _to_cpu(opt.state_dict()),
        "history": history,
        "epochs": epoch_count,
    }
    boundary = _boundary_fig(model, device, x_train, y_train,
                             x_test, y_test, features)
    status = (f"Device: {device} | Epochs: {epoch_count} | "
              f"Train acc: {acc_train:.0%} | Test acc: {acc_test:.0%} | "
              f"Loss: {history[-1][1]:.4f}")
    return boundary, _loss_fig(history), status, new_state


def _reset(dataset, noise, test_ratio):
    x_train, y_train, x_test, y_test = _generate(dataset, noise, test_ratio)
    fig = _boundary_fig(None, "cpu", x_train, y_train, x_test, y_test, ["x"])
    return fig, _loss_fig([]), "Reset. Click Train to start.", {}


# ------------------------------------------------------------ UI
with gr.Blocks(title="Neural Network Playground") as demo:
    gr.Markdown(
        "# Neural Network Playground\n"
        "Pick a dataset, pick features, pick a small network, then train it "
        "and watch the decision boundary form. Click Train repeatedly to keep "
        "training from the current weights; Reset starts over. "
        "In the boundary plot, the background shading is the model's "
        "prediction, filled dots are training points, hollow circles are "
        "test points, and color is the class."
    )
    with gr.Row():
        dataset = gr.Radio(_DATASETS, value="Spiral", label="Dataset")
        activation = gr.Radio(_ACTIVATIONS, value="tanh", label="Activation")
    features = gr.CheckboxGroup(_FEATURES, value=["x", "y"],
                                label="Input features")
    with gr.Row():
        noise = gr.Slider(0, 0.5, value=0.1, step=0.05, label="Noise")
        test_ratio = gr.Slider(0.1, 0.5, value=0.3, step=0.05,
                               label="Test fraction")
    with gr.Row():
        hidden1 = gr.Slider(0, 6, value=4, step=1,
                            label="Hidden layer 1 neurons (0 = skip)")
        hidden2 = gr.Slider(0, 6, value=3, step=1,
                            label="Hidden layer 2 neurons (0 = skip)")
    with gr.Row():
        lr = gr.Slider(0.001, 0.3, value=0.03, step=0.001,
                       label="Learning rate")
        epochs = gr.Slider(10, 500, value=100, step=10,
                           label="Epochs per Train click")
    with gr.Row():
        train_btn = gr.Button("Train", variant="primary")
        reset_btn = gr.Button("Reset")
    with gr.Row():
        boundary = gr.Plot(label="Decision boundary")
        loss_plot = gr.Plot(label="Loss")
    status = gr.Textbox(label="Status", value="Click Train to start.")
    state = gr.State({})
    train_btn.click(
        _train,
        inputs=[dataset, noise, test_ratio, features, hidden1, hidden2,
                activation, lr, epochs, state],
        outputs=[boundary, loss_plot, status, state],
    )
    reset_btn.click(
        _reset,
        inputs=[dataset, noise, test_ratio],
        outputs=[boundary, loss_plot, status, state],
    )
