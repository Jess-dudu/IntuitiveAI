import gradio as gr
import torch

try:
    import spaces

    _gpu = spaces.GPU  # ZeroGPU: runs the decorated fn on GPU hardware
except Exception:  # no `spaces` package (local dev) -> plain call, no error
    def _gpu(fn):
        return fn


# zero = torch.Tensor([0]).cuda()
zero = torch.Tensor([0])
print(zero.device)  # <-- 'cpu' 🤔

@_gpu
def greet(n):
    # On the Space this runs inside @spaces.GPU, so zero.device is 'cuda:0';
    # without a GPU it runs plain on CPU and zero.device is 'cpu'.
    device = str(zero.device)
    print(zero.device)  # <-- 'cuda:0' 🤗 on the Space, 'cpu' locally
    return f"Hello {zero + n} Tensor (running on {device})"


demo = gr.Interface(fn=greet, inputs=gr.Number(), outputs=gr.Text())
