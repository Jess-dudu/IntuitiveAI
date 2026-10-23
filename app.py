import gradio as gr

from pages import gpu_demo
from pages import mandelbrot
from pages import yolo_detect

with gr.Blocks(title="IntuitiveAI") as demo:
    gr.Markdown("# Home")
    gr.Markdown("Welcome. Use the navbar above to visit other pages.")

with demo.route("GPU Demo", "/gpu-demo"):
    gpu_demo.demo.render()

with demo.route("Mandelbrot GPU", "/mandelbrot"):
    mandelbrot.demo.render()

with demo.route("YOLO Detection", "/yolo"):
    yolo_detect.demo.render()
    
if __name__ == "__main__":
    demo.launch()