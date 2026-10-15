import gradio as gr

from pages import gpu_demo
from pages.nav import navbar

with gr.Blocks(title="IntuitiveAI") as demo:
    navbar()
    gr.Markdown("# Home")
    gr.Markdown("Welcome. Use the navbar above to visit other pages.")

with demo.route("GPU Demo", "/gpu-demo"):
    navbar(main_page_name="Home")
    gpu_demo.demo.render()

if __name__ == "__main__":
    demo.launch()
