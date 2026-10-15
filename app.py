import gradio as gr

from pages import gpu_demo
from pages.nav import sidebar

with gr.Blocks(title="IntuitiveAI") as demo:
    sidebar()
    gr.Markdown("# Home")
    gr.Markdown("Welcome. Use the sidebar on the left to visit other pages.")

with demo.route("GPU Demo", "/gpu-demo"):
    sidebar()
    gpu_demo.demo.render()

if __name__ == "__main__":
    demo.launch()
