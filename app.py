import gradio as gr

from pages import gpu_demo
from pages.nav import NAVBAR_CSS, navbar

with gr.Blocks(title="IntuitiveAI", css=NAVBAR_CSS) as demo:
    navbar()
    gr.Markdown("# Home")
    gr.Markdown("Welcome. Use the sidebar on the left to visit other pages.")

with demo.route("GPU Demo", "/gpu-demo"):
    navbar(main_page_name="Home")
    gpu_demo.demo.render()

if __name__ == "__main__":
    demo.launch()
