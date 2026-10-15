import gradio as gr

PAGES = [
    ("Home", "/"),
    ("GPU Demo", "/gpu-demo"),
]


def sidebar():
    gr.Navbar(visible=False)
    with gr.Sidebar(position="left"):
        gr.Markdown("## Navigation")
        for label, path in PAGES:
            gr.Markdown(f"[{label}]({path})")
