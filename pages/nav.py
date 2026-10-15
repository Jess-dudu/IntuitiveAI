import gradio as gr

PAGES = [
    ("GPU Demo", "/gpu-demo"),
]


def navbar(main_page_name="Home"):
    return gr.Navbar(value=PAGES, main_page_name=main_page_name)
