import gradio as gr

PAGES = [
    ("GPU Demo", "/gpu-demo"),
]

NAVBAR_CSS = """
.left-navbar {
    position: fixed !important;
    top: 0;
    left: 0;
    height: 100vh;
    width: 220px;
    flex-direction: column !important;
    align-items: flex-start !important;
    gap: 8px;
    padding: 20px 12px;
    overflow-y: auto;
    border-right: 1px solid var(--border-color-primary);
}

.gradio-container {
    margin-left: 220px !important;
}
"""


def navbar(main_page_name="Home"):
    return gr.Navbar(value=PAGES, main_page_name=main_page_name, elem_classes=["left-navbar"])
