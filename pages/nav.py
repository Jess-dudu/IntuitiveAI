import gradio as gr

# Extra navbar links that are NOT already registered via demo.route(...)
# e.g. ("About", "https://example.com")
EXTRA_LINKS = []


def navbar(main_page_name="Home"):
    return gr.Navbar(value=EXTRA_LINKS, main_page_name=main_page_name)
