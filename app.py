"""Hugging Face Spaces entry point for ZLSDE."""

from zlsde.ui import create_ui

demo = create_ui()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
