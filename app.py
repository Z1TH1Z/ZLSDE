"""Hugging Face Spaces entry point for ZLSDE."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from zlsde.ui import create_ui

demo = create_ui()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
