#!/usr/bin/env python3
"""
Compiles docs/slides.html into docs/slides.pdf (16:9 widescreen landscape)
using headless Chromium.
"""

import os
import subprocess
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HTML_PATH = os.path.join(ROOT_DIR, "docs", "slides.html")
PDF_PATH = os.path.join(ROOT_DIR, "docs", "slides.pdf")

def main():
    if not os.path.exists(HTML_PATH):
        print(f"Error: {HTML_PATH} does not exist.")
        sys.exit(1)

    print(f"Compiling {HTML_PATH} to {PDF_PATH}...")
    cmd = [
        "chromium",
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=3000",
        "--no-pdf-header-footer",
        f"--print-to-pdf={PDF_PATH}",
        f"file://{HTML_PATH}"
    ]
    subprocess.run(cmd, check=True)
    print(f"Successfully generated {PDF_PATH}")

    # Inspect page count
    res = subprocess.run(["pdfinfo", PDF_PATH], capture_output=True, text=True, check=True)
    for line in res.stdout.splitlines():
        if "Pages:" in line or "Page size:" in line:
            print(f"  {line}")

if __name__ == "__main__":
    main()
