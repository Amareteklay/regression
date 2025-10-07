from pathlib import Path
import textwrap
import re

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

INPUT = Path("report.md")
OUTPUT = Path("report.pdf")

if not INPUT.exists():
    raise SystemExit("report.md not found")

image_pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")

lines = INPUT.read_text(encoding="utf-8").splitlines()
entries = []
for line in lines:
    stripped = line.strip("\ufeff")
    if not stripped:
        entries.append({"type": "text", "text": "", "style": "body"})
        continue
    image_match = image_pattern.match(stripped)
    if image_match:
        caption = image_match.group(1).strip()
        path = image_match.group(2).strip()
        entries.append({"type": "image", "path": path, "caption": caption})
        continue
    if stripped.startswith("#"):
        heading = stripped.lstrip("#").strip()
        if heading:
            entries.append({"type": "text", "text": heading, "style": "heading"})
            entries.append({"type": "text", "text": "", "style": "body"})
        continue
    if stripped.startswith("- "):
        content = stripped[2:].strip()
        wrapped = textwrap.fill(content, width=90, subsequent_indent="   ")
        for idx, seg in enumerate(wrapped.splitlines()):
            bullet = "* " if idx == 0 else "  "
            entries.append({"type": "text", "text": bullet + seg, "style": "body"})
        continue
    if stripped.startswith((">", "|")):
        entries.append({"type": "text", "text": stripped, "style": "code"})
        continue
    wrapped = textwrap.fill(stripped, width=90)
    for seg in wrapped.splitlines():
        entries.append({"type": "text", "text": seg, "style": "body"})

line_limit = 42
font_sizes = {"heading": 14, "body": 10, "code": 9}
line_spacing = {"heading": 0.045, "body": 0.027, "code": 0.030}
page_size = (8.27, 11.69)
text_margin_left = 0.06

output_dir = INPUT.parent

with PdfPages(OUTPUT) as pdf:
    state = {"buffer": [], "count": 0}

    def flush_text_page():
        if not state["buffer"]:
            return
        fig = plt.figure(figsize=page_size)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        y = 0.95
        for item in state["buffer"]:
            text = item["text"]
            style = item["style"]
            if text == "":
                y -= 0.02
                continue
            size = font_sizes.get(style, 10)
            weight = "bold" if style == "heading" else "normal"
            fig.text(text_margin_left, y, text, fontsize=size, fontweight=weight, family="DejaVu Sans")
            y -= line_spacing.get(style, 0.027)
            if y < 0.05:
                break
        pdf.savefig(fig)
        plt.close(fig)
        state["buffer"] = []
        state["count"] = 0

    for entry in entries:
        if entry["type"] == "text":
            state["buffer"].append(entry)
            if entry["text"]:
                state["count"] += 1
            if state["count"] >= line_limit:
                flush_text_page()
            continue

        flush_text_page()

        img_path = (output_dir / entry["path"]).resolve()
        fig = plt.figure(figsize=page_size)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")

        try:
            img = plt.imread(img_path)
            img_ax = fig.add_axes([0.08, 0.25, 0.84, 0.6])
            img_ax.imshow(img)
            img_ax.axis("off")
        except FileNotFoundError:
            fig.text(0.1, 0.6, f"Image not found: {entry['path']}", fontsize=12, color="red")
        fig.text(0.08, 0.18, entry["caption"], fontsize=10, fontweight="bold")
        pdf.savefig(fig)
        plt.close(fig)

    flush_text_page()

print(f"Wrote {OUTPUT}")
