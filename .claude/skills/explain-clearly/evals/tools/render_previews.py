"""Render each run's response.md and HTML pages to PNG so the review viewer shows them as the user would see them."""
import re
import subprocess
import sys
from pathlib import Path

from markdown_it import MarkdownIt
from PIL import Image

CHAT_CSS = """
body{margin:0;background:#fff;color:#1f2328;font:16px/1.65 system-ui,'Noto Sans CJK KR',sans-serif;word-break:keep-all}
main{max-width:780px;margin:0 auto;padding:28px 20px 40px}
table{border-collapse:collapse;margin:12px 0;font-size:.93rem}th,td{border:1px solid #d0d7de;padding:6px 10px;text-align:left}
th{background:#f6f8fa}code{background:#f0f2f4;padding:1px 5px;border-radius:4px;font-size:.88em}
pre{background:#f6f8fa;padding:12px;border-radius:6px;overflow-x:auto}pre code{background:none;padding:0}
img{max-width:100%}h1{font-size:1.5rem}h2{font-size:1.25rem}h3{font-size:1.05rem}blockquote{border-left:4px solid #d0d7de;margin:10px 0;padding:2px 14px;color:#57606a}
"""
# measure after images have loaded, otherwise the page height is underestimated and the screenshot is cut short
PROBE = "<script>window.addEventListener('load',()=>document.documentElement.setAttribute('data-h',Math.ceil(Math.max(document.documentElement.scrollHeight,document.body.scrollHeight))))</script>"
WIDTH, MAX_H = 900, 15000


def chrome(*args):
    return subprocess.run(["google-chrome", "--headless=new", "--no-sandbox", "--disable-gpu", "--hide-scrollbars", *args],
                          capture_output=True, text=True, timeout=120)


def shoot(html_path: Path, png_path: Path):
    probe = html_path.with_name(f".probe_{html_path.name}")
    probe.write_text(html_path.read_text(encoding="utf-8", errors="replace") + PROBE, encoding="utf-8")
    try:
        dom = chrome(f"--window-size={WIDTH},1000", "--virtual-time-budget=3000", "--dump-dom", probe.as_uri()).stdout
        match = re.search(r'data-h="(\d+)"', dom)
        height = min(int(match.group(1)) if match else 4000, MAX_H)
        chrome(f"--window-size={WIDTH},{max(height, 300)}", "--virtual-time-budget=3000", f"--screenshot={png_path}", html_path.as_uri())
    finally:
        probe.unlink(missing_ok=True)
    with Image.open(png_path) as image:
        return image.size, (int(match.group(1)) if match else None)


def main(root: Path):
    md = MarkdownIt("commonmark").enable("table")
    for outputs in sorted(root.glob("eval-*/*/run-*/outputs")):
        for old in outputs.glob("0?_rendered_*.png"):
            old.unlink()
        response = outputs / "response.md"
        if response.is_file():
            wrapper = outputs / ".response_render.html"
            wrapper.write_text(f"<!doctype html><meta charset='utf-8'><style>{CHAT_CSS}</style><main>{md.render(response.read_text(encoding='utf-8'))}</main>", encoding="utf-8")
            size, full = shoot(wrapper, outputs / "00_rendered_chat_reply.png")
            wrapper.unlink()
            print(outputs.relative_to(root), "chat reply", size, "full height", full)
        for index, page in enumerate(sorted(p for p in outputs.glob("*.html") if not p.name.startswith(".")), start=1):
            size, full = shoot(page, outputs / f"0{index}_rendered_{page.stem}.png")
            print(outputs.relative_to(root), page.name, size, "full height", full, "(TRUNCATED)" if full and full > MAX_H else "")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
