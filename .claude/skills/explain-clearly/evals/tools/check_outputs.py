"""Programmatic checks per run: reply size, files, external dependencies, rendered height, and how many decimal numbers the visible body shows and repeats."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

NUM = re.compile(r"\d+\.\d+")
SOURCES = {"eval-1": "docs/19-tocom-cl-cifar100-three-seed-results.md", "eval-3": "docs/22-tocom-task-and-inference-budget-analysis.md"}
REPO = Path(__file__).resolve().parents[5]  # .claude/skills/explain-clearly/evals/tools/ -> repository root
EXTERNAL = re.compile(r"""<script[^>]+src=["']https?://|<link[^>]+href=["']https?://|<img[^>]+src=["']https?://|@import\s+url\(["']?https?://|fetch\(""", re.I)


def visible_text(path: Path) -> str:
    """Text a reader sees without opening folded details or hovering."""
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".html":
        text = re.sub(r"<(script|style|title)\b.*?</\1>", " ", text, flags=re.S | re.I)
        text = re.sub(r"<details\b(?![^>]*\bopen\b).*?</details>", " ", text, flags=re.S | re.I)
        text = re.sub(r"<[^>]+>", " ", text)
    return text


def main(root: Path):
    report = {}
    for outputs in sorted(root.glob("eval-*/*/run-*/outputs")):
        key = str(outputs.parent.relative_to(root))
        files = sorted(p for p in outputs.iterdir() if p.is_file() and not p.name.startswith(".") and "_rendered_" not in p.name)
        response = outputs / "response.md"
        entry = {
            "files": [f"{p.name} ({p.stat().st_size:,} B)" for p in files],
            "response_lines": len(response.read_text(encoding="utf-8").splitlines()) if response.is_file() else None,
            "external_dependencies": {p.name: sorted(set(EXTERNAL.findall(p.read_text(encoding="utf-8", errors="replace")))) for p in files if p.suffix == ".html"},
            "rendered_height_px_at_900w": {},
            "visible_body": {},
        }
        for png in sorted(outputs.glob("0?_rendered_*.png")):
            with Image.open(png) as image:
                entry["rendered_height_px_at_900w"][png.name] = image.height
        for p in files:
            if p.suffix in (".md", ".html"):
                found = NUM.findall(visible_text(p))
                repeats = {value: count for value, count in Counter(found).most_common(8) if count > 3}
                entry["visible_body"][p.name] = {"chars": len(re.sub(r"\s+", "", visible_text(p))), "decimal_numbers": len(found), "unique": len(set(found)), "values_shown_more_than_3_times": repeats}
        source = next((v for k, v in SOURCES.items() if key.startswith(k)), None)
        if source:
            # The source docs live on the archive/tocom-deit-tiny branch, not in every checkout.
            source_path = REPO / source
            entry["source_doc_decimal_numbers"] = len(NUM.findall(source_path.read_text(encoding="utf-8"))) if source_path.is_file() else None
        report[key] = entry
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
