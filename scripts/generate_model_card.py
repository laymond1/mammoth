#!/usr/bin/env python3
"""
Generate a Hugging Face model card README.md from local templates.

Examples:
  # Generate TAK v2 card into current directory
  uv run python scripts/generate_model_card.py \
    --model tak \
    --repo-id aimagelab-ta/TAK

  # Generate generic card and write to custom output
  uv run python scripts/generate_model_card.py \
    --model my_model \
    --repo-id user/my-repo \
    --output /tmp/README.md
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate model card README.md")
    parser.add_argument("--model", required=True, help="Model id, e.g. tak")
    parser.add_argument(
        "--repo-id", required=True, help="HF repo id, e.g. user-or-org/repo"
    )
    parser.add_argument(
        "--templates-dir",
        default="hub/model_cards",
        help="Directory containing markdown templates",
    )
    parser.add_argument(
        "--default-template",
        default="default.md",
        help="Fallback template filename",
    )
    parser.add_argument(
        "--output",
        default="README.md",
        help="Output markdown path",
    )
    parser.add_argument(
        "--license",
        default="mit",
        help="License used only by templates that expose {{LICENSE}}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output if it already exists",
    )
    return parser.parse_args()


def model_to_display_name(model_id: str) -> str:
    return model_id.replace("_", " ").replace("-", " ").strip().title()


def resolve_template(templates_dir: Path, model_id: str, default_template: str) -> Path:
    candidate = templates_dir / f"{model_id}.md"
    if candidate.exists():
        return candidate

    fallback = templates_dir / default_template
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"No template found for model `{model_id}` and fallback `{default_template}` missing in `{templates_dir}`"
    )


def render_template(raw: str, args: argparse.Namespace) -> str:
    replacements = {
        "{{MODEL_ID}}": args.model,
        "{{MODEL_NAME_DISPLAY}}": model_to_display_name(args.model),
        "{{REPO_ID}}": args.repo_id,
        "{{LICENSE}}": args.license,
    }

    rendered = raw
    for key, value in replacements.items():
        rendered = rendered.replace(key, value)
    return rendered


def main() -> None:
    args = parse_args()

    templates_dir = Path(args.templates_dir)
    if not templates_dir.exists() or not templates_dir.is_dir():
        raise ValueError(f"Templates directory not found: {templates_dir}")

    template_path = resolve_template(templates_dir, args.model, args.default_template)
    raw_template = template_path.read_text(encoding="utf-8")
    rendered = render_template(raw_template, args)

    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite to replace it."
        )

    if output_path.parent and not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(rendered, encoding="utf-8")
    print(f"Template: {template_path}")
    print(f"Generated: {output_path}")


if __name__ == "__main__":
    main()
