"""Run one upstream ARA analysis inside the upstream project's locked environment.

This module intentionally imports ARA only from an external checkout. Keeping the
GPL-licensed upstream tree out of this repository preserves a clean provenance and
licensing boundary while retaining the exact published six-query implementation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "minimal", "low", "medium", "high", "xhigh"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.checkout.resolve() / "src"
    if not (source / "ara_pipeline" / "online_llm_pipelines.py").is_file():
        raise FileNotFoundError(f"Not an ARA checkout: {args.checkout}")
    if not args.pdf.is_file():
        raise FileNotFoundError(args.pdf)
    sys.path.insert(0, str(source))

    from ara_pipeline import GPTPaperAnalyst

    reasoning_model = args.model.startswith(("gpt-5", "o1", "o3", "o4"))
    analyst = GPTPaperAnalyst(
        model=args.model,
        temperature=None if reasoning_model else args.temperature,
    )
    if args.reasoning_effort:
        original_parse = analyst.client.responses.parse

        def parse_with_reasoning(**kwargs):
            kwargs["reasoning"] = {"effort": args.reasoning_effort}
            return original_parse(**kwargs)

        analyst.client.responses.parse = parse_with_reasoning
    profile = analyst.analyze(args.pdf.resolve(), verbose=False)
    payload = {
        "bridge_version": "1.1.0" if args.reasoning_effort else "1.0.0",
        "request": {
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
        },
        "profile": profile.model_dump(mode="json"),
        "usage": analyst.usage_summary(),
        "responses": [
            {"query": name, "response": response.model_dump(mode="json")}
            for name, response in analyst.usage_log
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
