#!/usr/bin/env python3
"""Write WikiText-103 training lines to a plain text file (for prepare_text.sh)."""
from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="Output file path")
    p.add_argument(
        "--max-lines",
        type=int,
        default=0,
        help="If >0, write at most this many non-empty lines",
    )
    args = p.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit(
            "Install with: pip install datasets\n" f"Original error: {e}"
        ) from e

    ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for row in ds:
            text = (row.get("text") or "").strip()
            if not text:
                continue
            f.write(text + "\n")
            n += 1
            if args.max_lines and n >= args.max_lines:
                break
    print(f"Wrote {n} lines to {args.out}")


if __name__ == "__main__":
    main()
