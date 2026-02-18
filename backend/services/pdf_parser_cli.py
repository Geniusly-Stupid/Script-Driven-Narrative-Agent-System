#!/usr/bin/env python3
"""Read a TRPG script PDF and output clean dialogue text."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from backend.services.pdf_parser import filter_by_role, parse_pdf_script, to_text


def main() -> int:
    parser = argparse.ArgumentParser(description="读取跑团 PDF 剧本并输出结构化台词")
    parser.add_argument("pdf", type=Path, help="PDF 文件路径")
    parser.add_argument("-o", "--output", type=Path, help="输出文件路径（.txt 或 .json）")
    parser.add_argument("--start-page", type=int, default=1, help="起始页（从 1 开始）")
    parser.add_argument("--end-page", type=int, help="结束页（包含该页）")
    parser.add_argument("--role", help="只输出指定角色，例如 KP 或 玩家A")
    parser.add_argument("--show-page", action="store_true", help="输出时附带页码")
    args = parser.parse_args()

    if not args.pdf.exists():
        print(f"文件不存在: {args.pdf}", file=sys.stderr)
        return 1

    try:
        lines = parse_pdf_script(args.pdf, start_page=args.start_page, end_page=args.end_page)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"读取 PDF 失败: {exc}", file=sys.stderr)
        return 1

    lines = filter_by_role(lines, args.role)

    if args.output:
        suffix = args.output.suffix.lower()
        if suffix == ".json":
            args.output.write_text(
                json.dumps([asdict(line) for line in lines], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        else:
            args.output.write_text(to_text(lines, with_page=args.show_page), encoding="utf-8")
        print(f"已写入: {args.output}")
    else:
        print(to_text(lines, with_page=args.show_page))

    print(f"\n共解析 {len(lines)} 行。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
