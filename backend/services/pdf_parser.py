from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass
class ScriptLine:
    page: int
    role: str
    text: str


def extract_pdf_text(pdf_path: Path, start_page: int = 1, end_page: Optional[int] = None) -> list[str]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Missing dependency: pypdf. Install it with `pip install pypdf`.") from exc

    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    if total_pages == 0:
        return []

    start_idx = max(1, start_page) - 1
    end_idx = total_pages if end_page is None else min(end_page, total_pages)
    if start_idx >= end_idx:
        return []

    pages: list[str] = []
    for idx in range(start_idx, end_idx):
        text = reader.pages[idx].extract_text() or ""
        pages.append(text)
    return pages


def parse_script_lines(page_text: str, page_num: int) -> list[ScriptLine]:
    """
    Parse lines like: 'KP: ...' or '玩家A：...'
    Unmatched lines are treated as narration.
    """
    results: list[ScriptLine] = []
    dialogue_pattern = re.compile(r"^\s*([A-Za-z0-9_\-\u4e00-\u9fff·]{1,24})\s*[:：]\s*(.+?)\s*$")

    for raw in page_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        match = dialogue_pattern.match(line)
        if match:
            role = match.group(1).strip()
            text = match.group(2).strip()
            results.append(ScriptLine(page=page_num, role=role, text=text))
        else:
            results.append(ScriptLine(page=page_num, role="旁白", text=line))
    return results


def parse_pdf_script(pdf_path: Path, start_page: int = 1, end_page: Optional[int] = None) -> list[ScriptLine]:
    pages = extract_pdf_text(pdf_path, start_page=start_page, end_page=end_page)
    all_lines: list[ScriptLine] = []
    for i, text in enumerate(pages, start=start_page):
        all_lines.extend(parse_script_lines(text, page_num=i))
    return all_lines


def filter_by_role(lines: Iterable[ScriptLine], role: Optional[str]) -> list[ScriptLine]:
    if not role:
        return list(lines)
    wanted = role.strip().lower()
    return [line for line in lines if line.role.lower() == wanted]


def to_text(lines: Iterable[ScriptLine], with_page: bool = False) -> str:
    output: list[str] = []
    for line in lines:
        prefix = f"[P{line.page}] " if with_page else ""
        if line.role == "旁白":
            output.append(f"{prefix}{line.text}")
        else:
            output.append(f"{prefix}{line.role}: {line.text}")
    return "\n".join(output)


class PDFParserService:
    @staticmethod
    def extract_text(file_path: str, start_page: int = 1, end_page: Optional[int] = None) -> str:
        """
        Backward-compatible wrapper used by current FastAPI workflow.
        Returns joined page text while internally relying on restored page-bounded extraction.
        """
        pages = extract_pdf_text(Path(file_path), start_page=start_page, end_page=end_page)
        return "\n".join(pages)
