from __future__ import annotations

import re
from collections import defaultdict
from io import BytesIO

from pypdf import PdfReader


def read_pdf_pages(file_bytes: bytes, start_page: int = 1, end_page: int | None = None) -> list[str]:
    reader = PdfReader(BytesIO(file_bytes))
    total = len(reader.pages)
    if total == 0:
        return []
    s = max(1, start_page) - 1
    e = total if end_page is None else min(end_page, total)
    if s >= e:
        return []
    return [(reader.pages[i].extract_text() or '').strip() for i in range(s, e)]


def parse_script(pages: list[str], pages_per_scene: int = 3) -> list[dict]:
    if not pages:
        return []

    scenes: list[dict] = []
    scene_chunks = [pages[i:i + pages_per_scene] for i in range(0, len(pages), pages_per_scene)]

    for s_idx, chunk in enumerate(scene_chunks, start=1):
        combined = '\n'.join(chunk)
        lines = [ln.strip() for ln in combined.splitlines() if ln.strip()]
        scene_id = f'scene_{s_idx}'
        scene_goal = lines[0][:180] if lines else f'Advance story in {scene_id}'

        plot_slices = _split_plots(lines)
        plots = []
        for p_idx, p_lines in enumerate(plot_slices, start=1):
            plot_text = ' '.join(p_lines)
            npc, locations = _extract_entities(plot_text)
            timeline_markers = _extract_timeline(plot_text)
            mandatory_events = _extract_events(p_lines) + timeline_markers
            plots.append(
                {
                    'plot_id': f'{scene_id}_plot_{p_idx}',
                    'plot_goal': p_lines[0][:140] if p_lines else f'Progress {scene_id} plot {p_idx}',
                    'mandatory_events': mandatory_events[:6],
                    'npc': npc[:4] if npc else ['Guide'],
                    'locations': locations[:4] if locations else ['Unknown Site'],
                    'status': 'pending',
                    'progress': 0.0,
                }
            )

        scenes.append(
            {
                'scene_id': scene_id,
                'scene_goal': scene_goal,
                'plots': plots or [
                    {
                        'plot_id': f'{scene_id}_plot_1',
                        'plot_goal': f'Resolve central conflict of {scene_id}',
                        'mandatory_events': ['Key decision point'],
                        'npc': ['Guide'],
                        'locations': ['Unknown Site'],
                        'status': 'pending',
                        'progress': 0.0,
                    }
                ],
                'status': 'pending',
                'scene_summary': '',
            }
        )
    return scenes


def _split_plots(lines: list[str]) -> list[list[str]]:
    if not lines:
        return [[]]
    marker_idx = [i for i, ln in enumerate(lines) if re.search(r'\bplot\b|\bact\b|\bchapter\b', ln, re.I)]
    if marker_idx:
        chunks = []
        starts = marker_idx + [len(lines)]
        for i in range(len(marker_idx)):
            chunks.append(lines[starts[i]:starts[i + 1]])
        return [c for c in chunks if c]
    half = max(1, len(lines) // 2)
    return [lines[:half], lines[half:]] if len(lines) > 8 else [lines]


def _extract_entities(text: str) -> tuple[list[str], list[str]]:
    words = re.findall(r'[A-Z][a-zA-Z]{2,}', text)
    freq = defaultdict(int)
    for w in words:
        freq[w] += 1
    ranked = [w for w, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)]
    return ranked[:5], ranked[5:10]


def _extract_events(lines: list[str]) -> list[str]:
    events = []
    for ln in lines:
        if re.search(r'fight|discover|find|decide|escape|investigate|reveal', ln, re.I):
            events.append(ln[:140])
    return events[:5]


def _extract_timeline(text: str) -> list[str]:
    markers = re.findall(r'\b(?:day\s*\d+|\d{1,2}:\d{2}|morning|night|dawn|dusk|year\s*\d{3,4})\b', text, re.I)
    return [f'Timeline marker: {m}' for m in markers[:5]]
