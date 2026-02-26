from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import Any, Callable

from pypdf import PdfReader

from app.llm_client import call_nvidia_llm


def read_pdf_pages(file_bytes: bytes, start_page: int = 1, end_page: int | None = None) -> list[str]:
    reader = PdfReader(BytesIO(file_bytes))
    total = len(reader.pages)
    if total == 0:
        return []
    s = max(1, start_page) - 1
    e = total if end_page is None else min(end_page, total)
    if s >= e:
        return []
    return [(reader.pages[i].extract_text() or "").strip() for i in range(s, e)]


def parse_script(
    pages: list[str],
    pages_per_scene: int = 10,
    llm_client: Callable[[str], str] | None = None,
) -> list[dict]:
    if not pages:
        return []

    env_chunk = os.getenv("PARSER_PAGES_PER_CHUNK")
    if env_chunk:
        try:
            pages_per_scene = max(1, int(env_chunk))
        except ValueError:
            pass

    llm = llm_client or call_nvidia_llm
    chunks = [pages[i:i + pages_per_scene] for i in range(0, len(pages), pages_per_scene)]

    scenes: list[dict[str, Any]] = []
    scene_counter = 0

    def process_chunk(chunk_idx: int, chunk: list[str]) -> list[dict[str, Any]]:
        prompt = _build_extraction_prompt(chunk, chunk_idx)
        raw_text = _call_llm_with_retries(llm, prompt)
        data = _parse_json_response(raw_text, llm)
        raw_scenes = data.get("scenes", []) if isinstance(data, dict) else []
        if not raw_scenes:
            raise ValueError("LLM returned no scenes for a chunk")
        return raw_scenes

    max_workers = int(os.getenv("PARSER_WORKERS", "1"))
    chunk_results: list[list[dict[str, Any]]] = []

    if max_workers > 1 and len(chunks) > 1:
        indexed_results: dict[int, list[dict[str, Any]]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_chunk, idx, ch): idx for idx, ch in enumerate(chunks, start=1)}
            for future in as_completed(futures):
                idx = futures[future]
                indexed_results[idx] = future.result()
        for idx in sorted(indexed_results.keys()):
            chunk_results.append(indexed_results[idx])
    else:
        for idx, ch in enumerate(chunks, start=1):
            chunk_results.append(process_chunk(idx, ch))

    for raw_scenes in chunk_results:
        for raw_scene in raw_scenes:
            scene_counter += 1
            scene_id = f"scene_{scene_counter}"
            scene_goal = _coerce_text(raw_scene.get("scene_goal"), f"Advance story in {scene_id}")
            scene_summary = _coerce_text(raw_scene.get("scene_summary"), "")

            raw_plots = raw_scene.get("plots", [])
            if not raw_plots:
                raise ValueError(f"LLM returned scene without plots: {scene_id}")

            plots = []
            for p_idx, raw_plot in enumerate(raw_plots, start=1):
                plot_id = f"{scene_id}_plot_{p_idx}"
                plot_goal = _coerce_text(raw_plot.get("plot_goal"), f"Progress {scene_id} plot {p_idx}")
                mandatory_events = _coerce_list(raw_plot.get("mandatory_events"))
                npc = _coerce_list(raw_plot.get("npc"))
                locations = _coerce_list(raw_plot.get("locations"))

                plots.append(
                    {
                        "plot_id": plot_id,
                        "plot_goal": plot_goal,
                        "mandatory_events": mandatory_events,
                        "npc": npc,
                        "locations": locations,
                        "status": "pending",
                        "progress": 0.0,
                    }
                )

            scenes.append(
                {
                    "scene_id": scene_id,
                    "scene_goal": scene_goal,
                    "plots": plots,
                    "status": "pending",
                    "scene_summary": scene_summary,
                }
            )

    return scenes


def _build_extraction_prompt(chunk: list[str], chunk_idx: int) -> str:
    pages_text = "\n\n".join([f"[PAGE {i + 1}]\n{p}" for i, p in enumerate(chunk)])
    schema = (
        "{\n"
        '  "scenes": [\n'
        "    {\n"
        '      "scene_goal": "...",\n'
        '      "scene_summary": "",\n'
        '      "plots": [\n'
        "        {\n"
        '          "plot_goal": "...",\n'
        '          "mandatory_events": ["..."],\n'
        '          "npc": ["..."],\n'
        '          "locations": ["..."]\n'
        "        }\n"
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}\n"
    )

    return (
        "You are an information extraction engine. Extract scenes and plots from the script pages below.\n"
        "Output strictly valid JSON that matches this schema and includes only these keys.\n"
        "Rules:\n"
        "- Use only information from the text.\n"
        "- If a scene contains multiple beats or phases, split it into multiple plots when possible.\n"
        "- Lists must be arrays of strings (use [] if none).\n"
        "- Do not include extra keys or any non-JSON text.\n"
        f"Schema:\n{schema}\n"
        f"Chunk: {chunk_idx}\n"
        "Script Pages:\n"
        f"{pages_text}\n"
    )


def _call_llm_with_retries(llm: Callable[[str], str], prompt: str) -> str:
    return llm(prompt)


def _parse_json_response(text: str, llm: Callable[[str], str]) -> dict[str, Any]:
    for attempt in range(3):
        try:
            json_text = _extract_json_text(text)
            return json.loads(json_text)
        except Exception as exc:  # noqa: BLE001
            if attempt >= 2:
                raise ValueError(f"LLM returned invalid JSON after retries: {exc}") from exc
            text = llm(_build_json_repair_prompt(text, str(exc)))
    raise ValueError("LLM JSON repair failed unexpectedly")


def _extract_json_text(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fenced:
        return fenced.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1].strip()
    return text.strip()


def _build_json_repair_prompt(bad_text: str, error: str) -> str:
    return (
        "Fix the following content into strictly valid JSON that matches the schema.\n"
        "Return only JSON. Do not add extra keys or commentary.\n"
        f"Error: {error}\n"
        "Content:\n"
        f"{bad_text}\n"
    )


def _coerce_text(value: Any, default: str) -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _coerce_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    return []
