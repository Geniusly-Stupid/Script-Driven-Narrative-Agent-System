from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import Any, Callable

from pypdf import PdfReader

from app.llm_client import call_nvidia_llm

KNOWLEDGE_TYPES = {
    "setting",
    "truth",
    "background",
    "npc",
    "rule",
    "location",
    "item",
    "clue",
    "other",
}

PARSE_MODES = {"quality", "balanced", "speed"}
TRANSITION_SCENE_MARKERS = {"transition", "transitional", "bridge", "过渡", "transition_scene"}


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
    bundle = parse_script_bundle(
        pages=pages,
        pages_per_scene=pages_per_scene,
        llm_client=llm_client,
    )
    return bundle["scenes"]


def parse_script_bundle(
    pages: list[str],
    pages_per_scene: int = 10,
    llm_client: Callable[[str], str] | None = None,
    story_start_page: int | None = None,
    story_end_page: int | None = None,
) -> dict[str, Any]:
    if not pages:
        return {
            "scenes": [],
            "knowledge": [],
            "structure": {
                "front_knowledge": None,
                "story": None,
                "appendix_knowledge": None,
            },
            "warnings": [],
            "parse_mode": "balanced",
        }

    env_chunk = os.getenv("PARSER_PAGES_PER_CHUNK")
    if env_chunk:
        try:
            pages_per_scene = max(1, int(env_chunk))
        except ValueError:
            pass

    parse_mode = os.getenv("PARSER_PARSE_MODE", "balanced").strip().lower() or "balanced"
    if parse_mode not in PARSE_MODES:
        parse_mode = "balanced"

    llm = llm_client or call_nvidia_llm
    max_workers = int(os.getenv("PARSER_WORKERS", "1"))
    warnings: list[str] = []

    structure = _build_manual_structure(len(pages), story_start_page, story_end_page, warnings)
    if structure is None:
        structure = _identify_structure_ranges(pages, llm, warnings)

    raw_knowledge: list[dict[str, Any]] = []
    raw_story_scenes: list[dict[str, Any]] = []

    front = structure.get("front_knowledge")
    story = structure.get("story")
    appendix = structure.get("appendix_knowledge")

    if front:
        raw_knowledge.extend(
            _extract_knowledge_segment(
                pages,
                front["start_page"],
                front["end_page"],
                pages_per_scene,
                llm,
                max_workers,
                phase_label="front_knowledge",
            )
        )

    if story:
        raw_story_scenes.extend(
            _extract_story_segment(
                pages,
                story["start_page"],
                story["end_page"],
                pages_per_scene,
                llm,
                max_workers,
            )
        )

    if appendix:
        raw_knowledge.extend(
            _extract_knowledge_segment(
                pages,
                appendix["start_page"],
                appendix["end_page"],
                pages_per_scene,
                llm,
                max_workers,
                phase_label="appendix_knowledge",
            )
        )

    scenes: list[dict[str, Any]] = []
    knowledge_entries: list[dict[str, Any]] = []

    knowledge_counter = 0
    for raw_item in raw_knowledge:
        if not isinstance(raw_item, dict):
            continue

        knowledge_counter += 1
        knowledge_type = _normalize_knowledge_type(raw_item.get("knowledge_type"))
        title = _coerce_text(raw_item.get("title"), f"{knowledge_type}_{knowledge_counter}")
        content = _coerce_text(raw_item.get("content"), title)

        default_start = _coerce_int(raw_item.get("_chunk_start"), 1)
        default_end = _coerce_int(raw_item.get("_chunk_end"), default_start)
        source_start = _coerce_int(raw_item.get("source_page_start"), default_start)
        source_end = _coerce_int(raw_item.get("source_page_end"), default_end)
        if source_end < source_start:
            source_start, source_end = source_end, source_start

        metadata = raw_item.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        knowledge_entries.append(
            {
                "knowledge_id": f"knowledge_{knowledge_counter}",
                "knowledge_type": knowledge_type,
                "title": title,
                "content": content,
                "source_page_start": source_start,
                "source_page_end": source_end,
                "metadata": metadata,
            }
        )

    scene_counter = 0
    for raw_scene in raw_story_scenes:
        if not isinstance(raw_scene, dict):
            continue

        default_start = _coerce_int(raw_scene.get("_chunk_start"), story["start_page"] if story else 1)
        default_end = _coerce_int(raw_scene.get("_chunk_end"), default_start)
        source_start = _coerce_int(raw_scene.get("source_page_start"), default_start)
        source_end = _coerce_int(raw_scene.get("source_page_end"), default_end)
        if source_end < source_start:
            source_start, source_end = source_end, source_start

        scene_type = _normalize_scene_type(raw_scene)

        scene_counter += 1
        scene_id = f"scene_{scene_counter}"
        scene_goal = _coerce_text(raw_scene.get("scene_goal"), f"Advance story in {scene_id}")
        scene_description = _coerce_text(
            raw_scene.get("scene_description"),
            f"{scene_goal}. Present the immediate conflict, who is involved, and concrete choices for the player.",
        )
        scene_summary = _coerce_text(raw_scene.get("scene_summary"), "")

        plot_seed = raw_scene.get("plots", [])
        raw_plots = [p for p in plot_seed if isinstance(p, dict)] if isinstance(plot_seed, list) else []
        if not raw_plots:
            raw_plots = [
                {
                    "plot_goal": scene_goal,
                    "mandatory_events": [],
                    "npc": [],
                    "locations": [],
                    "source_page_start": source_start,
                    "source_page_end": source_end,
                }
            ]

        if _should_refine_single_plot(
            parse_mode=parse_mode,
            scene_type=scene_type,
            raw_plots=raw_plots,
            scene_description=scene_description,
            source_start=source_start,
            source_end=source_end,
        ):
            refined_scene_type, refined_plots = _refine_scene_plots(
                llm=llm,
                pages=pages,
                scene_goal=scene_goal,
                scene_description=scene_description,
                scene_summary=scene_summary,
                source_start=source_start,
                source_end=source_end,
            )
            if refined_scene_type:
                scene_type = refined_scene_type
            if refined_plots:
                raw_plots = refined_plots
            elif scene_type != "transition":
                warnings.append(f"scene {scene_id}: refinement returned empty plots; rule fallback applied")

        if scene_type != "transition" and len(raw_plots) < 2 and parse_mode != "speed":
            raw_plots = _rule_split_plots(
                scene_goal=scene_goal,
                scene_description=scene_description,
                source_start=source_start,
                source_end=source_end,
                base_plot=raw_plots[0] if raw_plots else None,
            )
            warnings.append(f"scene {scene_id}: single-plot fallback split applied")

        plots: list[dict[str, Any]] = []
        for p_idx, raw_plot in enumerate(raw_plots, start=1):
            plot_id = f"{scene_id}_plot_{p_idx}"
            p_default_start = _coerce_int(raw_plot.get("source_page_start"), source_start)
            p_default_end = _coerce_int(raw_plot.get("source_page_end"), source_end)
            if p_default_end < p_default_start:
                p_default_start, p_default_end = p_default_end, p_default_start

            plots.append(
                {
                    "plot_id": plot_id,
                    "plot_goal": _coerce_text(raw_plot.get("plot_goal"), f"Progress {scene_id} plot {p_idx}"),
                    "mandatory_events": _coerce_list(raw_plot.get("mandatory_events")),
                    "npc": _coerce_list(raw_plot.get("npc")),
                    "locations": _coerce_list(raw_plot.get("locations")),
                    "status": "pending",
                    "progress": 0.0,
                    "source_page_start": p_default_start,
                    "source_page_end": p_default_end,
                }
            )

        scenes.append(
            {
                "scene_id": scene_id,
                "scene_goal": scene_goal,
                "scene_description": scene_description,
                "scene_summary": scene_summary,
                "scene_type": scene_type,
                "source_page_start": source_start,
                "source_page_end": source_end,
                "plots": plots,
                "status": "pending",
            }
        )

    return {
        "scenes": scenes,
        "knowledge": knowledge_entries,
        "structure": structure,
        "warnings": warnings,
        "parse_mode": parse_mode,
    }





def _build_manual_structure(
    total_pages: int,
    story_start_page: int | None,
    story_end_page: int | None,
    warnings: list[str],
) -> dict[str, dict[str, int] | None] | None:
    start_raw = _coerce_int(story_start_page, 0)
    end_raw = _coerce_int(story_end_page, 0)

    has_start = start_raw > 0
    has_end = end_raw > 0
    if not has_start and not has_end:
        return None

    if has_start != has_end:
        warnings.append("structure: manual story range ignored because both story_start_page and story_end_page are required")
        return None

    start_page = max(1, min(total_pages, start_raw))
    end_page = max(1, min(total_pages, end_raw))
    if end_page < start_page:
        start_page, end_page = end_page, start_page
        warnings.append("structure: manual story range start/end swapped to keep ascending order")

    warnings.append(f"structure: manual story range applied ({start_page}-{end_page})")
    front_range = {"start_page": 1, "end_page": start_page - 1} if start_page > 1 else None
    appendix_range = {"start_page": end_page + 1, "end_page": total_pages} if end_page < total_pages else None
    return {
        "front_knowledge": front_range,
        "story": {"start_page": start_page, "end_page": end_page},
        "appendix_knowledge": appendix_range,
    }


def _identify_structure_ranges(
    pages: list[str],
    llm: Callable[[str], str],
    warnings: list[str],
) -> dict[str, dict[str, int] | None]:
    total_pages = len(pages)
    preview_chars = _read_int_env("PARSER_STRUCTURE_PREVIEW_CHARS", 100, 50)

    previews = []
    for idx, content in enumerate(pages, start=1):
        normalized = re.sub(r"\s+", " ", content or "").strip()
        previews.append(f"[PAGE {idx}] {normalized[:preview_chars]}")

    prompt = _build_structure_prompt(previews, total_pages)

    try:
        raw_text = _call_llm_with_retries(llm, prompt)
        data = _parse_json_response(raw_text, llm)
        if not isinstance(data, dict):
            raise ValueError("structure response root must be object")

        story_range = _coerce_range(data.get("story"), total_pages)
        if story_range is None:
            warnings.append("structure: invalid story range from LLM, fallback to full document story")
            story_range = {"start_page": 1, "end_page": total_pages}

        story_start = story_range["start_page"]
        story_end = story_range["end_page"]
        front_range = {"start_page": 1, "end_page": story_start - 1} if story_start > 1 else None
        appendix_range = {"start_page": story_end + 1, "end_page": total_pages} if story_end < total_pages else None

        return {
            "front_knowledge": front_range,
            "story": {"start_page": story_start, "end_page": story_end},
            "appendix_knowledge": appendix_range,
        }
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"structure: fallback to full-story mode ({exc})")
        return {
            "front_knowledge": None,
            "story": {"start_page": 1, "end_page": total_pages},
            "appendix_knowledge": None,
        }


def _build_structure_prompt(previews: list[str], total_pages: int) -> str:
    preview_text = "\n".join(previews)
    schema = (
        "{\n"
        '  "story": {"start_page": 1, "end_page": 1}\n'
        "}\n"
    )

    return (
        "TASK: IDENTIFY_SCRIPT_STRUCTURE\n"
        "You are a script-structure classifier.\n"
        "Given page previews, identify the SINGLE continuous story range.\n"
        "Pages before story are front knowledge, pages after story are appendix knowledge.\n"
        "Return strict JSON only.\n"
        "Rules:\n"
        "- story must be one continuous range.\n"
        "- If uncertain, include pages in story instead of excluding them.\n"
        "- Use page numbers from the preview labels.\n"
        f"- Total pages: {total_pages}.\n"
        f"Schema:\n{schema}\n"
        "Page previews:\n"
        f"{preview_text}\n"
    )


def _read_int_env(name: str, default: int, minimum: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        return default


def _extract_knowledge_segment(
    pages: list[str],
    start_page: int,
    end_page: int,
    pages_per_chunk: int,
    llm: Callable[[str], str],
    max_workers: int,
    phase_label: str,
) -> list[dict[str, Any]]:
    chunks = _build_chunks_for_range(pages, start_page, end_page, pages_per_chunk)

    def process_chunk(chunk_idx: int, chunk_start: int, chunk_end: int, chunk: list[str]) -> list[dict[str, Any]]:
        prompt = _build_knowledge_prompt(chunk, chunk_idx, chunk_start, chunk_end, phase_label)
        raw_text = _call_llm_with_retries(llm, prompt)
        data = _parse_json_response(raw_text, llm)
        if not isinstance(data, dict):
            return []
        items = _extract_knowledge_items(data)
        out = []
        for item in items:
            if isinstance(item, dict):
                normalized = item.copy()
                normalized.setdefault("_chunk_start", chunk_start)
                normalized.setdefault("_chunk_end", chunk_end)
                out.append(normalized)
        return out

    return _run_chunk_jobs(chunks, process_chunk, max_workers)


def _extract_story_segment(
    pages: list[str],
    start_page: int,
    end_page: int,
    pages_per_chunk: int,
    llm: Callable[[str], str],
    max_workers: int,
) -> list[dict[str, Any]]:
    chunks = _build_chunks_for_range(pages, start_page, end_page, pages_per_chunk)

    def process_chunk(chunk_idx: int, chunk_start: int, chunk_end: int, chunk: list[str]) -> list[dict[str, Any]]:
        prompt = _build_story_prompt(chunk, chunk_idx, chunk_start, chunk_end)
        raw_text = _call_llm_with_retries(llm, prompt)
        data = _parse_json_response(raw_text, llm)
        if not isinstance(data, dict):
            return []
        scenes = _extract_scene_items(data)
        out = []
        for scene in scenes:
            if isinstance(scene, dict):
                normalized = scene.copy()
                normalized.setdefault("_chunk_start", chunk_start)
                normalized.setdefault("_chunk_end", chunk_end)
                out.append(normalized)
        return out

    return _run_chunk_jobs(chunks, process_chunk, max_workers)


def _build_chunks_for_range(
    pages: list[str],
    start_page: int,
    end_page: int,
    pages_per_chunk: int,
) -> list[tuple[int, int, int, list[str]]]:
    chunks: list[tuple[int, int, int, list[str]]] = []
    if start_page > end_page:
        return chunks

    idx = 0
    for chunk_start in range(start_page, end_page + 1, max(1, pages_per_chunk)):
        chunk_end = min(end_page, chunk_start + max(1, pages_per_chunk) - 1)
        idx += 1
        chunk = pages[chunk_start - 1 : chunk_end]
        chunks.append((idx, chunk_start, chunk_end, chunk))
    return chunks


def _run_chunk_jobs(
    chunks: list[tuple[int, int, int, list[str]]],
    process_chunk: Callable[[int, int, int, list[str]], list[dict[str, Any]]],
    max_workers: int,
) -> list[dict[str, Any]]:
    if not chunks:
        return []

    results: list[list[dict[str, Any]]] = []

    if max_workers > 1 and len(chunks) > 1:
        indexed_results: dict[int, list[dict[str, Any]]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_chunk, chunk_idx, chunk_start, chunk_end, chunk): chunk_idx
                for chunk_idx, chunk_start, chunk_end, chunk in chunks
            }
            for future in as_completed(futures):
                idx = futures[future]
                indexed_results[idx] = future.result()
        for idx in sorted(indexed_results.keys()):
            results.append(indexed_results[idx])
    else:
        for chunk_idx, chunk_start, chunk_end, chunk in chunks:
            results.append(process_chunk(chunk_idx, chunk_start, chunk_end, chunk))

    flattened: list[dict[str, Any]] = []
    for items in results:
        flattened.extend(items)
    return flattened


def _build_knowledge_prompt(
    chunk: list[str],
    chunk_idx: int,
    chunk_start: int,
    chunk_end: int,
    phase_label: str,
) -> str:
    pages_text = "\n\n".join([f"[PAGE {chunk_start + i}]\n{p}" for i, p in enumerate(chunk)])
    schema = (
        "{\n"
        '  "knowledge": [\n'
        "    {\n"
        '      "knowledge_type": "setting|truth|background|npc|rule|location|item|clue|other",\n'
        '      "title": "...",\n'
        '      "content": "...",\n'
        '      "source_page_start": 1,\n'
        '      "source_page_end": 1\n'
        "    }\n"
        "  ]\n"
        "}\n"
    )

    return (
        "TASK: EXTRACT_KNOWLEDGE_ONLY\n"
        "You are an information extraction engine.\n"
        "Extract only NON-PLAYABLE knowledge from these pages.\n"
        "Do not output scenes or plots.\n"
        "Return strict JSON only.\n"
        f"Phase: {phase_label}\n"
        f"Chunk: {chunk_idx} (pages {chunk_start}-{chunk_end})\n"
        f"Schema:\n{schema}\n"
        "Pages:\n"
        f"{pages_text}\n"
    )


def _build_story_prompt(
    chunk: list[str],
    chunk_idx: int,
    chunk_start: int,
    chunk_end: int,
) -> str:
    pages_text = "\n\n".join([f"[PAGE {chunk_start + i}]\n{p}" for i, p in enumerate(chunk)])
    schema = (
        "{\n"
        '  "scenes": [\n'
        "    {\n"
        '      "scene_goal": "...",\n'
        '      "scene_description": "...",\n'
        '      "scene_summary": "",\n'
        '      "scene_type": "normal|transition",\n'
        '      "source_page_start": 1,\n'
        '      "source_page_end": 1,\n'
        '      "plots": [\n'
        "        {\n"
        '          "plot_goal": "...",\n'
        '          "mandatory_events": ["..."],\n'
        '          "npc": ["..."],\n'
        '          "locations": ["..."],\n'
        '          "source_page_start": 1,\n'
        '          "source_page_end": 1\n'
        "        },\n"
        "        {\n"
        '          "plot_goal": "...",\n'
        '          "mandatory_events": ["..."],\n'
        '          "npc": ["..."],\n'
        '          "locations": ["..."],\n'
        '          "source_page_start": 1,\n'
        '          "source_page_end": 1\n'
        "        }\n"
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}\n"
    )

    return (
        "TASK: EXTRACT_STORY_SCENES\n"
        "You are an information extraction engine for story sections only.\n"
        "Rules:\n"
        "- Output SCENES only. Do NOT output knowledge entries.\n"
        "- scene_goal should be short. scene_description should be a detailed paragraph (3-6 sentences).\n"
        "- scene_type = transition only for brief bridge scenes; transition scenes may have one plot.\n"
        "- Non-transition scenes should have at least 2 plots.\n"
        "- Plots are fine-grained action/progression units and can split within the same page.\n"
        "- Use only information from these pages.\n"
        "- Return strict JSON only.\n"
        f"Chunk: {chunk_idx} (pages {chunk_start}-{chunk_end})\n"
        f"Schema:\n{schema}\n"
        "Pages:\n"
        f"{pages_text}\n"
    )


def _refine_scene_plots(
    llm: Callable[[str], str],
    pages: list[str],
    scene_goal: str,
    scene_description: str,
    scene_summary: str,
    source_start: int,
    source_end: int,
) -> tuple[str | None, list[dict[str, Any]]]:
    excerpt = _render_pages_excerpt(pages, source_start, source_end)
    prompt = _build_plot_refine_prompt(
        scene_goal=scene_goal,
        scene_description=scene_description,
        scene_summary=scene_summary,
        source_start=source_start,
        source_end=source_end,
        scene_excerpt=excerpt,
    )

    try:
        raw_text = _call_llm_with_retries(llm, prompt)
        data = _parse_json_response(raw_text, llm)
        if not isinstance(data, dict):
            return None, []
        scene_type = _normalize_scene_type(data)

        plots = data.get("plots", [])
        if not isinstance(plots, list):
            plots = []
        normalized = [p for p in plots if isinstance(p, dict)]
        return scene_type, normalized
    except Exception:
        return None, []


def _build_plot_refine_prompt(
    scene_goal: str,
    scene_description: str,
    scene_summary: str,
    source_start: int,
    source_end: int,
    scene_excerpt: str,
) -> str:
    schema = (
        "{\n"
        '  "scene_type": "normal|transition",\n'
        '  "plots": [\n'
        "    {\n"
        '      "plot_goal": "...",\n'
        '      "mandatory_events": ["..."],\n'
        '      "npc": ["..."],\n'
        '      "locations": ["..."],\n'
        '      "source_page_start": 1,\n'
        '      "source_page_end": 1\n'
        "    }\n"
        "  ]\n"
        "}\n"
    )

    return (
        "TASK: REFINE_SCENE_PLOTS\n"
        "Split this scene into fine-grained plots.\n"
        "Rules:\n"
        "- If scene_type is normal, output at least 2 plots.\n"
        "- If scene is truly transitional, set scene_type=transition and output 1 plot.\n"
        "- Keep plot goals concrete and action-progress oriented.\n"
        "- Return strict JSON only.\n"
        f"Scene Goal: {scene_goal}\n"
        f"Scene Description: {scene_description}\n"
        f"Scene Summary: {scene_summary}\n"
        f"Scene Source Pages: {source_start}-{source_end}\n"
        f"Schema:\n{schema}\n"
        "Scene Excerpt:\n"
        f"{scene_excerpt}\n"
    )


def _render_pages_excerpt(pages: list[str], start_page: int, end_page: int, max_chars: int = 3500) -> str:
    if start_page < 1:
        start_page = 1
    if end_page > len(pages):
        end_page = len(pages)
    if start_page > end_page:
        return ""

    parts: list[str] = []
    total = 0
    for page_no in range(start_page, end_page + 1):
        text = pages[page_no - 1]
        snippet = f"[PAGE {page_no}]\n{text}\n"
        total += len(snippet)
        if total > max_chars:
            remain = max(0, max_chars - (total - len(snippet)))
            parts.append(snippet[:remain])
            break
        parts.append(snippet)
    return "\n".join(parts)


def _should_refine_single_plot(
    parse_mode: str,
    scene_type: str,
    raw_plots: list[dict[str, Any]],
    scene_description: str,
    source_start: int,
    source_end: int,
) -> bool:
    if scene_type == "transition":
        return False
    if len(raw_plots) >= 2:
        return False
    if parse_mode == "speed":
        return False
    if parse_mode == "quality":
        return True
    return source_end > source_start or len(scene_description or "") >= 180


def _rule_split_plots(
    scene_goal: str,
    scene_description: str,
    source_start: int,
    source_end: int,
    base_plot: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    midpoint = (source_start + source_end) // 2
    left_end = midpoint
    right_start = midpoint + 1 if midpoint < source_end else source_end

    if base_plot and isinstance(base_plot.get("mandatory_events"), list):
        events = [str(e).strip() for e in base_plot.get("mandatory_events", []) if str(e).strip()]
    else:
        events = []

    if len(events) >= 2:
        split_at = max(1, len(events) // 2)
        first_events = events[:split_at]
        second_events = events[split_at:]
    else:
        sentences = [s for s in re.split(r"[。！？!?；;\n]+", scene_description or "") if s.strip()]
        if len(sentences) >= 2:
            half = max(1, len(sentences) // 2)
            first_events = [f"{sentences[0].strip()}" if half == 1 else "；".join(s.strip() for s in sentences[:half])]
            second_events = ["；".join(s.strip() for s in sentences[half:])]
        else:
            first_events = ["进入并建立局势"]
            second_events = ["推进冲突并形成下一步行动"]

    first_goal = first_events[0] if first_events else f"{scene_goal}（前段推进）"
    second_goal = second_events[0] if second_events else f"{scene_goal}（后段推进）"

    base_npc = _coerce_list(base_plot.get("npc") if isinstance(base_plot, dict) else [])
    base_locations = _coerce_list(base_plot.get("locations") if isinstance(base_plot, dict) else [])

    return [
        {
            "plot_goal": _coerce_text(first_goal, f"{scene_goal}（前段）"),
            "mandatory_events": first_events,
            "npc": base_npc,
            "locations": base_locations,
            "source_page_start": source_start,
            "source_page_end": left_end,
        },
        {
            "plot_goal": _coerce_text(second_goal, f"{scene_goal}（后段）"),
            "mandatory_events": second_events,
            "npc": base_npc,
            "locations": base_locations,
            "source_page_start": right_start,
            "source_page_end": source_end,
        },
    ]


def _extract_knowledge_items(data: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(data.get("knowledge"), list):
        return [k for k in data.get("knowledge", []) if isinstance(k, dict)]

    if isinstance(data.get("units"), list):
        out = []
        for unit in data.get("units", []):
            if not isinstance(unit, dict):
                continue
            if str(unit.get("unit_type", "")).strip().lower() == "knowledge":
                out.append(unit)
        return out

    return []


def _extract_scene_items(data: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(data.get("scenes"), list):
        return [s for s in data.get("scenes", []) if isinstance(s, dict)]

    if isinstance(data.get("units"), list):
        out = []
        for unit in data.get("units", []):
            if not isinstance(unit, dict):
                continue
            if str(unit.get("unit_type", "")).strip().lower() == "scene":
                out.append(unit)
        return out

    return []


def _coerce_range(value: Any, total_pages: int) -> dict[str, int] | None:
    if not isinstance(value, dict):
        return None
    start_page = _coerce_int(value.get("start_page"), 0)
    end_page = _coerce_int(value.get("end_page"), 0)
    if start_page < 1 or end_page < 1:
        return None
    start_page = max(1, min(total_pages, start_page))
    end_page = max(1, min(total_pages, end_page))
    if end_page < start_page:
        start_page, end_page = end_page, start_page
    return {"start_page": start_page, "end_page": end_page}


def _call_llm_with_retries(llm: Callable[[str], str], prompt: str) -> str:
    return llm(prompt)


def _parse_json_response(text: str, llm: Callable[[str], str]) -> dict[str, Any]:
    for attempt in range(3):
        try:
            json_text = _extract_json_text(text)
            parsed = json.loads(json_text)
            if isinstance(parsed, dict):
                return parsed
            raise ValueError("JSON root must be object")
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


def _normalize_scene_type(value: Any) -> str:
    if isinstance(value, dict):
        raw = str(value.get("scene_type", "")).strip().lower()
        if not raw and value.get("is_transition") is True:
            raw = "transition"
    else:
        raw = str(value or "").strip().lower()
    return "transition" if raw in TRANSITION_SCENE_MARKERS else "normal"


def _normalize_knowledge_type(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in KNOWLEDGE_TYPES:
        return raw

    aliases = {
        "world_setting": "setting",
        "world": "setting",
        "settings": "setting",
        "lore": "background",
        "secret": "truth",
        "npc_profile": "npc",
        "character": "npc",
        "characters": "npc",
        "rules": "rule",
        "game_rule": "rule",
        "clues": "clue",
        "items": "item",
        "设定": "setting",
        "真相": "truth",
        "背景": "background",
        "人物": "npc",
        "角色": "npc",
        "规则": "rule",
        "地点": "location",
        "道具": "item",
        "线索": "clue",
    }
    if raw in aliases:
        return aliases[raw]

    fuzzy_aliases = [
        ("truth", "truth"),
        ("secret", "truth"),
        ("setting", "setting"),
        ("background", "background"),
        ("npc", "npc"),
        ("character", "npc"),
        ("rule", "rule"),
        ("location", "location"),
        ("place", "location"),
        ("item", "item"),
        ("clue", "clue"),
        ("设定", "setting"),
        ("真相", "truth"),
        ("背景", "background"),
        ("人物", "npc"),
        ("角色", "npc"),
        ("规则", "rule"),
        ("地点", "location"),
        ("道具", "item"),
        ("线索", "clue"),
    ]
    for needle, mapped in fuzzy_aliases:
        if needle in raw:
            return mapped
    return "other"


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _coerce_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    return []
