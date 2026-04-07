from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Callable

from pypdf import PdfReader

from app.llm_client import call_nvidia_llm
from app.navigation import default_navigation, ensure_scene_navigation_defaults, make_target

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
MARKDOWN_EXTENSIONS = {".md", ".markdown"}
DECORATIVE_HTML_RE = re.compile(r"^</?div\b[^>]*>$|^<img\b[^>]*>$|^<div\b[^>]*><img\b.*</div>$", re.I)
MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
MARKDOWN_HINT_HEADINGS = ("keeper note", "keeper information", "two investigators", "next steps")
MARKDOWN_STANDALONE_SCENE_HEADINGS = {"start", "next steps", "conclusion"}
MARKDOWN_AUXILIARY_PLOT_MARKERS = (
    "two investigators",
    "keeper note",
    "keeper notes",
    "keeper information",
    "concerning ",
    "appendix",
)
SCENE_TRANSITION_TITLE_MARKERS = {"transition", "conclusion", "epilogue", "aftermath"}
PLOT_OPTION_TITLE_PREFIXES = ("about ", "asking about ", "if ", "exploring ", "ignoring ", "meeting ", "the investigator ")


def _truncate_text(text: str, limit: int = 500) -> str:
    normalized = (text or "").strip()
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 3].rstrip()}..."


@dataclass(frozen=True)
class SourceSegment:
    text: str
    source_start: int
    source_end: int
    heading_path: tuple[str, ...] = ()
    heading_level: int = 0
    segment_kind: str = "body"


@dataclass(frozen=True)
class SourceDocument:
    source_type: str
    unit_label: str
    segments: tuple[SourceSegment, ...]
    display_start: int
    display_end: int
    outline: tuple[str, ...] = ()
    source_file_name: str = ""
    raw_units: tuple[str, ...] = ()


@dataclass(frozen=True)
class MarkdownHeading:
    line_no: int
    level: int
    title: str
    path: tuple[str, ...]


@dataclass(frozen=True)
class MarkdownSceneCandidate:
    start: int
    end: int
    title: str
    path: tuple[str, ...]
    heading_level: int
    reason: str


@dataclass(frozen=True)
class MarkdownPlotBlock:
    start: int
    end: int
    title: str
    path: tuple[str, ...]
    kind: str
    heading_level: int
    raw_text: str


def detect_source_type(file_name: str, mime_type: str | None = None) -> str:
    suffix = ""
    if file_name:
        parts = file_name.lower().rsplit(".", 1)
        if len(parts) == 2:
            suffix = f".{parts[1]}"

    normalized_mime = (mime_type or "").strip().lower()
    if suffix == ".pdf" or normalized_mime == "application/pdf":
        return "pdf"
    if suffix in MARKDOWN_EXTENSIONS or "markdown" in normalized_mime:
        return "markdown"
    raise ValueError(f"Unsupported file type for parsing: {file_name or mime_type or 'unknown'}")


def read_pdf_pages(file_bytes: bytes, start_page: int = 1, end_page: int | None = None) -> list[str]:
    return [segment.text for segment in read_pdf_document(file_bytes, start_page, end_page).segments]


def read_pdf_document(
    file_bytes: bytes,
    start_page: int = 1,
    end_page: int | None = None,
    file_name: str = "",
) -> SourceDocument:
    reader = PdfReader(BytesIO(file_bytes))
    total = len(reader.pages)
    if total == 0:
        return SourceDocument("pdf", "page", (), max(1, start_page), max(1, start_page), (), file_name, ())

    start = max(1, start_page)
    end = total if end_page is None else min(end_page, total)
    if start > end:
        return SourceDocument("pdf", "page", (), start, end, (), file_name, ())

    raw_units = tuple((reader.pages[page_no - 1].extract_text() or "").strip() for page_no in range(start, end + 1))
    segments = tuple(
        SourceSegment(
            text=raw_units[idx],
            source_start=page_no,
            source_end=page_no,
        )
        for idx, page_no in enumerate(range(start, end + 1))
    )
    return SourceDocument("pdf", "page", segments, start, end, (), file_name, raw_units)


def read_markdown_document(
    file_bytes: bytes,
    start_unit: int = 1,
    end_unit: int | None = None,
    file_name: str = "",
) -> SourceDocument:
    text = _decode_text_bytes(file_bytes).replace("\r\n", "\n").replace("\r", "\n")
    raw_lines = text.split("\n")
    total_lines = len(raw_lines)
    if total_lines == 0:
        return SourceDocument("markdown", "line", (), max(1, start_unit), max(1, start_unit), (), file_name, ())

    start_line = max(1, start_unit)
    end_line = total_lines if end_unit is None else min(end_unit, total_lines)
    if start_line > end_line:
        return SourceDocument("markdown", "line", (), start_line, end_line, (), file_name, ())

    cropped_lines = [(line_no, raw_lines[line_no - 1]) for line_no in range(start_line, end_line + 1)]
    raw_units = tuple(raw_lines[line_no - 1] for line_no in range(start_line, end_line + 1))
    segments: list[SourceSegment] = []
    outline: list[str] = []
    heading_stack: list[tuple[int, str]] = []
    block_lines: list[str] = []
    block_start: int | None = None
    block_end: int | None = None

    def current_heading_path() -> tuple[str, ...]:
        return tuple(title for _, title in heading_stack)

    def flush_block() -> None:
        nonlocal block_lines, block_start, block_end
        if block_lines and block_start is not None and block_end is not None:
            block_text = "\n".join(block_lines).strip()
            if block_text:
                heading_path = current_heading_path()
                heading_level = heading_stack[-1][0] if heading_stack else 0
                segments.append(
                    SourceSegment(
                        text=block_text,
                        source_start=block_start,
                        source_end=block_end,
                        heading_path=heading_path,
                        heading_level=heading_level,
                        segment_kind="body",
                    )
                )
        block_lines = []
        block_start = None
        block_end = None

    for line_no, raw_line in cropped_lines:
        line = raw_line.rstrip()
        stripped = line.strip()

        if _is_decorative_markdown_line(stripped):
            continue

        heading_match = MARKDOWN_HEADING_RE.match(stripped)
        if heading_match:
            flush_block()
            level = len(heading_match.group(1))
            title = _normalize_markdown_heading(heading_match.group(2))
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            heading_path = current_heading_path()
            segments.append(
                SourceSegment(
                    text=f"{'#' * level} {title}",
                    source_start=line_no,
                    source_end=line_no,
                    heading_path=heading_path,
                    heading_level=level,
                    segment_kind="heading",
                )
            )
            outline_text = " > ".join(heading_path)
            if outline_text and outline_text not in outline:
                outline.append(outline_text)
            continue

        if not stripped:
            flush_block()
            continue

        if block_start is None:
            block_start = line_no
        block_end = line_no
        block_lines.append(line)

    flush_block()

    return SourceDocument(
        "markdown",
        "line",
        tuple(segments),
        start_line,
        end_line,
        tuple(outline),
        file_name,
        raw_units,
    )


def read_uploaded_document(
    file_name: str,
    file_bytes: bytes,
    start_unit: int = 1,
    end_unit: int | None = None,
    mime_type: str | None = None,
) -> SourceDocument:
    source_type = detect_source_type(file_name, mime_type)
    if source_type == "pdf":
        return read_pdf_document(file_bytes, start_unit, end_unit, file_name=file_name)
    if source_type == "markdown":
        return read_markdown_document(file_bytes, start_unit, end_unit, file_name=file_name)
    raise ValueError(f"Unsupported file type for parsing: {file_name}")


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
    pages: list[str] | None = None,
    pages_per_scene: int = 10,
    llm_client: Callable[[str], str] | None = None,
    story_start_page: int | None = None,
    story_end_page: int | None = None,
    source_document: SourceDocument | None = None,
) -> dict[str, Any]:
    document = source_document or _source_document_from_pages(pages or [])
    if not document.segments:
        return _empty_parse_bundle(document)

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

    structure = _build_manual_structure(document, story_start_page, story_end_page, warnings)
    if structure is None:
        structure = _identify_structure_ranges(document, llm, warnings)

    raw_knowledge: list[dict[str, Any]] = []

    front = structure.get("front_knowledge")
    story = structure.get("story")
    appendix = structure.get("appendix_knowledge")

    if front:
        raw_knowledge.extend(
            _extract_knowledge_segment(
                document,
                front["start_page"],
                front["end_page"],
                pages_per_scene,
                llm,
                max_workers,
                phase_label="front_knowledge",
            )
        )

    if appendix:
        raw_knowledge.extend(
            _extract_knowledge_segment(
                document,
                appendix["start_page"],
                appendix["end_page"],
                pages_per_scene,
                llm,
                max_workers,
                phase_label="appendix_knowledge",
            )
        )

    knowledge_entries = _normalize_knowledge_entries(raw_knowledge)

    scenes: list[dict[str, Any]] = []
    if story:
        if document.source_type == "markdown":
            scenes = _extract_markdown_story_scenes(
                document=document,
                story_start=story["start_page"],
                story_end=story["end_page"],
                llm=llm,
                warnings=warnings,
            )
        else:
            raw_story_scenes = _extract_story_segment(
                document,
                story["start_page"],
                story["end_page"],
                pages_per_scene,
                llm,
                max_workers,
            )
            scenes = _normalize_story_scenes(
                raw_story_scenes=raw_story_scenes,
                story=story,
                parse_mode=parse_mode,
                document=document,
                llm=llm,
                warnings=warnings,
            )

    scenes = _attach_navigation_metadata(scenes, document, llm, warnings)

    return {
        "scenes": scenes,
        "knowledge": knowledge_entries,
        "structure": structure,
        "warnings": warnings,
        "parse_mode": parse_mode,
        "source_metadata": _build_source_metadata(document),
    }





def _source_document_from_pages(pages: list[str]) -> SourceDocument:
    raw_units = tuple((page or "").strip() for page in pages)
    segments = tuple(
        SourceSegment(text=page_text, source_start=idx, source_end=idx)
        for idx, page_text in enumerate(raw_units, start=1)
    )
    end = len(segments) if segments else 1
    return SourceDocument("pdf", "page", segments, 1, end, (), "", raw_units)


def _empty_parse_bundle(document: SourceDocument | None = None) -> dict[str, Any]:
    doc = document or SourceDocument("pdf", "page", (), 1, 1)
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
        "source_metadata": _build_source_metadata(doc),
    }


def _build_source_metadata(document: SourceDocument) -> dict[str, Any]:
    return {
        "source_file_name": document.source_file_name,
        "source_type": document.source_type,
        "source_unit_label": document.unit_label,
        "display_start": document.display_start,
        "display_end": document.display_end,
        "heading_outline_preview": list(document.outline[:20]),
    }


def _normalize_knowledge_entries(raw_knowledge: list[dict[str, Any]]) -> list[dict[str, Any]]:
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

    return knowledge_entries


def _normalize_story_scenes(
    raw_story_scenes: list[dict[str, Any]],
    story: dict[str, int],
    parse_mode: str,
    document: SourceDocument,
    llm: Callable[[str], str],
    warnings: list[str],
) -> list[dict[str, Any]]:
    scenes: list[dict[str, Any]] = []
    scene_counter = 0
    for raw_scene in raw_story_scenes:
        if not isinstance(raw_scene, dict):
            continue

        default_start = _coerce_int(raw_scene.get("_chunk_start"), story["start_page"])
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
                document=document,
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
                    "raw_text": _slice_document_text(document, p_default_start, p_default_end),
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

    return scenes


def _slice_document_text(document: SourceDocument, start_unit: int, end_unit: int) -> str:
    if start_unit > end_unit:
        return ""
    start_unit = max(document.display_start, start_unit)
    end_unit = min(document.display_end, end_unit)
    if end_unit < start_unit:
        return ""

    if document.raw_units:
        offset_start = start_unit - document.display_start
        offset_end = end_unit - document.display_start + 1
        units = document.raw_units[offset_start:offset_end]
        separator = "\n" if document.source_type == "markdown" else "\n\n"
        return separator.join(units)

    parts: list[str] = []
    for segment in document.segments:
        if segment.source_end < start_unit or segment.source_start > end_unit:
            continue
        parts.append(segment.text)
    separator = "\n" if document.source_type == "markdown" else "\n\n"
    return separator.join(parts)


def _get_document_unit(document: SourceDocument, unit_no: int) -> str | None:
    if unit_no < document.display_start or unit_no > document.display_end or not document.raw_units:
        return None
    return document.raw_units[unit_no - document.display_start]


def _truncate_prompt_text(text: str, max_chars: int) -> str:
    normalized = (text or "").strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3] + "..."


def _is_auxiliary_plot_title(title: str) -> bool:
    normalized = (title or "").strip().lower()
    return any(normalized.startswith(marker) or marker in normalized for marker in MARKDOWN_AUXILIARY_PLOT_MARKERS)


def _extract_markdown_story_scenes(
    document: SourceDocument,
    story_start: int,
    story_end: int,
    llm: Callable[[str], str],
    warnings: list[str],
) -> list[dict[str, Any]]:
    headings = _collect_markdown_headings(document)
    candidates = _build_markdown_scene_candidates(document, headings, story_start, story_end)
    merged_candidates = candidates

    scenes: list[dict[str, Any]] = []
    for scene_idx, candidate in enumerate(merged_candidates, start=1):
        scene_id = f"scene_{scene_idx}"
        blocks = _build_markdown_plot_blocks(document, headings, candidate)
        plot_decisions = _classify_markdown_plot_blocks(llm, document, candidate, blocks, warnings)
        plot_spans = _materialize_markdown_plot_spans(candidate, blocks, plot_decisions)
        metadata = _extract_fixed_scene_metadata(llm, document, candidate, plot_spans, warnings)
        scenes.append(_assemble_markdown_scene(scene_id, document, candidate, plot_spans, metadata))

    _validate_scene_plot_coverage(scenes, story_start, story_end)
    return scenes


def _attach_navigation_metadata(
    scenes: list[dict[str, Any]],
    document: SourceDocument,
    llm: Callable[[str], str],
    warnings: list[str],
) -> list[dict[str, Any]]:
    if not scenes:
        return scenes

    ensure_scene_navigation_defaults(scenes)
    if document.source_type != "markdown":
        return scenes

    _attach_markdown_scene_navigation(scenes)
    for scene in scenes:
        _attach_markdown_plot_navigation(scene, scenes)
    ensure_scene_navigation_defaults(scenes)
    return scenes


def _attach_markdown_scene_navigation(scenes: list[dict[str, Any]]) -> None:
    anchor_indexes: list[int] = []
    scene_titles = [_scene_heading_title(scene) for scene in scenes]

    for idx, scene in enumerate(scenes):
        normalized = scene_titles[idx]
        if normalized in {"start", "next steps", "conclusion"}:
            anchor_indexes.append(idx)
            if normalized == "conclusion":
                scene["node_kind"] = "terminal"
                scene["navigation"] = default_navigation(completion_policy="terminal_on_resolve")

    if not anchor_indexes:
        return

    for anchor_pos, scene_idx in enumerate(anchor_indexes):
        scene = scenes[scene_idx]
        title = scene_titles[scene_idx]
        if title == "conclusion":
            continue

        next_anchor_idx = anchor_indexes[anchor_pos + 1] if anchor_pos + 1 < len(anchor_indexes) else len(scenes)
        child_indexes = list(range(scene_idx + 1, next_anchor_idx))
        child_scenes = [scenes[idx] for idx in child_indexes]
        child_titles = [scene_titles[idx] for idx in child_indexes]
        if not child_scenes:
            continue

        completion_policy = "all_required_then_advance" if title == "start" else "optional_until_exit"
        scene["node_kind"] = "hub"
        scene["navigation"] = default_navigation(completion_policy=completion_policy)
        scene["navigation"]["allowed_targets"] = [
            make_target(
                "scene",
                str(child_scene.get("scene_id", "")),
                label=child_title or str(child_scene.get("scene_goal", "")),
                required=title == "start",
                role="branch",
                goal=str(child_scene.get("scene_goal", "")),
                excerpt=_truncate_text(_scene_raw_text(child_scene), 500),
            )
            for child_scene, child_title in zip(child_scenes, child_titles, strict=True)
        ]

        if title in {"start", "next steps"} and next_anchor_idx < len(scenes):
            downstream_scene = scenes[next_anchor_idx]
            scene["navigation"]["allowed_targets"].append(
                make_target(
                    "scene",
                    str(downstream_scene.get("scene_id", "")),
                    label=scene_titles[next_anchor_idx] or str(downstream_scene.get("scene_goal", "")),
                    role="exit",
                    prerequisites=[str(child_scene.get("scene_id", "")) for child_scene in child_scenes] if title == "start" else [],
                    goal=str(downstream_scene.get("scene_goal", "")),
                    excerpt=_truncate_text(_scene_raw_text(downstream_scene), 500),
                )
            )

        for child_scene in child_scenes:
            if str(child_scene.get("node_kind", "")) not in {"hub", "terminal"}:
                child_scene["node_kind"] = "branch"
            child_navigation = default_navigation(
                completion_policy="optional_until_exit" if title == "next steps" else "terminal_on_resolve"
            )
            child_navigation["return_target"] = make_target(
                "scene",
                str(scene.get("scene_id", "")),
                label=title or str(scene.get("scene_goal", "")),
                role="return",
                goal=str(scene.get("scene_goal", "")),
                excerpt=_truncate_text(_scene_raw_text(scene), 500),
            )
            if title == "next steps" and next_anchor_idx < len(scenes):
                downstream_scene = scenes[next_anchor_idx]
                child_navigation["allowed_targets"] = [
                    make_target(
                        "scene",
                        str(downstream_scene.get("scene_id", "")),
                        label=scene_titles[next_anchor_idx] or str(downstream_scene.get("scene_goal", "")),
                        role="exit",
                        goal=str(downstream_scene.get("scene_goal", "")),
                        excerpt=_truncate_text(_scene_raw_text(downstream_scene), 500),
                    )
                ]
            child_scene["navigation"] = child_navigation


def _attach_markdown_plot_navigation(scene: dict[str, Any], scenes: list[dict[str, Any]]) -> None:
    plots = scene.get("plots", [])
    if not isinstance(plots, list) or not plots:
        return

    if len(plots) == 1:
        plots[0]["node_kind"] = scene.get("node_kind", "linear")
        plots[0]["navigation"] = json.loads(json.dumps(scene.get("navigation", default_navigation())))
        return

    scene_return_target = _scene_return_target(scene)
    next_scene_target = _scene_exit_target(scene, scenes)
    plot_titles = [_plot_heading_title(plot) for plot in plots]
    branch_flags = [_plot_branch_mode(title) for title in plot_titles]

    optional_topic_scene = all(flag in {"optional", "none"} for flag in branch_flags[1:]) and any(
        flag == "optional" for flag in branch_flags[1:]
    )
    conditional_scene = any(flag in {"conditional", "followup"} for flag in branch_flags[1:])

    if optional_topic_scene:
        _apply_optional_plot_hub(scene, plots, plot_titles, scene_return_target or next_scene_target)
        return

    if conditional_scene:
        _apply_conditional_plot_hub(scene, plots, plot_titles, branch_flags, next_scene_target)


def _apply_optional_plot_hub(
    scene: dict[str, Any],
    plots: list[dict[str, Any]],
    plot_titles: list[str],
    next_scene_target: dict[str, Any] | None,
) -> None:
    hub_plot = plots[0]
    hub_plot["node_kind"] = "hub"
    hub_plot["navigation"] = default_navigation(completion_policy="optional_until_exit")
    hub_plot["navigation"]["allowed_targets"] = []

    for plot, title in zip(plots[1:], plot_titles[1:], strict=True):
        plot["node_kind"] = "branch"
        plot["navigation"] = default_navigation(completion_policy="terminal_on_resolve")
        plot["navigation"]["return_target"] = make_target(
            "plot",
            str(hub_plot.get("plot_id", "")),
            label=plot_titles[0] or str(hub_plot.get("plot_goal", "")),
            role="return",
            goal=str(hub_plot.get("plot_goal", "")),
            excerpt=_truncate_text(str(hub_plot.get("raw_text", "")), 500),
        )
        hub_plot["navigation"]["allowed_targets"].append(
            make_target(
                "plot",
                str(plot.get("plot_id", "")),
                label=title or str(plot.get("plot_goal", "")),
                role="branch",
                goal=str(plot.get("plot_goal", "")),
                excerpt=_truncate_text(str(plot.get("raw_text", "")), 500),
            )
        )

    if next_scene_target:
        hub_plot["navigation"]["allowed_targets"].append(next_scene_target)


def _apply_conditional_plot_hub(
    scene: dict[str, Any],
    plots: list[dict[str, Any]],
    plot_titles: list[str],
    branch_flags: list[str],
    next_scene_target: dict[str, Any] | None,
) -> None:
    hub_plot = plots[0]
    downstream_plot = _conditional_downstream_plot(plots, plot_titles, branch_flags)
    hub_plot["node_kind"] = "hub"
    hub_plot["navigation"] = default_navigation(
        completion_policy="exclusive_choice",
        close_unselected_on_advance=True,
    )
    hub_plot["navigation"]["allowed_targets"] = []

    attached_to_previous: dict[str, str] = {}
    for idx in range(1, len(plots)):
        if branch_flags[idx] == "followup":
            previous_idx = max(1, idx - 1)
            attached_to_previous[str(plots[idx].get("plot_id", ""))] = str(plots[previous_idx].get("plot_id", ""))

    for idx, plot in enumerate(plots[1:], start=1):
        plot_id = str(plot.get("plot_id", ""))
        title = plot_titles[idx]
        plot["node_kind"] = "branch"
        plot["navigation"] = default_navigation(completion_policy="optional_until_exit")
        plot["navigation"]["return_target"] = make_target(
            "plot",
            str(hub_plot.get("plot_id", "")),
            label=plot_titles[0] or str(hub_plot.get("plot_goal", "")),
            role="return",
            goal=str(hub_plot.get("plot_goal", "")),
            excerpt=_truncate_text(str(hub_plot.get("raw_text", "")), 500),
        )

        if plot_id in attached_to_previous:
            continue

        if downstream_plot and plot_id != str(downstream_plot.get("plot_id", "")):
            plot["navigation"]["allowed_targets"] = [
                make_target(
                    "plot",
                    str(downstream_plot.get("plot_id", "")),
                    label=_plot_heading_title(downstream_plot) or str(downstream_plot.get("plot_goal", "")),
                    role="exit",
                    goal=str(downstream_plot.get("plot_goal", "")),
                    excerpt=_truncate_text(str(downstream_plot.get("raw_text", "")), 500),
                )
            ]
            plot["navigation"]["completion_policy"] = "optional_until_exit"
        elif next_scene_target:
            plot["navigation"]["allowed_targets"] = [next_scene_target]
            plot["navigation"]["completion_policy"] = "optional_until_exit"

        hub_plot["navigation"]["allowed_targets"].append(
            make_target(
                "plot",
                plot_id,
                label=title or str(plot.get("plot_goal", "")),
                role="branch",
                goal=str(plot.get("plot_goal", "")),
                excerpt=_truncate_text(str(plot.get("raw_text", "")), 500),
            )
        )

    for plot in plots[1:]:
        plot_id = str(plot.get("plot_id", ""))
        previous_plot_id = attached_to_previous.get(plot_id)
        if not previous_plot_id:
            continue
        previous_plot = next((item for item in plots if str(item.get("plot_id", "")) == previous_plot_id), None)
        if previous_plot is None:
            continue
        previous_plot.setdefault("navigation", default_navigation())
        previous_plot["navigation"] = previous_plot.get("navigation", default_navigation())
        previous_plot["navigation"].setdefault("allowed_targets", [])
        previous_plot["navigation"]["allowed_targets"].append(
            make_target(
                "plot",
                plot_id,
                label=_plot_heading_title(plot) or str(plot.get("plot_goal", "")),
                role="exit",
                goal=str(plot.get("plot_goal", "")),
                excerpt=_truncate_text(str(plot.get("raw_text", "")), 500),
            )
        )
        if next_scene_target:
            plot["navigation"]["allowed_targets"] = [next_scene_target]
            plot["navigation"]["completion_policy"] = "optional_until_exit"

    if downstream_plot:
        downstream_plot["node_kind"] = "branch"
        downstream_plot["navigation"] = default_navigation(completion_policy="optional_until_exit")
        if next_scene_target:
            downstream_plot["navigation"]["allowed_targets"] = [next_scene_target]


def _scene_heading_title(scene: dict[str, Any]) -> str:
    plots = scene.get("plots", [])
    if isinstance(plots, list) and plots:
        raw_title = _extract_heading_title(str(plots[0].get("raw_text", "")))
        if raw_title:
            return raw_title
    return str(scene.get("scene_goal", "")).strip().lower()


def _plot_heading_title(plot: dict[str, Any]) -> str:
    raw_title = _extract_heading_title(str(plot.get("raw_text", "")))
    if raw_title:
        return raw_title
    return str(plot.get("plot_goal", "")).strip().lower()


def _extract_heading_title(raw_text: str) -> str:
    for line in (raw_text or "").splitlines():
        stripped = line.strip()
        match = MARKDOWN_HEADING_RE.match(stripped)
        if match:
            return _normalize_markdown_heading(match.group(2)).strip().lower()
        if stripped:
            break
    return ""


def _scene_raw_text(scene: dict[str, Any]) -> str:
    plots = scene.get("plots", [])
    if not isinstance(plots, list):
        return ""
    return "\n".join(str(plot.get("raw_text", "")).strip() for plot in plots if str(plot.get("raw_text", "")).strip())


def _scene_exit_target(scene: dict[str, Any], scenes: list[dict[str, Any]]) -> dict[str, Any] | None:
    scene_id = str(scene.get("scene_id", ""))
    scene_idx = next((idx for idx, item in enumerate(scenes) if str(item.get("scene_id", "")) == scene_id), -1)
    if scene_idx < 0:
        return None

    current_navigation = scene.get("navigation", {})
    exit_target = next(
        (
            target
            for target in current_navigation.get("allowed_targets", [])
            if target.get("target_kind") == "scene" and target.get("role") == "exit"
        ),
        None,
    )
    if exit_target:
        return exit_target

    if scene_idx + 1 >= len(scenes):
        return None
    next_scene = scenes[scene_idx + 1]
    return make_target(
        "scene",
        str(next_scene.get("scene_id", "")),
        label=_scene_heading_title(next_scene) or str(next_scene.get("scene_goal", "")),
        role="exit",
        goal=str(next_scene.get("scene_goal", "")),
        excerpt=_truncate_text(_scene_raw_text(next_scene), 500),
    )


def _scene_return_target(scene: dict[str, Any]) -> dict[str, Any] | None:
    navigation = scene.get("navigation", {})
    target = navigation.get("return_target")
    if isinstance(target, dict) and target.get("target_kind") and target.get("target_id"):
        return target
    return None


def _plot_branch_mode(title: str) -> str:
    normalized = (title or "").strip().lower()
    if not normalized:
        return "none"
    if normalized.startswith("about ") or normalized.startswith("asking about "):
        return "optional"
    if normalized.startswith("if "):
        if "manages to" in normalized or "kill or incapacitate" in normalized:
            return "followup"
        return "conditional"
    if normalized.startswith("exploring ") or normalized.startswith("ignoring ") or normalized.startswith("meeting ") or normalized.startswith("the investigator "):
        return "conditional"
    return "none"


def _conditional_downstream_plot(
    plots: list[dict[str, Any]],
    plot_titles: list[str],
    branch_flags: list[str],
) -> dict[str, Any] | None:
    if len(plots) < 3:
        return None
    last_title = plot_titles[-1]
    if branch_flags[-1] == "conditional" and last_title.startswith("meeting "):
        return plots[-1]
    return None


def _collect_markdown_headings(document: SourceDocument) -> list[MarkdownHeading]:
    if document.source_type != "markdown":
        return []

    headings: list[MarkdownHeading] = []
    heading_stack: list[tuple[int, str]] = []
    for line_no, raw_line in enumerate(document.raw_units, start=document.display_start):
        stripped = raw_line.strip()
        if not stripped or _is_decorative_markdown_line(stripped):
            continue
        match = MARKDOWN_HEADING_RE.match(stripped)
        if not match:
            continue
        level = len(match.group(1))
        title = _normalize_markdown_heading(match.group(2))
        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        heading_stack.append((level, title))
        headings.append(MarkdownHeading(line_no=line_no, level=level, title=title, path=tuple(title for _, title in heading_stack)))
    return headings


def _build_markdown_scene_candidates(
    document: SourceDocument,
    headings: list[MarkdownHeading],
    story_start: int,
    story_end: int,
) -> list[MarkdownSceneCandidate]:
    in_story = [heading for heading in headings if story_start <= heading.line_no <= story_end]
    start_map: dict[int, MarkdownSceneCandidate] = {}

    for idx, heading in enumerate(in_story):
        if heading.level == 4:
            start_map[heading.line_no] = MarkdownSceneCandidate(
                start=heading.line_no,
                end=heading.line_no,
                title=heading.title,
                path=heading.path,
                heading_level=heading.level,
                reason="level4",
            )
            continue

        normalized_title = heading.title.strip().lower()
        if heading.level == 3 and normalized_title in MARKDOWN_STANDALONE_SCENE_HEADINGS:
            section_end = _find_heading_section_end(in_story, idx, story_end)
            first_descendant = _find_first_descendant_heading(in_story, idx, section_end)
            direct_body_end = first_descendant.line_no - 1 if first_descendant else section_end
            if _has_meaningful_markdown_content(document, heading.line_no, direct_body_end):
                start_map[heading.line_no] = MarkdownSceneCandidate(
                    start=heading.line_no,
                    end=heading.line_no,
                    title=heading.title,
                    path=heading.path,
                    heading_level=heading.level,
                    reason="eligible_level3",
                )

    if story_start not in start_map:
        nearest = next((heading for heading in in_story if heading.line_no >= story_start), None)
        start_map[story_start] = MarkdownSceneCandidate(
            start=story_start,
            end=story_start,
            title=nearest.title if nearest else "Story",
            path=nearest.path if nearest else (),
            heading_level=nearest.level if nearest else 0,
            reason="story_start_fallback",
        )

    starts = sorted(start_map.values(), key=lambda candidate: candidate.start)
    candidates: list[MarkdownSceneCandidate] = []
    for idx, candidate in enumerate(starts):
        next_start = starts[idx + 1].start if idx + 1 < len(starts) else story_end + 1
        candidates.append(
            MarkdownSceneCandidate(
                start=candidate.start,
                end=max(candidate.start, next_start - 1),
                title=candidate.title,
                path=candidate.path,
                heading_level=candidate.heading_level,
                reason=candidate.reason,
            )
        )
    return candidates


def _find_heading_section_end(headings: list[MarkdownHeading], index: int, fallback_end: int) -> int:
    current = headings[index]
    for next_heading in headings[index + 1 :]:
        if next_heading.level <= current.level:
            return next_heading.line_no - 1
    return fallback_end


def _find_first_descendant_heading(
    headings: list[MarkdownHeading],
    index: int,
    section_end: int,
) -> MarkdownHeading | None:
    current = headings[index]
    for next_heading in headings[index + 1 :]:
        if next_heading.line_no > section_end:
            return None
        if next_heading.level > current.level:
            return next_heading
        if next_heading.level <= current.level:
            return None
    return None


def _has_meaningful_markdown_content(document: SourceDocument, start_line: int, end_line: int) -> bool:
    if start_line > end_line:
        return False
    for line_no in range(start_line, end_line + 1):
        raw_line = _get_document_unit(document, line_no)
        if raw_line is None:
            continue
        stripped = raw_line.strip()
        if stripped and not _is_decorative_markdown_line(stripped):
            return True
    return False


def _classify_markdown_scene_candidates(
    llm: Callable[[str], str],
    document: SourceDocument,
    candidates: list[MarkdownSceneCandidate],
    warnings: list[str],
) -> list[str]:
    if len(candidates) <= 1:
        return ["keep"] * len(candidates)

    prompt = _build_markdown_scene_classification_prompt(document, candidates)
    try:
        raw_text = _call_llm_with_retries(llm, prompt)
        data = _parse_json_response(raw_text, llm)
        decisions = data.get("decisions", [])
        if not isinstance(decisions, list) or len(decisions) != len(candidates):
            raise ValueError("scene decisions length mismatch")

        actions: list[str] = []
        for idx, decision in enumerate(decisions, start=1):
            if not isinstance(decision, dict):
                raise ValueError("scene decision must be object")
            if _coerce_int(decision.get("candidate_index"), 0) != idx:
                raise ValueError("scene candidate indexes must stay ordered")
            if idx == 1:
                actions.append("keep")
                continue
            action = str(decision.get("action", "keep")).strip().lower()
            if action not in {"keep", "merge_with_previous"}:
                raise ValueError("unsupported scene action")
            actions.append(action)
        return actions
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"markdown scene classification fallback: {exc}")
        return ["keep"] * len(candidates)


def _materialize_markdown_scene_candidates(
    candidates: list[MarkdownSceneCandidate],
    actions: list[str],
) -> list[MarkdownSceneCandidate]:
    if not candidates:
        return []
    merged: list[MarkdownSceneCandidate] = [candidates[0]]
    for candidate, action in zip(candidates[1:], actions[1:], strict=False):
        if action == "merge_with_previous":
            previous = merged[-1]
            merged[-1] = MarkdownSceneCandidate(
                start=previous.start,
                end=candidate.end,
                title=previous.title,
                path=previous.path,
                heading_level=previous.heading_level,
                reason=f"{previous.reason}+merge",
            )
        else:
            merged.append(candidate)
    return merged


def _build_markdown_plot_blocks(
    document: SourceDocument,
    headings: list[MarkdownHeading],
    scene: MarkdownSceneCandidate,
) -> list[MarkdownPlotBlock]:
    level5_headings = [
        heading
        for heading in headings
        if heading.level == 5 and scene.start < heading.line_no <= scene.end
    ]
    starts = [scene.start] + [heading.line_no for heading in level5_headings]
    blocks: list[MarkdownPlotBlock] = []
    for idx, start in enumerate(starts):
        end = starts[idx + 1] - 1 if idx + 1 < len(starts) else scene.end
        if start == scene.start:
            title = scene.title
            path = scene.path
            level = scene.heading_level
            kind = "prefix"
        else:
            heading = level5_headings[idx - 1]
            title = heading.title
            path = heading.path
            level = heading.level
            kind = "heading5"
        blocks.append(
            MarkdownPlotBlock(
                start=start,
                end=end,
                title=title,
                path=path,
                kind=kind,
                heading_level=level,
                raw_text=_slice_document_text(document, start, end),
            )
        )
    return blocks


def _classify_markdown_plot_blocks(
    llm: Callable[[str], str],
    document: SourceDocument,
    scene: MarkdownSceneCandidate,
    blocks: list[MarkdownPlotBlock],
    warnings: list[str],
) -> list[dict[str, str]]:
    heuristic = [_default_plot_block_decision(blocks, idx) for idx in range(len(blocks))]
    if not any(block.kind == "heading5" for block in blocks):
        return heuristic

    prompt = _build_markdown_plot_classification_prompt(scene, blocks, heuristic)
    try:
        raw_text = _call_llm_with_retries(llm, prompt)
        data = _parse_json_response(raw_text, llm)
        decisions = data.get("decisions", [])
        if not isinstance(decisions, list) or len(decisions) != len(blocks):
            raise ValueError("plot decisions length mismatch")

        normalized: list[dict[str, str]] = []
        for idx, decision in enumerate(decisions, start=1):
            if not isinstance(decision, dict):
                raise ValueError("plot decision must be object")
            if _coerce_int(decision.get("block_index"), 0) != idx:
                raise ValueError("plot block indexes must stay ordered")
            block = blocks[idx - 1]
            if block.kind == "prefix":
                normalized.append({"role": "auxiliary", "attach_to": "next"})
                continue
            role = str(decision.get("role", "")).strip().lower()
            attach_to = str(decision.get("attach_to", "")).strip().lower()
            if role not in {"plot", "auxiliary"}:
                raise ValueError("unsupported plot role")
            if role == "auxiliary" and attach_to not in {"previous", "next"}:
                raise ValueError("auxiliary blocks must declare attach_to")
            normalized.append({"role": role, "attach_to": attach_to or "previous"})

        if not _plot_decisions_are_valid(blocks, normalized):
            raise ValueError("plot decisions violate continuity constraints")
        return normalized
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"markdown plot classification fallback in {scene.title}: {exc}")
        return heuristic


def _default_plot_block_decision(blocks: list[MarkdownPlotBlock], index: int) -> dict[str, str]:
    block = blocks[index]
    if block.kind == "prefix":
        return {"role": "plot", "attach_to": ""} if _prefix_block_should_be_plot(blocks, block) else {"role": "auxiliary", "attach_to": "next"}
    if _is_auxiliary_plot_title(block.title):
        return {"role": "auxiliary", "attach_to": "previous" if index > 0 else "next"}
    return {"role": "plot", "attach_to": ""}


def _plot_decisions_are_valid(blocks: list[MarkdownPlotBlock], decisions: list[dict[str, str]]) -> bool:
    if len(blocks) != len(decisions):
        return False
    if not blocks:
        return True
    first_role = decisions[0].get("role")
    first_attach = decisions[0].get("attach_to")
    if first_role == "auxiliary":
        if first_attach != "next":
            return False
    elif first_role != "plot":
        return False
    for decision in decisions[1:]:
        role = decision.get("role")
        attach_to = decision.get("attach_to")
        if role not in {"plot", "auxiliary"}:
            return False
        if role == "auxiliary" and attach_to not in {"previous", "next"}:
            return False
        if role == "plot" and attach_to not in {"", None}:
            return False
    return True


def _prefix_block_should_be_plot(blocks: list[MarkdownPlotBlock], prefix_block: MarkdownPlotBlock) -> bool:
    if not any(block.kind == "heading5" and not _is_auxiliary_plot_title(block.title) for block in blocks):
        return False

    lines = [line.strip() for line in (prefix_block.raw_text or "").splitlines()]
    body_lines = [line for line in lines[1:] if line and not _is_decorative_markdown_line(line)]
    return bool(body_lines)


def _materialize_markdown_plot_spans(
    scene: MarkdownSceneCandidate,
    blocks: list[MarkdownPlotBlock],
    decisions: list[dict[str, str]],
) -> list[dict[str, Any]]:
    if not blocks or not any(decision.get("role") == "plot" for decision in decisions):
        return [{"start": scene.start, "end": scene.end, "title": scene.title, "seed_titles": [block.title for block in blocks] or [scene.title]}]

    groups: list[list[int]] = []
    pending_for_next: list[int] = []
    for idx, decision in enumerate(decisions):
        if decision.get("role") == "plot":
            groups.append(pending_for_next + [idx])
            pending_for_next = []
            continue
        attach_to = decision.get("attach_to", "previous")
        if attach_to == "previous" and groups:
            groups[-1].append(idx)
        else:
            pending_for_next.append(idx)

    if pending_for_next and groups:
        groups[-1].extend(pending_for_next)

    plot_spans: list[dict[str, Any]] = []
    for group in groups:
        group_blocks = [blocks[idx] for idx in group]
        heading_titles = [
            block.title
            for block in group_blocks
            if block.kind == "heading5" and not _is_auxiliary_plot_title(block.title)
        ]
        plot_spans.append(
            {
                "start": group_blocks[0].start,
                "end": group_blocks[-1].end,
                "title": heading_titles[0] if heading_titles else group_blocks[0].title,
                "seed_titles": [block.title for block in group_blocks],
            }
        )
    return plot_spans


def _extract_fixed_scene_metadata(
    llm: Callable[[str], str],
    document: SourceDocument,
    scene: MarkdownSceneCandidate,
    plot_spans: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    prompt = _build_fixed_scene_metadata_prompt(document, scene, plot_spans)
    try:
        raw_text = _call_llm_with_retries(llm, prompt)
        data = _parse_json_response(raw_text, llm)
        plots = data.get("plots", [])
        if not isinstance(data, dict) or not isinstance(plots, list) or len(plots) != len(plot_spans):
            raise ValueError("scene metadata plot count mismatch")
        return data
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"markdown scene metadata fallback in {scene.title}: {exc}")
        return _fallback_scene_metadata(scene, plot_spans)


def _fallback_scene_metadata(scene: MarkdownSceneCandidate, plot_spans: list[dict[str, Any]]) -> dict[str, Any]:
    scene_goal = scene.title or "Advance the investigation"
    return {
        "scene_goal": scene_goal,
        "scene_description": f"{scene_goal}. This scene covers the material assigned to lines {scene.start}-{scene.end}.",
        "scene_summary": scene_goal,
        "scene_type": "transition" if any(marker in scene.title.lower() for marker in SCENE_TRANSITION_TITLE_MARKERS) else "normal",
        "plots": [
            {
                "plot_index": idx,
                "plot_goal": plot_span.get("title") or f"{scene_goal} plot {idx}",
                "mandatory_events": [],
                "npc": [],
                "locations": [],
            }
            for idx, plot_span in enumerate(plot_spans, start=1)
        ],
    }


def _assemble_markdown_scene(
    scene_id: str,
    document: SourceDocument,
    scene: MarkdownSceneCandidate,
    plot_spans: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    plot_meta = metadata.get("plots", []) if isinstance(metadata.get("plots"), list) else []
    plots: list[dict[str, Any]] = []
    for idx, plot_span in enumerate(plot_spans, start=1):
        meta = plot_meta[idx - 1] if idx - 1 < len(plot_meta) and isinstance(plot_meta[idx - 1], dict) else {}
        plots.append(
            {
                "plot_id": f"{scene_id}_plot_{idx}",
                "plot_goal": _coerce_text(meta.get("plot_goal"), plot_span.get("title") or f"{scene_id} plot {idx}"),
                "mandatory_events": _coerce_list(meta.get("mandatory_events")),
                "npc": _coerce_list(meta.get("npc")),
                "locations": _coerce_list(meta.get("locations")),
                "raw_text": _slice_document_text(document, plot_span["start"], plot_span["end"]),
                "status": "pending",
                "progress": 0.0,
                "source_page_start": plot_span["start"],
                "source_page_end": plot_span["end"],
            }
        )

    return {
        "scene_id": scene_id,
        "scene_goal": _coerce_text(metadata.get("scene_goal"), scene.title or f"Advance story in {scene_id}"),
        "scene_description": _coerce_text(
            metadata.get("scene_description"),
            f"{scene.title}. This scene covers the material assigned to lines {scene.start}-{scene.end}.",
        ),
        "scene_summary": _coerce_text(metadata.get("scene_summary"), ""),
        "scene_type": _normalize_scene_type(metadata),
        "source_page_start": scene.start,
        "source_page_end": scene.end,
        "plots": plots,
        "status": "pending",
    }


def _validate_scene_plot_coverage(scenes: list[dict[str, Any]], story_start: int, story_end: int) -> None:
    if not scenes:
        raise ValueError("story produced no scenes")

    previous_scene_end = story_start - 1
    for scene in scenes:
        scene_start = _coerce_int(scene.get("source_page_start"), story_start)
        scene_end = _coerce_int(scene.get("source_page_end"), scene_start)
        if scene_start != previous_scene_end + 1:
            raise ValueError(f"scene coverage gap/overlap near {scene.get('scene_id')}: expected {previous_scene_end + 1}, got {scene_start}")
        if scene_end < scene_start:
            raise ValueError(f"scene span inverted for {scene.get('scene_id')}")

        plots = scene.get("plots", [])
        if not isinstance(plots, list) or not plots:
            raise ValueError(f"scene {scene.get('scene_id')} has no plots")

        previous_plot_end = scene_start - 1
        for plot in plots:
            plot_start = _coerce_int(plot.get("source_page_start"), scene_start)
            plot_end = _coerce_int(plot.get("source_page_end"), plot_start)
            if plot_start != previous_plot_end + 1:
                raise ValueError(f"plot coverage gap/overlap near {plot.get('plot_id')}: expected {previous_plot_end + 1}, got {plot_start}")
            if plot_end < plot_start:
                raise ValueError(f"plot span inverted for {plot.get('plot_id')}")
            if plot_start < scene_start or plot_end > scene_end:
                raise ValueError(f"plot {plot.get('plot_id')} escapes parent scene span")
            previous_plot_end = plot_end

        if previous_plot_end != scene_end:
            raise ValueError(f"scene {scene.get('scene_id')} plots do not cover the full scene span")
        previous_scene_end = scene_end

    if _coerce_int(scenes[0].get("source_page_start"), story_start) != story_start:
        raise ValueError("scene coverage does not start at story_start")
    if previous_scene_end != story_end:
        raise ValueError("scene coverage does not reach story_end")


def _build_markdown_scene_classification_prompt(
    document: SourceDocument,
    candidates: list[MarkdownSceneCandidate],
) -> str:
    schema = (
        "{\n"
        '  "decisions": [\n'
        '    {"candidate_index": 1, "action": "keep"},\n'
        '    {"candidate_index": 2, "action": "keep|merge_with_previous"}\n'
        "  ]\n"
        "}\n"
    )
    blocks: list[str] = []
    for idx, candidate in enumerate(candidates, start=1):
        excerpt = _truncate_prompt_text(_slice_document_text(document, candidate.start, candidate.end), 900)
        blocks.append(
            "\n".join(
                [
                    f"Candidate {idx}",
                    f"Span: {candidate.start}-{candidate.end}",
                    f"Heading Level: {candidate.heading_level}",
                    f"Heading Path: {' > '.join(candidate.path) if candidate.path else candidate.title}",
                    f"Reason: {candidate.reason}",
                    "Excerpt:",
                    excerpt,
                ]
            )
        )

    return (
        "TASK: CLASSIFY_MARKDOWN_SCENE_CANDIDATES\n"
        "You are deciding whether heading-derived markdown scene candidates should stay separate or merge with the previous candidate.\n"
        "Rules:\n"
        "- Do not invent new spans.\n"
        "- Do not reorder candidates.\n"
        "- Use merge_with_previous only when adjacent candidates are semantically one playable scene.\n"
        "- Prefer keeping candidates separate when headings suggest distinct investigation locations, actions, or outcomes.\n"
        "- Higher-level headings are strong hints, not absolute boundaries.\n"
        "Return strict JSON only.\n"
        f"Schema:\n{schema}\n"
        "Candidates:\n"
        + "\n\n".join(blocks)
        + "\n"
    )


def _build_markdown_plot_classification_prompt(
    scene: MarkdownSceneCandidate,
    blocks: list[MarkdownPlotBlock],
    heuristic: list[dict[str, str]],
) -> str:
    schema = (
        "{\n"
        '  "decisions": [\n'
        '    {"block_index": 1, "role": "auxiliary", "attach_to": "next"},\n'
        '    {"block_index": 2, "role": "plot|auxiliary", "attach_to": "previous|next"}\n'
        "  ]\n"
        "}\n"
    )
    rendered: list[str] = []
    for idx, block in enumerate(blocks, start=1):
        rendered.append(
            "\n".join(
                [
                    f"Block {idx}",
                    f"Span: {block.start}-{block.end}",
                    f"Kind: {block.kind}",
                    f"Heading Path: {' > '.join(block.path) if block.path else block.title}",
                    f"Heuristic Role: {heuristic[idx - 1]['role']}",
                    f"Heuristic Attach: {heuristic[idx - 1].get('attach_to', '')}",
                    "Excerpt:",
                    _truncate_prompt_text(block.raw_text, 700),
                ]
            )
        )

    return (
        "TASK: CLASSIFY_MARKDOWN_PLOT_BLOCKS\n"
        "You are classifying fixed markdown plot blocks inside one scene.\n"
        "Rules:\n"
        "- Do not invent new spans.\n"
        "- Do not split or merge blocks; only label each block as plot or auxiliary.\n"
        "- Auxiliary note-like blocks should attach to previous or next plot.\n"
        "- Blocks such as keeper notes, two investigators notes, or lore sidebars are usually auxiliary.\n"
        "- Distinct investigation actions, branches, or outcomes are usually plots.\n"
        "- The first prefix block may be its own plot when the scene-level intro contains meaningful playable setup before any ##### subsection.\n"
        "Return strict JSON only.\n"
        f"Scene Span: {scene.start}-{scene.end}\n"
        f"Scene Heading: {scene.title}\n"
        f"Schema:\n{schema}\n"
        "Blocks:\n"
        + "\n\n".join(rendered)
        + "\n"
    )


def _build_fixed_scene_metadata_prompt(
    document: SourceDocument,
    scene: MarkdownSceneCandidate,
    plot_spans: list[dict[str, Any]],
) -> str:
    schema = (
        "{\n"
        '  "scene_goal": "...",\n'
        '  "scene_description": "...",\n'
        '  "scene_summary": "",\n'
        '  "scene_type": "normal|transition",\n'
        '  "plots": [\n'
        "    {\n"
        '      "plot_index": 1,\n'
        '      "plot_goal": "...",\n'
        '      "mandatory_events": ["..."],\n'
        '      "npc": ["..."],\n'
        '      "locations": ["..."]\n'
        "    }\n"
        "  ]\n"
        "}\n"
    )
    plot_sections: list[str] = []
    for idx, plot_span in enumerate(plot_spans, start=1):
        plot_sections.append(
            "\n".join(
                [
                    f"Plot {idx}",
                    f"Span: {plot_span['start']}-{plot_span['end']}",
                    f"Seed Titles: {', '.join(plot_span.get('seed_titles', []))}",
                    "Raw Text:",
                    _truncate_prompt_text(_slice_document_text(document, plot_span["start"], plot_span["end"]), 1200),
                ]
            )
        )

    return (
        "TASK: EXTRACT_FIXED_SCENE_METADATA\n"
        "You are extracting metadata for one markdown scene and its fixed plot spans.\n"
        "Rules:\n"
        "- Do not change the number of plots.\n"
        "- Do not change plot order or spans.\n"
        "- scene_goal should be concise.\n"
        "- scene_description should be a clear playable summary paragraph.\n"
        "- scene_type may be transition only for brief bridge/conclusion material.\n"
        "- plot_goal should reflect concrete progression inside that exact span.\n"
        "Return strict JSON only.\n"
        f"Schema:\n{schema}\n"
        f"Scene Span: {scene.start}-{scene.end}\n"
        f"Scene Heading: {scene.title}\n"
        "Scene Raw Text:\n"
        f"{_truncate_prompt_text(_slice_document_text(document, scene.start, scene.end), 1800)}\n"
        "Plots:\n"
        + "\n\n".join(plot_sections)
        + "\n"
    )


def _build_manual_structure(
    document: SourceDocument,
    story_start_page: int | None,
    story_end_page: int | None,
    warnings: list[str],
) -> dict[str, dict[str, int] | None] | None:
    start_raw = _coerce_int(story_start_page, 0)
    end_raw = _coerce_int(story_end_page, 0)
    min_unit = document.display_start
    max_unit = document.display_end

    has_start = start_raw > 0
    has_end = end_raw > 0
    if not has_start and not has_end:
        return None

    if has_start != has_end:
        warnings.append("structure: manual story range ignored because both story_start_page and story_end_page are required")
        return None

    start_page = max(min_unit, min(max_unit, start_raw))
    end_page = max(min_unit, min(max_unit, end_raw))
    if end_page < start_page:
        start_page, end_page = end_page, start_page
        warnings.append("structure: manual story range start/end swapped to keep ascending order")

    warnings.append(f"structure: manual story range applied ({start_page}-{end_page})")
    front_range = {"start_page": min_unit, "end_page": start_page - 1} if start_page > min_unit else None
    appendix_range = {"start_page": end_page + 1, "end_page": max_unit} if end_page < max_unit else None
    return {
        "front_knowledge": front_range,
        "story": {"start_page": start_page, "end_page": end_page},
        "appendix_knowledge": appendix_range,
    }


def _identify_structure_ranges(
    document: SourceDocument,
    llm: Callable[[str], str],
    warnings: list[str],
) -> dict[str, dict[str, int] | None]:
    min_unit = document.display_start
    max_unit = document.display_end
    preview_chars = _read_int_env("PARSER_STRUCTURE_PREVIEW_CHARS", 100, 50)

    previews = [_build_structure_preview(segment, document.unit_label, document.source_type, preview_chars) for segment in document.segments]

    prompt = _build_structure_prompt(previews, document)

    try:
        raw_text = _call_llm_with_retries(llm, prompt)
        data = _parse_json_response(raw_text, llm)
        if not isinstance(data, dict):
            raise ValueError("structure response root must be object")

        story_range = _coerce_range(data.get("story"), min_unit, max_unit)
        if story_range is None:
            warnings.append("structure: invalid story range from LLM, fallback to full document story")
            story_range = {"start_page": min_unit, "end_page": max_unit}

        story_start = story_range["start_page"]
        story_end = story_range["end_page"]
        front_range = {"start_page": min_unit, "end_page": story_start - 1} if story_start > min_unit else None
        appendix_range = {"start_page": story_end + 1, "end_page": max_unit} if story_end < max_unit else None

        return {
            "front_knowledge": front_range,
            "story": {"start_page": story_start, "end_page": story_end},
            "appendix_knowledge": appendix_range,
        }
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"structure: fallback to full-story mode ({exc})")
        return {
            "front_knowledge": None,
            "story": {"start_page": min_unit, "end_page": max_unit},
            "appendix_knowledge": None,
        }


def _build_structure_prompt(previews: list[str], document: SourceDocument) -> str:
    preview_text = "\n".join(previews)
    schema = (
        "{\n"
        '  "story": {"start_page": 1, "end_page": 1}\n'
        "}\n"
    )
    unit_name = document.unit_label
    unit_plural = _unit_plural(unit_name)
    intro = "Given page previews, identify the SINGLE continuous story range.\n"
    extra_rules = ""
    section_header = "Page previews:\n"
    if document.source_type == "markdown":
        intro = "Given markdown previews, identify the SINGLE continuous playable story range.\n"
        extra_rules = (
            "- Higher-level headings are strong structural hints.\n"
            "- Headings such as START or investigation section titles often mark playable story.\n"
            "- Keeper/background/rules/next steps style headings often belong to knowledge, not story.\n"
            "- Even for markdown, keep JSON keys start_page/end_page; the values must be line numbers.\n"
        )
        section_header = "Markdown previews:\n"

    return (
        "TASK: IDENTIFY_SCRIPT_STRUCTURE\n"
        "You are a script-structure classifier.\n"
        f"{intro}"
        f"{unit_plural.capitalize()} before story are front knowledge, {unit_plural} after story are appendix knowledge.\n"
        "Return strict JSON only.\n"
        "Rules:\n"
        "- story must be one continuous range.\n"
        f"- If uncertain, include {unit_plural} in story instead of excluding them.\n"
        f"- Use {unit_name} numbers from the preview labels.\n"
        f"- Available {unit_name} range: {document.display_start}-{document.display_end}.\n"
        f"{extra_rules}"
        f"Schema:\n{schema}\n"
        f"{section_header}"
        f"{preview_text}\n"
    )


def _build_structure_preview(segment: SourceSegment, unit_label: str, source_type: str, preview_chars: int) -> str:
    normalized = re.sub(r"\s+", " ", segment.text or "").strip()
    heading_hint = ""
    if source_type == "markdown":
        if segment.heading_path:
            heading_hint = f" | Heading Path: {' > '.join(segment.heading_path)}"
        if _is_markdown_hint_segment(segment):
            heading_hint += " | Heuristic: likely knowledge/supporting note"
    return f"[{_source_tag(unit_label)} {_format_source_span(segment.source_start, segment.source_end)}]{heading_hint} {normalized[:preview_chars]}".strip()


def _render_segments_for_prompt(chunk: list[SourceSegment], unit_label: str, source_type: str) -> str:
    return "\n\n".join(_render_segment_for_prompt(segment, unit_label, source_type) for segment in chunk)


def _render_segment_for_prompt(segment: SourceSegment, unit_label: str, source_type: str) -> str:
    label = f"[{_source_tag(unit_label)} {_format_source_span(segment.source_start, segment.source_end)}]"
    if source_type != "markdown":
        return f"{label}\n{segment.text}"

    lines = [label]
    if segment.heading_path:
        lines.append(f"Heading Path: {' > '.join(segment.heading_path)}")
    if segment.heading_level:
        lines.append(f"Heading Level: {segment.heading_level}")
    if segment.segment_kind:
        lines.append(f"Segment Kind: {segment.segment_kind}")
    if _is_markdown_hint_segment(segment):
        lines.append("Heuristic: likely knowledge/supporting note")
    lines.append("Text:")
    lines.append(segment.text)
    return "\n".join(lines)


def _markdown_prompt_rules(source_type: str, task: str) -> str:
    if source_type != "markdown":
        return ""
    if task == "knowledge":
        return (
            "- Treat headings as strong structural hints.\n"
            "- Keeper/background/rules/two investigators/next steps sections are often knowledge.\n"
        )
    if task == "story":
        return (
            "- Treat headings as strong structural hints, not hard boundaries.\n"
            "- START and investigation-style section headings often mark playable scene material.\n"
            "- You may merge adjacent subheadings into one scene if the action is semantically continuous.\n"
            "- Do not turn every heading into a separate scene by default.\n"
        )
    if task == "plot_refine":
        return (
            "- Respect heading path and excerpt structure when splitting plots.\n"
            "- Use headings as strong hints, but do not force every heading to become one plot.\n"
        )
    return ""


def _read_int_env(name: str, default: int, minimum: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        return default


def _extract_knowledge_segment(
    document: SourceDocument,
    start_page: int,
    end_page: int,
    pages_per_chunk: int,
    llm: Callable[[str], str],
    max_workers: int,
    phase_label: str,
) -> list[dict[str, Any]]:
    chunks = _build_chunks_for_range(document.segments, start_page, end_page, pages_per_chunk)

    def process_chunk(
        chunk_idx: int, chunk_start: int, chunk_end: int, chunk: list[SourceSegment]
    ) -> list[dict[str, Any]]:
        prompt = _build_knowledge_prompt(
            chunk,
            chunk_idx,
            chunk_start,
            chunk_end,
            phase_label,
            document.unit_label,
            document.source_type,
        )
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
    document: SourceDocument,
    start_page: int,
    end_page: int,
    pages_per_chunk: int,
    llm: Callable[[str], str],
    max_workers: int,
) -> list[dict[str, Any]]:
    chunks = _build_chunks_for_range(document.segments, start_page, end_page, pages_per_chunk)

    def process_chunk(
        chunk_idx: int, chunk_start: int, chunk_end: int, chunk: list[SourceSegment]
    ) -> list[dict[str, Any]]:
        prompt = _build_story_prompt(
            chunk,
            chunk_idx,
            chunk_start,
            chunk_end,
            document.unit_label,
            document.source_type,
        )
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
    segments: tuple[SourceSegment, ...] | list[SourceSegment],
    start_page: int,
    end_page: int,
    pages_per_chunk: int,
) -> list[tuple[int, int, int, list[SourceSegment]]]:
    chunks: list[tuple[int, int, int, list[SourceSegment]]] = []
    if start_page > end_page:
        return chunks

    relevant = [segment for segment in segments if segment.source_end >= start_page and segment.source_start <= end_page]
    if not relevant:
        return chunks

    idx = 0
    chunk_size = max(1, pages_per_chunk)
    for offset in range(0, len(relevant), chunk_size):
        idx += 1
        chunk = relevant[offset : offset + chunk_size]
        chunk_start = max(start_page, chunk[0].source_start)
        chunk_end = min(end_page, chunk[-1].source_end)
        chunks.append((idx, chunk_start, chunk_end, chunk))
    return chunks


def _run_chunk_jobs(
    chunks: list[tuple[int, int, int, list[SourceSegment]]],
    process_chunk: Callable[[int, int, int, list[SourceSegment]], list[dict[str, Any]]],
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
    chunk: list[SourceSegment],
    chunk_idx: int,
    chunk_start: int,
    chunk_end: int,
    phase_label: str,
    unit_label: str,
    source_type: str,
) -> str:
    pages_text = _render_segments_for_prompt(chunk, unit_label, source_type)
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
        f"Extract only NON-PLAYABLE knowledge from these {_unit_plural(unit_label)}.\n"
        "Do not output scenes or plots.\n"
        "Return strict JSON only.\n"
        f"Phase: {phase_label}\n"
        f"Chunk: {chunk_idx} ({_unit_plural(unit_label)} {chunk_start}-{chunk_end})\n"
        "Compatibility note: keep keys source_page_start/source_page_end in JSON.\n"
        f"For this document type, those values mean {unit_label} numbers.\n"
        f"{_markdown_prompt_rules(source_type, task='knowledge')}"
        f"Schema:\n{schema}\n"
        f"{_unit_plural(unit_label).capitalize()}:\n"
        f"{pages_text}\n"
    )


def _build_story_prompt(
    chunk: list[SourceSegment],
    chunk_idx: int,
    chunk_start: int,
    chunk_end: int,
    unit_label: str,
    source_type: str,
) -> str:
    pages_text = _render_segments_for_prompt(chunk, unit_label, source_type)
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
        f"- Plots are fine-grained action/progression units and can split within the same {unit_label}.\n"
        f"- Use only information from these {_unit_plural(unit_label)}.\n"
        "- Keep keys source_page_start/source_page_end in JSON for compatibility.\n"
        f"- For this document type, those numeric values mean {unit_label} numbers.\n"
        f"{_markdown_prompt_rules(source_type, task='story')}"
        "- Return strict JSON only.\n"
        f"Chunk: {chunk_idx} ({_unit_plural(unit_label)} {chunk_start}-{chunk_end})\n"
        f"Schema:\n{schema}\n"
        f"{_unit_plural(unit_label).capitalize()}:\n"
        f"{pages_text}\n"
    )


def _refine_scene_plots(
    llm: Callable[[str], str],
    document: SourceDocument,
    scene_goal: str,
    scene_description: str,
    scene_summary: str,
    source_start: int,
    source_end: int,
) -> tuple[str | None, list[dict[str, Any]]]:
    excerpt = _render_source_excerpt(document, source_start, source_end)
    prompt = _build_plot_refine_prompt(
        scene_goal=scene_goal,
        scene_description=scene_description,
        scene_summary=scene_summary,
        source_start=source_start,
        source_end=source_end,
        scene_excerpt=excerpt,
        unit_label=document.unit_label,
        source_type=document.source_type,
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
    unit_label: str,
    source_type: str,
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
        "- Keep keys source_page_start/source_page_end in JSON for compatibility.\n"
        f"- For this document type, those numeric values mean {unit_label} numbers.\n"
        f"{_markdown_prompt_rules(source_type, task='plot_refine')}"
        "- Return strict JSON only.\n"
        f"Scene Goal: {scene_goal}\n"
        f"Scene Description: {scene_description}\n"
        f"Scene Summary: {scene_summary}\n"
        f"Scene Source {_unit_plural(unit_label).capitalize()}: {source_start}-{source_end}\n"
        f"Schema:\n{schema}\n"
        "Scene Excerpt:\n"
        f"{scene_excerpt}\n"
    )


def _render_source_excerpt(document: SourceDocument, start_page: int, end_page: int, max_chars: int = 3500) -> str:
    if start_page > end_page:
        return ""

    parts: list[str] = []
    total = 0
    for segment in document.segments:
        if segment.source_end < start_page or segment.source_start > end_page:
            continue
        snippet = _render_segment_for_prompt(segment, document.unit_label, document.source_type) + "\n"
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


def _coerce_range(value: Any, min_unit: int, max_unit: int) -> dict[str, int] | None:
    if not isinstance(value, dict):
        return None
    start_page = _coerce_int(value.get("start_page"), 0)
    end_page = _coerce_int(value.get("end_page"), 0)
    if start_page < 1 or end_page < 1:
        return None
    start_page = max(min_unit, min(max_unit, start_page))
    end_page = max(min_unit, min(max_unit, end_page))
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


def _decode_text_bytes(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        return file_bytes.decode("utf-8", errors="replace")


def _is_decorative_markdown_line(stripped_line: str) -> bool:
    if not stripped_line:
        return False
    if DECORATIVE_HTML_RE.match(stripped_line):
        return True
    lowered = stripped_line.lower()
    if "<img" in lowered and stripped_line.startswith("<"):
        return True
    return False


def _normalize_markdown_heading(text: str) -> str:
    normalized = re.sub(r"\s+", " ", (text or "").strip())
    normalized = normalized.strip("# ").strip()
    return normalized or "Untitled"


def _source_tag(unit_label: str) -> str:
    return unit_label.upper()


def _unit_plural(unit_label: str) -> str:
    return "pages" if unit_label == "page" else "lines"


def _format_source_span(start: int, end: int) -> str:
    return str(start) if start == end else f"{start}-{end}"


def _is_markdown_hint_segment(segment: SourceSegment) -> bool:
    heading_text = " ".join(segment.heading_path).lower()
    text = (segment.text or "").lower()
    return any(marker in heading_text or marker in text for marker in MARKDOWN_HINT_HEADINGS)
