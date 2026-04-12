from __future__ import annotations

from collections import Counter
import json5
import re
from dataclasses import dataclass
from typing import Any, Callable

from app.llm_client import call_llm

MARKDOWN_EXTENSIONS = {".md", ".markdown"}
MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
DECORATIVE_HTML_RE = re.compile(r"^</?div\b[^>]*>$|^<img\b[^>]*>$|^<div\b[^>]*><img\b.*</div>$", re.I)
KNOWLEDGE_TYPES = {"setting", "npc", "clue", "other"}
LLM_PROMPT_TEMPLATE = """You are an information extraction assistant.

Given a piece of narrative text, extract structured information in JSON format.

Definitions (IMPORTANT):

- NOT all content should be converted into scenes or plots.
- Only extract a scene if the text clearly describes a **player-related activity at a specific location**.
  - A scene should involve the player taking action, such as accepting a task, making a decision, interact with NPC/environment or investigate something.
  - Do NOT extract a scene if the text only provides background information (e.g., NPC descriptions or lore).
- Only extract plots if there are **clear player actions or tasks within that scene**. Each scene at least has 1 plot.

- If the content is:
  - background information
  - narrative exposition
  - NPC descriptions
  - hidden information for the Keeper
  - world lore or rules
→ store it as **knowledge**, NOT as a scene or plot.

---

### Scene

A scene represents a player’s activity at a specific location.

Only create a scene if:
- the player is actively doing something
- the location/activity is clear and actionable

Fields:
- scene_name (location-based player activity)
- scene_goal
- scene_description

If no valid scene exists, return:
"scene": {}

---

### Plots

- A scene may contain one or multiple plots
- Only extract plots if there are **clear actionable steps**
- For scenes that present multiple directions or choices:
  - Do NOT extract each direction as a separate plot
  - Instead, extract only the high-level "choice" action

Each plot:
- plot_name
- plot_goal

If no valid plots exist, return:
"plots": []

---

### Knowledge

Types: setting | npc | clue | other

Rules:

- Use **knowledge** for:
  - Keeper-only information
  - background lore
  - descriptions not directly tied to player actions
  - hidden facts or explanations

- npc:
  - must include descriptive information (not just names)

- clue:
  - must include:
    - what the clue is
    - how it is obtained
  - avoid spoilers

- setting:
  - describes world/background

---

### Priority Rule (CRITICAL)

When uncertain:
→ Prefer **knowledge** over scene/plot

Do NOT force structure.

---

### Output format (STRICT JSON)

{
  "scene": {
    "scene_name": "string",
    "scene_goal": "string",
    "scene_description": "string"
  },
  "plots": [
    {
      "plot_name": "string",
      "plot_goal": "string"
    }
  ],
  "knowledge": [
    {
      "knowledge_type": "npc | clue | setting | other",
      "title": "string",
      "content": "string"
    }
  ]
}

---

### Example output

{
  "scene": {
    "scene_name": "Explore the Kimball House",
    "scene_goal": "Investigate Douglas Kimball’s disappearance",
    "scene_description": "The player explores the Kimball residence, examining rooms and searching for clues."
  },
  "plots": [
    {
      "plot_name": "Search the study room",
      "plot_goal": "Find documents related to Douglas Kimball"
    }
  ],
  "knowledge": [
    {
      "knowledge_type": "npc",
      "title": "Douglas Kimball",
      "content": "Douglas Kimball is a reclusive man obsessed with books and frequently visits the cemetery."
    },
    {
      "knowledge_type": "clue",
      "title": "Strange footprints",
      "content": "Unusual footprints were found in the cemetery, reported by locals."
    },
    {
      "knowledge_type": "setting",
      "title": "Arnoldsburg",
      "content": "A quiet town with a cemetery near the Kimball residence."
    },
    {
      "knowledge_type": "other",
      "title": "Keeper Information",
      "content": "The creature behind the events lives underground but this is not yet revealed to the player."
    }
  ]
}

---

Return only JSON. Do not include explanations.

--- BEGIN INPUT ---
{Content}
--- END INPUT ---
"""

SCRIPT_SUMMARY_PROMPT_TEMPLATE = """You are summarizing a narrative script.

Existing summary:
{script_summary}

New input (scene & plots):
{structured_scene_plot_json}

Task:
Update the global story summary.

Requirements:
- The final summary must be 3–4 sentences in total
- Focus on main storyline, key actions, and discoveries
- If existing summary is not empty, refine and compress it while incorporating new information
- Avoid repeating content
- Keep the summary concise and coherent
"""

SCRIPT_SUMMARY_INPUT_LIMIT = 10000


@dataclass(frozen=True)
class SourceSegment:
    text: str
    segment_kind: str
    heading_level: int = 0
    heading_path: tuple[str, ...] = ()


@dataclass(frozen=True)
class SourceDocument:
    source_type: str
    source_file_name: str
    text: str
    raw_units: tuple[str, ...]
    outline: tuple[str, ...]
    segments: tuple[SourceSegment, ...]


def detect_source_type(file_name: str, mime_type: str | None = None) -> str:
    suffix = ""
    if file_name:
        parts = file_name.lower().rsplit(".", 1)
        if len(parts) == 2:
            suffix = f".{parts[1]}"
    normalized_mime = (mime_type or "").strip().lower()
    if suffix in MARKDOWN_EXTENSIONS or "markdown" in normalized_mime:
        return "markdown"
    raise ValueError(f"Only Markdown input is supported: {file_name or mime_type or 'unknown'}")


def read_markdown_document(file_bytes: bytes, file_name: str = "") -> SourceDocument:
    text = _decode_text_bytes(file_bytes).replace("\r\n", "\n").replace("\r", "\n")
    raw_lines = text.split("\n")
    outline: list[str] = []
    segments: list[SourceSegment] = []
    heading_stack: list[tuple[int, str]] = []
    block_lines: list[str] = []

    def current_heading_path() -> tuple[str, ...]:
        return tuple(title for _, title in heading_stack)

    def flush_block() -> None:
        nonlocal block_lines
        if not block_lines:
            return
        block_text = "\n".join(block_lines).strip()
        if block_text:
            segments.append(
                SourceSegment(
                    text=block_text,
                    segment_kind="body",
                    heading_level=heading_stack[-1][0] if heading_stack else 0,
                    heading_path=current_heading_path(),
                )
            )
        block_lines = []

    for raw_line in raw_lines:
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
                    segment_kind="heading",
                    heading_level=level,
                    heading_path=heading_path,
                )
            )
            outline_text = " > ".join(heading_path)
            if outline_text and outline_text not in outline:
                outline.append(outline_text)
            continue

        if not stripped:
            flush_block()
            continue

        block_lines.append(line)

    flush_block()
    return SourceDocument(
        source_type="markdown",
        source_file_name=file_name,
        text=text,
        raw_units=tuple(raw_lines),
        outline=tuple(outline),
        segments=tuple(segments),
    )


def read_uploaded_document(file_name: str, file_bytes: bytes, mime_type: str | None = None) -> SourceDocument:
    detect_source_type(file_name, mime_type)
    return read_markdown_document(file_bytes, file_name=file_name)


def parse_script(
    pages: list[str],
    llm_client: Callable[[str], str] | None = None,
    scene_heading_levels: tuple[int, ...] | list[int] | set[int] | None = None,
) -> list[dict[str, Any]]:
    return parse_script_bundle(
        pages=pages,
        llm_client=llm_client,
        scene_heading_levels=scene_heading_levels,
    )["scenes"]


def parse_script_bundle(
    pages: list[str] | None = None,
    llm_client: Callable[[str], str] | None = None,
    source_document: SourceDocument | None = None,
    scene_heading_levels: tuple[int, ...] | list[int] | set[int] | None = None,
) -> dict[str, Any]:
    # Read the full Markdown document once and parse every scene section with the same prompt.
    if source_document is None:
        source_document = read_markdown_document("\n\n".join((page or "").strip() for page in (pages or [])).encode("utf-8"), file_name="inline.md")

    if source_document.source_type != "markdown":
        raise ValueError("Only Markdown input is supported")

    llm = llm_client or (lambda prompt: call_llm(prompt, step_name="parser_extract"))
    sections = _build_scene_sections(source_document, scene_heading_levels=scene_heading_levels)
    scenes: list[dict[str, Any]] = []
    knowledge: list[dict[str, Any]] = []
    knowledge_counter = 0

    # Split the document by heading structure so plot raw_text stays tied to the original Markdown.
    for scene_index, section in enumerate(sections, start=1):
        raw_text = section["raw_text"].strip()
        if not raw_text:
            continue

        # Run one extraction step for the section and store the JSON fields directly.
        parsed = _parse_json_response(llm(LLM_PROMPT_TEMPLATE.replace("{Content}", raw_text[:10000])))
        scene_payload = parsed.get("scene", {}) if isinstance(parsed.get("scene"), dict) else {}
        parsed_plots = parsed.get("plots", []) if isinstance(parsed.get("plots"), list) else []
        has_scene = _has_scene_payload(scene_payload)
        normalized_plots = [plot for plot in parsed_plots if isinstance(plot, dict) and _has_plot_payload(plot)]
        if has_scene:
            scene_name = _coerce_text(scene_payload.get("scene_name"), section["title"] or f"Scene {scene_index}")
            scene_record = {
                "scene_id": f"scene_{scene_index}",
                "scene_index": scene_index,
                "scene_name": scene_name,
                "scene_goal": _coerce_text(scene_payload.get("scene_goal"), scene_name),
                "scene_description": _coerce_text(scene_payload.get("scene_description"), raw_text[:500] or scene_name),
                "scene_summary": "",
                "status": "pending",
                "plots": [],
            }

            for plot_index, plot_payload in enumerate(normalized_plots, start=1):
                plot_name = _coerce_text(plot_payload.get("plot_name"), f"Plot {plot_index}")
                scene_record["plots"].append(
                    {
                        "plot_id": f"scene_{scene_index}_plot_{plot_index}",
                        "plot_index": plot_index,
                        "plot_name": plot_name,
                        "plot_goal": _coerce_text(plot_payload.get("plot_goal"), plot_name),
                        "raw_text": raw_text,
                        "status": "pending",
                        "progress": 0.0,
                    }
                )

            scenes.append(scene_record)

        for item in parsed.get("knowledge", []) if isinstance(parsed.get("knowledge"), list) else []:
            if not isinstance(item, dict):
                continue
            knowledge_counter += 1
            knowledge.append(
                {
                    "knowledge_id": f"knowledge_{knowledge_counter}",
                    "knowledge_type": _normalize_knowledge_type(item.get("knowledge_type")),
                    "title": _coerce_text(item.get("title"), f"Knowledge {knowledge_counter}"),
                    "content": _coerce_text(item.get("content"), ""),
                }
            )

    script_summary = _build_script_summary(scenes, summary_llm)

    return {
        "scenes": scenes,
        "knowledge": knowledge,
        "script_summary": script_summary,
        "source_metadata": {
            "source_file_name": source_document.source_file_name,
            "source_type": source_document.source_type,
            "line_count": len(source_document.raw_units),
            "heading_outline_preview": list(source_document.outline[:20]),
        },
    }


def _build_scene_sections(
    document: SourceDocument,
    scene_heading_levels: tuple[int, ...] | list[int] | set[int] | None = None,
) -> list[dict[str, Any]]:
    headings = _collect_headings(document.raw_units)
    if not headings:
        raw_text = document.text.strip()
        return [{"title": "Scene 1", "raw_text": raw_text}] if raw_text else []

    allowed_levels = _resolve_scene_heading_levels(
        headings,
        document.raw_units,
        scene_heading_levels=scene_heading_levels,
    )
    scene_starts: list[dict[str, Any]] = []
    for index, heading in enumerate(headings):
        if heading["level"] not in allowed_levels:
            continue
        section_end = len(document.raw_units)
        for next_heading in headings[index + 1 :]:
            if next_heading["level"] <= heading["level"]:
                section_end = next_heading["line_no"] - 1
                break

        direct_body_end = section_end
        for next_heading in headings[index + 1 :]:
            if next_heading["line_no"] > section_end:
                break
            if next_heading["level"] > heading["level"]:
                direct_body_end = next_heading["line_no"] - 1
                break

        if _has_meaningful_markdown_content(document.raw_units, heading["line_no"], direct_body_end):
            scene_starts.append({"line_no": heading["line_no"], "title": heading["title"], "level": heading["level"]})

    if not scene_starts:
        raw_text = document.text.strip()
        return [{"title": "Scene 1", "raw_text": raw_text}] if raw_text else []

    sections: list[dict[str, Any]] = []
    for index, scene_start in enumerate(scene_starts):
        section_start = scene_start["line_no"]
        section_end = scene_starts[index + 1]["line_no"] - 1 if index + 1 < len(scene_starts) else len(document.raw_units)
        raw_text = _slice_lines(document.raw_units, section_start, section_end).strip()
        if not raw_text:
            continue
        sections.append({"title": scene_start["title"], "raw_text": raw_text})
    return sections


def _resolve_scene_heading_levels(
    headings: list[dict[str, Any]],
    raw_units: tuple[str, ...],
    scene_heading_levels: tuple[int, ...] | list[int] | set[int] | None = None,
) -> set[int]:
    if scene_heading_levels is not None:
        normalized_levels = {int(level) for level in scene_heading_levels if 1 <= int(level) <= 6}
        if not normalized_levels:
            raise ValueError("scene_heading_levels must contain Markdown heading levels between 1 and 6")
        return normalized_levels

    candidate_headings = [
        heading
        for index, heading in enumerate(headings)
        if _heading_has_direct_body_content(headings, raw_units, index)
    ]
    if not candidate_headings:
        return {heading["level"] for heading in headings}

    candidate_levels = sorted({heading["level"] for heading in candidate_headings})
    level_counts = Counter(heading["level"] for heading in candidate_headings)

    # Skip a lone document title such as "# Scenario Name" when deeper sections exist.
    if len(candidate_levels) > 1 and level_counts.get(candidate_levels[0], 0) == 1:
        candidate_levels = candidate_levels[1:] or candidate_levels

    # When the document uses many nested heading depths, treat the deepest level as detail text rather than scene splits.
    if len(candidate_levels) > 2:
        candidate_levels = candidate_levels[:-1]

    return set(candidate_levels)


def _heading_has_direct_body_content(
    headings: list[dict[str, Any]],
    raw_units: tuple[str, ...],
    heading_index: int,
) -> bool:
    heading = headings[heading_index]
    section_end = len(raw_units)
    for next_heading in headings[heading_index + 1 :]:
        if next_heading["level"] <= heading["level"]:
            section_end = next_heading["line_no"] - 1
            break

    direct_body_end = section_end
    for next_heading in headings[heading_index + 1 :]:
        if next_heading["line_no"] > section_end:
            break
        if next_heading["level"] > heading["level"]:
            direct_body_end = next_heading["line_no"] - 1
            break

    return _has_meaningful_markdown_content(raw_units, heading["line_no"], direct_body_end)


def _collect_headings(raw_units: tuple[str, ...]) -> list[dict[str, Any]]:
    headings: list[dict[str, Any]] = []
    heading_stack: list[tuple[int, str]] = []
    for line_no, raw_line in enumerate(raw_units, start=1):
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
        headings.append({"line_no": line_no, "level": level, "title": title, "path": tuple(part for _, part in heading_stack)})
    return headings


def _build_plot_spans(
    raw_units: tuple[str, ...],
    headings: list[dict[str, Any]],
    section_start: int,
    section_end: int,
    scene_title: str,
) -> list[dict[str, str]]:
    level5_headings = [heading for heading in headings if heading["level"] == 5 and section_start < heading["line_no"] <= section_end]
    block_starts = [section_start] + [heading["line_no"] for heading in level5_headings]
    blocks: list[dict[str, str | int]] = []
    for block_index, block_start in enumerate(block_starts):
        block_end = block_starts[block_index + 1] - 1 if block_index + 1 < len(block_starts) else section_end
        if block_index == 0:
            blocks.append({"title": scene_title, "kind": "prefix", "raw_text": _slice_lines(raw_units, block_start, block_end)})
        else:
            blocks.append(
                {
                    "title": level5_headings[block_index - 1]["title"],
                    "kind": "heading5",
                    "raw_text": _slice_lines(raw_units, block_start, block_end),
                }
            )

    substantive_heading5 = any(block["kind"] == "heading5" and not _is_auxiliary_plot_title(str(block["title"])) for block in blocks)
    decisions: list[dict[str, str]] = []
    for block_index, block in enumerate(blocks):
        if block["kind"] == "prefix":
            body_lines = [line.strip() for line in str(block["raw_text"]).splitlines()[1:] if line.strip()]
            decisions.append({"role": "plot" if substantive_heading5 and bool(body_lines) else "auxiliary", "attach_to": "" if substantive_heading5 and bool(body_lines) else "next"})
            continue
        if _is_auxiliary_plot_title(str(block["title"])):
            decisions.append({"role": "auxiliary", "attach_to": "previous" if block_index > 0 else "next"})
            continue
        decisions.append({"role": "plot", "attach_to": ""})

    if not any(decision["role"] == "plot" for decision in decisions):
        raw_text = _slice_lines(raw_units, section_start, section_end)
        return [{"title": scene_title, "raw_text": raw_text}]

    groups: list[list[int]] = []
    pending_for_next: list[int] = []
    for block_index, decision in enumerate(decisions):
        if decision["role"] == "plot":
            groups.append(pending_for_next + [block_index])
            pending_for_next = []
            continue
        if decision["attach_to"] == "previous" and groups:
            groups[-1].append(block_index)
        else:
            pending_for_next.append(block_index)
    if pending_for_next and groups:
        groups[-1].extend(pending_for_next)

    plot_spans: list[dict[str, str]] = []
    for group in groups:
        group_blocks = [blocks[index] for index in group]
        titles = [
            str(block["title"])
            for block in group_blocks
            if block["kind"] == "heading5" and not _is_auxiliary_plot_title(str(block["title"]))
        ]
        plot_spans.append(
            {
                "title": titles[0] if titles else str(group_blocks[0]["title"]),
                "raw_text": "\n".join(str(block["raw_text"]).strip() for block in group_blocks if str(block["raw_text"]).strip()),
            }
        )
    return plot_spans or [{"title": scene_title, "raw_text": _slice_lines(raw_units, section_start, section_end)}]


def _build_script_summary(
    scenes: list[dict[str, Any]],
    llm: Callable[[str], str],
) -> str:
    structured_scenes = _build_script_summary_payload(scenes)
    if not structured_scenes:
        return ""

    script_summary = ""
    for batch in _split_script_summary_batches(structured_scenes):
        prompt = SCRIPT_SUMMARY_PROMPT_TEMPLATE.format(
            script_summary=script_summary,
            structured_scene_plot_json=json.dumps(batch, ensure_ascii=False),
        )
        updated_summary = str(llm(prompt) or "").strip()
        if updated_summary:
            script_summary = updated_summary
    return script_summary


def _build_script_summary_payload(scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for scene in scenes:
        payload.append(
            {
                "scene_name": _coerce_text(scene.get("scene_name"), ""),
                "scene_goal": _coerce_text(scene.get("scene_goal"), ""),
                "scene_description": _coerce_text(scene.get("scene_description"), ""),
                "plots": [
                    {
                        "plot_name": _coerce_text(plot.get("plot_name"), ""),
                        "plot_goal": _coerce_text(plot.get("plot_goal"), ""),
                    }
                    for plot in scene.get("plots", [])
                    if isinstance(plot, dict)
                ],
            }
        )
    return payload


def _split_script_summary_batches(
    structured_scenes: list[dict[str, Any]],
    max_chars: int = SCRIPT_SUMMARY_INPUT_LIMIT,
) -> list[list[dict[str, Any]]]:
    batches: list[list[dict[str, Any]]] = []
    current_batch: list[dict[str, Any]] = []

    for scene in structured_scenes:
        candidate_batch = current_batch + [scene]
        candidate_text = json.dumps(candidate_batch, ensure_ascii=False)
        if current_batch and len(candidate_text) > max_chars:
            batches.append(current_batch)
            current_batch = [scene]
            continue
        current_batch = candidate_batch

    if current_batch:
        batches.append(current_batch)
    return batches


def _slice_lines(raw_units: tuple[str, ...], first_row: int, last_row: int) -> str:
    if not raw_units or first_row > last_row:
        return ""
    start = max(1, first_row) - 1
    end = min(len(raw_units), last_row)
    return "\n".join(raw_units[start:end])


def _has_meaningful_markdown_content(raw_units: tuple[str, ...], first_row: int, last_row: int) -> bool:
    if first_row > last_row:
        return False
    for line in raw_units[first_row - 1 : last_row]:
        stripped = line.strip()
        if stripped and not _is_decorative_markdown_line(stripped):
            return True
    return False


def _parse_json_response(text: str) -> dict[str, Any]:
    json_text = _extract_json_text(text)
    parsed = json5.loads(json_text)
    if not isinstance(parsed, dict):
        raise ValueError("LLM response root must be a JSON object")
    return parsed


def _extract_json_text(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fenced:
        return fenced.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1].strip()
    return text.strip()


def _normalize_knowledge_type(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in KNOWLEDGE_TYPES else "other"


def _has_scene_payload(value: dict[str, Any]) -> bool:
    if not isinstance(value, dict):
        return False
    return any(str(value.get(key, "")).strip() for key in ("scene_name", "scene_goal", "scene_description"))


def _has_plot_payload(value: dict[str, Any]) -> bool:
    if not isinstance(value, dict):
        return False
    return any(str(value.get(key, "")).strip() for key in ("plot_name", "plot_goal"))


def _coerce_text(value: Any, default: str) -> str:
    text = str(value or "").strip()
    return text if text else default


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
    return "<img" in stripped_line.lower() and stripped_line.startswith("<")


def _normalize_markdown_heading(text: str) -> str:
    normalized = re.sub(r"\s+", " ", (text or "").strip())
    normalized = normalized.strip("# ").strip()
    return normalized or "Untitled"
