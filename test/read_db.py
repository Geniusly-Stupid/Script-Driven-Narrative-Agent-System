import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path


def _safe_json_loads(value: str):
    try:
        return json.loads(value)
    except Exception:
        return value


def _scene_sort_key(scene_id: str):
    if scene_id.startswith("scene_"):
        tail = scene_id.split("_")[-1]
        if tail.isdigit():
            return (0, int(tail))
    return (1, scene_id)


def _truncate(text: str, max_len: int = 220) -> str:
    normalized = (text or "").replace("\r\n", "\n").strip()
    if len(normalized) <= max_len:
        return normalized
    return normalized[: max_len - 3] + "..."


def _display_text(text: str, full_text: bool) -> str:
    return text if full_text else _truncate(text)


def _format_obj(value) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)


def _span_count(start: int, end: int) -> int:
    if start > end:
        return 0
    return end - start + 1


def _unit_title(unit_label: str) -> str:
    return "Line" if unit_label == "line" else "Page"


def _sort_by_span(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: (
            int(row.get("source_page_start", 0)),
            int(row.get("source_page_end", 0)),
            str(row.get("scene_id") or row.get("plot_id") or ""),
        ),
    )


def _build_span_audit(scene_rows: list[dict], plot_map: dict[str, list[dict]], unit_label: str) -> dict:
    issues: list[str] = []
    ordered_scenes = _sort_by_span(scene_rows)

    previous_scene = None
    for scene in ordered_scenes:
        start = int(scene.get("source_page_start", 0))
        end = int(scene.get("source_page_end", 0))
        if previous_scene is not None:
            prev_end = int(previous_scene.get("source_page_end", 0))
            if start <= prev_end:
                issues.append(
                    f"scene overlap: {previous_scene.get('scene_id')} ({previous_scene.get('source_page_start')}-{prev_end}) -> "
                    f"{scene.get('scene_id')} ({start}-{end})"
                )
            elif start > prev_end + 1:
                issues.append(
                    f"scene gap: {previous_scene.get('scene_id')} ends at {prev_end}, {scene.get('scene_id')} starts at {start}"
                )
        previous_scene = scene

        plots = _sort_by_span(plot_map.get(scene.get("scene_id", ""), []))
        if not plots:
            issues.append(f"scene missing plots: {scene.get('scene_id')}")
            continue

        previous_plot_end = start - 1
        for plot in plots:
            plot_start = int(plot.get("source_page_start", 0))
            plot_end = int(plot.get("source_page_end", 0))
            if plot_start < start or plot_end > end:
                issues.append(
                    f"plot outside scene: {plot.get('plot_id')} ({plot_start}-{plot_end}) not within {scene.get('scene_id')} ({start}-{end})"
                )
            if plot_start <= previous_plot_end:
                issues.append(
                    f"plot overlap in {scene.get('scene_id')}: previous ends {previous_plot_end}, {plot.get('plot_id')} starts {plot_start}"
                )
            elif plot_start > previous_plot_end + 1:
                issues.append(
                    f"plot gap in {scene.get('scene_id')}: expected {previous_plot_end + 1}, got {plot_start}"
                )
            previous_plot_end = plot_end

        if previous_plot_end < end:
            issues.append(
                f"scene trailing gap: {scene.get('scene_id')} ends at {end}, last plot ends at {previous_plot_end}"
            )

    return {
        "unit_label": unit_label,
        "unit_title": _unit_title(unit_label),
        "issues": issues,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect narrative SQLite database content.")
    parser.add_argument("--db", default="narrative.db", help="Path to SQLite database file.")
    parser.add_argument("--memory-limit", type=int, default=20, help="How many recent memory rows to show.")
    parser.add_argument("--summary-limit", type=int, default=20, help="How many summary rows to show.")
    parser.add_argument("--full-text", action="store_true", help="Show full text fields without truncation.")
    parser.add_argument(
        "--output",
        help="Path to txt output file. Defaults to <db-stem>_dump.txt next to the database file.",
    )
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    output_lines = []
    stdout_encoding = (getattr(sys.stdout, "encoding", None) or "").lower()

    db_path = Path(args.db).resolve()
    output_path = Path(args.output).resolve() if args.output else db_path.with_name(f"{db_path.stem}_dump.txt")

    def emit(value=""):
        text = _format_obj(value)
        if "utf" in stdout_encoding:
            print(text)
        else:
            sys.stdout.buffer.write((text + "\n").encode("utf-8", errors="replace"))
        output_lines.append(text)

    tables = ["scenes", "plots", "knowledge_base", "memory", "summaries", "system_state"]
    counts = {}
    for table in tables:
        try:
            counts[table] = cur.execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()["c"]
        except Exception:
            counts[table] = "N/A"

    parse_mode_row = cur.execute(
        "SELECT content FROM summaries WHERE summary_type='parse_mode' AND scene_id='' AND plot_id=''"
    ).fetchone()
    parse_structure_row = cur.execute(
        "SELECT content FROM summaries WHERE summary_type='parse_structure' AND scene_id='' AND plot_id=''"
    ).fetchone()
    parse_warnings_row = cur.execute(
        "SELECT content FROM summaries WHERE summary_type='parse_warnings' AND scene_id='' AND plot_id=''"
    ).fetchone()
    parse_source_meta_row = cur.execute(
        "SELECT content FROM summaries WHERE summary_type='parse_source_meta' AND scene_id='' AND plot_id=''"
    ).fetchone()

    parse_source_meta = _safe_json_loads(parse_source_meta_row["content"]) if parse_source_meta_row else {}
    if not isinstance(parse_source_meta, dict):
        parse_source_meta = {}
    unit_label = str(parse_source_meta.get("source_unit_label", "page")).strip().lower() or "page"
    unit_title = _unit_title(unit_label)

    scene_rows = [dict(row) for row in cur.execute("SELECT * FROM scenes").fetchall()]
    plot_rows = [dict(row) for row in cur.execute("SELECT * FROM plots").fetchall()]
    knowledge_rows = [
        dict(row)
        for row in cur.execute(
            "SELECT * FROM knowledge_base ORDER BY source_page_start, source_page_end, knowledge_id"
        ).fetchall()
    ]
    memory_rows = [
        dict(row)
        for row in cur.execute(
            "SELECT id, scene_id, plot_id, user, agent, visit_id, timestamp FROM memory ORDER BY id DESC LIMIT ?",
            (max(0, args.memory_limit),),
        ).fetchall()
    ]
    summary_rows = [
        dict(row)
        for row in cur.execute(
            "SELECT id, summary_type, scene_id, plot_id, content FROM summaries ORDER BY id DESC LIMIT ?",
            (max(0, args.summary_limit),),
        ).fetchall()
    ]

    for plot in plot_rows:
        plot["mandatory_events"] = _safe_json_loads(plot.get("mandatory_events", "[]") or "[]")
        plot["npc"] = _safe_json_loads(plot.get("npc", "[]") or "[]")
        plot["locations"] = _safe_json_loads(plot.get("locations", "[]") or "[]")
        plot["navigation"] = _safe_json_loads(plot.get("navigation_json", "{}") or "{}")

    for scene in scene_rows:
        scene["navigation"] = _safe_json_loads(scene.get("navigation_json", "{}") or "{}")

    plot_map = defaultdict(list)
    for plot in plot_rows:
        plot_map[plot["scene_id"]].append(plot)

    for scene_id in plot_map:
        plot_map[scene_id] = _sort_by_span(plot_map[scene_id])

    scene_rows.sort(key=lambda row: _scene_sort_key(row.get("scene_id", "")))
    audit = _build_span_audit(scene_rows, plot_map, unit_label)

    emit(f"DB: {db_path}")
    emit(f"OUTPUT TXT: {output_path}")

    emit("\n=== TABLE COUNTS ===")
    emit(counts)

    emit("\n=== PARSE METADATA ===")
    emit({"parse_mode": parse_mode_row["content"] if parse_mode_row else ""})
    emit({"parse_source_meta": parse_source_meta})
    emit({"structure": _safe_json_loads(parse_structure_row["content"]) if parse_structure_row else {}})
    emit({"warnings": _safe_json_loads(parse_warnings_row["content"]) if parse_warnings_row else []})

    state = cur.execute("SELECT * FROM system_state WHERE id = 1").fetchone()
    emit("\n=== SYSTEM STATE ===")
    if state:
        state_data = dict(state)
        state_data["player_profile"] = _safe_json_loads(state_data.get("player_profile", "{}") or "{}")
        state_data["navigation_state"] = _safe_json_loads(state_data.get("navigation_state_json", "{}") or "{}")
        state_data["current_scene_intro"] = _display_text(state_data.get("current_scene_intro", ""), args.full_text)
        emit(state_data)
    else:
        emit("No system_state row found (id=1).")

    emit("\n=== SCENES + PLOTS ===")
    if not scene_rows:
        emit("No scenes.")
    for scene in scene_rows:
        scene_start = int(scene.get("source_page_start", 0))
        scene_end = int(scene.get("source_page_end", 0))
        scene_plots = plot_map.get(scene.get("scene_id", ""), [])
        emit(
            {
                "scene_id": scene.get("scene_id", ""),
                "scene_goal": scene.get("scene_goal", ""),
                f"{unit_label}_start": scene_start,
                f"{unit_label}_end": scene_end,
                f"{unit_label}_count": _span_count(scene_start, scene_end),
                "scene_description": _display_text(scene.get("scene_description", ""), args.full_text),
                "scene_summary": _display_text(scene.get("scene_summary", ""), args.full_text),
                "status": scene.get("status", ""),
                "node_kind": scene.get("node_kind", ""),
                "navigation": scene.get("navigation", {}),
                "plot_count": len(scene_plots),
            }
        )
        for plot in scene_plots:
            plot_start = int(plot.get("source_page_start", 0))
            plot_end = int(plot.get("source_page_end", 0))
            emit(
                {
                    "plot_id": plot.get("plot_id", ""),
                    "scene_id": plot.get("scene_id", ""),
                    "plot_goal": plot.get("plot_goal", ""),
                    f"{unit_label}_start": plot_start,
                    f"{unit_label}_end": plot_end,
                    f"{unit_label}_count": _span_count(plot_start, plot_end),
                    "mandatory_events": plot.get("mandatory_events", []),
                    "npc": plot.get("npc", []),
                    "locations": plot.get("locations", []),
                    "raw_text": _display_text(plot.get("raw_text", ""), args.full_text),
                    "status": plot.get("status", ""),
                    "progress": plot.get("progress", 0.0),
                    "node_kind": plot.get("node_kind", ""),
                    "navigation": plot.get("navigation", {}),
                }
            )

    emit("\n=== SPAN AUDIT ===")
    if not scene_rows:
        emit("No spans to audit.")
    for scene in _sort_by_span(scene_rows):
        scene_start = int(scene.get("source_page_start", 0))
        scene_end = int(scene.get("source_page_end", 0))
        scene_plots = _sort_by_span(plot_map.get(scene.get("scene_id", ""), []))
        emit(
            {
                "scene_id": scene.get("scene_id", ""),
                f"{unit_title.lower()}_span": f"{scene_start}-{scene_end}",
                f"{unit_title.lower()}_count": _span_count(scene_start, scene_end),
                "plot_count": len(scene_plots),
            }
        )
        for plot in scene_plots:
            plot_start = int(plot.get("source_page_start", 0))
            plot_end = int(plot.get("source_page_end", 0))
            emit(
                {
                    "plot_id": plot.get("plot_id", ""),
                    f"{unit_title.lower()}_span": f"{plot_start}-{plot_end}",
                    f"{unit_title.lower()}_count": _span_count(plot_start, plot_end),
                }
            )

    emit({"issues": audit["issues"] if audit["issues"] else []})

    emit("\n=== KNOWLEDGE BASE ===")
    if not knowledge_rows:
        emit("No knowledge records.")
    for knowledge in knowledge_rows:
        emit(
            {
                "knowledge_id": knowledge.get("knowledge_id", ""),
                "knowledge_type": knowledge.get("knowledge_type", ""),
                "title": knowledge.get("title", ""),
                "content": _display_text(knowledge.get("content", ""), args.full_text),
                f"{unit_label}_start": knowledge.get("source_page_start", ""),
                f"{unit_label}_end": knowledge.get("source_page_end", ""),
                "metadata": _safe_json_loads(knowledge.get("metadata", "{}") or "{}"),
            }
        )

    emit("\n=== RECENT MEMORY ===")
    if not memory_rows:
        emit("No memory rows.")
    for memory in reversed(memory_rows):
        emit(
            {
                "id": memory.get("id", ""),
                "scene_id": memory.get("scene_id", ""),
                "plot_id": memory.get("plot_id", ""),
                "user": _display_text(memory.get("user", ""), args.full_text),
                "agent": _display_text(memory.get("agent", ""), args.full_text),
                "visit_id": memory.get("visit_id", ""),
                "timestamp": memory.get("timestamp", ""),
            }
        )

    emit("\n=== SUMMARIES ===")
    if not summary_rows:
        emit("No summary rows.")
    for summary in reversed(summary_rows):
        emit(
            {
                "id": summary.get("id", ""),
                "summary_type": summary.get("summary_type", ""),
                "scene_id": summary.get("scene_id", ""),
                "plot_id": summary.get("plot_id", ""),
                "content": _display_text(summary.get("content", ""), args.full_text),
            }
        )

    conn.close()
    emit(f"\nSaved report to: {output_path}")
    output_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
