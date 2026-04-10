import argparse
import json
import sqlite3
import sys
from pathlib import Path


def _safe_json_loads(value: str):
    try:
        return json.loads(value)
    except Exception:
        return value


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

    parse_source_meta_row = cur.execute(
        "SELECT content FROM summaries WHERE summary_type='parse_source_meta' AND scene_id='' AND plot_id=''"
    ).fetchone()
    parse_source_meta = _safe_json_loads(parse_source_meta_row["content"]) if parse_source_meta_row else {}
    if not isinstance(parse_source_meta, dict):
        parse_source_meta = {}

    scene_rows = [dict(row) for row in cur.execute("SELECT * FROM scenes ORDER BY scene_index, scene_id").fetchall()]
    plot_rows = [dict(row) for row in cur.execute("SELECT * FROM plots ORDER BY scene_id, plot_index, plot_id").fetchall()]
    knowledge_rows = [dict(row) for row in cur.execute("SELECT * FROM knowledge_base ORDER BY knowledge_id").fetchall()]
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

    plots_by_scene: dict[str, list[dict]] = {}
    for plot in plot_rows:
        plots_by_scene.setdefault(str(plot.get("scene_id", "")), []).append(plot)

    emit(f"DB: {db_path}")
    emit(f"OUTPUT TXT: {output_path}")

    emit("\n=== TABLE COUNTS ===")
    emit(counts)

    emit("\n=== PARSE SOURCE METADATA ===")
    emit(parse_source_meta)

    state = cur.execute("SELECT * FROM system_state WHERE id = 1").fetchone()
    emit("\n=== SYSTEM STATE ===")
    if state:
        state_data = dict(state)
        state_data["player_profile"] = _safe_json_loads(state_data.get("player_profile", "{}") or "{}")
        state_data["runtime_state"] = _safe_json_loads(state_data.get("navigation_state_json", "{}") or "{}")
        state_data["current_scene_intro"] = _display_text(state_data.get("current_scene_intro", ""), args.full_text)
        state_data.pop("navigation_state_json", None)
        emit(state_data)
    else:
        emit("No system_state row found (id=1).")

    emit("\n=== SCENES + PLOTS ===")
    if not scene_rows:
        emit("No scenes.")
    for scene in scene_rows:
        scene_plots = plots_by_scene.get(str(scene.get("scene_id", "")), [])
        emit(
            {
                "scene_id": scene.get("scene_id", ""),
                "scene_index": scene.get("scene_index", ""),
                "scene_name": scene.get("scene_name", ""),
                "scene_goal": scene.get("scene_goal", ""),
                "scene_description": _display_text(scene.get("scene_description", ""), args.full_text),
                "scene_summary": _display_text(scene.get("scene_summary", ""), args.full_text),
                "status": scene.get("status", ""),
                "plot_count": len(scene_plots),
            }
        )
        for plot in scene_plots:
            emit(
                {
                    "plot_id": plot.get("plot_id", ""),
                    "scene_id": plot.get("scene_id", ""),
                    "plot_index": plot.get("plot_index", ""),
                    "plot_name": plot.get("plot_name", ""),
                    "plot_goal": plot.get("plot_goal", ""),
                    "raw_text": _display_text(plot.get("raw_text", ""), args.full_text),
                    "status": plot.get("status", ""),
                    "progress": plot.get("progress", 0.0),
                }
            )

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

    emit("\n=== RECENT SUMMARIES ===")
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

    output_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
