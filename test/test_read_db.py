import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.database import Database


def main() -> int:
    db_path = ROOT / "test" / "debug_read_db.db"
    output_path = ROOT / "test" / "debug_read_db_dump.txt"
    print(f"[test_read_db] input: db_path={db_path}")

    try:
        if db_path.exists():
            os.remove(db_path)
        if output_path.exists():
            os.remove(output_path)

        db = Database(str(db_path))
        db.insert_scenes(
            [
                {
                    "scene_id": "scene_1",
                    "scene_index": 1,
                    "scene_name": "Opening Investigation",
                    "scene_goal": "Open the mystery",
                    "scene_description": "Scene one description.",
                    "scene_summary": "Scene one summary.",
                    "status": "in_progress",
                    "plots": [
                        {
                            "plot_id": "scene_1_plot_1",
                            "plot_index": 1,
                            "plot_name": "First beat",
                            "plot_goal": "Inspect the foyer",
                            "raw_text": "Line 10\nLine 11\nLine 12",
                            "status": "in_progress",
                            "progress": 0.0,
                        },
                        {
                            "plot_id": "scene_1_plot_2",
                            "plot_index": 2,
                            "plot_name": "Second beat",
                            "plot_goal": "Search the study",
                            "raw_text": "Line 13\nLine 14",
                            "status": "pending",
                            "progress": 0.0,
                        },
                    ],
                }
            ]
        )
        db.insert_knowledge(
            [
                {
                    "knowledge_id": "knowledge_1",
                    "knowledge_type": "clue",
                    "title": "Ledger",
                    "content": "A damaged ledger was hidden behind the coats.",
                }
            ]
        )
        db.append_memory("scene_1", "scene_1_plot_1", "hello", "world", visit_id=3)
        db.update_system_state(
            {
                "current_scene_id": "scene_1",
                "current_plot_id": "scene_1_plot_1",
                "navigation_state": {
                    "latest_transition": {
                        "source_scene_id": "",
                        "source_plot_id": "",
                        "target_scene_id": "scene_1",
                        "target_plot_id": "scene_1_plot_1",
                        "relationship": "scene_opening",
                    }
                },
                "current_visit_id": 3,
            }
        )
        db.save_summary(
            "parse_source_meta",
            '{"source_type":"markdown","line_count":20,"heading_outline_preview":["Book > Scene 1"]}',
        )
        db.close()

        command = [
            sys.executable,
            str(ROOT / "test" / "read_db.py"),
            "--db",
            str(db_path),
            "--output",
            str(output_path),
        ]
        print(f"[test_read_db] input: command={command}")
        subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8")
        assert output_path.exists(), "read_db should create txt output"

        dump_text = output_path.read_text(encoding="utf-8")
        assert "=== PARSE SOURCE METADATA ===" in dump_text
        assert "scene_name" in dump_text
        assert "plot_name" in dump_text
        assert "raw_text" in dump_text
        assert "knowledge_type" in dump_text
        assert "line_count" in dump_text
        assert "runtime_state" in dump_text
        assert "source_page_start" not in dump_text
        assert "source_page_end" not in dump_text
        assert "node_kind" not in dump_text
        assert "navigation" not in dump_text
        assert "=== SPAN AUDIT ===" not in dump_text
        assert "parse_mode" not in dump_text
        assert "parse_structure" not in dump_text
        assert "parse_warnings" not in dump_text

        print("[test_read_db] result: PASS")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[test_read_db] result: FAIL -> {exc}")
        return 1
    finally:
        if db_path.exists():
            os.remove(db_path)
        if output_path.exists():
            os.remove(output_path)


if __name__ == "__main__":
    raise SystemExit(main())
