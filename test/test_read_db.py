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
                    "scene_goal": "Open the mystery",
                    "scene_description": "Scene one description.",
                    "scene_summary": "Scene one summary.",
                    "source_page_start": 10,
                    "source_page_end": 14,
                    "status": "pending",
                    "plots": [
                        {
                            "plot_id": "scene_1_plot_1",
                            "plot_goal": "First beat",
                            "mandatory_events": ["Event A"],
                            "npc": ["Thomas"],
                            "locations": ["House"],
                            "raw_text": "Line 10\nLine 11\nLine 12",
                            "source_page_start": 10,
                            "source_page_end": 12,
                            "status": "pending",
                            "progress": 0.0,
                        },
                        {
                            "plot_id": "scene_1_plot_2",
                            "plot_goal": "Second beat",
                            "mandatory_events": ["Event B"],
                            "npc": ["Jefferson"],
                            "locations": ["Cemetery"],
                            "raw_text": "Line 13\nLine 14",
                            "source_page_start": 13,
                            "source_page_end": 14,
                            "status": "pending",
                            "progress": 0.0,
                        },
                    ],
                }
            ]
        )
        db.save_summary(
            "parse_source_meta",
            '{"source_type":"markdown","source_unit_label":"line","display_start":1,"display_end":20}',
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
        completed = subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8")
        print("[test_read_db] output: stdout captured")
        assert output_path.exists(), "read_db should create txt output"

        dump_text = output_path.read_text(encoding="utf-8")
        assert "=== SPAN AUDIT ===" in dump_text, "span audit section missing"
        assert "raw_text" in dump_text, "plot raw_text should be shown"
        assert "canonical_text" not in dump_text, "canonical_text should not be shown anymore"
        assert "compression_meta" not in dump_text, "compression_meta should not be shown anymore"
        assert '"issues": []' in dump_text, "healthy sample should have no audit issues"
        assert "line_count" in dump_text, "line counts should be shown"

        print("[test_read_db] result: PASS")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[test_read_db] result: FAIL -> {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
