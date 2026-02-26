import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.parser import parse_script


def main() -> int:
    mock_pages = [
        "Scene: A small village at dawn. Li Ming meets a traveler.\nThey decide to head to the river.",
        "At the river, a hidden chest is discovered.\nA shadowy figure watches from the trees.",
    ]

    print("[test_parser] 输入: mock_pages")
    for i, p in enumerate(mock_pages, start=1):
        print(f"  page{i}: {p}")

    try:
        scenes = parse_script(mock_pages, pages_per_scene=2)
        print("[test_parser] 输出: 解析结果 scenes ->")
        for s in scenes:
            print(f"  scene_id={s['scene_id']} scene_goal={s['scene_goal']}")
            for p in s["plots"]:
                print(f"    plot_id={p['plot_id']} plot_goal={p['plot_goal']}")
                print(f"    mandatory_events={p['mandatory_events']}")
                print(f"    npc={p['npc']} locations={p['locations']}")

        print("[test_parser] 结果: PASS")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[test_parser] 结果: FAIL -> {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
