import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.pdf_parser import parse_script_lines


def main() -> int:
    print('[test_script_parser] START')
    try:
        print('[test_script_parser] Using sample script text (no frontend/server required)...')
        page1 = """
KP: Welcome to the story.
玩家A：我调查房间。
A cold wind passes through the hall.
"""
        page2 = """
玩家B: I guard the door.
Narration line without role prefix.
KP：Roll for perception.
"""

        lines = []
        lines.extend(parse_script_lines(page1, page_num=1))
        lines.extend(parse_script_lines(page2, page_num=2))

        print('[test_script_parser] First 10 structured lines:')
        for idx, line in enumerate(lines[:10], start=1):
            print(f'  {idx}. page={line.page} role={line.role} text={line.text}')

        print('[test_script_parser] Validating page tracking...')
        if not any(line.page == 1 for line in lines) or not any(line.page == 2 for line in lines):
            raise RuntimeError('Page tracking validation failed.')

        print('[test_script_parser] SUCCESS')
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f'[test_script_parser] FAILED: {exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
