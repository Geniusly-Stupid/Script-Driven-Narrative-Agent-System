import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    print('[test_init] 输入: 导入 app 包')
    try:
        import app  # noqa: F401
        print('[test_init] 输出: app 包导入成功')
        print('[test_init] 结果: PASS')
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f'[test_init] 结果: FAIL -> {exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
