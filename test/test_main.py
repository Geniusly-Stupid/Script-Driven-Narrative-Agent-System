import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    print('[test_main] 输入: 导入 main 模块并检查 run_app')
    try:
        import main as app_main
        ok = hasattr(app_main, 'run_app')
        print('[test_main] 输出: has run_app ->', ok)
        print('[test_main] 结果:', 'PASS' if ok else 'FAIL')
        return 0 if ok else 1
    except Exception as exc:  # noqa: BLE001
        print(f'[test_main] 结果: FAIL -> {exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
