import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / 'test'

SCRIPTS = [
    'test_init.py',
    'test_database.py',
    'test_parser.py',
    'test_rag.py',
    'test_state.py',
    'test_vector_store.py',
    'test_agent_graph.py',
    'test_main.py',
]


def main() -> int:
    print('[run_all] START')
    results = []
    for script in SCRIPTS:
        print(f'\n[run_all] 运行 {script}')
        code = subprocess.run([sys.executable, str(TEST_DIR / script)], cwd=str(ROOT)).returncode
        results.append((script, code))

    print('\n[run_all] SUMMARY')
    for name, code in results:
        print(f'  {name}: {"PASS" if code == 0 else "FAIL"} ({code})')

    return 0 if all(code == 0 for _, code in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
