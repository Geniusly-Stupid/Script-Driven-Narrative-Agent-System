import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = Path(__file__).resolve().parent

TEST_SCRIPTS = [
    'test_database.py',
    'test_vector_store.py',
    'test_script_parser.py',
    'test_agent_graph.py',
    'test_rag_pipeline.py',
    'test_state_progression.py',
]


def main() -> int:
    print('[run_all_tests] START')
    results: list[tuple[str, int]] = []

    for script in TEST_SCRIPTS:
        path = TEST_DIR / script
        print(f'\n[run_all_tests] Running {script}...')
        proc = subprocess.run([sys.executable, str(path)], cwd=str(ROOT))
        results.append((script, proc.returncode))

    print('\n[run_all_tests] SUMMARY')
    for name, code in results:
        status = 'PASS' if code == 0 else 'FAIL'
        print(f'  - {name}: {status} (exit={code})')

    failed = [name for name, code in results if code != 0]
    if failed:
        print(f"[run_all_tests] FAILED SCRIPTS: {', '.join(failed)}")
        return 1

    print('[run_all_tests] ALL TESTS PASSED')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
