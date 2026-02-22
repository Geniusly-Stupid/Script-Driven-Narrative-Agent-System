import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.vector_store import ChromaStore


def main() -> int:
    try:
        print('[test_vector_store] 输入: 初始化 ChromaStore(path=test/.chroma_test)')
        store = ChromaStore(path=str(ROOT / 'test' / '.chroma_test'))
        store.reset()

        scenes = [
            {
                'scene_id': 's1',
                'plots': [
                    {
                        'plot_id': 'p1',
                        'npc': ['Archivist'],
                        'locations': ['Temple'],
                        'mandatory_events': ['发现地图'],
                    }
                ],
            }
        ]
        print('[test_vector_store] 输入: scenes ->', scenes)
        store.add_from_scenes(scenes)

        query = '谁知道神殿地图信息'
        print('[test_vector_store] 输入: query ->', query)
        result = store.search(query, k=3)
        print('[test_vector_store] 输出: search result ->', result)

        print('[test_vector_store] 结果: PASS')
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f'[test_vector_store] 结果: FAIL -> {exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
