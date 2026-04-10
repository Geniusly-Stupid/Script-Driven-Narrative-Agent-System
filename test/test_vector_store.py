import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.vector_store import ChromaStore


def main() -> int:
    try:
        print('[test_vector_store] input: initialize ChromaStore(path=test/.chroma_test)')
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
                    }
                ],
            }
        ]
        knowledge = [
            {
                'knowledge_id': 'knowledge_1',
                'knowledge_type': 'setting',
                'title': 'Temple History',
                'content': 'The temple was sealed after a faction conflict.',
            },
            {
                'knowledge_id': 'knowledge_2',
                'knowledge_type': 'other',
                'title': 'Hidden Truth',
                'content': 'The archivist forged records to hide a pact.',
            },
        ]

        print('[test_vector_store] input: scenes ->', scenes)
        print('[test_vector_store] input: knowledge ->', knowledge)
        store.add_from_scenes(scenes, knowledge=knowledge)

        query = 'Who knows about the temple history and faction conflict'
        print('[test_vector_store] input: query ->', query)
        result = store.search(query, k=5)
        print('[test_vector_store] output: search result ->', result)

        assert result, 'search should return documents'
        assert any(r.get('metadata', {}).get('type') == 'setting' for r in result), 'setting docs should be searchable'

        print('[test_vector_store] result: PASS')
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f'[test_vector_store] result: FAIL -> {exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
