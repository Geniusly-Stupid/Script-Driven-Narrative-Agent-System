import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.vector_store.chroma_store import chroma_store


def main() -> int:
    print('[test_vector_store] START')
    try:
        print('[test_vector_store] Resetting collection...')
        chroma_store.reset()

        print('[test_vector_store] Adding sample documents...')
        docs = [
            {'type': 'npc', 'name': 'Elder', 'description': 'The village elder knows ancient lore.', 'metadata': {'scene_id': 's1', 'plot_id': 'p1'}},
            {'type': 'location', 'name': 'Ruins', 'description': 'The ruins are filled with traps.', 'metadata': {'scene_id': 's1', 'plot_id': 'p1'}},
            {'type': 'rule', 'name': 'Stealth Rule', 'description': 'Stealth checks use a d100 roll.', 'metadata': {'scene_id': 's1', 'plot_id': 'p1'}},
        ]
        chroma_store.add_documents(docs)

        print('[test_vector_store] Querying top-k results...')
        results = chroma_store.search('How does stealth check work in ruins?', k=3)
        for idx, item in enumerate(results, start=1):
            print(f"  {idx}. type={item['metadata'].get('type')} name={item['metadata'].get('name')} dist={item.get('distance')}")
            print(f"     content={item['content']}")

        print('[test_vector_store] Resetting collection for cleanup...')
        chroma_store.reset()

        print('[test_vector_store] SUCCESS')
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f'[test_vector_store] FAILED: {exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
