import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.rules_loader import load_game_rules_knowledge
from app.vector_store import ChromaStore


def main() -> int:
    try:
        print('[test_rules_loader] input: load GameRules.md via rules loader')
        chunks = load_game_rules_knowledge()
        print(f'[test_rules_loader] output: loaded chunks -> {len(chunks)}')

        assert chunks, 'rules loader should produce at least one chunk'
        assert any(item.get('knowledge_type') == 'rule' for item in chunks), 'chunks should be tagged as rule knowledge'
        assert any(item.get('title') for item in chunks), 'chunks should preserve section titles'

        store_path = ROOT / 'test' / '.chroma_rules_test'
        print(f'[test_rules_loader] input: initialize ChromaStore(path={store_path})')
        store = ChromaStore(path=str(store_path))
        store.reset()

        print('[test_rules_loader] input: insert rule chunks into vector store')
        store.add_from_scenes([], knowledge=chunks)

        query = 'sanity points and magic points'
        print('[test_rules_loader] input: query ->', query)
        result = store.search(query, k=5)
        print('[test_rules_loader] output: search result ->', result)

        assert result, 'search should return indexed rule chunks'
        assert any(r.get('metadata', {}).get('type') == 'rule' for r in result), 'retrieval should include rule documents'
        assert any(
            'sanity' in r.get('content', '').lower() or 'magic points' in r.get('content', '').lower()
            for r in result
        ), 'retrieval should surface relevant Call of Cthulhu rules content'

        print('[test_rules_loader] result: PASS')
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f'[test_rules_loader] result: FAIL -> {exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
