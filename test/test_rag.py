import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.rag import categorize_docs, generate_retrieval_queries


def main() -> int:
    try:
        user_input = 'I want to inspect the alley behind the market.'
        plot_goal = 'Find the suspect route'
        history = [
            {'user': 'I head to the market', 'agent': 'You arrive at the market.'},
            {'user': 'I ask a vendor', 'agent': 'The vendor points to a narrow alley.'},
        ]

        print('[test_rag] input:')
        print('  user_input=', user_input)
        print('  plot_goal=', plot_goal)
        print('  history=', history)

        queries = generate_retrieval_queries(user_input, plot_goal, history)
        print('[test_rag] output: retrieval queries ->', queries)

        docs = [
            {'content': 'Witness saw cloaked figure near market.', 'metadata': {'type': 'npc'}},
            {'content': 'Harbor has hidden warehouse.', 'metadata': {'type': 'location'}},
            {'content': 'Stealth checks use d100.', 'metadata': {'type': 'rule'}},
            {'content': 'Map shows tunnel route.', 'metadata': {'type': 'event'}},
            {
                'content': 'Town council controls public records and suppresses rumors.',
                'metadata': {'type': 'world_context', 'knowledge_type': 'background'},
            },
            {
                'content': 'Victim was silenced after uncovering the faction pact.',
                'metadata': {'type': 'world_context', 'knowledge_type': 'truth'},
            },
        ]
        print('[test_rag] input: docs ->', docs)
        categorized = categorize_docs(docs)
        print('[test_rag] output: categorized ->', categorized)

        assert categorized['world_context_info'] != 'None', 'world_context bucket should not be empty'
        assert categorized['truth_related_info'] != 'None', 'truth bucket should not be empty'

        print('[test_rag] result: PASS')
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f'[test_rag] result: FAIL -> {exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
