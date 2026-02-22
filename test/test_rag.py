import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.rag import categorize_docs, generate_retrieval_queries


def main() -> int:
    try:
        user_input = '我想调查市场和港口之间的通道'
        plot_goal = '找到嫌疑人的藏身点'
        mandatory_events = ['询问证人', '检查地图']
        history = [{'user': '我先去市场', 'agent': '你到了市场'}, {'user': '我问路人', 'agent': '路人说见过黑袍人'}]

        print('[test_rag] 输入:')
        print('  user_input=', user_input)
        print('  plot_goal=', plot_goal)
        print('  mandatory_events=', mandatory_events)
        print('  history=', history)

        queries = generate_retrieval_queries(user_input, plot_goal, mandatory_events, history)
        print('[test_rag] 输出: retrieval queries ->', queries)

        docs = [
            {'content': 'Witness saw cloaked figure near market.', 'metadata': {'type': 'npc'}},
            {'content': 'Harbor has hidden warehouse.', 'metadata': {'type': 'location'}},
            {'content': 'Stealth checks use d100.', 'metadata': {'type': 'rule'}},
            {'content': 'Map shows tunnel route.', 'metadata': {'type': 'event'}},
        ]
        print('[test_rag] 输入: docs ->', docs)
        categorized = categorize_docs(docs)
        print('[test_rag] 输出: categorized ->', categorized)

        print('[test_rag] 结果: PASS')
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f'[test_rag] 结果: FAIL -> {exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
