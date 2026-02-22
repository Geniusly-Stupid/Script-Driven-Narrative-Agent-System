import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent_graph import NarrativeAgent
from app.database import Database
from app.vector_store import ChromaStore


def main() -> int:
    db_path = ROOT / 'test' / 'debug_agent.db'
    chroma_path = ROOT / 'test' / '.chroma_agent'
    try:
        if db_path.exists():
            db_path.unlink()

        db = Database(str(db_path))
        store = ChromaStore(path=str(chroma_path))
        store.reset()

        scenes = [
            {
                'scene_id': 'scene_a',
                'scene_goal': '找到遗迹入口',
                'status': 'in_progress',
                'scene_summary': '队伍抵达森林边缘',
                'plots': [
                    {
                        'plot_id': 'scene_a_plot_1',
                        'plot_goal': '调查石碑并获取线索',
                        'mandatory_events': ['调查石碑', '记录符文'],
                        'npc': ['Guide'],
                        'locations': ['Forest Ruin'],
                        'status': 'in_progress',
                        'progress': 0.2,
                    }
                ],
            }
        ]
        db.insert_scenes(scenes)
        store.add_from_scenes(scenes)
        db.update_system_state(
            {
                'stage': 'session',
                'current_scene_id': 'scene_a',
                'current_plot_id': 'scene_a_plot_1',
                'plot_progress': 0.2,
                'scene_progress': 0.0,
                'player_profile': {'name': 'Tester', 'traits': ['calm']},
            }
        )

        agent = NarrativeAgent(db, store)
        user_input = '我调查石碑并进行1d20检定'
        print('[test_agent_graph] 输入: user_input ->', user_input)
        result = agent.run_turn(user_input)

        print('[test_agent_graph] 输出: prompt(前500字符) ->')
        print((result.get('prompt') or '')[:500])
        print('[test_agent_graph] 输出: retrieved_docs ->', result.get('retrieved_docs'))
        print('[test_agent_graph] 输出: response ->', result.get('response'))
        print('[test_agent_graph] 输出: dice_result ->', result.get('dice_result'))

        new_state = db.get_system_state()
        print('[test_agent_graph] 输出: system_state ->', new_state)

        db.close()
        print('[test_agent_graph] 结果: PASS')
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f'[test_agent_graph] 结果: FAIL -> {exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
