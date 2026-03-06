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
                'scene_goal': 'Find the ruin entrance',
                'scene_description': 'The team reaches the forest edge under low visibility. A guide urges quick action while concealing political tension around the ruins. Players can examine old markings, interrogate locals, or scout dangerous routes before rival groups arrive.',
                'status': 'in_progress',
                'scene_summary': 'The party reached the forest edge.',
                'plots': [
                    {
                        'plot_id': 'scene_a_plot_1',
                        'plot_goal': 'Inspect the stone tablet and obtain clues',
                        'mandatory_events': ['Inspect tablet', 'Record runes'],
                        'npc': ['Guide'],
                        'locations': ['Forest Ruin'],
                        'status': 'in_progress',
                        'progress': 0.2,
                    }
                ],
            }
        ]
        knowledge = [
            {
                'knowledge_id': 'knowledge_1',
                'knowledge_type': 'background',
                'title': 'Faction Pressure',
                'content': 'Two factions compete for control of ruin artifacts.',
                'source_page_start': 1,
                'source_page_end': 1,
            },
            {
                'knowledge_id': 'knowledge_2',
                'knowledge_type': 'truth',
                'title': 'Hidden Pact',
                'content': 'The guide secretly works for a faction leader.',
                'source_page_start': 2,
                'source_page_end': 2,
            },
        ]
        db.insert_scenes(scenes)
        db.insert_knowledge(knowledge)
        store.add_from_scenes(scenes, knowledge=knowledge)
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
        user_input = 'I inspect the stone tablet and roll 1d20'
        print('[test_agent_graph] input: user_input ->', user_input)
        result = agent.run_turn(user_input)

        prompt = result.get('prompt') or ''
        print('[test_agent_graph] output: prompt(first 500 chars) ->')
        print(prompt[:500])
        print('[test_agent_graph] output: retrieved_docs ->', result.get('retrieved_docs'))
        print('[test_agent_graph] output: response ->', result.get('response'))
        print('[test_agent_graph] output: dice_result ->', result.get('dice_result'))

        assert 'Scene Description:' in prompt, 'prompt must include scene_description section'
        assert 'World Context:' in prompt, 'prompt must include world_context section'
        assert 'Hidden Truth (Keeper-only):' in prompt, 'prompt must include truth context section'

        new_state = db.get_system_state()
        print('[test_agent_graph] output: system_state ->', new_state)

        db.close()
        print('[test_agent_graph] result: PASS')
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f'[test_agent_graph] result: FAIL -> {exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
