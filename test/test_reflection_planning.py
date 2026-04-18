import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app.agent_graph as agent_graph_module
from app.agent_graph import NarrativeAgent
from app.database import Database


class DummyVectorStore:
    def search(self, query: str, k: int = 3) -> list[dict]:
        return []


def _build_db(db_path: Path) -> Database:
    if db_path.exists():
        db_path.unlink()
    db = Database(str(db_path))
    db.insert_scenes(
        [
            {
                'scene_id': 'scene_1',
                'scene_index': 1,
                'scene_name': 'Town Archive',
                'scene_goal': 'Investigate the missing records',
                'scene_description': 'Dusty shelves and sealed cabinets line the archive hall.',
                'scene_summary': '',
                'status': 'in_progress',
                'plots': [
                    {
                        'plot_id': 'scene_1_plot_1',
                        'plot_index': 1,
                        'plot_name': 'Search the reading room',
                        'plot_goal': 'Find clues about the missing ledger',
                        'raw_text': 'The archive reading room contains old ledgers, index cards, and guarded cabinets.',
                        'status': 'in_progress',
                        'progress': 0.0,
                    }
                ],
            },
            {
                'scene_id': 'scene_2',
                'scene_index': 2,
                'scene_name': 'River Docks',
                'scene_goal': 'Follow the escape route',
                'scene_description': 'Night fog drifts between moored boats.',
                'scene_summary': '',
                'status': 'pending',
                'plots': [
                    {
                        'plot_id': 'scene_2_plot_1',
                        'plot_index': 1,
                        'plot_name': 'Question the dock watchman',
                        'plot_goal': 'Learn who moved the ledger',
                        'raw_text': 'The watchman has seen suspicious traffic at night.',
                        'status': 'pending',
                        'progress': 0.0,
                    }
                ],
            },
        ]
    )
    db.save_summary('script', 'A missing ledger points toward a wider conspiracy around the town archive.')
    db.update_system_state(
        {
            'stage': 'session',
            'current_scene_id': 'scene_1',
            'current_plot_id': 'scene_1_plot_1',
            'output_language': 'English',
            'player_profile': {
                'name': 'Tester',
                'chosen_skill_allocations': {'occupation': ['Spot Hidden:60', 'Library Use:55']},
            },
        }
    )
    return db


def main() -> int:
    db_path = ROOT / 'test' / 'debug_reflection_planning.db'
    original_llm = agent_graph_module.call_llm
    db = None
    branch_prompts: list[str] = []
    response_prompts: list[str] = []

    try:
        def fake_llm(prompt: str, model=None, *, step_name: str = 'generation', max_retries: int = 3, timeout: int | float = 120) -> str:
            if step_name == 'branch_transition_decision':
                branch_prompts.append(prompt)
                return '{"switch": false, "target_plot_id": ""}'
            if step_name == 'check_whether_roll_dice':
                return '{"need_check": false, "skill": "", "reason": "", "dice_type": ""}'
            if step_name == 'generate_retrieval_queries':
                return '{"queries": ["ledger", "archive"]}'
            if step_name == 'generate_response':
                response_prompts.append(prompt)
                return 'The archivist watches your search in tense silence.'
            if step_name == 'long_term_memory_update':
                return 'The investigator searched the archive and confirmed the missing ledger is central to the case.'
            if step_name == 'reflection_planning_generation':
                return json.dumps(
                    {
                        'missing_info': 'The ledger destination and who removed it are still unclear.',
                        'evaluation': 'Recent guidance has kept attention on the archive but needs a clearer lead outward.',
                        'guidance': 'Use NPC hesitation and references to the docks to nudge the investigation forward.',
                    },
                    ensure_ascii=False,
                )
            return '{}'

        agent_graph_module.call_llm = fake_llm
        db = _build_db(db_path)
        agent = NarrativeAgent(db, DummyVectorStore())
        agent.set_debug_mode(True)

        print('[test_reflection_planning] case 1: reflection should be generated on the third turn')
        agent.run_turn('I inspect the index cards.')
        agent.run_turn('I search the ledger shelf.')
        agent.run_turn('I ask the archivist what is missing.')

        stored = db.get_summary('reflection_planning')
        assert stored, 'reflection planning should be stored after three turns'
        stored_data = json.loads(stored)
        assert stored_data['guidance'].startswith('Use NPC hesitation'), 'stored guidance should match reflection output'
        nav_state = db.get_system_state().get('navigation_state', {})
        assert nav_state.get('reflection_guidance', '').startswith('Use NPC hesitation'), 'latest reflection should persist in navigation state'

        print('[test_reflection_planning] case 2: stored evaluation and guidance should be injected on later turns')
        agent.run_turn('I check the desk for dock records.')

        assert branch_prompts, 'branch prompts should be recorded'
        assert response_prompts, 'response prompts should be recorded'
        assert 'Use NPC hesitation and references to the docks' in branch_prompts[-1], 'branch prompt should include reflection guidance'
        assert 'Recent guidance has kept attention on the archive' in response_prompts[-1], 'response prompt should include reflection evaluation'
        assert 'Use NPC hesitation and references to the docks' in response_prompts[-1], 'response prompt should include reflection guidance'

        print('[test_reflection_planning] case 3: after switching plots without a new reflection, the latest stored reflection should still be reused')
        db.update_system_state(
            {
                'current_scene_id': 'scene_2',
                'current_plot_id': 'scene_2_plot_1',
            }
        )
        agent.run_turn('I ask the dock watchman about the missing ledger.')
        assert 'Use NPC hesitation and references to the docks' in branch_prompts[-1], 'branch prompt should reuse latest stored guidance after scene change'
        assert 'Recent guidance has kept attention on the archive' in response_prompts[-1], 'response prompt should reuse latest stored evaluation after scene change'

        print('[test_reflection_planning] result: PASS')
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f'[test_reflection_planning] result: FAIL -> {exc}')
        return 1
    finally:
        agent_graph_module.call_llm = original_llm
        if db is not None:
            db.close()
        if db_path.exists():
            db_path.unlink()


if __name__ == '__main__':
    raise SystemExit(main())
