import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app.agent_graph as agent_graph_module
import app.state as state_module
from app.agent_graph import NarrativeAgent
from app.database import Database
from app.vector_store import ChromaStore


def _build_agent(db_name: str, chroma_name: str) -> tuple[NarrativeAgent, Database, ChromaStore]:
    db_path = ROOT / 'test' / db_name
    chroma_path = ROOT / 'test' / chroma_name
    if db_path.exists():
        db_path.unlink()

    db = Database(str(db_path))
    store = ChromaStore(path=str(chroma_path))
    store.reset()

    scenes = [
        {
            'scene_id': 'scene_1',
            'scene_goal': 'Investigate the library disturbance',
            'scene_description': 'The library is silent except for scraping sounds behind the archive shelves.',
            'status': 'in_progress',
            'scene_summary': '',
            'plots': [
                {
                    'plot_id': 'scene_1_plot_1',
                    'plot_goal': 'Find the source of the disturbance',
                    'mandatory_events': ['Inspect the archive shelves'],
                    'npc': ['Archivist'],
                    'locations': ['Library'],
                    'status': 'in_progress',
                    'progress': 0.0,
                }
            ],
        }
    ]
    db.insert_scenes(scenes)
    db.update_system_state(
        {
            'stage': 'session',
            'current_scene_id': 'scene_1',
            'current_plot_id': 'scene_1_plot_1',
            'plot_progress': 0.0,
            'scene_progress': 0.0,
            'output_language': 'English',
            'player_profile': {
                'name': 'Tester',
                'chosen_skill_allocations': {
                    'occupation': ['Spot Hidden:60', 'Library Use:50'],
                    'personal_interest': ['Occult:40'],
                },
                'characteristics': {'DEX': 55, 'STR': 50, 'SAN': 65},
                'derived_attributes': {'SAN': 65, 'HP': 11, 'MP': 13},
            },
        }
    )
    return NarrativeAgent(db, store), db, store


def main() -> int:
    old_agent_llm = agent_graph_module.call_nvidia_llm
    old_state_llm = state_module.call_nvidia_llm
    old_randint = agent_graph_module.random.randint

    try:
        def fake_agent_llm(prompt: str, model=None, *, step_name: str = 'generation', max_retries: int = 3, timeout: int | float = 120) -> str:
            if step_name == 'check_whether_roll_dice':
                if 'search the archive shelves' in prompt.lower():
                    return '{"need_check": true, "skill": "Spot Hidden", "reason": "Spot Hidden check", "dice_type": "1d100"}'
                return '{"need_check": false, "skill": "", "reason": "", "dice_type": ""}'
            if step_name == 'generate_response':
                if 'Skill Check Result:\nSpot Hidden 60: Extreme Success' in prompt:
                    return 'You quickly notice fresh scratches and a hidden blood mark behind the shelves.'
                return 'You move carefully through the library, but nothing forces a roll at this moment.'
            return 'OK'

        def fake_state_llm(prompt: str, model=None, *, step_name: str = 'generation', max_retries: int = 3, timeout: int | float = 120) -> str:
            if '"progress_delta"' in prompt:
                return '{"completed": false, "progress_delta": 0.1}'
            if '"scene_progress"' in prompt:
                return '{"scene_progress": 0.1}'
            return '{}'

        agent_graph_module.call_nvidia_llm = fake_agent_llm
        state_module.call_nvidia_llm = fake_state_llm
        agent_graph_module.random.randint = lambda a, b: 12

        agent, db, _store = _build_agent('debug_roll_workflow.db', '.chroma_roll_workflow')

        result_with_check = agent.run_turn('I search the archive shelves for hidden marks.')
        print('[test_roll_workflow] output with check ->', result_with_check)
        assert result_with_check.get('need_check') is True, 'check should be triggered'
        assert '1d100' in (result_with_check.get('dice_result') or ''), 'dice result should be recorded'
        assert result_with_check.get('skill_check_result') == 'Spot Hidden 60: Extreme Success', 'skill evaluation should match roll'

        result_without_check = agent.run_turn('I wait quietly and listen to the room.')
        print('[test_roll_workflow] output without check ->', result_without_check)
        assert result_without_check.get('need_check') is False, 'check should not be triggered'
        assert result_without_check.get('dice_result') is None, 'dice result should stay empty when no check is needed'
        assert result_without_check.get('skill_check_result') is None, 'skill result should stay empty when no check is needed'

        assert agent._evaluate_skill_check(12, 60) == 'Extreme Success'
        assert agent._evaluate_skill_check(25, 60) == 'Hard Success'
        assert agent._evaluate_skill_check(54, 60) == 'Regular Success'
        assert agent._evaluate_skill_check(80, 60) == 'Fail'
        assert agent._evaluate_skill_check(96, 60) == 'Worst Fail'

        db.close()
        print('[test_roll_workflow] result: PASS')
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f'[test_roll_workflow] result: FAIL -> {exc}')
        return 1
    finally:
        agent_graph_module.call_nvidia_llm = old_agent_llm
        state_module.call_nvidia_llm = old_state_llm
        agent_graph_module.random.randint = old_randint


if __name__ == '__main__':
    raise SystemExit(main())
