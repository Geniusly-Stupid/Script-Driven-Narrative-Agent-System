import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app.agent_graph as agent_graph_module
import app.state as state_module
from app.agent_graph import NarrativeAgent
from app.database import Database


class DummyVectorStore:
    def search(self, query: str, k: int = 3) -> list[dict]:
        return []


def _extract_player_input(prompt: str) -> str:
    marker = 'Player Input:\n'
    if marker not in prompt:
        return ''
    tail = prompt.split(marker, 1)[1]
    return tail.split('\n\n', 1)[0].strip().lower()


def _build_agent(db_name: str, chroma_name: str) -> tuple[NarrativeAgent, Database, DummyVectorStore]:
    db_path = ROOT / 'test' / db_name
    if db_path.exists():
        db_path.unlink()

    db = Database(str(db_path))
    store = DummyVectorStore()

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
                if _extract_player_input(prompt) == 'i search the archive shelves for hidden marks.':
                    return '```json\n{"need_check": true, "skill": "Spot Hidden", "reason": "Spot Hidden check", "dice_type": "1d100"}\n```'
                if _extract_player_input(prompt) == 'i search the back issues carefully.':
                    return 'Will likely need a follow-up roll.'
                if _extract_player_input(prompt) == 'i head deeper into the records room.':
                    return 'Not a JSON payload.'
                return '{"need_check": false, "skill": "", "reason": "", "dice_type": ""}'
            if step_name == 'generate_response':
                if 'Skill Check Result:\nSpot Hidden 60: Extreme Success' in prompt:
                    return 'You quickly notice fresh scratches and a hidden blood mark behind the shelves.'
                if 'Skill Check Result:\nLibrary Use 50: Extreme Success' in prompt:
                    assert 'Resolved Check Summary:\nThe Library Use is already resolved this turn:' in prompt
                    return 'You quickly trace the relevant article and pull the right bound volume from the shelf.'
                if 'Player Input:\nI head deeper into the records room.' in prompt:
                    return 'You pull out a stack of bound registers and pause at a dusty index drawer. (Make a Library Use check.)'
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

        result_prompted_check = agent.run_turn('I head deeper into the records room.')
        print('[test_roll_workflow] output with keeper-prompted check ->', result_prompted_check)
        assert result_prompted_check.get('need_check') is False, 'first setup turn should just present the prompted check'
        assert '(Make a Library Use check.)' in (result_prompted_check.get('response') or '')

        result_followup_check = agent.run_turn('I search the back issues carefully.')
        print('[test_roll_workflow] output after malformed roll-check JSON ->', result_followup_check)
        assert result_followup_check.get('need_check') is True, 'explicit keeper prompt should trigger fallback roll handling'
        assert result_followup_check.get('check_skill') == 'Library Use'
        assert result_followup_check.get('skill_check_result') == 'Library Use 50: Hard Success'
        assert result_followup_check.get('resolved_check_summary'), 'resolved check summary should be populated'
        assert 'Make a Library Use check' not in (result_followup_check.get('response') or ''), 'response should not ask for the same roll again'
        assert 'Resolved Check Summary:' in (result_followup_check.get('prompt') or '')

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
