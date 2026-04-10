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


def _fake_llm(prompt: str, *args, **kwargs) -> str:
    step_name = kwargs.get('step_name', '')
    if step_name == 'scene_opening_generation':
        return 'The foyer smells of damp wool and old varnish. A fresh scrape marks the coat stand.'
    if step_name == 'check_whether_roll_dice':
        return '{"need_check": false, "skill": "", "reason": "", "dice_type": ""}'
    if step_name == 'player_alignment_classification':
        if 'Latest Player Input:\nI ask about the weather.' in prompt:
            return '{"alignment": "off_topic", "reason": "llm_detected_off_topic", "confidence": 0.88}'
        return '{"alignment": "current_plot", "reason": "continue_current_plot", "confidence": 0.72}'
    if step_name == 'turn_state_extraction':
        if 'You recover the ledger from behind the coats.' in prompt:
            return '{"beat_status": "wrapped", "summary": "the foyer beat is resolved"}'
        if 'The caretaker admits he saw a man limping toward the bridge.' in prompt:
            return '{"beat_status": "wrapped", "summary": "the study beat is resolved"}'
        return '{"beat_status": "open", "summary": "the beat remains open"}'
    if step_name == 'plot_completion_evaluation':
        if 'Latest User Input:\nI inspect the coat stand carefully.' in prompt:
            return (
                '{"completed": true, "objective_satisfied": true, "progress_delta": 1.0, '
                '"close_current": true, "reason": "foyer_resolved"}'
            )
        if 'Latest User Input:\nI question the caretaker about the ledger.' in prompt:
            return (
                '{"completed": true, "objective_satisfied": true, "progress_delta": 1.0, '
                '"close_current": true, "reason": "study_resolved"}'
            )
        return (
            '{"completed": false, "objective_satisfied": false, "progress_delta": 0.1, '
            '"close_current": false, "reason": "plot_continues"}'
        )
    if step_name == 'generate_response':
        if 'Plot ID: scene_1_plot_1' in prompt and 'Player Input:\n\n' in prompt:
            return 'The foyer smells of damp wool and old varnish. A fresh scrape marks the coat stand.'
        if 'Plot ID: scene_1_plot_1' in prompt:
            return 'You recover the ledger from behind the coats.'
        if 'Plot ID: scene_1_plot_2' in prompt:
            return 'The caretaker admits he saw a man limping toward the bridge.'
        if 'Plot ID: scene_2_plot_1' in prompt:
            return 'At the bridge, the witness waits beneath a dim lamp.'
        return 'The investigation continues.'
    if step_name == 'plot_summary_generation':
        return '- The investigator resolved the active plot and secured a useful clue.'
    if step_name == 'scene_summary_generation':
        return '- The scene concluded after both ordered plots were resolved.'
    return '{}'


def _build_db(db_path: Path) -> Database:
    if db_path.exists():
        db_path.unlink()
    db = Database(str(db_path))
    db.insert_scenes(
        [
            {
                'scene_id': 'scene_1',
                'scene_index': 1,
                'scene_name': 'Kimball House',
                'scene_goal': 'Search the house for evidence',
                'scene_description': 'The investigators move room by room through the quiet house.',
                'scene_summary': '',
                'status': 'in_progress',
                'plots': [
                    {
                        'plot_id': 'scene_1_plot_1',
                        'plot_index': 1,
                        'plot_name': 'Check the foyer',
                        'plot_goal': 'Find the first sign of disturbance',
                        'raw_text': 'The foyer is dim, with muddy prints near the coat stand.',
                        'status': 'in_progress',
                        'progress': 0.0,
                    },
                    {
                        'plot_id': 'scene_1_plot_2',
                        'plot_index': 2,
                        'plot_name': 'Search the study',
                        'plot_goal': 'Recover any written evidence',
                        'raw_text': 'The study desk is locked, but papers are scattered nearby.',
                        'status': 'pending',
                        'progress': 0.0,
                    },
                ],
            },
            {
                'scene_id': 'scene_2',
                'scene_index': 2,
                'scene_name': 'Riverside Bridge',
                'scene_goal': 'Meet the witness',
                'scene_description': 'Fog hangs low over the bridge.',
                'scene_summary': '',
                'status': 'pending',
                'plots': [
                    {
                        'plot_id': 'scene_2_plot_1',
                        'plot_index': 1,
                        'plot_name': 'Question the witness',
                        'plot_goal': 'Learn who fled the house',
                        'raw_text': 'A nervous witness waits beneath the nearest lamp.',
                        'status': 'pending',
                        'progress': 0.0,
                    }
                ],
            },
        ]
    )
    db.update_system_state(
        {
            'stage': 'session',
            'current_scene_id': 'scene_1',
            'current_plot_id': 'scene_1_plot_1',
            'plot_progress': 0.0,
            'scene_progress': 0.0,
            'player_profile': {
                'name': 'Tester',
                'chosen_skill_allocations': {'occupation': ['Spot Hidden:70']},
            },
        }
    )
    return db


def main() -> int:
    db_path = ROOT / 'test' / 'debug_agent_linear.db'
    original_agent_llm = agent_graph_module.call_llm
    original_state_llm = state_module.call_nvidia_llm
    db = None
    try:
        agent_graph_module.call_llm = _fake_llm
        state_module.call_nvidia_llm = _fake_llm

        db = _build_db(db_path)
        agent = NarrativeAgent(db, DummyVectorStore())
        agent.set_debug_mode(True)

        print('[test_agent_graph] case 1: initial response should open the first plot')
        initial = agent.generate_initial_response()
        assert initial.get('response'), 'opening response should exist'
        opening_turns = db.get_recent_turns('scene_1', 'scene_1_plot_1', limit=5, visit_id=0)
        assert opening_turns, 'opening narration should be written to memory'

        print('[test_agent_graph] case 2: resolving the first plot should advance to the next plot in order')
        first_turn = agent.run_turn('I inspect the coat stand carefully.')
        system_after_first = db.get_system_state()
        assert first_turn.get('pre_response_transition_applied') is False
        assert system_after_first['current_scene_id'] == 'scene_1'
        assert system_after_first['current_plot_id'] == 'scene_1_plot_2'
        assert db.get_summary('plot', scene_id='scene_1', plot_id='scene_1_plot_1'), 'plot 1 summary should be stored'

        print('[test_agent_graph] case 3: resolving the last plot should advance to the next scene')
        second_turn = agent.run_turn('I question the caretaker about the ledger.')
        system_after_second = db.get_system_state()
        assert second_turn.get('pre_response_transition_applied') is False
        assert system_after_second['current_scene_id'] == 'scene_2'
        assert system_after_second['current_plot_id'] == 'scene_2_plot_1'
        assert db.get_summary('scene', scene_id='scene_1'), 'scene 1 summary should be stored'

        print('[test_agent_graph] result: PASS')
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f'[test_agent_graph] result: FAIL -> {exc}')
        return 1
    finally:
        agent_graph_module.call_llm = original_agent_llm
        state_module.call_nvidia_llm = original_state_llm
        if db is not None:
            db.close()
        if db_path.exists():
            db_path.unlink()


if __name__ == '__main__':
    raise SystemExit(main())
