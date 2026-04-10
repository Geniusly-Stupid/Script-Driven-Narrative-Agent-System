import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app.state as state_module
from app.database import Database
from app.state import (
    evaluate_plot_completion,
    evaluate_pre_response_transition,
    evaluate_scene_completion,
    next_story_position,
    story_position_context,
)


def _fake_state_llm(prompt: str, *args, **kwargs) -> str:
    step_name = kwargs.get('step_name', '')
    if step_name == 'player_alignment_classification':
        if 'Latest Player Input:\nI ask about the weather.' in prompt:
            return '{"alignment": "off_topic", "reason": "llm_detected_off_topic", "confidence": 0.87}'
        return '{"alignment": "current_plot", "reason": "continue_current_plot", "confidence": 0.61}'
    if step_name == 'plot_completion_evaluation':
        if 'Latest User Input:\nI search the files.' in prompt:
            return (
                '{"completed": true, "objective_satisfied": true, "progress_delta": 1.0, '
                '"close_current": true, "reason": "lead_resolved"}'
            )
        return (
            '{"completed": false, "objective_satisfied": false, "progress_delta": 0.1, '
            '"close_current": false, "reason": "plot_continues"}'
        )
    if step_name == 'turn_state_extraction':
        if 'You recover the ledger.' in prompt:
            return '{"beat_status": "wrapped", "summary": "beat wrapped"}'
        return '{"beat_status": "open", "summary": "beat open"}'
    raise AssertionError(f'unexpected step_name={step_name}')


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
                'scene_description': 'Fog hangs low over the water beneath the bridge.',
                'scene_summary': '',
                'status': 'pending',
                'plots': [
                    {
                        'plot_id': 'scene_2_plot_1',
                        'plot_index': 1,
                        'plot_name': 'Question the witness',
                        'plot_goal': 'Learn who fled the house',
                        'raw_text': 'A nervous witness waits under the nearest lamp.',
                        'status': 'pending',
                        'progress': 0.0,
                    }
                ],
            },
        ]
    )
    return db


def main() -> int:
    db_path = ROOT / 'test' / 'debug_state_linear.db'
    original_llm = state_module.call_nvidia_llm
    db = None
    try:
        state_module.call_nvidia_llm = _fake_state_llm
        db = _build_db(db_path)

        print('[test_state] case 1: story_position_context should expose the current plot and the ordered next plot')
        ctx = story_position_context(db, 'scene_1', 'scene_1_plot_1', navigation_state={}, current_visit_id=0)
        assert ctx['current_plot_raw_text'].startswith('The foyer is dim')
        assert ctx['next_plot_goal'] == 'Recover any written evidence'
        assert ctx['opening_choice_allowed'] is False
        assert ctx['allowed_targets'] == []

        print('[test_state] case 2: pre-response transitions should stay disabled in the linear model')
        pre = evaluate_pre_response_transition(
            'I head straight to the bridge.',
            plot_goal='Find the first sign of disturbance',
            scene_goal='Search the house for evidence',
            current_plot_raw_text=ctx['current_plot_raw_text'],
        )
        assert pre['action'] == 'stay'
        assert pre['reason'] == 'linear_progression_only'

        print('[test_state] case 3: plot completion should close the current plot without selecting a target')
        plot_eval = evaluate_plot_completion(
            'I search the files.',
            'You recover the ledger.',
            0.0,
            plot_goal='Find the first sign of disturbance',
            scene_goal='Search the house for evidence',
            scene_description='The investigators move room by room through the quiet house.',
            current_plot_raw_text=ctx['current_plot_raw_text'],
        )
        assert plot_eval['completed'] is True
        assert plot_eval['close_current'] is True
        assert plot_eval['action'] == 'stay'
        assert plot_eval['target_id'] == ''

        print('[test_state] case 4: completing a non-final plot should advance to the next plot in the same scene')
        db.update_plot('scene_1_plot_1', status='completed', progress=1.0)
        scene_eval = evaluate_scene_completion(
            db,
            'scene_1',
            current_plot_id='scene_1_plot_1',
            plot_transition=plot_eval,
        )
        assert scene_eval['completed'] is False
        next_pos = next_story_position(
            db,
            'scene_1',
            'scene_1_plot_1',
            navigation_state={},
            advance_decision=plot_eval,
            current_visit_id=0,
        )
        assert next_pos['current_scene_id'] == 'scene_1'
        assert next_pos['current_plot_id'] == 'scene_1_plot_2'
        assert next_pos['current_visit_id'] == 1

        print('[test_state] case 5: finishing the last plot should complete the scene and move to the next scene')
        db.update_plot('scene_1_plot_2', status='completed', progress=1.0)
        final_scene_eval = evaluate_scene_completion(
            db,
            'scene_1',
            current_plot_id='scene_1_plot_2',
            plot_transition={'close_current': True},
        )
        assert final_scene_eval['completed'] is True
        next_scene_pos = next_story_position(
            db,
            'scene_1',
            'scene_1_plot_2',
            navigation_state={},
            advance_decision={'close_current': True},
            current_visit_id=1,
        )
        assert next_scene_pos['current_scene_id'] == 'scene_2'
        assert next_scene_pos['current_plot_id'] == 'scene_2_plot_1'
        assert next_scene_pos['current_visit_id'] == 2

        print('[test_state] result: PASS')
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f'[test_state] result: FAIL -> {exc}')
        return 1
    finally:
        state_module.call_nvidia_llm = original_llm
        if db is not None:
            db.close()
        if db_path.exists():
            db_path.unlink()


if __name__ == '__main__':
    raise SystemExit(main())
