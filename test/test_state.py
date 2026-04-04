import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.database import Database
import app.state as state_module
from app.state import evaluate_plot_completion, evaluate_scene_completion, next_story_position


def _failing_llm(*args, **kwargs):
    raise RuntimeError('offline test fallback')


def main() -> int:
    db_path = ROOT / 'test' / 'debug_state.db'
    original_llm = state_module.call_nvidia_llm
    try:
        if db_path.exists():
            db_path.unlink()

        state_module.call_nvidia_llm = _failing_llm

        db = Database(str(db_path))
        db.insert_scenes(
            [
                {
                    'scene_id': 'scene_setup',
                    'scene_goal': 'Prepare the investigator to begin the investigation.',
                    'status': 'in_progress',
                    'scene_summary': '',
                    'scene_description': 'A setup scene that gives the investigator several places to visit next.',
                    'plots': [
                        {
                            'plot_id': 'scene_setup_plot_1',
                            'plot_goal': 'Introduce the investigation and provide initial options for the investigator.',
                            'mandatory_events': [],
                            'npc': ['Thomas'],
                            'locations': ['House'],
                            'raw_text': 'The investigator is ready to dig into the mystery and can choose to ask around, visit the library, or speak to the police.',
                            'status': 'in_progress',
                            'progress': 0.0,
                        }
                    ],
                },
                {
                    'scene_id': 'scene_library',
                    'scene_goal': 'Research the old newspaper records in the library.',
                    'status': 'pending',
                    'scene_summary': '',
                    'scene_description': 'A follow-up investigation scene in the local library.',
                    'plots': [
                        {
                            'plot_id': 'scene_library_plot_1',
                            'plot_goal': 'Find and read the article about the cemetery sighting.',
                            'mandatory_events': ['Read the article'],
                            'npc': [],
                            'locations': ['Library'],
                            'raw_text': 'A library investigation scene.',
                            'status': 'pending',
                            'progress': 0.0,
                        }
                    ],
                },
                {
                    'scene_id': 'scene_police',
                    'scene_goal': 'Ask several follow-up questions at the police station.',
                    'status': 'pending',
                    'scene_summary': '',
                    'scene_description': 'A multi-plot scene where the investigator asks the desk officer about different topics.',
                    'plots': [
                        {
                            'plot_id': 'scene_police_plot_1',
                            'plot_goal': 'Introduce the police station conversation and first topic.',
                            'mandatory_events': [],
                            'npc': ['Desk officer'],
                            'locations': ['Police station'],
                            'raw_text': 'The investigator can ask the desk officer about burglaries, the cemetery, or the missing uncle.',
                            'status': 'pending',
                            'progress': 0.0,
                        },
                        {
                            'plot_id': 'scene_police_plot_2',
                            'plot_goal': 'Ask about the cemetery and receive police information about strange noises.',
                            'mandatory_events': ['Ask about the cemetery'],
                            'npc': ['Desk officer'],
                            'locations': ['Police station'],
                            'raw_text': 'A follow-up question about the cemetery.',
                            'status': 'pending',
                            'progress': 0.0,
                        },
                    ],
                },
            ]
        )

        print('[test_state] case 1: setup plot should hand off to next scene')
        plot_eval = evaluate_plot_completion(
            'I head to the library and start reading old newspaper files.',
            'Thomas nods and leaves you to follow that lead.',
            0.0,
            [],
            plot_goal='Introduce the investigation and provide initial options for the investigator.',
            scene_goal='Prepare the investigator to begin the investigation.',
            next_plot_goal='',
            next_scene_goal='Research the old newspaper records in the library.',
            current_plot_raw_text='The investigator is ready to dig into the mystery and can choose to ask around, visit the library, or speak to the police.',
        )
        print('[test_state] output:', plot_eval)
        assert plot_eval['completed'] is True
        assert plot_eval['advance_target'] == 'next_scene'

        print('[test_state] case 2: same-scene multi-plot should hand off to next plot')
        next_plot_eval = evaluate_plot_completion(
            'I ask the desk officer specifically about the cemetery and those strange noises.',
            'The officer glances up and starts recounting the reports.',
            0.0,
            [],
            plot_goal='Introduce the police station conversation and first topic.',
            scene_goal='Ask several follow-up questions at the police station.',
            next_plot_goal='Ask about the cemetery and receive police information about strange noises.',
            next_scene_goal='Research the old newspaper records in the library.',
            current_plot_raw_text='The investigator can ask the desk officer about burglaries, the cemetery, or the missing uncle.',
        )
        print('[test_state] output:', next_plot_eval)
        assert next_plot_eval['completed'] is True
        assert next_plot_eval['advance_target'] == 'next_plot'

        print('[test_state] case 3: deeper investigation in current plot should not misadvance')
        stay_eval = evaluate_plot_completion(
            'I keep pressing the desk officer for more detail about the burglaries in the area.',
            'The officer repeats the break-in timeline and the suspect description.',
            0.2,
            ['Ask about burglaries'],
            plot_goal='Ask about recent burglaries in the area.',
            scene_goal='Ask several follow-up questions at the police station.',
            next_plot_goal='Ask about the cemetery and receive police information about strange noises.',
            next_scene_goal='Research the old newspaper records in the library.',
            current_plot_raw_text='This sub-plot is specifically about the burglary reports and the arrested suspect.',
        )
        print('[test_state] output:', stay_eval)
        assert stay_eval['advance_target'] == 'stay'
        assert stay_eval['completed'] is False

        print('[test_state] case 4: scene should complete when single plot handed off to next scene')
        db.update_plot('scene_setup_plot_1', status='completed', progress=1.0)
        scene_eval = evaluate_scene_completion(
            db,
            'scene_setup',
            latest_turn_text='User: I head to the library.\nAgent: Thomas nods and lets you go.',
            next_scene_goal='Research the old newspaper records in the library.',
            plot_advance_target='next_scene',
            plot_advance_reason='fallback_shift_detected(score=1.00)',
        )
        print('[test_state] output:', scene_eval)
        assert scene_eval['completed'] is True

        nxt = next_story_position(db, 'scene_setup', 'scene_setup_plot_1')
        print('[test_state] next_story_position ->', nxt)
        assert nxt['current_scene_id'] == 'scene_library'

        db.close()
        print('[test_state] result: PASS')
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f'[test_state] result: FAIL -> {exc}')
        return 1
    finally:
        state_module.call_nvidia_llm = original_llm


if __name__ == '__main__':
    raise SystemExit(main())
