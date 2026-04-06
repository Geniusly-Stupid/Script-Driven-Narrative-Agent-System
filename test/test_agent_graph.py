import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app.agent_graph as agent_graph_module
import app.state as state_module
from app.agent_graph import NarrativeAgent
from app.database import Database
from app.navigation import default_navigation, make_target


class DummyVectorStore:
    def search(self, query: str, k: int = 3) -> list[dict]:
        return []


def _agent_llm(prompt: str, *args, **kwargs):
    step_name = kwargs.get('step_name', '')
    if step_name == 'check_whether_roll_dice':
        if 'ROLL_TRIGGER_TEST' in prompt:
            return '{"need_check": true, "skill": "Spot Hidden", "reason": "llm_requests_check", "dice_type": "1d100"}'
        if 'ROLL_INVALID_TEST' in prompt:
            return 'not-json'
        return '{"need_check": false, "skill": "", "reason": "", "dice_type": ""}'
    if step_name == 'player_alignment_classification':
        if 'Latest Player Input:\nI go to the library.' in prompt:
            return (
                '{"alignment": "target_choice", "action": "target", "target_kind": "scene", '
                '"target_id": "scene_library", "transition_path": "direct", "close_current": false, '
                '"confidence": 0.97, "reason": "llm_selected_library"}'
            )
        if 'Latest Player Input:\nI ask about the weather.' in prompt:
            return (
                '{"alignment": "off_topic", "action": "stay", "target_kind": "", "target_id": "", '
                '"transition_path": "stay", "close_current": false, "confidence": 0.88, '
                '"reason": "llm_detected_off_topic"}'
            )
        return (
            '{"alignment": "current_plot", "action": "stay", "target_kind": "", "target_id": "", '
            '"transition_path": "stay", "close_current": false, "confidence": 0.34, '
            '"reason": "llm_kept_current_plot"}'
        )
    if step_name == 'plot_completion_evaluation':
        if 'User: I search the shelves for the obituary.' in prompt:
            return (
                '{"completed": true, "objective_satisfied": true, "progress_delta": 0.6, '
                '"action": "stay", "target_kind": "", "target_id": "", "transition_path": "stay", '
                '"close_current": true, "confidence": 0.93, "reason": "llm_wrapped_library"}'
            )
        return (
            '{"completed": false, "objective_satisfied": false, "progress_delta": 0.0, '
            '"action": "stay", "target_kind": "", "target_id": "", "transition_path": "stay", '
            '"close_current": false, "confidence": 0.31, "reason": "llm_keep_current_plot"}'
        )
    if step_name == 'turn_state_extraction':
        if 'scene_library' in prompt and 'scene_police' in prompt:
            return (
                '{"choice_open": true, "offered_targets": ['
                '{"target_kind": "scene", "target_id": "scene_library"}, '
                '{"target_kind": "scene", "target_id": "scene_police"}], '
                '"beat_status": "open", "summary": "two legal leads are open"}'
            )
        if 'You find the obituary and this lead is wrapped.' in prompt:
            return (
                '{"choice_open": false, "offered_targets": [], '
                '"beat_status": "wrapped", "summary": "library lead wrapped"}'
            )
        return (
            '{"choice_open": false, "offered_targets": [], '
            '"beat_status": "open", "summary": "no active choice"}'
        )
    if step_name == 'generate_response':
        if 'Scene ID: scene_start' in prompt and 'Player Input:\n\n' in prompt:
            return 'You can follow either lead from here.'
        if 'Scene ID: scene_library' in prompt:
            if 'Input Consumed By Transition:\ntrue' in prompt and 'Transition Trigger Input:\nI go to the library.' in prompt:
                return 'You step into the library archives, where dust hangs over the obituary index.'
            if 'Player Input:\nI ask about the weather.' in prompt:
                return 'The librarian ignores the weather and points you back toward the obituary shelves.'
            if 'Player Input:\nI search the shelves for the obituary.' in prompt:
                return 'You find the obituary and this lead is wrapped.'
            return 'Dust hangs over the obituary index.'
        return 'The investigation continues.'
    if step_name == 'plot_summary_generation':
        return '- The library lead ended with the obituary in hand.'
    if step_name == 'scene_summary_generation':
        return '- The scene closed after its active branch resolved.'
    raise AssertionError(f'unexpected step_name={step_name}')


def _make_scene(
    scene_id: str,
    scene_goal: str,
    *,
    node_kind: str,
    navigation: dict,
    status: str = 'pending',
) -> dict:
    plot_status = 'in_progress' if status == 'in_progress' else status
    return {
        'scene_id': scene_id,
        'scene_goal': scene_goal,
        'scene_description': scene_goal,
        'status': status,
        'scene_summary': '',
        'node_kind': node_kind,
        'navigation': navigation,
        'plots': [
            {
                'plot_id': f'{scene_id}_plot_1',
                'plot_goal': scene_goal,
                'mandatory_events': [],
                'npc': [],
                'locations': [],
                'raw_text': scene_goal,
                'status': plot_status,
                'progress': 0.0,
                'node_kind': node_kind,
                'navigation': navigation,
            }
        ],
    }


def _build_db(db_path: Path) -> Database:
    if db_path.exists():
        db_path.unlink()
    db = Database(str(db_path))
    start_nav = default_navigation(completion_policy='all_required_then_advance')
    start_nav['allowed_targets'] = [
        make_target('scene', 'scene_library', label='Library', role='branch', required=True),
        make_target('scene', 'scene_police', label='Police', role='branch', required=True),
    ]
    start_return = make_target('scene', 'scene_start', label='Start', role='return')
    scenes = [
        _make_scene('scene_start', 'Start hub', node_kind='hub', navigation=start_nav, status='in_progress'),
        _make_scene(
            'scene_library',
            'Library branch',
            node_kind='branch',
            navigation={
                'allowed_targets': [],
                'return_target': start_return,
                'completion_policy': 'terminal_on_resolve',
                'prerequisites': [],
                'close_unselected_on_advance': False,
            },
        ),
        _make_scene(
            'scene_police',
            'Police branch',
            node_kind='branch',
            navigation={
                'allowed_targets': [],
                'return_target': start_return,
                'completion_policy': 'terminal_on_resolve',
                'prerequisites': [],
                'close_unselected_on_advance': False,
            },
        ),
    ]
    db.insert_scenes(scenes)
    db.update_system_state(
        {
            'stage': 'session',
            'current_scene_id': 'scene_start',
            'current_plot_id': 'scene_start_plot_1',
            'plot_progress': 0.0,
            'scene_progress': 0.0,
            'navigation_state': {},
            'current_visit_id': 0,
            'player_profile': {'name': 'Tester'},
        }
    )
    return db


def main() -> int:
    db_path = ROOT / 'test' / 'debug_agent_runtime_turn_state.db'
    original_agent_llm = agent_graph_module.call_nvidia_llm
    original_state_llm = state_module.call_nvidia_llm
    db = None
    try:
        agent_graph_module.call_nvidia_llm = _agent_llm
        state_module.call_nvidia_llm = _agent_llm

        db = _build_db(db_path)
        agent = NarrativeAgent(db, DummyVectorStore())
        agent.set_debug_mode(True)

        print('[test_agent_graph] case 1: initial response should store extracted turn_state in memory')
        initial_result = agent.generate_initial_response()
        opening_turns = db.get_recent_turns('scene_start', 'scene_start_plot_1', limit=5, visit_id=0)
        assert initial_result.get('response'), 'initial response should exist'
        assert opening_turns, 'initial opening should be written to memory'
        assert opening_turns[-1]['turn_state']['choice_open'] is True
        offered_ids = {target['target_id'] for target in opening_turns[-1]['turn_state']['offered_targets']}
        assert offered_ids == {'scene_library', 'scene_police'}
        pre_prompt = next(
            (item.get('prompt', '') for item in agent.latest_debug_prompts if item.get('name') == 'turn_state_prompt'),
            '',
        )
        assert pre_prompt, 'turn_state extraction prompt should be recorded in debug mode'

        print('[test_agent_graph] case 2: pre-response transition should consume structured intent and enter the branch')
        handoff = agent.run_turn('I go to the library.')
        handoff_prompt = next(
            (item.get('prompt', '') for item in handoff.get('debug_prompts', []) if item.get('name') == 'pre_response_transition_prompt'),
            '',
        )
        handoff_response_prompt = next(
            (item.get('prompt', '') for item in handoff.get('debug_prompts', []) if item.get('name') == 'generate_response_prompt'),
            '',
        )
        system_after_handoff = db.get_system_state()
        assert handoff.get('pre_response_transition_applied') is True
        assert system_after_handoff['current_scene_id'] == 'scene_library'
        assert handoff_prompt and 'choice_open=true' in handoff_prompt
        assert handoff.get('response') == 'You step into the library archives, where dust hangs over the obituary index.'
        assert 'Input Consumed By Transition:\ntrue' in handoff_response_prompt
        assert 'Transition Trigger Input:\nI go to the library.' in handoff_response_prompt
        assert 'Player Input:\n\n' in handoff_response_prompt

        print('[test_agent_graph] case 3: off-topic turns should stay put and update redirect guidance')
        off_topic = agent.run_turn('I ask about the weather.')
        library_turns = db.get_recent_turns('scene_library', 'scene_library_plot_1', limit=5, visit_id=1)
        off_topic_state = db.get_system_state()
        guidance = off_topic_state['navigation_state']['plot_guidance']['scene_library_plot_1::1']
        assert off_topic_state['current_scene_id'] == 'scene_library'
        assert guidance['redirect_streak'] == 1
        assert library_turns[-1]['turn_state']['choice_open'] is False
        assert off_topic.get('plot_advance_action') == 'stay'

        print('[test_agent_graph] case 4: wrapped branches should return to the parent hub after write_memory + completion')
        wrapped = agent.run_turn('I search the shelves for the obituary.')
        wrapped_state = db.get_system_state()
        library_turns = db.get_recent_turns('scene_library', 'scene_library_plot_1', limit=8, visit_id=1)
        assert wrapped_state['current_scene_id'] == 'scene_start'
        assert wrapped_state['current_plot_id'] == 'scene_start_plot_1'
        assert wrapped_state['current_visit_id'] == 2
        assert library_turns[-1]['turn_state']['beat_status'] == 'wrapped'
        assert wrapped.get('scene_id') == 'scene_start'

        print('[test_agent_graph] case 5: roll checks should trigger only from llm output')
        roll_state = {
            'roll_check_prompt': 'ROLL_TRIGGER_TEST',
            'need_check': False,
            'check_skill': '',
            'check_reason': '',
            'dice_type': '',
            'dice_result': None,
            'skill_check_result': None,
            'resolved_check_summary': '',
        }
        roll_state = agent.check_whether_roll_dice(roll_state)
        assert roll_state.get('need_check') is True
        assert roll_state.get('check_skill') == 'Spot Hidden'
        assert roll_state.get('check_reason') == 'llm_requests_check'
        assert roll_state.get('dice_type') == '1d100'

        print('[test_agent_graph] case 6: invalid roll-check json should conservatively avoid marker fallback')
        invalid_roll_state = {
            'roll_check_prompt': 'ROLL_INVALID_TEST',
            'need_check': False,
            'check_skill': '',
            'check_reason': '',
            'dice_type': '',
            'dice_result': None,
            'skill_check_result': None,
            'resolved_check_summary': '',
            'latest_agent_turn_excerpt': 'Make a Spot Hidden check.',
            'current_plot_raw_text': 'This scene requires a Spot Hidden roll.',
            'latest_alignment': 'current_plot',
            'player_profile': {'chosen_skill_allocations': {'occupation': ['Spot Hidden:70']}},
        }
        invalid_roll_state = agent.check_whether_roll_dice(invalid_roll_state)
        assert invalid_roll_state.get('need_check') is False
        assert invalid_roll_state.get('check_skill') == ''
        assert invalid_roll_state.get('check_reason') == ''
        assert invalid_roll_state.get('dice_type') == ''

        print('[test_agent_graph] result: PASS')
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f'[test_agent_graph] result: FAIL -> {exc}')
        return 1
    finally:
        agent_graph_module.call_nvidia_llm = original_agent_llm
        state_module.call_nvidia_llm = original_state_llm
        if db is not None:
            db.close()
        if db_path.exists():
            db_path.unlink()


if __name__ == '__main__':
    raise SystemExit(main())
