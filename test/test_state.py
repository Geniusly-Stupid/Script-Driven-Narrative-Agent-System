import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.database import Database
from app.navigation import default_navigation, make_target
import app.state as state_module
from app.state import (
    evaluate_plot_completion,
    evaluate_pre_response_transition,
    next_story_position,
    story_position_context,
)


def _state_llm(prompt: str, *args, **kwargs):
    step_name = kwargs.get('step_name', '')
    if step_name == 'player_alignment_classification':
        if 'Latest Player Input:\n1' in prompt and 'choice_open=true' in prompt and 'scene_library' in prompt:
            return (
                '{"alignment": "target_choice", "action": "target", "target_kind": "scene", '
                '"target_id": "scene_library", "transition_path": "direct", "close_current": false, '
                '"confidence": 0.92, "reason": "llm_resolved_open_choice"}'
            )
        if 'Latest Player Input:\nI go to the library.' in prompt:
            return (
                '{"alignment": "target_choice", "action": "target", "target_kind": "scene", '
                '"target_id": "scene_library", "transition_path": "direct", "close_current": false, '
                '"confidence": 0.97, "reason": "llm_selected_library"}'
            )
        if 'Latest Player Input:\nI head to the police station instead.' in prompt and 'Current Node Kind:\nbranch' in prompt:
            return (
                '{"alignment": "target_choice", "action": "target", "target_kind": "scene", '
                '"target_id": "scene_police", "transition_path": "via_return", "close_current": true, '
                '"confidence": 0.95, "reason": "llm_selected_via_return"}'
            )
        if 'Latest Player Input:\nI ask about the weather.' in prompt:
            return (
                '{"alignment": "off_topic", "action": "stay", "target_kind": "", "target_id": "", '
                '"transition_path": "stay", "close_current": false, "confidence": 0.88, '
                '"reason": "llm_detected_off_topic"}'
            )
        return (
            '{"alignment": "current_plot", "action": "stay", "target_kind": "", "target_id": "", '
            '"transition_path": "stay", "close_current": false, "confidence": 0.41, '
            '"reason": "llm_kept_current_plot"}'
        )
    if step_name == 'plot_completion_evaluation':
        if 'User: I search the shelves for the obituary.' in prompt:
            return (
                '{"completed": true, "objective_satisfied": true, "progress_delta": 0.6, '
                '"action": "stay", "target_kind": "", "target_id": "", "transition_path": "stay", '
                '"close_current": true, "confidence": 0.93, "reason": "llm_wrapped_branch"}'
            )
        return (
            '{"completed": false, "objective_satisfied": false, "progress_delta": 0.0, '
            '"action": "stay", "target_kind": "", "target_id": "", "transition_path": "stay", '
            '"close_current": false, "confidence": 0.32, "reason": "llm_keep_current_plot"}'
        )
    if step_name == 'turn_state_extraction':
        return (
            '{"choice_open": false, "offered_targets": [], "beat_status": "open", '
            '"summary": "current beat remains open"}'
        )
    raise AssertionError(f'unexpected step_name={step_name}')


def _failing_llm(*args, **kwargs):
    raise RuntimeError('offline')


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


def _build_scene_branch_db(db_path: Path) -> Database:
    if db_path.exists():
        db_path.unlink()
    db = Database(str(db_path))

    start_nav = default_navigation(completion_policy='all_required_then_advance')
    start_nav['allowed_targets'] = [
        make_target('scene', 'scene_library', label='Library', role='branch', required=True),
        make_target('scene', 'scene_police', label='Police', role='branch', required=True),
        make_target(
            'scene',
            'scene_next_steps',
            label='Next Steps',
            role='exit',
            prerequisites=['scene_library', 'scene_police'],
        ),
    ]

    start_return = make_target('scene', 'scene_start', label='Start', role='return')

    next_steps_nav = default_navigation(completion_policy='optional_until_exit')
    next_steps_nav['allowed_targets'] = [
        make_target('scene', 'scene_graveyard', label='Graveyard', role='branch'),
        make_target('scene', 'scene_university', label='University', role='branch'),
        make_target('scene', 'scene_conclusion', label='Conclusion', role='exit'),
    ]
    next_steps_return = make_target('scene', 'scene_next_steps', label='Next Steps', role='return')

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
        _make_scene('scene_next_steps', 'Next steps hub', node_kind='hub', navigation=next_steps_nav),
        _make_scene(
            'scene_graveyard',
            'Graveyard branch',
            node_kind='branch',
            navigation={
                'allowed_targets': [],
                'return_target': next_steps_return,
                'completion_policy': 'terminal_on_resolve',
                'prerequisites': [],
                'close_unselected_on_advance': False,
            },
        ),
        _make_scene(
            'scene_university',
            'University branch',
            node_kind='branch',
            navigation={
                'allowed_targets': [],
                'return_target': next_steps_return,
                'completion_policy': 'terminal_on_resolve',
                'prerequisites': [],
                'close_unselected_on_advance': False,
            },
        ),
        _make_scene(
            'scene_conclusion',
            'Conclusion',
            node_kind='terminal',
            navigation=default_navigation(completion_policy='terminal_on_resolve'),
        ),
    ]
    db.insert_scenes(scenes)
    return db


def _build_exclusive_choice_db(db_path: Path) -> Database:
    if db_path.exists():
        db_path.unlink()
    db = Database(str(db_path))
    choice_nav = default_navigation(completion_policy='exclusive_choice')
    choice_nav['close_unselected_on_advance'] = True
    choice_nav['allowed_targets'] = [
        make_target('scene', 'scene_left', label='Left', role='branch'),
        make_target('scene', 'scene_right', label='Right', role='branch'),
    ]
    scenes = [
        _make_scene('scene_choice', 'Exclusive hub', node_kind='hub', navigation=choice_nav, status='in_progress'),
        _make_scene(
            'scene_left',
            'Left branch',
            node_kind='branch',
            navigation={'allowed_targets': [], 'completion_policy': 'terminal_on_resolve', 'prerequisites': [], 'close_unselected_on_advance': False},
        ),
        _make_scene(
            'scene_right',
            'Right branch',
            node_kind='branch',
            navigation={'allowed_targets': [], 'completion_policy': 'terminal_on_resolve', 'prerequisites': [], 'close_unselected_on_advance': False},
        ),
    ]
    db.insert_scenes(scenes)
    return db


def _build_nested_return_db(db_path: Path) -> Database:
    if db_path.exists():
        db_path.unlink()
    db = Database(str(db_path))
    outer_nav = default_navigation(completion_policy='optional_until_exit')
    outer_nav['allowed_targets'] = [
        make_target('scene', 'scene_inner_hub', label='Inner Hub', role='branch'),
        make_target('scene', 'scene_outer_alt', label='Outer Alternate', role='branch'),
    ]
    inner_nav = default_navigation(completion_policy='optional_until_exit')
    inner_nav['allowed_targets'] = [
        make_target('scene', 'scene_inner_a', label='Inner A', role='branch'),
        make_target('scene', 'scene_inner_b', label='Inner B', role='branch'),
    ]
    inner_nav['return_target'] = make_target('scene', 'scene_outer', label='Outer Hub', role='return')
    inner_return = make_target('scene', 'scene_inner_hub', label='Inner Hub', role='return')
    scenes = [
        _make_scene('scene_outer', 'Outer hub', node_kind='hub', navigation=outer_nav, status='in_progress'),
        _make_scene('scene_inner_hub', 'Inner hub', node_kind='hub', navigation=inner_nav),
        _make_scene(
            'scene_inner_a',
            'Inner A branch',
            node_kind='branch',
            navigation={'allowed_targets': [], 'return_target': inner_return, 'completion_policy': 'terminal_on_resolve', 'prerequisites': [], 'close_unselected_on_advance': False},
        ),
        _make_scene(
            'scene_inner_b',
            'Inner B branch',
            node_kind='branch',
            navigation={'allowed_targets': [], 'return_target': inner_return, 'completion_policy': 'terminal_on_resolve', 'prerequisites': [], 'close_unselected_on_advance': False},
        ),
        _make_scene(
            'scene_outer_alt',
            'Outer alternate branch',
            node_kind='branch',
            navigation={'allowed_targets': [], 'completion_policy': 'terminal_on_resolve', 'prerequisites': [], 'close_unselected_on_advance': False},
        ),
    ]
    db.insert_scenes(scenes)
    return db


def main() -> int:
    original_llm = state_module.call_nvidia_llm
    db = None
    exclusive_db = None
    nested_db = None
    try:
        state_module.call_nvidia_llm = _state_llm

        print('[test_state] case 1: story_position_context should read structured turn_state instead of parsing agent text')
        db = _build_scene_branch_db(ROOT / 'test' / 'debug_state_scene_branch.db')
        db.append_memory(
            'scene_start',
            'scene_start_plot_1',
            '',
            'You can follow either lead next.',
            visit_id=0,
            turn_state={
                'choice_open': True,
                'offered_targets': [
                    {'target_kind': 'scene', 'target_id': 'scene_library'},
                    {'target_kind': 'scene', 'target_id': 'scene_police'},
                ],
                'beat_status': 'open',
                'summary': 'the keeper left the player at a hub choice',
            },
        )
        start_ctx = story_position_context(db, 'scene_start', 'scene_start_plot_1', {}, current_visit_id=0)
        assert start_ctx['choice_prompt_active'] is True
        assert start_ctx['latest_turn_state']['offered_targets'][0]['target_id'] == 'scene_library'
        assert start_ctx['opening_choice_allowed'] is True

        print('[test_state] case 2: pre-response transition should follow llm-selected direct branch targets')
        library_eval = evaluate_pre_response_transition(
            'I go to the library.',
            plot_goal='Start hub',
            scene_goal='Start hub',
            scene_description='Two legal leads are open.',
            current_plot_raw_text='Start hub',
            current_node_kind='hub',
            allowed_targets=start_ctx['allowed_targets'],
            indirect_targets_via_return=start_ctx['indirect_targets_via_return'],
            remaining_required_targets=start_ctx['remaining_required_targets'],
            latest_agent_turn_excerpt=start_ctx['latest_agent_turn_excerpt'],
            latest_turn_state=start_ctx['latest_turn_state'],
            choice_prompt_active=start_ctx['choice_prompt_active'],
        )
        assert library_eval['action'] == 'target'
        assert library_eval['target_id'] == 'scene_library'
        assert library_eval['transition_path'] == 'direct'

        print('[test_state] case 3: open-choice answers should resolve through llm + turn_state, not numeric fallback code')
        numeric_eval = evaluate_pre_response_transition(
            '1',
            plot_goal='Start hub',
            scene_goal='Start hub',
            scene_description='Two legal leads are open.',
            current_plot_raw_text='Start hub',
            current_node_kind='hub',
            allowed_targets=start_ctx['allowed_targets'],
            indirect_targets_via_return=start_ctx['indirect_targets_via_return'],
            remaining_required_targets=start_ctx['remaining_required_targets'],
            latest_agent_turn_excerpt=start_ctx['latest_agent_turn_excerpt'],
            latest_turn_state=start_ctx['latest_turn_state'],
            choice_prompt_active=start_ctx['choice_prompt_active'],
        )
        assert numeric_eval['action'] == 'target'
        assert numeric_eval['target_id'] == 'scene_library'

        print('[test_state] case 4: branch-to-branch travel should route via return when llm selects an indirect target')
        to_library = next_story_position(
            db,
            'scene_start',
            'scene_start_plot_1',
            navigation_state={},
            advance_decision=library_eval,
            current_visit_id=0,
        )
        library_ctx = story_position_context(
            db,
            to_library['current_scene_id'],
            to_library['current_plot_id'],
            to_library['navigation_state'],
            current_visit_id=to_library['current_visit_id'],
        )
        via_return_eval = evaluate_pre_response_transition(
            'I head to the police station instead.',
            plot_goal='Library branch',
            scene_goal='Library branch',
            scene_description='The library is quiet.',
            current_plot_raw_text='Library branch',
            current_node_kind='branch',
            allowed_targets=library_ctx['allowed_targets'],
            indirect_targets_via_return=library_ctx['indirect_targets_via_return'],
            remaining_required_targets=library_ctx['remaining_required_targets'],
            latest_agent_turn_excerpt=library_ctx['latest_agent_turn_excerpt'],
            latest_turn_state=library_ctx['latest_turn_state'],
            choice_prompt_active=library_ctx['choice_prompt_active'],
        )
        assert via_return_eval['action'] == 'target'
        assert via_return_eval['target_id'] == 'scene_police'
        assert via_return_eval['transition_path'] == 'via_return'

        print('[test_state] case 5: off-topic input should conservatively stay in the current plot')
        off_topic_eval = evaluate_pre_response_transition(
            'I ask about the weather.',
            plot_goal='Library branch',
            scene_goal='Library branch',
            scene_description='The library is quiet.',
            current_plot_raw_text='Library branch',
            current_node_kind='branch',
            allowed_targets=library_ctx['allowed_targets'],
            indirect_targets_via_return=library_ctx['indirect_targets_via_return'],
            latest_turn_state=library_ctx['latest_turn_state'],
            choice_prompt_active=library_ctx['choice_prompt_active'],
        )
        assert off_topic_eval['action'] == 'stay'
        assert off_topic_eval['reason'] == 'llm_detected_off_topic'

        print('[test_state] case 6: llm-unavailable transition and completion evaluations should stay conservative')
        state_module.call_nvidia_llm = _failing_llm
        failed_pre_eval = evaluate_pre_response_transition(
            'I go to the library.',
            plot_goal='Start hub',
            scene_goal='Start hub',
            scene_description='Two legal leads are open.',
            current_plot_raw_text='Start hub',
            current_node_kind='hub',
            allowed_targets=start_ctx['allowed_targets'],
            indirect_targets_via_return=start_ctx['indirect_targets_via_return'],
            latest_turn_state=start_ctx['latest_turn_state'],
            choice_prompt_active=start_ctx['choice_prompt_active'],
        )
        failed_plot_eval = evaluate_plot_completion(
            'I search the shelves for the obituary.',
            'The search continues.',
            0.4,
            [],
            plot_goal='Library branch',
            scene_goal='Library branch',
            scene_description='The library is quiet.',
            current_plot_raw_text='Library branch',
            current_node_kind='branch',
            allowed_targets=library_ctx['allowed_targets'],
            indirect_targets_via_return=library_ctx['indirect_targets_via_return'],
            latest_turn_state=library_ctx['latest_turn_state'],
            choice_prompt_active=library_ctx['choice_prompt_active'],
        )
        assert failed_pre_eval['action'] == 'stay' and failed_pre_eval['reason'] == 'llm_unavailable_stay'
        assert failed_plot_eval['action'] == 'stay' and failed_plot_eval['progress'] == 0.4
        assert failed_plot_eval['reason'] == 'llm_unavailable_stay'

        state_module.call_nvidia_llm = _state_llm

        print('[test_state] case 7: branch completion should close and return to the parent hub when llm says the beat is wrapped')
        completion_eval = evaluate_plot_completion(
            'I search the shelves for the obituary.',
            'You recover the obituary and the lead is wrapped.',
            0.4,
            [],
            plot_goal='Library branch',
            scene_goal='Library branch',
            scene_description='The library is quiet.',
            current_plot_raw_text='Library branch',
            current_node_kind='branch',
            allowed_targets=library_ctx['allowed_targets'],
            indirect_targets_via_return=library_ctx['indirect_targets_via_return'],
            latest_turn_state=library_ctx['latest_turn_state'],
            choice_prompt_active=library_ctx['choice_prompt_active'],
        )
        assert completion_eval['completed'] is True
        assert completion_eval['close_current'] is True
        assert completion_eval['progress'] == 1.0

        print('[test_state] case 8: required hub exits should stay blocked until all required branches are complete')
        back_to_start = next_story_position(
            db,
            to_library['current_scene_id'],
            to_library['current_plot_id'],
            navigation_state=to_library['navigation_state'],
            advance_decision={'action': 'stay', 'close_current': True},
            current_visit_id=to_library['current_visit_id'],
        )
        start_after_library = story_position_context(
            db,
            back_to_start['current_scene_id'],
            back_to_start['current_plot_id'],
            back_to_start['navigation_state'],
            current_visit_id=back_to_start['current_visit_id'],
        )
        exit_ids_after_library = {target['target_id'] for target in start_after_library['eligible_targets'] if target.get('role') == 'exit'}
        assert 'scene_next_steps' not in exit_ids_after_library

        to_police = next_story_position(
            db,
            back_to_start['current_scene_id'],
            back_to_start['current_plot_id'],
            navigation_state=back_to_start['navigation_state'],
            advance_decision={
                'action': 'target',
                'target_kind': 'scene',
                'target_id': 'scene_police',
                'transition_path': 'direct',
                'close_current': False,
            },
            current_visit_id=back_to_start['current_visit_id'],
        )
        back_from_police = next_story_position(
            db,
            to_police['current_scene_id'],
            to_police['current_plot_id'],
            navigation_state=to_police['navigation_state'],
            advance_decision={'action': 'stay', 'close_current': True},
            current_visit_id=to_police['current_visit_id'],
        )
        start_after_police = story_position_context(
            db,
            back_from_police['current_scene_id'],
            back_from_police['current_plot_id'],
            back_from_police['navigation_state'],
            current_visit_id=back_from_police['current_visit_id'],
        )
        exit_ids_after_police = {target['target_id'] for target in start_after_police['eligible_targets'] if target.get('role') == 'exit'}
        assert 'scene_next_steps' in exit_ids_after_police

        print('[test_state] case 9: optional hub branches should return and keep exits available')
        to_next_steps = next_story_position(
            db,
            back_from_police['current_scene_id'],
            back_from_police['current_plot_id'],
            navigation_state=back_from_police['navigation_state'],
            advance_decision={
                'action': 'target',
                'target_kind': 'scene',
                'target_id': 'scene_next_steps',
                'transition_path': 'direct',
                'close_current': True,
            },
            current_visit_id=back_from_police['current_visit_id'],
        )
        to_graveyard = next_story_position(
            db,
            to_next_steps['current_scene_id'],
            to_next_steps['current_plot_id'],
            navigation_state=to_next_steps['navigation_state'],
            advance_decision={
                'action': 'target',
                'target_kind': 'scene',
                'target_id': 'scene_graveyard',
                'transition_path': 'direct',
                'close_current': False,
            },
            current_visit_id=to_next_steps['current_visit_id'],
        )
        back_to_next_steps = next_story_position(
            db,
            to_graveyard['current_scene_id'],
            to_graveyard['current_plot_id'],
            navigation_state=to_graveyard['navigation_state'],
            advance_decision={'action': 'stay', 'close_current': True},
            current_visit_id=to_graveyard['current_visit_id'],
        )
        next_steps_ctx = story_position_context(
            db,
            back_to_next_steps['current_scene_id'],
            back_to_next_steps['current_plot_id'],
            back_to_next_steps['navigation_state'],
            current_visit_id=back_to_next_steps['current_visit_id'],
        )
        next_steps_exit_ids = {target['target_id'] for target in next_steps_ctx['eligible_targets'] if target.get('role') == 'exit'}
        assert 'scene_conclusion' in next_steps_exit_ids

        print('[test_state] case 10: exclusive-choice hubs should skip unselected siblings on advance')
        exclusive_db = _build_exclusive_choice_db(ROOT / 'test' / 'debug_state_exclusive_choice.db')
        next_story_position(
            exclusive_db,
            'scene_choice',
            'scene_choice_plot_1',
            navigation_state={},
            advance_decision={
                'action': 'target',
                'target_kind': 'scene',
                'target_id': 'scene_left',
                'transition_path': 'direct',
                'close_current': False,
            },
            current_visit_id=0,
        )
        assert exclusive_db.get_scene('scene_right')['status'] == 'skipped'

        print('[test_state] case 11: nested via_return should expose both sibling and outer-hub targets')
        nested_db = _build_nested_return_db(ROOT / 'test' / 'debug_state_nested_return.db')
        to_inner_hub = next_story_position(
            nested_db,
            'scene_outer',
            'scene_outer_plot_1',
            navigation_state={},
            advance_decision={
                'action': 'target',
                'target_kind': 'scene',
                'target_id': 'scene_inner_hub',
                'transition_path': 'direct',
                'close_current': False,
            },
            current_visit_id=0,
        )
        to_inner_a = next_story_position(
            nested_db,
            to_inner_hub['current_scene_id'],
            to_inner_hub['current_plot_id'],
            navigation_state=to_inner_hub['navigation_state'],
            advance_decision={
                'action': 'target',
                'target_kind': 'scene',
                'target_id': 'scene_inner_a',
                'transition_path': 'direct',
                'close_current': False,
            },
            current_visit_id=to_inner_hub['current_visit_id'],
        )
        nested_ctx = story_position_context(
            nested_db,
            to_inner_a['current_scene_id'],
            to_inner_a['current_plot_id'],
            to_inner_a['navigation_state'],
            current_visit_id=to_inner_a['current_visit_id'],
        )
        nested_indirect_ids = {target['target_id'] for target in nested_ctx['indirect_targets_via_return']}
        assert 'scene_inner_b' in nested_indirect_ids
        assert 'scene_outer_alt' in nested_indirect_ids

        print('[test_state] result: PASS')
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f'[test_state] result: FAIL -> {exc}')
        return 1
    finally:
        state_module.call_nvidia_llm = original_llm
        for handle in (db, exclusive_db, nested_db):
            if handle is not None:
                handle.close()


if __name__ == '__main__':
    raise SystemExit(main())
