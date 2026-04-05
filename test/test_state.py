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
    evaluate_scene_completion,
    next_story_position,
    story_position_context,
)


def _failing_llm(*args, **kwargs):
    raise RuntimeError('offline test fallback')


def _capturing_llm(prompt: str, *args, **kwargs):
    step_name = kwargs.get('step_name', '')
    model = kwargs.get('model', '')
    allow_env_override = kwargs.get('allow_env_override', True)
    if step_name == 'pre_response_transition_evaluation':
        assert model == 'qwen/qwen3.5-397b-a17b'
        assert allow_env_override is False
        return (
            '{"action": "target", "target_kind": "scene", "target_id": "scene_library", '
            '"transition_path": "direct", "close_current": false, "confidence": 0.96, '
            '"reason": "high-confidence library choice"}'
        )
    if step_name == 'plot_completion_evaluation':
        assert model == 'qwen/qwen3.5-397b-a17b'
        assert allow_env_override is False
        return (
            '{"completed": false, "progress_delta": 0.0, "action": "stay", '
            '"target_kind": "", "target_id": "", "close_current": false, '
            '"transition_path": "stay", "objective_satisfied": false, "confidence": 0.45, '
            '"reason": "still working current plot"}'
        )
    raise AssertionError(f'unexpected step_name={step_name}')


def _chinese_transition_llm(prompt: str, *args, **kwargs):
    step_name = kwargs.get('step_name', '')
    model = kwargs.get('model', '')
    allow_env_override = kwargs.get('allow_env_override', True)
    assert model == 'qwen/qwen3.5-397b-a17b'
    assert allow_env_override is False
    if step_name == 'pre_response_transition_evaluation':
        if 'Current Node Kind:\nhub' in prompt:
            return (
                '{"action": "target", "target_kind": "scene", "target_id": "scene_police", '
                '"transition_path": "direct", "close_current": false, "confidence": 0.98, '
                '"reason": "player explicitly chose the police lead in Chinese"}'
            )
        if 'Current Node Kind:\nbranch' in prompt:
            return (
                '{"action": "target", "target_kind": "scene", "target_id": "scene_police", '
                '"transition_path": "via_return", "close_current": true, "confidence": 0.94, '
                '"reason": "player chose a lead reachable after returning"}'
            )
    if step_name == 'plot_completion_evaluation':
        return (
            '{"completed": false, "progress_delta": 0.0, "action": "stay", '
            '"target_kind": "", "target_id": "", "transition_path": "stay", '
            '"objective_satisfied": false, "close_current": false, "confidence": 0.32, '
            '"reason": "keep current plot"}'
        )
    raise AssertionError(f'unexpected step_name={step_name}')


def _choice_resolution_llm(prompt: str, *args, **kwargs):
    step_name = kwargs.get('step_name', '')
    if step_name != 'pre_response_transition_evaluation':
        raise AssertionError(f'unexpected step_name={step_name}')
    if 'immediately previous choice prompt' in prompt:
        return (
            '{"action": "target", "target_kind": "scene", "target_id": "scene_library", '
            '"transition_path": "direct", "close_current": false, "confidence": 0.93, '
            '"reason": "resolved the short answer against the prior choice prompt"}'
        )
    return (
        '{"action": "stay", "target_kind": "", "target_id": "", '
        '"transition_path": "stay", "close_current": false, "confidence": 0.15, '
        '"reason": "generic evaluator stayed conservative"}'
    )


def _progress_full_llm(prompt: str, *args, **kwargs):
    step_name = kwargs.get('step_name', '')
    if step_name != 'plot_completion_evaluation':
        raise AssertionError(f'unexpected step_name={step_name}')
    return (
        '{"completed": false, "progress_delta": 0.6, "action": "stay", '
        '"target_kind": "", "target_id": "", "transition_path": "stay", '
        '"objective_satisfied": false, "close_current": false, "confidence": 0.41, '
        '"reason": "model left completion undecided"}'
    )


def _make_mirrored_scene(
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
    terminal_nav = default_navigation(completion_policy='terminal_on_resolve')

    scenes = [
        _make_mirrored_scene('scene_start', 'Start hub', node_kind='hub', navigation=start_nav, status='in_progress'),
        _make_mirrored_scene(
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
        _make_mirrored_scene(
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
        _make_mirrored_scene('scene_next_steps', 'Next steps hub', node_kind='hub', navigation=next_steps_nav),
        _make_mirrored_scene(
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
        _make_mirrored_scene(
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
        _make_mirrored_scene('scene_conclusion', 'Conclusion', node_kind='terminal', navigation=terminal_nav),
    ]
    db.insert_scenes(scenes)
    return db


def _build_plot_branch_db(db_path: Path) -> Database:
    if db_path.exists():
        db_path.unlink()

    db = Database(str(db_path))
    hub_nav = default_navigation(completion_policy='exclusive_choice', close_unselected_on_advance=True)
    hub_nav['allowed_targets'] = [
        make_target('plot', 'scene_choice_plot_2', label='Exploring', role='branch'),
        make_target('plot', 'scene_choice_plot_3', label='Ignoring', role='branch'),
    ]

    db.insert_scenes(
        [
            {
                'scene_id': 'scene_choice',
                'scene_goal': 'A conditional plot choice scene.',
                'scene_description': 'The investigator must choose one route.',
                'status': 'in_progress',
                'scene_summary': '',
                'plots': [
                    {
                        'plot_id': 'scene_choice_plot_1',
                        'plot_goal': 'Present the conditional choice.',
                        'mandatory_events': [],
                        'npc': [],
                        'locations': [],
                        'raw_text': 'The investigator may explore or ignore the sound.',
                        'status': 'in_progress',
                        'progress': 0.0,
                        'node_kind': 'hub',
                        'navigation': hub_nav,
                    },
                    {
                        'plot_id': 'scene_choice_plot_2',
                        'plot_goal': 'Exploring the sound.',
                        'mandatory_events': [],
                        'npc': [],
                        'locations': [],
                        'raw_text': 'Exploring the sound reveals a hidden chamber.',
                        'status': 'pending',
                        'progress': 0.0,
                        'node_kind': 'branch',
                        'navigation': default_navigation(completion_policy='optional_until_exit'),
                    },
                    {
                        'plot_id': 'scene_choice_plot_3',
                        'plot_goal': 'Ignoring the sound.',
                        'mandatory_events': [],
                        'npc': [],
                        'locations': [],
                        'raw_text': 'Ignoring the sound keeps the danger unseen.',
                        'status': 'pending',
                        'progress': 0.0,
                        'node_kind': 'branch',
                        'navigation': default_navigation(completion_policy='optional_until_exit'),
                    },
                ],
            }
        ]
    )
    return db


def _build_scene_plot_dead_end_db(db_path: Path) -> Database:
    if db_path.exists():
        db_path.unlink()

    db = Database(str(db_path))

    start_nav = default_navigation(completion_policy='all_required_then_advance')
    start_nav['allowed_targets'] = [
        make_target('scene', 'scene_dead_end', label='Dead End', role='branch', required=True),
        make_target('scene', 'scene_fresh', label='Fresh Lead', role='branch', required=True),
        make_target(
            'scene',
            'scene_next_steps',
            label='Next Steps',
            role='exit',
            prerequisites=['scene_dead_end', 'scene_fresh'],
        ),
    ]
    start_return = make_target('scene', 'scene_start', label='Start', role='return')

    db.insert_scenes(
        [
            {
                'scene_id': 'scene_start',
                'scene_goal': 'Start hub',
                'scene_description': 'Choose where to investigate next.',
                'status': 'in_progress',
                'scene_summary': '',
                'node_kind': 'hub',
                'navigation': start_nav,
                'plots': [
                    {
                        'plot_id': 'scene_start_plot_1',
                        'plot_goal': 'Choose a lead',
                        'mandatory_events': [],
                        'npc': [],
                        'locations': [],
                        'raw_text': 'You may revisit the dead-end lead or pursue the fresh lead.',
                        'status': 'in_progress',
                        'progress': 0.0,
                        'node_kind': 'hub',
                        'navigation': start_nav,
                    }
                ],
            },
            {
                'scene_id': 'scene_dead_end',
                'scene_goal': 'Dead-end branch scene',
                'scene_description': 'A branch whose final plot has no legal plot-level next step left.',
                'status': 'in_progress',
                'scene_summary': '',
                'node_kind': 'branch',
                'navigation': {
                    'allowed_targets': [],
                    'return_target': start_return,
                    'completion_policy': 'terminal_on_resolve',
                    'prerequisites': [],
                    'close_unselected_on_advance': False,
                },
                'plots': [
                    {
                        'plot_id': 'scene_dead_end_plot_1',
                        'plot_goal': 'Follow a stale lead',
                        'mandatory_events': [],
                        'npc': [],
                        'locations': [],
                        'raw_text': 'The stale lead only points back to an already exhausted branch.',
                        'status': 'in_progress',
                        'progress': 1.0,
                        'node_kind': 'linear',
                        'navigation': {
                            'allowed_targets': [
                                make_target('scene', 'scene_done', label='Already exhausted lead', role=''),
                            ],
                            'return_target': None,
                            'completion_policy': 'all_required_then_advance',
                            'prerequisites': [],
                            'close_unselected_on_advance': False,
                        },
                    }
                ],
            },
            {
                'scene_id': 'scene_fresh',
                'scene_goal': 'Fresh branch scene',
                'scene_description': 'A remaining START branch.',
                'status': 'pending',
                'scene_summary': '',
                'node_kind': 'branch',
                'navigation': {
                    'allowed_targets': [],
                    'return_target': start_return,
                    'completion_policy': 'terminal_on_resolve',
                    'prerequisites': [],
                    'close_unselected_on_advance': False,
                },
                'plots': [
                    {
                        'plot_id': 'scene_fresh_plot_1',
                        'plot_goal': 'Investigate the fresh lead',
                        'mandatory_events': [],
                        'npc': [],
                        'locations': [],
                        'raw_text': 'There is still something new to investigate here.',
                        'status': 'pending',
                        'progress': 0.0,
                        'node_kind': 'branch',
                        'navigation': {
                            'allowed_targets': [],
                            'return_target': start_return,
                            'completion_policy': 'terminal_on_resolve',
                            'prerequisites': [],
                            'close_unselected_on_advance': False,
                        },
                    }
                ],
            },
            {
                'scene_id': 'scene_done',
                'scene_goal': 'Already exhausted lead',
                'scene_description': 'This branch is already complete.',
                'status': 'completed',
                'scene_summary': '',
                'node_kind': 'branch',
                'navigation': {
                    'allowed_targets': [],
                    'return_target': start_return,
                    'completion_policy': 'terminal_on_resolve',
                    'prerequisites': [],
                    'close_unselected_on_advance': False,
                },
                'plots': [
                    {
                        'plot_id': 'scene_done_plot_1',
                        'plot_goal': 'An exhausted branch',
                        'mandatory_events': [],
                        'npc': [],
                        'locations': [],
                        'raw_text': 'Nothing new remains here.',
                        'status': 'completed',
                        'progress': 1.0,
                        'node_kind': 'branch',
                        'navigation': {
                            'allowed_targets': [],
                            'return_target': start_return,
                            'completion_policy': 'terminal_on_resolve',
                            'prerequisites': [],
                            'close_unselected_on_advance': False,
                        },
                    }
                ],
            },
            _make_mirrored_scene('scene_next_steps', 'Next steps hub', node_kind='hub', navigation=default_navigation(completion_policy='optional_until_exit')),
        ]
    )
    return db


def _build_nested_via_return_db(db_path: Path) -> Database:
    if db_path.exists():
        db_path.unlink()

    db = Database(str(db_path))

    start_nav = default_navigation(completion_policy='all_required_then_advance')
    start_nav['allowed_targets'] = [
        make_target('scene', 'scene_police', label='Police station', role='branch', required=True),
        make_target('scene', 'scene_newspaper', label='Arnoldsburg Advertiser', role='branch', required=True),
    ]
    start_return = make_target('scene', 'scene_start', label='Start hub', role='return')
    police_hub_return = make_target('scene', 'scene_start', label='Back to town leads', role='return')
    police_plot_hub_nav = default_navigation(completion_policy='optional_until_exit')
    police_plot_hub_nav['allowed_targets'] = [
        make_target('plot', 'scene_police_plot_2', label='Ask about recent burglaries', role='branch'),
        police_hub_return,
    ]
    police_plot_return = make_target('plot', 'scene_police_plot_1', label='Police desk', role='return')

    db.insert_scenes(
        [
            _make_mirrored_scene('scene_start', 'Start hub', node_kind='hub', navigation=start_nav, status='in_progress'),
            {
                'scene_id': 'scene_police',
                'scene_goal': 'Police station',
                'scene_description': 'The desk officer can point you toward the next lead.',
                'status': 'pending',
                'scene_summary': '',
                'node_kind': 'branch',
                'navigation': {
                    'allowed_targets': [],
                    'return_target': start_return,
                    'completion_policy': 'terminal_on_resolve',
                    'prerequisites': [],
                    'close_unselected_on_advance': False,
                },
                'plots': [
                    {
                        'plot_id': 'scene_police_plot_1',
                        'plot_goal': 'Choose which police angle to pursue.',
                        'mandatory_events': [],
                        'npc': [],
                        'locations': [],
                        'raw_text': 'The officer offers a burglary file and can also send you back to the town leads.',
                        'status': 'pending',
                        'progress': 0.0,
                        'node_kind': 'hub',
                        'navigation': police_plot_hub_nav,
                    },
                    {
                        'plot_id': 'scene_police_plot_2',
                        'plot_goal': 'Ask about recent burglaries.',
                        'mandatory_events': [],
                        'npc': [],
                        'locations': [],
                        'raw_text': 'The officer reviews a burglary pattern and mentions the paper may have old morgue files.',
                        'status': 'pending',
                        'progress': 0.0,
                        'node_kind': 'branch',
                        'navigation': {
                            'allowed_targets': [],
                            'return_target': police_plot_return,
                            'completion_policy': 'terminal_on_resolve',
                            'prerequisites': [],
                            'close_unselected_on_advance': False,
                        },
                    },
                ],
            },
            _make_mirrored_scene(
                'scene_newspaper',
                'Search the Arnoldsburg Advertiser archives.',
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
    )
    return db


def main() -> int:
    original_llm = state_module.call_nvidia_llm
    try:
        state_module.call_nvidia_llm = _failing_llm

        scene_db = _build_scene_branch_db(ROOT / 'test' / 'debug_state_scene_graph.db')

        print('[test_state] case 1: START hub should expose required branch scenes and gate exit')
        start_ctx = story_position_context(scene_db, 'scene_start', 'scene_start_plot_1', {})
        start_targets = {(target['target_id'], target['role']): target for target in start_ctx['allowed_targets']}
        assert start_ctx['current_node_kind'] == 'hub'
        assert start_ctx['remaining_required_targets_summary'] != 'None'
        assert start_targets[('scene_library', 'branch')]['eligible'] is True
        assert start_targets[('scene_police', 'branch')]['eligible'] is True
        assert start_targets[('scene_next_steps', 'exit')]['eligible'] is False
        assert 'scene_library' in start_ctx['eligible_targets_summary']
        assert 'scene_next_steps' in start_ctx['blocked_targets_summary']

        print('[test_state] case 2: choosing a START branch should enter that branch and push return stack')
        to_library = next_story_position(
            scene_db,
            'scene_start',
            'scene_start_plot_1',
            {},
            {
                'action': 'target',
                'target_kind': 'scene',
                'target_id': 'scene_library',
                'close_current': False,
                'reason': 'player chose library',
            },
            current_visit_id=0,
        )
        assert to_library['current_scene_id'] == 'scene_library'
        assert to_library['current_plot_id'] == 'scene_library_plot_1'
        assert to_library['current_visit_id'] == 1
        assert to_library['navigation_state']['return_stack'] == [
            {'target_kind': 'scene', 'target_id': 'scene_start', 'visit_id': 0}
        ]

        print('[test_state] case 3: finishing first branch should return to START instead of advancing')
        scene_db.update_plot('scene_library_plot_1', status='completed', progress=1.0)
        scene_db.update_scene('scene_library', {'status': 'completed'})
        back_to_start = next_story_position(
            scene_db,
            'scene_library',
            'scene_library_plot_1',
            to_library['navigation_state'],
            {
                'action': 'stay',
                'target_kind': '',
                'target_id': '',
                'close_current': True,
                'reason': 'branch complete',
            },
            current_visit_id=to_library['current_visit_id'],
        )
        assert back_to_start['current_scene_id'] == 'scene_start'
        assert back_to_start['current_plot_id'] == 'scene_start_plot_1'
        assert back_to_start['current_visit_id'] == 2
        assert back_to_start['navigation_state']['return_stack'] == []
        start_return_ctx = story_position_context(
            scene_db,
            'scene_start',
            'scene_start_plot_1',
            back_to_start['navigation_state'],
        )
        assert 'progress=50%' in start_return_ctx['current_hub_status']
        assert 'scene_police' in start_return_ctx['eligible_targets_summary']
        assert 'scene_library' in start_return_ctx['exhausted_targets_summary']

        print('[test_state] case 4: finishing all START branches should return to START and expose the exit')
        to_police = next_story_position(
            scene_db,
            'scene_start',
            'scene_start_plot_1',
            back_to_start['navigation_state'],
            {
                'action': 'target',
                'target_kind': 'scene',
                'target_id': 'scene_police',
                'close_current': False,
                'reason': 'player chose police',
            },
            current_visit_id=back_to_start['current_visit_id'],
        )
        assert to_police['current_scene_id'] == 'scene_police'
        scene_db.update_plot('scene_police_plot_1', status='completed', progress=1.0)
        scene_db.update_scene('scene_police', {'status': 'completed'})
        to_next_steps = next_story_position(
            scene_db,
            'scene_police',
            'scene_police_plot_1',
            to_police['navigation_state'],
            {
                'action': 'stay',
                'target_kind': '',
                'target_id': '',
                'close_current': True,
                'reason': 'branch complete',
            },
            current_visit_id=to_police['current_visit_id'],
        )
        assert to_next_steps['current_scene_id'] == 'scene_start'
        assert to_next_steps['current_plot_id'] == 'scene_start_plot_1'
        completed_start_ctx = story_position_context(
            scene_db,
            'scene_start',
            'scene_start_plot_1',
            to_next_steps['navigation_state'],
        )
        completed_start_targets = {(target['target_id'], target['role']): target for target in completed_start_ctx['allowed_targets']}
        assert completed_start_targets[('scene_next_steps', 'exit')]['eligible'] is True
        assert 'scene_next_steps' in completed_start_ctx['eligible_targets_summary']
        assert 'scene_library' in completed_start_ctx['exhausted_targets_summary']
        assert 'scene_police' in completed_start_ctx['exhausted_targets_summary']

        print('[test_state] case 5: a completed hub should stay open until the player explicitly takes the exit')
        completed_hub_eval = evaluate_plot_completion(
            'I pause and think through the leads.',
            'Thomas lets the remaining options hang in the air.',
            1.0,
            [],
            plot_goal='Start hub',
            scene_goal='Start hub',
            current_plot_raw_text='The remaining leads still wait.',
            current_node_kind='hub',
            allowed_targets=completed_start_ctx['allowed_targets'],
            remaining_required_targets=completed_start_ctx['remaining_required_targets'],
            conversation_history=[],
        )
        assert completed_hub_eval['completed'] is False
        assert completed_hub_eval['close_current'] is False
        completed_hub_scene_eval = evaluate_scene_completion(
            scene_db,
            'scene_start',
            current_plot_id='scene_start_plot_1',
            plot_transition={
                'action': 'stay',
                'target_kind': '',
                'target_id': '',
                'close_current': False,
                'reason': 'waiting on player choice',
            },
            navigation_state=to_next_steps['navigation_state'],
            conversation_history=[],
            latest_turn_text='User: I think quietly.\nAgent: The remaining leads still wait.',
        )
        assert completed_hub_scene_eval['completed'] is False
        assert completed_hub_scene_eval['reason'] == 'hub_still_open'
        to_next_steps = next_story_position(
            scene_db,
            'scene_start',
            'scene_start_plot_1',
            to_next_steps['navigation_state'],
            {
                'action': 'target',
                'target_kind': 'scene',
                'target_id': 'scene_next_steps',
                'close_current': True,
                'reason': 'player is ready for next steps',
            },
            current_visit_id=to_next_steps['current_visit_id'],
        )
        assert to_next_steps['current_scene_id'] == 'scene_next_steps'
        assert to_next_steps['current_plot_id'] == 'scene_next_steps_plot_1'

        print('[test_state] case 6: NEXT STEPS branch should return to hub until player exits')
        to_graveyard = next_story_position(
            scene_db,
            'scene_next_steps',
            'scene_next_steps_plot_1',
            to_next_steps['navigation_state'],
            {
                'action': 'target',
                'target_kind': 'scene',
                'target_id': 'scene_graveyard',
                'close_current': False,
                'reason': 'player chose graveyard',
            },
            current_visit_id=to_next_steps['current_visit_id'],
        )
        assert to_graveyard['current_scene_id'] == 'scene_graveyard'
        scene_db.update_plot('scene_graveyard_plot_1', status='completed', progress=1.0)
        scene_db.update_scene('scene_graveyard', {'status': 'completed'})
        back_to_next_steps = next_story_position(
            scene_db,
            'scene_graveyard',
            'scene_graveyard_plot_1',
            to_graveyard['navigation_state'],
            {
                'action': 'stay',
                'target_kind': '',
                'target_id': '',
                'close_current': True,
                'reason': 'branch complete',
            },
            current_visit_id=to_graveyard['current_visit_id'],
        )
        assert back_to_next_steps['current_scene_id'] == 'scene_next_steps'
        next_steps_ctx = story_position_context(
            scene_db,
            'scene_next_steps',
            'scene_next_steps_plot_1',
            back_to_next_steps['navigation_state'],
        )
        next_targets = {(target['target_id'], target['role']): target for target in next_steps_ctx['allowed_targets']}
        assert next_steps_ctx['current_node_kind'] == 'hub'
        assert next_targets[('scene_conclusion', 'exit')]['eligible'] is True

        print('[test_state] case 7: optional hub exit should skip unvisited sibling scenes')
        to_conclusion = next_story_position(
            scene_db,
            'scene_next_steps',
            'scene_next_steps_plot_1',
            back_to_next_steps['navigation_state'],
            {
                'action': 'target',
                'target_kind': 'scene',
                'target_id': 'scene_conclusion',
                'close_current': True,
                'reason': 'enough evidence for ending',
            },
            current_visit_id=back_to_next_steps['current_visit_id'],
        )
        assert to_conclusion['current_scene_id'] == 'scene_conclusion'
        assert scene_db.get_scene('scene_university')['status'] == 'skipped'
        assert scene_db.get_plot('scene_university_plot_1')['status'] == 'skipped'

        print('[test_state] case 8: explicit abandonment should also close a branch scene and return to its hub')
        abandon_db = _build_scene_branch_db(ROOT / 'test' / 'debug_state_scene_abandon.db')
        to_library_again = next_story_position(
            abandon_db,
            'scene_start',
            'scene_start_plot_1',
            {},
            {
                'action': 'target',
                'target_kind': 'scene',
                'target_id': 'scene_library',
                'close_current': False,
                'reason': 'player chose library',
            },
            current_visit_id=0,
        )
        abandon_eval = evaluate_plot_completion(
            '算了，这里没什么可查的，我们回去吧。',
            'You step away from the archive tables.',
            0.2,
            [],
            plot_goal='Library branch',
            scene_goal='Library branch',
            current_plot_raw_text='Dusty shelves and old newsprint surround you.',
            current_node_kind='branch',
            allowed_targets=[],
            remaining_required_targets=[],
            conversation_history=[],
        )
        assert abandon_eval['completed'] is True
        assert abandon_eval['close_current'] is True
        abandon_db.update_plot('scene_library_plot_1', status='completed', progress=1.0)
        abandon_db.update_scene('scene_library', {'status': 'completed'})
        abandoned_return = next_story_position(
            abandon_db,
            'scene_library',
            'scene_library_plot_1',
            to_library_again['navigation_state'],
            {
                'action': 'stay',
                'target_kind': '',
                'target_id': '',
                'close_current': True,
                'reason': 'explicit abandonment',
            },
            current_visit_id=to_library_again['current_visit_id'],
        )
        assert abandoned_return['current_scene_id'] == 'scene_start'
        assert abandoned_return['current_plot_id'] == 'scene_start_plot_1'
        abandon_db.close()

        print('[test_state] case 9: off-topic unrelated input should not end the current plot')
        off_topic_eval = evaluate_plot_completion(
            'I suddenly discuss the weather and nearby restaurants.',
            'The archive remains quiet around you.',
            0.35,
            ['Search the obituary archive'],
            plot_goal='Library branch',
            scene_goal='Investigate the library branch',
            scene_description='You are inside the newspaper archive room.',
            current_plot_raw_text='Dusty newspaper shelves and bound ledgers fill the archive room.',
            current_node_kind='branch',
            allowed_targets=[],
            remaining_required_targets=[],
            redirect_streak=1,
            conversation_history=[],
        )
        assert off_topic_eval['completed'] is False
        assert off_topic_eval['close_current'] is False
        assert off_topic_eval['action'] == 'stay'
        assert off_topic_eval['progress'] == 0.35
        assert off_topic_eval['reason'] == 'off_topic_unrelated'

        print('[test_state] case 10: intro setup with a single downstream target should auto-advance on a generic reply')
        intro_allowed_targets = [dict(make_target('scene', 'scene_start', label='Start hub', role='exit'), eligible=True)]
        intro_pre_eval = evaluate_pre_response_transition(
            '好的。',
            plot_goal='Introduce the case to the investigator',
            scene_goal='Initial setup with the client',
            scene_description='Thomas explains the burglary and his missing uncle.',
            current_plot_raw_text='Thomas Kimball introduces the case and asks for help.',
            current_node_kind='linear',
            allowed_targets=intro_allowed_targets,
            remaining_required_targets=[],
            mandatory_events=[],
            redirect_streak=0,
            latest_agent_turn_excerpt='Thomas finishes the initial briefing and waits for your response.',
            choice_prompt_active=False,
            conversation_history=[],
        )
        assert intro_pre_eval['action'] == 'target'
        assert intro_pre_eval['target_id'] == 'scene_start'
        assert intro_pre_eval['close_current'] is True
        assert intro_pre_eval['reason'] == 'intro_generic_reply_transition'

        intro_plot_eval = evaluate_plot_completion(
            '好的。',
            '托马斯点点头，准备把后续可调查的方向告诉你。',
            0.0,
            [],
            plot_goal='Introduce the case to the investigator',
            scene_goal='Initial setup with the client',
            scene_description='Thomas explains the burglary and his missing uncle.',
            current_plot_raw_text='Thomas Kimball introduces the case and asks for help.',
            current_node_kind='linear',
            allowed_targets=intro_allowed_targets,
            remaining_required_targets=[],
            redirect_streak=0,
            latest_agent_turn_excerpt='Thomas finishes the initial briefing and waits for your response.',
            choice_prompt_active=False,
            conversation_history=[],
        )
        assert intro_plot_eval['completed'] is True
        assert intro_plot_eval['action'] == 'target'
        assert intro_plot_eval['target_id'] == 'scene_start'
        assert intro_plot_eval['close_current'] is True
        assert intro_plot_eval['reason'] == 'intro_generic_reply_transition'

        print('[test_state] case 11: progression evaluators should explicitly use the 397b model')
        state_module.call_nvidia_llm = _capturing_llm
        pre_response_eval = evaluate_pre_response_transition(
            'I go straight to the library.',
            plot_goal='Choose a lead',
            scene_goal='Start hub',
            scene_description='Thomas lays out the case in his office.',
            current_plot_raw_text='You can visit the library or speak to the police.',
            current_node_kind='hub',
            allowed_targets=start_ctx['allowed_targets'],
            remaining_required_targets=start_ctx['remaining_required_targets'],
            mandatory_events=[],
            redirect_streak=0,
            conversation_history=[],
        )
        assert pre_response_eval['action'] == 'target'
        assert pre_response_eval['target_id'] == 'scene_library'
        model_eval = evaluate_plot_completion(
            'I keep searching the archive indexes.',
            'You scan the bound pages for another clue.',
            0.2,
            ['Search the obituary archive'],
            plot_goal='Library branch',
            scene_goal='Investigate the library branch',
            scene_description='You are inside the newspaper archive room.',
            current_plot_raw_text='Dusty newspaper shelves and bound ledgers fill the archive room.',
            current_node_kind='branch',
            allowed_targets=[],
            remaining_required_targets=[],
            redirect_streak=0,
            conversation_history=[],
        )
        assert model_eval['completed'] is False
        state_module.call_nvidia_llm = _failing_llm

        print('[test_state] case 12: Chinese legal targets from the LLM should be accepted even without lexical overlap')
        state_module.call_nvidia_llm = _chinese_transition_llm
        chinese_pre_eval = evaluate_pre_response_transition(
            '前往警局，询问近期是否有类似的入室盗窃或失踪案件记录。',
            plot_goal='Choose a lead',
            scene_goal='Start hub',
            scene_description='Thomas lays out the case in his office.',
            current_plot_raw_text='You can visit the library or speak to the police.',
            current_node_kind='hub',
            allowed_targets=start_ctx['allowed_targets'],
            remaining_required_targets=start_ctx['remaining_required_targets'],
            mandatory_events=[],
            redirect_streak=0,
            latest_agent_turn_excerpt='Thomas points out the library and the police as the two best leads.',
            choice_prompt_active=True,
            conversation_history=[],
        )
        assert chinese_pre_eval['action'] == 'target'
        assert chinese_pre_eval['target_id'] == 'scene_police'
        assert chinese_pre_eval['transition_path'] == 'direct'

        print('[test_state] case 13: choice-prompt answers should get a second focused transition pass')
        state_module.call_nvidia_llm = _choice_resolution_llm
        choice_resolution_eval = evaluate_pre_response_transition(
            '查阅报纸。',
            plot_goal='Choose a lead',
            scene_goal='Start hub',
            scene_description='Thomas lays out the case in his office.',
            current_plot_raw_text='You can visit the library or speak to the police.',
            current_node_kind='hub',
            allowed_targets=start_ctx['allowed_targets'],
            remaining_required_targets=start_ctx['remaining_required_targets'],
            mandatory_events=[],
            redirect_streak=0,
            latest_agent_turn_excerpt='You could question the police or consult the town newspaper files.',
            choice_prompt_active=True,
            conversation_history=[],
        )
        assert choice_resolution_eval['action'] == 'target'
        assert choice_resolution_eval['target_id'] == 'scene_library'
        assert choice_resolution_eval['transition_path'] == 'direct'

        print('[test_state] case 14: via-return targets should land in the new branch on the same turn')
        state_module.call_nvidia_llm = _chinese_transition_llm
        via_return_db = _build_scene_branch_db(ROOT / 'test' / 'debug_state_scene_via_return.db')
        via_return_entry = next_story_position(
            via_return_db,
            'scene_start',
            'scene_start_plot_1',
            {},
            {
                'action': 'target',
                'target_kind': 'scene',
                'target_id': 'scene_library',
                'close_current': False,
                'reason': 'player chose library first',
            },
            current_visit_id=0,
        )
        via_return_ctx = story_position_context(
            via_return_db,
            'scene_library',
            'scene_library_plot_1',
            via_return_entry['navigation_state'],
            current_visit_id=via_return_entry['current_visit_id'],
        )
        via_return_eval = evaluate_pre_response_transition(
            '去警局看看。',
            plot_goal='Library branch',
            scene_goal='Library branch',
            scene_description='Search the library archives for a lead.',
            current_plot_raw_text='Old newspapers and bound ledgers cover the archive table.',
            current_node_kind='branch',
            allowed_targets=via_return_ctx['allowed_targets'],
            indirect_targets_via_return=via_return_ctx['indirect_targets_via_return'],
            remaining_required_targets=[],
            mandatory_events=[],
            redirect_streak=0,
            latest_agent_turn_excerpt='The old article points toward the next lead.',
            choice_prompt_active=False,
            conversation_history=[],
        )
        assert via_return_eval['action'] == 'target'
        assert via_return_eval['target_id'] == 'scene_police'
        assert via_return_eval['transition_path'] == 'via_return'
        via_return_next = next_story_position(
            via_return_db,
            'scene_library',
            'scene_library_plot_1',
            via_return_entry['navigation_state'],
            via_return_eval,
            current_visit_id=via_return_entry['current_visit_id'],
        )
        assert via_return_next['current_scene_id'] == 'scene_police'
        assert via_return_next['current_plot_id'] == 'scene_police_plot_1'
        via_return_db.close()
        scene_db.close()

        print('[test_state] case 15: non-hub plots should force close when progress reaches 1.0')
        state_module.call_nvidia_llm = _progress_full_llm
        progress_eval = evaluate_plot_completion(
            'I continue searching the obituary archive until I finish the last relevant index.',
            'You have what you came for, and there is nothing more here to uncover.',
            0.45,
            ['Search the obituary archive'],
            plot_goal='Library branch',
            scene_goal='Investigate the library branch',
            scene_description='You are inside the newspaper archive room.',
            current_plot_raw_text='Dusty newspaper shelves and bound ledgers fill the archive room.',
            current_node_kind='branch',
            allowed_targets=[],
            remaining_required_targets=[],
            redirect_streak=0,
            latest_agent_turn_excerpt='The archive table is covered with the relevant clippings.',
            choice_prompt_active=False,
            conversation_history=[],
        )
        assert progress_eval['progress'] == 1.0
        assert progress_eval['completed'] is True
        assert progress_eval['close_current'] is True
        state_module.call_nvidia_llm = _failing_llm

        print('[test_state] case 16: a closed dead-end plot should fall back to scene-level return routing')
        dead_end_db = _build_scene_plot_dead_end_db(ROOT / 'test' / 'debug_state_plot_dead_end.db')
        dead_end_ctx = story_position_context(dead_end_db, 'scene_dead_end', 'scene_dead_end_plot_1', {})
        assert dead_end_ctx['return_target']['target_id'] == 'scene_start'
        assert 'scene_fresh' in dead_end_ctx['indirect_targets_summary']
        dead_end_next = next_story_position(
            dead_end_db,
            'scene_dead_end',
            'scene_dead_end_plot_1',
            {},
            {
                'action': 'stay',
                'target_kind': '',
                'target_id': '',
                'transition_path': 'stay',
                'close_current': True,
                'reason': 'dead-end plot finished and should return to the START hub',
            },
            current_visit_id=0,
        )
        assert dead_end_next['current_scene_id'] == 'scene_start'
        assert dead_end_next['current_plot_id'] == 'scene_start_plot_1'
        dead_end_db.close()

        print('[test_state] case 17: plot-level exclusive choice should skip sibling plots immediately')
        plot_db = _build_plot_branch_db(ROOT / 'test' / 'debug_state_plot_graph.db')
        choice_ctx = story_position_context(plot_db, 'scene_choice', 'scene_choice_plot_1', {})
        assert choice_ctx['current_node_kind'] == 'hub'
        assert len(choice_ctx['allowed_targets']) == 2
        chosen_plot = next_story_position(
            plot_db,
            'scene_choice',
            'scene_choice_plot_1',
            {},
            {
                'action': 'target',
                'target_kind': 'plot',
                'target_id': 'scene_choice_plot_2',
                'close_current': False,
                'reason': 'player explores',
            },
            current_visit_id=0,
        )
        assert chosen_plot['current_scene_id'] == 'scene_choice'
        assert chosen_plot['current_plot_id'] == 'scene_choice_plot_2'
        assert plot_db.get_plot('scene_choice_plot_3')['status'] == 'skipped'
        assert chosen_plot['navigation_state']['return_stack'] == [
            {'target_kind': 'plot', 'target_id': 'scene_choice_plot_1', 'visit_id': 0}
        ]
        plot_db.close()

        print('[test_state] case 18: explicit newspaper-office input should prefer the legal newspaper lead over the study')
        state_module.call_nvidia_llm = _failing_llm
        newspaper_targets = [
            dict(
                make_target(
                    'scene',
                    'scene_newspaper',
                    label='The Arnoldsburg Advertiser',
                    role='branch',
                    goal='Search the newspaper office archives with Artie Malloy.',
                    excerpt='Artie Malloy can grant access to the back issues and morgue files.',
                ),
                eligible=True,
            ),
            dict(
                make_target(
                    'scene',
                    'scene_study',
                    label='The Kimball House Study',
                    role='branch',
                    goal='Examine the study for clues about Douglas Kimball.',
                    excerpt='The Kimball house study is full of journals and stacked books.',
                ),
                eligible=True,
            ),
        ]
        newspaper_eval = evaluate_pre_response_transition(
            'I go to the Arnoldsburg Advertiser office and ask Artie Malloy for old newspaper files about the cemetery and the Kimball house.',
            plot_goal='Choose a lead',
            scene_goal='Start hub',
            scene_description='The next leads are the newspaper office and the Kimball study.',
            current_plot_raw_text='You can search old newspaper back issues or inspect the Kimball house study.',
            current_node_kind='hub',
            allowed_targets=newspaper_targets,
            remaining_required_targets=[],
            mandatory_events=[],
            redirect_streak=0,
            latest_agent_turn_excerpt='The remaining leads are the newspaper office and the study.',
            choice_prompt_active=False,
            conversation_history=[],
        )
        assert newspaper_eval['action'] == 'target'
        assert newspaper_eval['target_id'] == 'scene_newspaper'
        assert newspaper_eval['transition_path'] == 'direct'

        print('[test_state] case 19: terminal completion should preserve scene progress even without a next target')
        terminal_db_path = ROOT / 'test' / 'debug_state_terminal_resolved.db'
        if terminal_db_path.exists():
            terminal_db_path.unlink()
        terminal_db = Database(str(terminal_db_path))
        terminal_nav = default_navigation(completion_policy='terminal_on_resolve')
        terminal_db.insert_scenes(
            [
                _make_mirrored_scene(
                    'scene_end',
                    'Adventure conclusion',
                    node_kind='terminal',
                    navigation=terminal_nav,
                    status='in_progress',
                )
            ]
        )
        terminal_db.update_plot('scene_end_plot_1', status='completed', progress=1.0)
        terminal_db.update_scene('scene_end', {'status': 'completed'})
        terminal_db.update_system_state(
            {
                'stage': 'session',
                'current_scene_id': 'scene_end',
                'current_plot_id': 'scene_end_plot_1',
                'plot_progress': 1.0,
                'scene_progress': 0.0,
                'navigation_state': {},
                'current_visit_id': 0,
                'player_profile': {'name': 'Tester'},
            }
        )
        terminal_next = next_story_position(
            terminal_db,
            'scene_end',
            'scene_end_plot_1',
            {},
            {
                'action': 'stay',
                'target_kind': '',
                'target_id': '',
                'transition_path': 'stay',
                'close_current': True,
                'reason': 'terminal done',
            },
            current_visit_id=0,
        )
        assert terminal_next['current_scene_id'] == 'scene_end'
        assert terminal_next['current_plot_id'] == 'scene_end_plot_1'
        assert terminal_next['plot_progress'] == 1.0
        assert terminal_next['scene_progress'] == 1.0
        terminal_eval = evaluate_scene_completion(
            terminal_db,
            'scene_end',
            current_plot_id='scene_end_plot_1',
            plot_transition={
                'action': 'stay',
                'target_kind': '',
                'target_id': '',
                'close_current': True,
                'reason': 'terminal done',
            },
            navigation_state={},
            conversation_history=[],
            latest_turn_text='User: I finish the adventure.\nAgent: The case ends here.',
        )
        assert terminal_eval['completed'] is True
        assert terminal_eval['reason'] == 'terminal_scene_resolved'
        terminal_db.close()

        print('[test_state] case 20: explicit non-hub travel should honor legal via-return targets in the same turn')
        state_module.call_nvidia_llm = _failing_llm
        via_return_linear_eval = evaluate_pre_response_transition(
            'I thank the officer, leave the station, and head to the Arnoldsburg Advertiser to search old files.',
            plot_goal='Ask about Douglas Kimball at the police station.',
            scene_goal='Police station',
            scene_description='You are following up police leads about the disappearance.',
            current_plot_raw_text='The officer can talk about burglaries, suspicious activity, or the missing-person report.',
            current_node_kind='linear',
            allowed_targets=[],
            indirect_targets_via_return=[
                dict(
                    make_target(
                        'scene',
                        'scene_7',
                        label='The Arnoldsburg Advertiser',
                        role='branch',
                        goal='Search the newspaper office archives with Artie Malloy.',
                        excerpt='The Arnoldsburg Advertiser keeps old back issues and morgue files.',
                    ),
                    eligible=True,
                ),
                dict(
                    make_target(
                        'scene',
                        'scene_8',
                        label='The Kimball House',
                        role='branch',
                        goal='Search the Kimball house study for clues.',
                        excerpt='The Kimball study is packed with books and personal papers.',
                    ),
                    eligible=True,
                ),
            ],
            remaining_required_targets=[],
            mandatory_events=[],
            redirect_streak=0,
            latest_agent_turn_excerpt='The missing-person report is exhausted, but you could leave for another lead.',
            choice_prompt_active=False,
            conversation_history=[],
        )
        assert via_return_linear_eval['action'] == 'target'
        assert via_return_linear_eval['target_id'] == 'scene_7'
        assert via_return_linear_eval['transition_path'] == 'via_return'

        print('[test_state] case 21: explicit numeric answers should still resolve legal options even if choice_prompt_active is false')
        numeric_without_flag = evaluate_pre_response_transition(
            'I choose option 3 and ask about the missing-person report.',
            plot_goal='Choose which police topic to pursue.',
            scene_goal='Police station',
            scene_description='The desk officer offers three case angles.',
            current_plot_raw_text='You can ask about burglaries, suspicious activity, or the missing-person report.',
            current_node_kind='linear',
            allowed_targets=[
                dict(make_target('plot', 'scene_6_plot_2', label='Ask about recent burglaries', role='branch'), eligible=True),
                dict(make_target('plot', 'scene_6_plot_3', label='Ask about suspicious activity', role='branch'), eligible=True),
                dict(make_target('plot', 'scene_6_plot_4', label='Ask about Douglas Kimball missing-person report', role='branch'), eligible=True),
            ],
            indirect_targets_via_return=[],
            remaining_required_targets=[],
            mandatory_events=[],
            redirect_streak=0,
            latest_agent_turn_excerpt='The officer lays out three angles: 1. burglaries 2. suspicious activity 3. missing-person report.',
            choice_prompt_active=False,
            conversation_history=[],
        )
        assert numeric_without_flag['action'] == 'target'
        assert numeric_without_flag['target_kind'] == 'plot'
        assert numeric_without_flag['target_id'] == 'scene_6_plot_4'
        assert numeric_without_flag['transition_path'] == 'direct'

        print('[test_state] case 22: nested via-return should expose only downstream actionable targets and land there in one turn')
        nested_db = _build_nested_via_return_db(ROOT / 'test' / 'debug_state_nested_via_return.db')
        to_police = next_story_position(
            nested_db,
            'scene_start',
            'scene_start_plot_1',
            {},
            {
                'action': 'target',
                'target_kind': 'scene',
                'target_id': 'scene_police',
                'transition_path': 'direct',
                'close_current': False,
                'reason': 'enter police branch',
            },
            current_visit_id=0,
        )
        to_burglaries = next_story_position(
            nested_db,
            'scene_police',
            'scene_police_plot_1',
            to_police['navigation_state'],
            {
                'action': 'target',
                'target_kind': 'plot',
                'target_id': 'scene_police_plot_2',
                'transition_path': 'direct',
                'close_current': False,
                'reason': 'choose burglary file',
            },
            current_visit_id=to_police['current_visit_id'],
        )
        nested_ctx = story_position_context(
            nested_db,
            'scene_police',
            'scene_police_plot_2',
            to_burglaries['navigation_state'],
            current_visit_id=to_burglaries['current_visit_id'],
        )
        nested_indirect_ids = {target['target_id'] for target in nested_ctx['indirect_targets_via_return']}
        assert 'scene_newspaper' in nested_indirect_ids
        assert 'scene_start' not in nested_indirect_ids
        nested_eval = evaluate_pre_response_transition(
            'I leave the station and head to the Arnoldsburg Advertiser to search the old morgue files.',
            plot_goal='Ask about recent burglaries.',
            scene_goal='Police station',
            scene_description='The desk officer can point you toward the next lead.',
            current_plot_raw_text='The officer reviews a burglary pattern and mentions the paper may have old morgue files.',
            current_node_kind='branch',
            allowed_targets=nested_ctx['allowed_targets'],
            indirect_targets_via_return=nested_ctx['indirect_targets_via_return'],
            remaining_required_targets=[],
            mandatory_events=[],
            redirect_streak=0,
            latest_agent_turn_excerpt='The officer mentions that the Arnoldsburg Advertiser keeps useful old morgue files.',
            choice_prompt_active=False,
            conversation_history=[],
        )
        assert nested_eval['action'] == 'target'
        assert nested_eval['target_kind'] == 'scene'
        assert nested_eval['target_id'] == 'scene_newspaper'
        assert nested_eval['transition_path'] == 'via_return'
        nested_next = next_story_position(
            nested_db,
            'scene_police',
            'scene_police_plot_2',
            to_burglaries['navigation_state'],
            nested_eval,
            current_visit_id=to_burglaries['current_visit_id'],
        )
        assert nested_next['current_scene_id'] == 'scene_newspaper'
        assert nested_next['current_plot_id'] == 'scene_newspaper_plot_1'
        nested_db.close()

        print('[test_state] case 23: explicit next-step intent should prefer a legal via-return exit over unrelated branches')
        explicit_next_step_eval = evaluate_pre_response_transition(
            'I am done here and want to move on to the next step.',
            plot_goal='Search the Kimball house study for clues.',
            scene_goal='Kimball house study',
            scene_description='The study is cluttered with books and scattered notes.',
            current_plot_raw_text='You have searched the obvious shelves and desk drawers.',
            current_node_kind='branch',
            allowed_targets=[],
            indirect_targets_via_return=[
                dict(make_target('scene', 'scene_3', label='Asking around the neighborhood', role='branch'), eligible=True),
                dict(make_target('scene', 'scene_9', label='next steps', role='exit'), eligible=True),
            ],
            remaining_required_targets=[],
            mandatory_events=[],
            redirect_streak=0,
            latest_agent_turn_excerpt='The study feels exhausted, and the broader investigation now points toward the next phase.',
            choice_prompt_active=False,
            conversation_history=[],
        )
        assert explicit_next_step_eval['action'] == 'target'
        assert explicit_next_step_eval['target_id'] == 'scene_9'
        assert explicit_next_step_eval['transition_path'] == 'via_return'
        assert explicit_next_step_eval['reason'] == 'explicit_exit_intent_via_return'

        print('[test_state] case 24: explicit next-step intent should return to the legal hub when no exit is unlocked yet')
        return_to_hub_eval = evaluate_pre_response_transition(
            'I am done here and want to move on to the next step.',
            plot_goal='Search the Kimball house study for clues.',
            scene_goal='Kimball house study',
            scene_description='The study is cluttered with books and scattered notes.',
            current_plot_raw_text='You have searched the obvious shelves and desk drawers.',
            current_node_kind='branch',
            allowed_targets=[],
            indirect_targets_via_return=[
                dict(make_target('scene', 'scene_3', label='Asking around the neighborhood', role='branch'), eligible=True),
                dict(make_target('scene', 'scene_4', label='Looking around the graveyard', role='branch'), eligible=True),
            ],
            remaining_required_targets=[],
            return_target=dict(make_target('scene', 'scene_2', label='start', role='return'), eligible=True),
            mandatory_events=[],
            redirect_streak=0,
            latest_agent_turn_excerpt='The study feels exhausted, but the next-step hub is not unlocked yet.',
            choice_prompt_active=False,
            conversation_history=[],
        )
        assert return_to_hub_eval['action'] == 'target'
        assert return_to_hub_eval['target_id'] == 'scene_2'
        assert return_to_hub_eval['transition_path'] == 'via_return'
        assert return_to_hub_eval['reason'] == 'explicit_exit_to_return_target'

        print('[test_state] result: PASS')
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f'[test_state] result: FAIL -> {exc}')
        return 1
    finally:
        state_module.call_nvidia_llm = original_llm


if __name__ == '__main__':
    raise SystemExit(main())
