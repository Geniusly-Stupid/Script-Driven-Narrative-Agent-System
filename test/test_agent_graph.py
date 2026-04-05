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


def _fake_llm(prompt: str, *args, **kwargs) -> str:
    step_name = kwargs.get('step_name', '')
    model = kwargs.get('model', '')
    allow_env_override = kwargs.get('allow_env_override', True)
    if step_name == 'check_whether_roll_dice':
        return '{"need_check": false, "skill": "", "reason": "", "dice_type": ""}'
    if step_name == 'pre_response_transition_evaluation':
        assert model == 'qwen/qwen3.5-397b-a17b'
        assert allow_env_override is False
        if 'Latest Player Input:\nI go straight to the library.' in prompt:
            return (
                '{"action": "target", "target_kind": "scene", "target_id": "scene_library", '
                '"transition_path": "direct", "close_current": false, "confidence": 0.97, '
                '"reason": "player explicitly chose the library branch"}'
            )
        if 'Latest Player Input:\n前往警局，询问近期是否有类似的入室盗窃或失踪案件记录。' in prompt:
            return (
                '{"action": "target", "target_kind": "scene", "target_id": "scene_police", '
                '"transition_path": "direct", "close_current": false, "confidence": 0.99, '
                '"reason": "player explicitly chose the police branch in Chinese"}'
            )
        if 'Latest Player Input:\n我去警局问问最近的入室盗窃案。' in prompt:
            return (
                '{"action": "target", "target_kind": "scene", "target_id": "scene_police", '
                '"transition_path": "via_return", "close_current": true, "confidence": 0.96, '
                '"reason": "player chose another legal lead while wrapping the current branch"}'
            )
        if 'Latest Player Input:\n我想问问最近的入室盗窃案。' in prompt:
            return (
                '{"action": "target", "target_kind": "plot", "target_id": "scene_police_plot_2", '
                '"transition_path": "direct", "close_current": false, "confidence": 0.98, '
                '"reason": "player answered the police topic choice directly"}'
            )
        return (
            '{"action": "stay", "target_kind": "", "target_id": "", '
            '"transition_path": "stay", "close_current": false, "confidence": 0.12, '
            '"reason": "stay put"}'
        )
    if step_name == 'generate_response':
        if 'Scene ID: scene_start' in prompt and 'Player Input:\n\n---' in prompt:
            return 'Thomas quietly lays out the case and leaves several leads open to you. (Will you ask about the books first, or go straight to the house?)'
        if 'Scene ID: scene_police' in prompt and 'Plot ID: scene_police_plot_2' in prompt:
            return 'The desk sergeant flips open the recent incident blotter and starts tracing a pattern of break-ins for you.'
        if 'Scene ID: scene_police' in prompt:
            return 'Inside the station, the duty officer glances up and waits for you to name which case angle you want to pursue.'
        if 'Scene ID: scene_library' in prompt and 'Player Input:\nI ask about the weather.' in prompt:
            return 'The room gives you nothing for small talk; the archive tables still draw your attention back to the missing obituary.'
        if 'Scene ID: scene_library' in prompt and 'Player Input:\nI start talking about baseball.' in prompt:
            return 'The idle thought dies quickly in the dusty silence, leaving the clipped obituary notices and index cards squarely in front of you.'
        if 'Scene ID: scene_library' in prompt and 'Player Input:\n我去警局问问最近的入室盗窃案。' in prompt:
            return 'You gather the clipping and leave the archive with the next lead firmly in mind.'
        if 'Scene ID: scene_library' in prompt:
            return 'Dust swirls through the lamp light as you pull a bound newspaper volume onto the table.'
        if 'Scene ID: scene_start' in prompt:
            return 'Back at the crossroads of the investigation, the remaining leads still wait.'
        return 'Stub response.'
    if step_name == 'plot_completion_evaluation':
        assert model == 'qwen/qwen3.5-397b-a17b'
        assert allow_env_override is False
        if 'Latest Turn:\nUser: I ask about the weather.' in prompt:
            return (
                '{"completed": false, "progress_delta": 0.0, "action": "stay", '
                '"target_kind": "", "target_id": "", "close_current": false, '
                '"transition_path": "stay", "objective_satisfied": false, "confidence": 0.15, '
                '"reason": "off topic, keep current plot active"}'
            )
        if 'Latest Turn:\nUser: I start talking about baseball.' in prompt:
            return (
                '{"completed": false, "progress_delta": 0.0, "action": "stay", '
                '"target_kind": "", "target_id": "", "close_current": false, '
                '"transition_path": "stay", "objective_satisfied": false, "confidence": 0.12, '
                '"reason": "still off topic, keep redirecting"}'
            )
        if 'Latest Turn:\nUser: I search the shelves for the missing obituary.' in prompt:
            return (
                '{"completed": true, "progress_delta": 0.6, "action": "stay", '
                '"target_kind": "", "target_id": "", "close_current": true, '
                '"transition_path": "stay", "objective_satisfied": true, "confidence": 0.93, '
                '"reason": "library branch resolved"}'
            )
        return (
            '{"completed": false, "progress_delta": 0.1, "action": "stay", '
            '"target_kind": "", "target_id": "", "close_current": false, '
            '"transition_path": "stay", "objective_satisfied": false, "confidence": 0.34, '
            '"reason": "continue current node"}'
        )
    if step_name == 'plot_summary_generation':
        return (
            '- The player entered the library branch.\n'
            '- The branch focused on archive research.\n'
            '- The branch closed cleanly and returned to the start hub.'
        )
    if step_name == 'scene_summary_generation':
        return (
            '- The scene framed the case.\n'
            '- The player followed one lead.\n'
            '- The scene can still return for remaining leads.\n'
            '- The investigation kept moving.'
        )
    return 'OK'


def main() -> int:
    db_path = ROOT / 'test' / 'debug_agent_branch.db'
    original_agent_llm = agent_graph_module.call_nvidia_llm
    original_state_llm = state_module.call_nvidia_llm
    try:
        if db_path.exists():
            db_path.unlink()

        agent_graph_module.call_nvidia_llm = _fake_llm
        state_module.call_nvidia_llm = _fake_llm

        db = Database(str(db_path))

        start_nav = default_navigation(completion_policy='all_required_then_advance')
        start_nav['allowed_targets'] = [
            make_target('scene', 'scene_library', label='Library', role='branch', required=True),
            make_target('scene', 'scene_police', label='Police', role='branch', required=True),
        ]
        start_return = make_target('scene', 'scene_start', label='Start', role='return')
        police_plot_hub_nav = default_navigation(completion_policy='optional_until_exit')
        police_plot_hub_nav['allowed_targets'] = [
            make_target('plot', 'scene_police_plot_2', label='Recent burglaries', role='branch'),
            make_target('plot', 'scene_police_plot_3', label='Missing-person reports', role='branch'),
        ]
        police_plot_return = make_target('plot', 'scene_police_plot_1', label='Police desk', role='return')

        scenes = [
            _make_mirrored_scene('scene_start', 'Start hub', node_kind='hub', navigation=start_nav, status='in_progress'),
            _make_mirrored_scene(
                'scene_library',
                'Search the library archives for the missing obituary.',
                node_kind='branch',
                navigation={
                    'allowed_targets': [],
                    'return_target': start_return,
                    'completion_policy': 'terminal_on_resolve',
                    'prerequisites': [],
                    'close_unselected_on_advance': False,
                },
            ),
            {
                'scene_id': 'scene_police',
                'scene_goal': 'Police branch',
                'scene_description': 'The station can help you pursue specific investigative angles.',
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
                        'plot_goal': 'Choose which police records or topic to pursue.',
                        'mandatory_events': [],
                        'npc': [],
                        'locations': [],
                        'raw_text': 'The duty officer can discuss recent burglaries or missing-person reports.',
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
                        'raw_text': 'The blotter lists a recent string of burglary reports.',
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
                    {
                        'plot_id': 'scene_police_plot_3',
                        'plot_goal': 'Ask about missing-person reports.',
                        'mandatory_events': [],
                        'npc': [],
                        'locations': [],
                        'raw_text': 'The case files include several unsettling disappearances.',
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

        agent = NarrativeAgent(db, DummyVectorStore())
        agent.set_debug_mode(True)

        print('[test_agent_graph] case 1: initial response should expose branch context')
        initial_result = agent.generate_initial_response()
        initial_prompt = initial_result.get('prompt') or ''
        assert initial_result.get('response'), 'initial response should be generated'
        assert '(Will you ask about the books first, or go straight to the house?)' not in initial_result.get('response', '')
        assert 'Allowed Targets:' in initial_prompt, 'response prompt should include allowed targets'
        assert 'Eligible Targets:' in initial_prompt, 'response prompt should include eligible targets'
        assert 'Indirect Targets Via Return:' in initial_prompt, 'response prompt should include indirect targets'
        assert 'Exhausted Targets:' in initial_prompt, 'response prompt should include exhausted targets'
        assert 'Current Hub Status:' in initial_prompt, 'response prompt should include current hub status'
        assert 'Do NOT narrate the player as already arriving in another scene or plot unless a legal state transition has already been applied for this turn.' in initial_prompt
        assert 'Opening Choice Allowed:' in initial_prompt
        opening_turns = db.get_recent_turns('scene_start', 'scene_start_plot_1', limit=5, visit_id=0)
        assert opening_turns, 'initial response should be written to visit 0 memory'

        print('[test_agent_graph] case 2: pre-response branch selection should hand off before keeper response')
        first_input = 'I go straight to the library.'
        result = agent.run_turn(first_input)
        prompt = result.get('prompt') or ''
        debug_prompts = result.get('debug_prompts', [])
        pre_transition_prompt = next((item.get('prompt', '') for item in debug_prompts if item.get('name') == 'pre_response_transition_prompt'), '')
        pre_scene_prompt = next((item.get('prompt', '') for item in debug_prompts if item.get('name') == 'pre_response_scene_completion_prompt'), '')

        assert result.get('pre_response_transition_applied') is True
        assert result.get('pre_response_transition_action') == 'target'
        assert result.get('pre_response_transition_target_kind') == 'scene'
        assert result.get('pre_response_transition_target_id') == 'scene_library'
        assert 'Scene ID: scene_library' in prompt, 'generation prompt should already target the library branch'
        assert 'Scene Entry Turn:\ntrue' in prompt, 'entered branch should render as a fresh visit'
        assert 'Allowed Targets:' in prompt, 'branch prompt should still include navigation state'
        assert 'Active Plot Objective:' in prompt
        assert 'Latest Player Alignment:' in prompt
        assert 'Indirect Targets Via Return:' in prompt
        assert 'Latest Agent Turn Excerpt:' in prompt
        assert pre_transition_prompt, 'debug should keep the pre-response transition prompt'
        assert pre_scene_prompt, 'debug should keep the pre-response scene completion prompt'

        system_after_handoff = db.get_system_state()
        assert system_after_handoff['current_scene_id'] == 'scene_library'
        assert system_after_handoff['current_plot_id'] == 'scene_library_plot_1'
        assert system_after_handoff['current_visit_id'] == 1
        assert db.get_recent_turns('scene_library', 'scene_library_plot_1', limit=5, visit_id=1), 'branch memory should be stored under visit 1'

        print('[test_agent_graph] case 3: off-topic reply should stay in the current plot and increase redirect state')
        off_topic_result = agent.run_turn('I ask about the weather.')
        off_topic_system = db.get_system_state()
        off_topic_prompt = off_topic_result.get('prompt') or ''
        assert off_topic_system['current_scene_id'] == 'scene_library'
        assert off_topic_system['current_plot_id'] == 'scene_library_plot_1'
        assert 'Latest Player Alignment:' in off_topic_prompt
        assert 'off_topic_unrelated' in off_topic_prompt
        assert 'Redirect State:' in off_topic_prompt
        assert off_topic_system['navigation_state']['plot_guidance']['scene_library_plot_1::1']['redirect_streak'] == 1

        print('[test_agent_graph] case 4: repeated off-topic replies should still not end the plot')
        second_off_topic_result = agent.run_turn('I start talking about baseball.')
        second_off_topic_system = db.get_system_state()
        second_off_topic_prompt = second_off_topic_result.get('prompt') or ''
        assert second_off_topic_system['current_scene_id'] == 'scene_library'
        assert second_off_topic_system['current_plot_id'] == 'scene_library_plot_1'
        assert second_off_topic_system['navigation_state']['plot_guidance']['scene_library_plot_1::1']['redirect_streak'] == 2
        assert 'redirect_streak=1' in second_off_topic_prompt or 'redirect_streak=2' in second_off_topic_prompt

        print('[test_agent_graph] case 5: finishing branch should return to hub with a fresh visit history')
        second_input = 'I search the shelves for the missing obituary.'
        result_after_return = agent.run_turn(second_input)
        system_after_return = db.get_system_state()
        assert system_after_return['current_scene_id'] == 'scene_start'
        assert system_after_return['current_plot_id'] == 'scene_start_plot_1'
        assert system_after_return['current_visit_id'] == 2
        assert system_after_return['navigation_state']['return_stack'] == []
        assert result_after_return.get('scene_id') == 'scene_start'
        assert result_after_return.get('plot_id') == 'scene_start_plot_1'
        assert result_after_return.get('scene_entry_turn') is True, 'hub revisit should start a fresh visit'
        assert result_after_return.get('conversation_history') == [], 'visit filtering should hide older hub turns on re-entry'
        assert db.get_recent_turns('scene_start', 'scene_start_plot_1', limit=5, visit_id=2) == []
        assert len(db.get_recent_turns('scene_start', 'scene_start_plot_1', limit=5, visit_id=0)) == 1
        assert len(db.get_recent_turns('scene_library', 'scene_library_plot_1', limit=8, visit_id=1)) == 4
        assert result_after_return.get('transition_context_summary')

        print('[test_agent_graph] case 6: revisiting an exhausted branch should stay at the hub and expose redirect context')
        revisit_result = agent.run_turn('I go straight to the library again.')
        revisit_prompt = revisit_result.get('prompt') or ''
        revisit_system = db.get_system_state()
        assert revisit_result.get('pre_response_transition_applied') is False
        assert revisit_system['current_scene_id'] == 'scene_start'
        assert revisit_system['current_plot_id'] == 'scene_start_plot_1'
        assert 'Scene ID: scene_start' in revisit_prompt
        assert 'Eligible Targets:' in revisit_prompt
        assert 'Exhausted Targets:' in revisit_prompt
        assert 'scene_police' in revisit_prompt
        assert 'scene_library' in revisit_prompt

        print('[test_agent_graph] case 7: Chinese hub choice should hand off to the police in the same turn')
        police_handoff = agent.run_turn('前往警局，询问近期是否有类似的入室盗窃或失踪案件记录。')
        police_system = db.get_system_state()
        police_prompt = police_handoff.get('prompt') or ''
        assert police_handoff.get('pre_response_transition_applied') is True
        assert police_handoff.get('pre_response_transition_target_id') == 'scene_police'
        assert police_handoff.get('pre_response_transition_path') == 'direct'
        assert police_system['current_scene_id'] == 'scene_police'
        assert police_system['current_plot_id'] == 'scene_police_plot_1'
        assert 'Scene ID: scene_police' in police_prompt
        assert 'Plot ID: scene_police_plot_1' in police_prompt

        print('[test_agent_graph] case 8: answering a police topic choice should jump straight into the child plot')
        police_topic = agent.run_turn('我想问问最近的入室盗窃案。')
        police_topic_system = db.get_system_state()
        police_topic_prompt = police_topic.get('prompt') or ''
        assert police_topic.get('pre_response_transition_applied') is True
        assert police_topic.get('pre_response_transition_target_kind') == 'plot'
        assert police_topic.get('pre_response_transition_target_id') == 'scene_police_plot_2'
        assert police_topic_system['current_scene_id'] == 'scene_police'
        assert police_topic_system['current_plot_id'] == 'scene_police_plot_2'
        assert 'Plot ID: scene_police_plot_2' in police_topic_prompt
        assert 'Scene ID: scene_police' in police_topic_prompt

        print('[test_agent_graph] case 9: branch via-return choices should enter the next legal scene in the same turn')
        db.close()
        if db_path.exists():
            db_path.unlink()
        db = Database(str(db_path))
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
        agent = NarrativeAgent(db, DummyVectorStore())
        agent.set_debug_mode(True)
        agent.generate_initial_response()
        agent.run_turn('I go straight to the library.')
        via_return_result = agent.run_turn('我去警局问问最近的入室盗窃案。')
        via_return_system = db.get_system_state()
        via_return_prompt = via_return_result.get('prompt') or ''
        assert via_return_result.get('pre_response_transition_applied') is True
        assert via_return_result.get('pre_response_transition_target_id') == 'scene_police'
        assert via_return_result.get('pre_response_transition_path') == 'via_return'
        assert via_return_system['current_scene_id'] == 'scene_police'
        assert via_return_system['current_plot_id'] == 'scene_police_plot_1'
        assert 'Scene ID: scene_police' in via_return_prompt
        assert 'Scene ID: scene_library' not in via_return_prompt

        db.close()
        print('[test_agent_graph] result: PASS')
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f'[test_agent_graph] result: FAIL -> {exc}')
        return 1
    finally:
        agent_graph_module.call_nvidia_llm = original_agent_llm
        state_module.call_nvidia_llm = original_state_llm


if __name__ == '__main__':
    raise SystemExit(main())
