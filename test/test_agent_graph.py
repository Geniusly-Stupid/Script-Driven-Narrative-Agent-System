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


def _fake_llm(prompt: str, *args, **kwargs) -> str:
    step_name = kwargs.get('step_name', '')
    if step_name == 'check_whether_roll_dice':
        return '{"need_check": false, "skill": "", "reason": "", "dice_type": ""}'
    if step_name == 'pre_response_transition_evaluation':
        return '{"advance_target": "next_scene", "reason": "player committed to the library lead"}'
    if step_name == 'generate_response':
        if 'Scene ID: scene_a' in prompt and ('Player Input:\n\n---' in prompt or 'Player Input:\n\n' in prompt):
            return 'Thomas lays out the case in a low, tired voice and leaves the first move to you.'
        if 'Scene ID: scene_b' in prompt:
            return 'The library air smells of dust and paste as the old newspaper files are brought to your table.'
        return 'You leave the house behind and make for the local library, following the most promising lead.'
    if step_name == 'plot_completion_evaluation':
        return '{"completed": false, "progress_delta": 0.2, "advance_target": "stay", "reason": "continue current plot"}'
    if step_name == 'scene_completion_evaluation':
        return '{"completed": true, "scene_progress": 1.0, "reason": "setup scene resolved by clear next-scene commitment"}'
    if step_name == 'plot_summary_generation':
        return '- Available options included the neighborhood, library, police, and house.\n- The investigator explicitly chose the library lead.\n- Thomas accepted that decision and did not block it.\n- The active branch is now the library follow-up scene.\n- The police and neighborhood leads remain unvisited.'
    if step_name == 'scene_summary_generation':
        return '- The case was framed.\n- The player committed to a lead.\n- The setup scene ended.\n- The investigation moved to the library.'
    return 'Stub response.'


def main() -> int:
    db_path = ROOT / 'test' / 'debug_agent.db'
    chroma_path = ROOT / 'test' / '.chroma_agent'
    original_agent_llm = agent_graph_module.call_nvidia_llm
    original_state_llm = state_module.call_nvidia_llm
    try:
        if db_path.exists():
            db_path.unlink()

        agent_graph_module.call_nvidia_llm = _fake_llm
        state_module.call_nvidia_llm = _fake_llm

        db = Database(str(db_path))
        store = ChromaStore(path=str(chroma_path))
        store.reset()

        scenes = [
            {
                'scene_id': 'scene_a',
                'scene_goal': 'Prepare the investigator to begin the investigation.',
                'scene_description': 'Thomas explains the theft and points the investigator toward several possible leads.',
                'status': 'in_progress',
                'scene_summary': 'The investigator has just heard the case.',
                'plots': [
                    {
                        'plot_id': 'scene_a_plot_1',
                        'plot_goal': 'Introduce the investigation and provide initial options for the investigator.',
                        'mandatory_events': [],
                        'npc': ['Thomas Kimball'],
                        'locations': ['Kimball house'],
                        'raw_text': 'The investigator can ask around the neighborhood, go to the library, talk to the police, or search the house.',
                        'status': 'in_progress',
                        'progress': 0.2,
                    }
                ],
            },
            {
                'scene_id': 'scene_b',
                'scene_goal': 'Research the old newspaper records in the library.',
                'scene_description': 'A library follow-up scene focused on old articles and local history.',
                'status': 'pending',
                'scene_summary': '',
                'plots': [
                    {
                        'plot_id': 'scene_b_plot_1',
                        'plot_goal': 'Find and read the article about the cemetery sighting.',
                        'mandatory_events': ['Read the article'],
                        'npc': [],
                        'locations': ['Library'],
                        'raw_text': 'The investigator searches old newspaper records.',
                        'status': 'pending',
                        'progress': 0.0,
                    }
                ],
            },
        ]
        knowledge = [
            {
                'knowledge_id': 'knowledge_1',
                'knowledge_type': 'background',
                'title': 'Local Rumor',
                'content': 'The library archives contain reports about the cemetery.',
                'source_page_start': 1,
                'source_page_end': 1,
            },
            {
                'knowledge_id': 'knowledge_2',
                'knowledge_type': 'truth',
                'title': 'Keeper Truth',
                'content': 'Thomas does not know the full truth about his uncle.',
                'source_page_start': 2,
                'source_page_end': 2,
            },
        ]
        db.insert_scenes(scenes)
        db.insert_knowledge(knowledge)
        store.add_from_scenes(scenes, knowledge=knowledge)
        db.update_system_state(
            {
                'stage': 'session',
                'current_scene_id': 'scene_a',
                'current_plot_id': 'scene_a_plot_1',
                'plot_progress': 0.2,
                'scene_progress': 0.0,
                'player_profile': {'name': 'Tester', 'traits': ['calm']},
            }
        )

        agent = NarrativeAgent(db, store)
        agent.set_debug_mode(True)
        initial_result = agent.generate_initial_response()
        initial_prompt = initial_result.get('prompt') or ''
        assert initial_result.get('response'), 'initial response should be generated'
        assert 'Scene Entry Turn:' in initial_prompt, 'initial response prompt must include scene entry flag'
        assert 'true' in initial_prompt, 'initial response should be marked as a scene entry turn'
        assert db.get_recent_turns('scene_a', 'scene_a_plot_1', limit=5), 'initial response should be written to memory'

        user_input = 'I go straight to the library and start checking old newspaper files.'
        print('[test_agent_graph] input:', user_input)
        result = agent.run_turn(user_input)

        prompt = result.get('prompt') or ''
        print('[test_agent_graph] response prompt(first 600 chars) ->')
        print(prompt[:600])
        assert 'Scene ID: scene_b' in prompt, 'response prompt should already target the next scene'
        assert 'Plot ID: scene_b_plot_1' in prompt, 'response prompt should already target the next plot'
        assert 'Current Plot Excerpt:' in prompt, 'response prompt must include current plot excerpt'
        assert 'Scene Entry Turn:' in prompt, 'response prompt must include scene entry flag'
        assert 'true' in prompt, 'the next scene first plot should be treated as a scene entry turn'
        assert 'Next Plot Goal:' not in prompt, 'response prompt should not include next plot lookahead'
        assert 'Next Scene Goal:' not in prompt, 'response prompt should not include next scene lookahead'
        assert 'Next Scene First Plot Goal:' not in prompt, 'response prompt should not include next scene plot lookahead'

        debug_prompts = result.get('debug_prompts', [])
        pre_transition_prompt = next((item.get('prompt', '') for item in debug_prompts if item.get('name') == 'pre_response_transition_prompt'), '')
        pre_scene_prompt = next((item.get('prompt', '') for item in debug_prompts if item.get('name') == 'pre_response_scene_completion_prompt'), '')
        roll_prompt = next((item.get('prompt', '') for item in debug_prompts if item.get('name') == 'roll_check_prompt'), '')
        plot_prompt = next((item.get('prompt', '') for item in debug_prompts if item.get('name') == 'plot_completion_prompt'), '')
        plot_summary_prompt = next((item.get('prompt', '') for item in debug_prompts if item.get('name') == 'plot_summary_prompt'), '')
        scene_prompt = next((item.get('prompt', '') for item in debug_prompts if item.get('name') == 'scene_completion_prompt'), '')
        print('[test_agent_graph] pre-response scene completion prompt(first 500 chars) ->')
        print(pre_scene_prompt[:500])
        assert pre_transition_prompt, 'pre-response transition prompt should be recorded in debug mode'
        assert pre_scene_prompt, 'pre-response scene completion prompt should be recorded when a handoff happens'
        assert 'Current Plot Excerpt:' in roll_prompt, 'roll check prompt must include current plot excerpt'
        assert 'Latest Turn:' in pre_scene_prompt, 'pre-response scene completion prompt must include latest turn section'
        assert user_input in pre_scene_prompt, 'pre-response scene completion prompt must include latest user text'
        assert '(transition pending)' in pre_scene_prompt, 'pre-response scene completion prompt must mark the agent response as pending'
        assert 'target=next_scene' in pre_scene_prompt, 'pre-response scene completion prompt must include plot handoff signal'
        assert 'Next Scene Goal:' in pre_scene_prompt, 'pre-response scene completion prompt must include next scene goal'
        assert not plot_prompt, 'post-response plot completion should be skipped when pre-response transition applies'
        assert not scene_prompt, 'post-response scene completion should be skipped when pre-response transition applies'
        assert 'If the plot presented options or branches' in plot_summary_prompt, 'plot summary prompt must preserve option and branch details'
        assert 'Current Plot Excerpt:' in plot_summary_prompt, 'plot summary prompt must include plot excerpt'
        assert 'Next Scene First Plot Goal:' in plot_summary_prompt, 'plot summary prompt must include next scene plot detail'

        new_state = db.get_system_state()
        print('[test_agent_graph] system_state ->', new_state)
        assert new_state['current_scene_id'] == 'scene_b', 'system state should advance to the next scene'
        assert new_state['current_plot_id'] == 'scene_b_plot_1', 'system state should advance to the next scene plot'

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
