from __future__ import annotations

import random
import re
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from backend.database.repository import repo
from backend.services.llm_service import llm_service
from backend.vector_store.chroma_store import chroma_store

PROMPT_TEMPLATE = """# SYSTEM PROMPT

You are {agent_role}, running an interactive {game_system} narrative experience.

## Core Responsibilities
- Maintain long-term narrative coherence.
- Progress toward current Scene and Plot goals.
- Respect world logic and past events.
- Do not reveal future plot content prematurely.
- Balance player freedom with script structure.

## Style
- Tone: {tone_style}
- Perspective: {narrative_perspective}
- Length: {response_length}
- Immersion: {immersion_level}

## Constraints
- Use only information provided in Context.
- Do not fabricate missing knowledge.
- Avoid meta commentary.
- Guide story subtly toward Plot goal.

## Tool Use

You may use tools when you need to perform actions such as dice rolling or other mechanical resolutions.

### Dice Roll Tool

Use when:
- An action requires probabilistic resolution.
- A skill check is needed.
- The outcome should not be determined purely narratively.

### Call Format

TOOL_CALL: roll_dice
{
  "dice_type": "{dice_type}",
  "reason": "{reason}"
}

Example:

TOOL_CALL: roll_dice
{
  "dice_type": "1d100",
  "reason": "Sanity check"
}

### Rules

- Call only when necessary.
- Do not fabricate results.
- Do not narrate the outcome before the tool returns.
- After receiving the result, integrate it naturally into the story.

---

# USER INPUT

Player Input:
{user_input}

Player Intent (optional):
{player_intent_optional}

---

# SCRIPT STATE

Scene ID: {scene_id}  
Plot ID: {plot_id}

Scene Goal:
{current_scene_goal}

Plot Goal:
{current_plot_goal}

Mandatory Events (if any):
{mandatory_events}

---

# MEMORY

Previous Plot Summary:
{previous_plot_summary}

Current Scene Summary:
{current_scene_summary}

---

# RETRIEVED KNOWLEDGE

NPC:
{npc_related_info}

Player:
{player_related_info}

Location:
{location_related_info}

World Rules:
{game_rule_info}

Items / Clues:
{item_or_clue_info}

---

# INTERNAL STATE

Narrative Constraints:
{narrative_constraints}

Plot Progress:
{plot_progress_percentage_or_state}

Scene Progress:
{scene_progress_percentage_or_state}

---

# INSTRUCTION

Generate the next narrative response.

Requirements:
- Move story toward Plot Goal.
- Maintain consistency with Memory.
- Use Retrieved Knowledge only if relevant.
- Preserve immersion.
"""


class NarrativeState(TypedDict, total=False):
    scene_id: str
    plot_id: str
    plot_progress: float
    scene_progress: float
    player_profile: dict[str, Any]
    conversation_history: list[dict[str, Any]]
    retrieved_docs: list[dict[str, Any]]
    latest_user_input: str
    system_prompt: str
    user_prompt: str
    prompt: str
    retrieval_queries: list[str]
    context: str
    response: str
    dice_result: str | None
    plot_completed: bool
    scene_completed: bool
    next_scene_id: str
    next_plot_id: str
    scene_goal: str
    plot_goal: str
    mandatory_events: list[str]
    previous_plot_summary: str
    current_scene_summary: str


class NarrativeAgent:
    def __init__(self) -> None:
        self.graph = self._build_graph()

    def roll_dice(self, dice_type: str) -> int:
        faces = {'d4': 4, 'd6': 6, 'd8': 8, 'd10': 10, 'd12': 12, 'd20': 20, 'd100': 100}
        if dice_type not in faces:
            raise ValueError(f'Unsupported dice type: {dice_type}')
        return random.randint(1, faces[dice_type])

    def _build_graph(self):
        workflow = StateGraph(NarrativeState)
        workflow.add_node('build_prompt', self.build_prompt)
        workflow.add_node('retrieve_memory', self.retrieve_memory)
        workflow.add_node('generate_retrieval_queries', self.generate_retrieval_queries)
        workflow.add_node('vector_retrieve', self.vector_retrieve)
        workflow.add_node('construct_context', self.construct_context)
        workflow.add_node('generate_response', self.generate_response)
        workflow.add_node('write_memory', self.write_memory)
        workflow.add_node('check_plot_completion', self.check_plot_completion)
        workflow.add_node('check_scene_completion', self.check_scene_completion)
        workflow.add_node('update_state', self.update_state)

        workflow.set_entry_point('build_prompt')
        workflow.add_edge('build_prompt', 'retrieve_memory')
        workflow.add_edge('retrieve_memory', 'generate_retrieval_queries')
        workflow.add_edge('generate_retrieval_queries', 'vector_retrieve')
        workflow.add_edge('vector_retrieve', 'construct_context')
        workflow.add_edge('construct_context', 'generate_response')
        workflow.add_edge('generate_response', 'write_memory')
        workflow.add_edge('write_memory', 'check_plot_completion')
        workflow.add_edge('check_plot_completion', 'check_scene_completion')
        workflow.add_edge('check_scene_completion', 'update_state')
        workflow.add_edge('update_state', END)
        return workflow.compile()

    async def run_turn(self, user_input: str) -> dict[str, Any]:
        system_state = await repo.get_system_state()
        player = await repo.get_player_profile() or {}
        state: NarrativeState = {
            'scene_id': system_state.get('current_scene_id', ''),
            'plot_id': system_state.get('current_plot_id', ''),
            'plot_progress': float(system_state.get('plot_progress', 0.0)),
            'scene_progress': float(system_state.get('scene_progress', 0.0)),
            'player_profile': player,
            'conversation_history': [],
            'retrieved_docs': [],
            'latest_user_input': user_input,
            'dice_result': None,
            'scene_goal': '',
            'plot_goal': '',
            'mandatory_events': [],
            'previous_plot_summary': '',
            'current_scene_summary': '',
        }
        final_state = await self.graph.ainvoke(state)
        return final_state

    async def build_prompt(self, state: NarrativeState) -> NarrativeState:
        state['system_prompt'] = ''
        state['user_prompt'] = state['latest_user_input']
        return state

    async def retrieve_memory(self, state: NarrativeState) -> NarrativeState:
        turns = await repo.get_turns(state['scene_id'], state['plot_id'], limit=12)
        state['conversation_history'] = turns

        scene = await repo.get_scene(state['scene_id'])
        if scene:
            state['scene_goal'] = str(scene.get('scene_goal', ''))
            state['current_scene_summary'] = str(scene.get('scene_summary', ''))
            plots = scene.get('plots', [])
            current_idx = -1
            for idx, plot in enumerate(plots):
                if plot.get('plot_id') == state['plot_id']:
                    current_idx = idx
                    state['plot_goal'] = str(plot.get('plot_goal', ''))
                    state['mandatory_events'] = [str(e) for e in plot.get('mandatory_events', [])]
                    break

            if current_idx > 0:
                previous_plot_id = str(plots[current_idx - 1].get('plot_id', ''))
                state['previous_plot_summary'] = await repo.get_plot_summary(previous_plot_id)
            else:
                state['previous_plot_summary'] = ''

        return state

    async def generate_retrieval_queries(self, state: NarrativeState) -> NarrativeState:
        history_tail = ' '.join((t.get('user', '') for t in state.get('conversation_history', [])[-4:]))
        events_text = '; '.join(state.get('mandatory_events', [])[:3])
        queries = [
            state['latest_user_input'],
            state.get('plot_goal', ''),
            events_text,
            history_tail or f"Scene context for {state['scene_id']}",
        ]
        state['retrieval_queries'] = [q for q in queries if q]
        return state

    async def vector_retrieve(self, state: NarrativeState) -> NarrativeState:
        docs = []
        for query in state.get('retrieval_queries', []):
            docs.extend(chroma_store.search(query, k=3))
        state['retrieved_docs'] = docs[:8]
        return state

    async def construct_context(self, state: NarrativeState) -> NarrativeState:
        memory_turns = '\n'.join(
            [f"U: {t.get('user', '')}\nA: {t.get('agent', '')}" for t in state.get('conversation_history', [])]
        ) or 'No prior memory.'
        previous_plot_summary = state.get('previous_plot_summary', '') or 'No previous plot summary.'
        docs = state.get('retrieved_docs', [])
        npc_related_info = '\n'.join([d['content'] for d in docs if d.get('metadata', {}).get('type') == 'npc']) or 'None'
        location_related_info = '\n'.join(
            [d['content'] for d in docs if d.get('metadata', {}).get('type') == 'location']
        ) or 'None'
        game_rule_info = '\n'.join([d['content'] for d in docs if d.get('metadata', {}).get('type') == 'rule']) or 'None'
        item_or_clue_info = '\n'.join(
            [d['content'] for d in docs if d.get('metadata', {}).get('type') in ['item', 'event']]
        ) or 'None'

        state['context'] = memory_turns
        state['prompt'] = PROMPT_TEMPLATE.format(
            agent_role='Narrative Agent',
            game_system='TRPG',
            tone_style='Immersive and grounded',
            narrative_perspective='Second person',
            response_length='Concise scene-forward turns',
            immersion_level='High',
            dice_type='{dice_type}',
            reason='{reason}',
            user_input=state['latest_user_input'],
            player_intent_optional='',
            scene_id=state['scene_id'],
            plot_id=state['plot_id'],
            current_scene_goal=state.get('scene_goal', '') or 'None',
            current_plot_goal=state.get('plot_goal', '') or 'None',
            mandatory_events=', '.join(state.get('mandatory_events', [])) or 'None',
            previous_plot_summary=previous_plot_summary,
            current_scene_summary=state.get('current_scene_summary', '') or 'None',
            npc_related_info=npc_related_info,
            player_related_info=str(state.get('player_profile', {})),
            location_related_info=location_related_info,
            game_rule_info=game_rule_info,
            item_or_clue_info=item_or_clue_info,
            narrative_constraints='Respect scene/plot ordering and established facts only.',
            plot_progress_percentage_or_state=f"{state['plot_progress']:.0%}",
            scene_progress_percentage_or_state=f"{state['scene_progress']:.0%}",
        )
        return state

    async def generate_response(self, state: NarrativeState) -> NarrativeState:
        user_input = state['latest_user_input']
        dice_match = re.search(r'(d4|d6|d8|d10|d12|d20|d100)', user_input.lower())
        dice_result_text = None
        if 'roll' in user_input.lower() and dice_match:
            dice_type = dice_match.group(1)
            dice_value = self.roll_dice(dice_type)
            dice_result_text = f'{dice_type} -> {dice_value}'

        prompt = state['prompt'] + (
            f"\n\nDice result available: {dice_result_text}. Integrate it into narration."
            if dice_result_text
            else ''
        )
        fallback = 'The world shifts forward. Describe your next precise action.'
        response = await llm_service.complete_text(prompt, fallback=fallback)

        state['response'] = response
        state['dice_result'] = dice_result_text
        return state

    async def write_memory(self, state: NarrativeState) -> NarrativeState:
        await repo.append_turn(state['scene_id'], state['plot_id'], state['latest_user_input'], state['response'])
        return state

    async def check_plot_completion(self, state: NarrativeState) -> NarrativeState:
        completion_prompt = (
            'Evaluate if current plot is completed based on user message and agent response. '
            'Return JSON {"plot_completed": bool, "progress_delta": float, "plot_summary": string}.\n\n'
            f"User: {state['latest_user_input']}\nAgent: {state['response']}"
        )
        fallback = {'plot_completed': False, 'progress_delta': 0.1, 'plot_summary': ''}
        result = await llm_service.complete_json('Plot evaluator', completion_prompt, fallback=fallback)

        delta = float(result.get('progress_delta', 0.1))
        new_progress = min(1.0, state.get('plot_progress', 0.0) + max(0.05, delta))
        completed = bool(result.get('plot_completed', False) or new_progress >= 1.0)

        state['plot_progress'] = 1.0 if completed else new_progress
        state['plot_completed'] = completed

        if completed:
            await repo.save_plot_summary(
                state['scene_id'],
                state['plot_id'],
                result.get('plot_summary', f"Plot {state['plot_id']} completed."),
            )
        return state

    async def check_scene_completion(self, state: NarrativeState) -> NarrativeState:
        scene = await repo.get_scene(state['scene_id'])
        if not scene:
            state['scene_completed'] = False
            return state

        plots = scene.get('plots', [])
        if state.get('plot_completed'):
            updated_plots = []
            for plot in plots:
                if plot.get('plot_id') == state['plot_id']:
                    plot['status'] = 'completed'
                    plot['progress'] = 1.0
                updated_plots.append(plot)
            plots = updated_plots
            await repo.update_scene(state['scene_id'], {'plots': plots})

        completed_count = len([p for p in plots if p.get('status') == 'completed'])
        scene_progress = completed_count / max(1, len(plots))
        state['scene_progress'] = scene_progress
        scene_completed = scene_progress >= 1.0
        state['scene_completed'] = scene_completed

        if scene_completed:
            summary_prompt = f"Summarize completed scene {state['scene_id']} in 3-4 sentences."
            scene_summary = await llm_service.complete_text(
                summary_prompt, fallback=f"Scene {state['scene_id']} completed."
            )
            await repo.save_scene_summary(state['scene_id'], scene_summary)
            await repo.update_scene(state['scene_id'], {'status': 'completed', 'scene_summary': scene_summary})
        else:
            await repo.update_scene(state['scene_id'], {'status': 'in_progress'})

        return state

    async def update_state(self, state: NarrativeState) -> NarrativeState:
        scenes = await repo.list_scenes()
        current_scene_id = state['scene_id']
        current_plot_id = state['plot_id']
        scene_intro = ''

        if state.get('plot_completed'):
            current_scene = next((s for s in scenes if s['scene_id'] == state['scene_id']), None)
            if current_scene:
                next_plot = next((p for p in current_scene.get('plots', []) if p.get('status') != 'completed'), None)
                if next_plot:
                    current_plot_id = next_plot['plot_id']
                    state['plot_progress'] = next_plot.get('progress', 0.0)
                else:
                    current_plot_id = ''

        if state.get('scene_completed'):
            next_scene = next((s for s in scenes if s.get('status') != 'completed'), None)
            if next_scene:
                current_scene_id = next_scene['scene_id']
                first_plot = next_scene.get('plots', [{}])[0]
                current_plot_id = first_plot.get('plot_id', '')
                state['plot_progress'] = first_plot.get('progress', 0.0)
                state['scene_progress'] = 0.0
                await repo.update_scene(current_scene_id, {'status': 'in_progress'})
                intro_prompt = (
                    f"Write a short scene introduction for {current_scene_id}. "
                    f"Scene goal: {next_scene.get('scene_goal', '')}."
                )
                scene_intro = await llm_service.complete_text(
                    intro_prompt, fallback=f"Scene {current_scene_id} begins."
                )
            else:
                current_scene_id = state['scene_id']

        await repo.update_system_state(
            {
                'current_scene_id': current_scene_id,
                'current_plot_id': current_plot_id,
                'plot_progress': state.get('plot_progress', 0.0),
                'scene_progress': state.get('scene_progress', 0.0),
                'stage': 'session',
                'current_scene_intro': scene_intro,
            }
        )
        state['scene_id'] = current_scene_id
        state['plot_id'] = current_plot_id
        return state


narrative_agent = NarrativeAgent()
