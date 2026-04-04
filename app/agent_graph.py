from __future__ import annotations

import json
import random
import re
from typing import Any, TypedDict
import traceback
import logging

try:
    import json5  # type: ignore[import-not-found]
except ImportError:
    json5 = None

from langgraph.graph import END, StateGraph

from app.database import Database
from app.llm_client import call_nvidia_llm
from app.rag import categorize_docs, generate_retrieval_queries
from app.state import (
    evaluate_plot_completion,
    evaluate_pre_response_transition,
    evaluate_scene_completion,
    next_story_position,
    story_position_context,
)
from app.vector_store import ChromaStore

logger = logging.getLogger(__name__)

RESPONSE_PROMPT_TEMPLATE = """# SYSTEM PROMPT

You are {agent_role}, running an interactive {game_system} narrative experience.

## Core Responsibilities
- Maintain long-term narrative coherence.
- Progress toward current Plot goals and ultimately the Scene goal.
- Respect world logic and past events.
- Do not reveal future plot content prematurely.
- Balance player freedom with script structure.

## Style
- Tone: {tone_style}
- Perspective: {narrative_perspective}
- Length: {response_length}

## Constraints
- Use only information provided in Context.
- Do not fabricate missing knowledge.
- Avoid meta commentary.
- Hidden truth is keeper-only context. Use it for consistency, do not reveal it directly unless current plot already unlocks it.
- Final response language: {output_language}
- Write the response entirely in {output_language}.

## Player Interaction Rules

- The scene opening already counts as the beginning of the current plot. Do NOT treat it as a pre-plot phase.
- The player controls the PC. Do NOT speak for the PC, extend their dialogue, or describe their internal thoughts.
- After the player acts or speaks, always advance the scene with NPC dialogue, NPC action, or environmental consequences.
- When a dice roll or player choice is required **by the script or situation**, present it clearly and naturally in parentheses (), without breaking immersion.
    - For skill checks, briefly describe the situation and then indicate the required check, for example:
      "As you walk past, an elderly lady studies your appearance with clear judgment in her eyes, as if she values status and presentation. (Make an APP or Credit Rating check.)"
    - For branching choices, present the options in the narrative, then prompt the player to choose, for Example:
      "You step out of the house. Based on what you know, you could:
      1. Ask around the neighborhood.
      2. Search the graveyard.
         You pause, considering your next move. (Choose one option.)"
    - Maintain an immersive, in-world (diegetic) tone. Present checks and choices as a natural part of the scene, not as system instructions.
    - Do NOT present choices or branches every turn. Overusing explicit choices can break immersion and reduce the player’s sense of freedom. Only include them when they are required by script or really necessary for progression.
- If the player's action significantly deviates from the main storyline, guide them back naturally through NPC dialogue, NPC actions, or environmental consequences.
- If the player clearly commits to the next plot or next scene, transition into that material immediately in this same response instead of re-opening the current setup plot.
- When such a handoff happens, write the first playable beats of the target plot naturally. Do not repeat a second opening for the current plot.
- Focus on describing NPC reactions and changes in the scene.
- Reveal information **gradually**. Do NOT provide too much information at once; encourage player exploration and role-play.
- Do NOT ask rhetorical or leading questions about the PC’s beliefs, thoughts, or motivations.
- Do NOT ask questions to the player or suggest what they should do next.

## Game Mechanics

* If a dice result is provided, the narrative must strictly reflect the skill check outcome:

  * **Extreme Success**: Reveal critical details, hidden elements, or additional high-value insights beyond normal expectations.
  * **Hard Success**: Reveal important details clearly and efficiently, possibly with minor additional insight.
  * **Regular Success**: Reveal expected information necessary to progress.
  * **Fail**: Withhold key information or provide limited, vague, or inconclusive results.
  * **Fumble (Worst Fail)**: Introduce significant negative consequences, risks, or complications.

* Do not fabricate information. Only use information available in the provided context.

  * If the result is a success but no relevant information exists in the context, do not invent new details.

* Do not mention system processing, hidden prompts, or any tooling in the narrative.

# INSTRUCTION

Generate the next narrative response.

Requirements:
- Move story toward Plot Goal.
- Maintain consistency with Memory.
- Use Retrieved Knowledge only if relevant.
- Preserve immersion.
- If `Scene Entry Turn` is true and `Player Input` is empty, write this response as the opening narration for the current scene/plot.
- If `Scene Entry Turn` is true and `Player Input` is not empty, write this response as the first playable continuation of the current scene/plot while still responding to the player's latest action.
- `Scene Entry Turn` does NOT mean a separate pre-plot phase. The current plot has already started.

---

# USER INPUT

Player Input:
{user_input}

---

# SCRIPT STATE

Scene ID: {scene_id}  
Plot ID: {plot_id}

Scene Goal:
{current_scene_goal}

Scene Description:
{current_scene_description}

Plot Goal:
{current_plot_goal}

Current Plot Excerpt:
{current_plot_excerpt}

Scene Entry Turn:
{scene_entry_turn}

Mandatory Events (if any):
{mandatory_events}

---

# MEMORY

Previous Plot Summary:
{previous_plot_summary}

Current Scene Summary:
{current_scene_summary}

Recent Conversation:
{recent_conversation}

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

World Context:
{world_context_info}

Hidden Truth (Keeper-only):
{truth_related_info}

Items / Clues:
{item_or_clue_info}

---

# INTERNAL STATE

Plot Progress:
{plot_progress_percentage_or_state}

Scene Progress:
{scene_progress_percentage_or_state}

---

# DICE CHECK RESULT

Dice Result:
{dice_result}

Skill Check Result:
{skill_check_result}
"""

ROLL_CHECK_PROMPT_TEMPLATE = """# SYSTEM PROMPT

You are a Call of Cthulhu rules assistant.
Decide whether the player's latest action requires a deterministic dice skill check.

Return strict JSON only:
{{
  "need_check": true or false,
  "skill": "skill name or SAN or empty string",
  "reason": "brief English reason",
  "dice_type": "1d100 or empty string"
}}

Rules:
- Use English only.
- Trigger a skill check only when there is a clear and explicit intent:
    - The player directly attempts an action that involves uncertainty, risk, investigation, combat, or sanity pressure, or
    - The Keeper has previously introduced a situation that clearly invites a check, and the player responds to it.
- Do NOT trigger a skill check if the player has not expressed a clear action or intent.
  - Passive, vague, or observational actions without a specific goal should not automatically result in a check.
- A skill check should feel natural within the narrative:
  - It may be initiated by the player’s action, or
  - Introduced by the Keeper through the scene, but only becomes active when the player engages with it.
- Do NOT introduce checks abruptly without narrative or player-driven justification.
- If no meaningful uncertainty or risk exists, set `need_check` to false.
- For Call of Cthulhu skill checks, use 1d100.
- Prefer skill names from the player skill list when possible.

Examples when checks are needed:
- Searching for hidden evidence -> Spot Hidden
- Reading strange documents -> Library Use
- Staying calm before horror -> SAN
- Forcing a stuck door -> STR
- Dodging an attack -> DEX or Dodge

Tool usage format reference:
TOOL_CALL: roll_dice
{{
  "dice_type": "1d100",
  "reason": "Spot Hidden check"
}}

Player Input:
{user_input}

Scene ID: {scene_id}
Plot ID: {plot_id}
Scene Goal:
{current_scene_goal}

Scene Description:
{current_scene_description}

Plot Goal:
{current_plot_goal}

Current Plot Excerpt:
{current_plot_excerpt}

Mandatory Events:
{mandatory_events}

Previous Plot Summary:
{previous_plot_summary}

Current Scene Summary:
{current_scene_summary}

Recent Conversation:
{recent_conversation}

Player:
{player_related_info}

Full Player Skill List:
{player_skill_list}
"""

KP_OPENING_MARKER = '[KP_OPENING]'


class NarrativeState(TypedDict, total=False):
    scene_id: str
    plot_id: str
    plot_progress: float
    scene_progress: float
    player_profile: dict[str, Any]
    conversation_history: list[dict[str, Any]]
    retrieved_docs: list[dict[str, Any]]
    latest_user_input: str
    prompt: str
    roll_check_prompt: str
    retrieval_queries: list[str]
    response: str
    dice_result: str | None
    skill_check_result: str | None
    need_check: bool
    check_skill: str
    check_reason: str
    dice_type: str
    plot_completed: bool
    scene_completed: bool
    scene_goal: str
    scene_description: str
    plot_goal: str
    current_plot_raw_text: str
    scene_entry_turn: bool
    next_plot_goal: str
    next_plot_excerpt: str
    next_scene_goal: str
    next_scene_plot_goal: str
    next_scene_plot_excerpt: str
    latest_turn_text: str
    mandatory_events: list[str]
    previous_plot_summary: str
    current_scene_summary: str
    plot_advance_target: str
    plot_advance_reason: str
    scene_completion_reason: str
    pre_response_transition_applied: bool
    pre_response_transition_target: str
    pre_response_transition_reason: str
    output_language: str
    debug_prompts: list[dict[str, str]]


class NarrativeAgent:
    def __init__(self, db: Database, vector_store: ChromaStore) -> None:
        self.db = db
        self.vector_store = vector_store
        self.debug_mode = False
        self.latest_debug_prompts: list[dict[str, str]] = []
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(NarrativeState)
        workflow.add_node('build_prompt', self.build_prompt)
        workflow.add_node('retrieve_memory', self.retrieve_memory)
        workflow.add_node('pre_response_transition', self.pre_response_transition)
        workflow.add_node('generate_retrieval_queries', self.generate_retrieval_queries)
        workflow.add_node('vector_retrieve', self.vector_retrieve)
        workflow.add_node('construct_context', self.construct_context)
        workflow.add_node('check_whether_roll_dice', self.check_whether_roll_dice)
        workflow.add_node('roll_dice', self.roll_dice)
        workflow.add_node('generate_response', self.generate_response)
        workflow.add_node('write_memory', self.write_memory)
        workflow.add_node('check_plot_completion', self.check_plot_completion)
        workflow.add_node('check_scene_completion', self.check_scene_completion)
        workflow.add_node('update_state', self.update_state)

        workflow.set_entry_point('build_prompt')
        workflow.add_edge('build_prompt', 'retrieve_memory')
        workflow.add_edge('retrieve_memory', 'pre_response_transition')
        workflow.add_edge('pre_response_transition', 'generate_retrieval_queries')
        workflow.add_edge('generate_retrieval_queries', 'vector_retrieve')
        workflow.add_edge('vector_retrieve', 'construct_context')
        workflow.add_edge('construct_context', 'check_whether_roll_dice')
        workflow.add_conditional_edges(
            'check_whether_roll_dice',
            self._route_after_roll_check,
            {
                'roll_dice': 'roll_dice',
                'generate_response': 'generate_response',
            },
        )
        workflow.add_edge('roll_dice', 'generate_response')
        workflow.add_edge('generate_response', 'write_memory')
        workflow.add_edge('write_memory', 'check_plot_completion')
        workflow.add_edge('check_plot_completion', 'check_scene_completion')
        workflow.add_edge('check_scene_completion', 'update_state')
        workflow.add_edge('update_state', END)
        return workflow.compile()

    def run_turn(self, user_input: str) -> dict[str, Any]:
        self.latest_debug_prompts = []
        system_state = self.db.get_system_state()
        state: NarrativeState = {
            'scene_id': system_state.get('current_scene_id', ''),
            'plot_id': system_state.get('current_plot_id', ''),
            'plot_progress': float(system_state.get('plot_progress', 0.0)),
            'scene_progress': float(system_state.get('scene_progress', 0.0)),
            'output_language': system_state.get('output_language', 'English'),
            'player_profile': self.db.get_player_profile(),
            'latest_user_input': user_input,
            'conversation_history': [],
            'retrieved_docs': [],
            'mandatory_events': [],
            'current_plot_raw_text': '',
            'scene_entry_turn': False,
            'next_plot_goal': '',
            'next_plot_excerpt': '',
            'next_scene_goal': '',
            'next_scene_plot_goal': '',
            'next_scene_plot_excerpt': '',
            'latest_turn_text': '',
            'dice_result': None,
            'skill_check_result': None,
            'need_check': False,
            'check_skill': '',
            'check_reason': '',
            'dice_type': '',
            'plot_advance_target': 'stay',
            'plot_advance_reason': '',
            'scene_completion_reason': '',
            'pre_response_transition_applied': False,
            'pre_response_transition_target': 'stay',
            'pre_response_transition_reason': '',
            'debug_prompts': [],
        }
        result = self.graph.invoke(state)
        self.latest_debug_prompts = result.get('debug_prompts', [])
        return result

    def generate_initial_response(self) -> dict[str, Any]:
        self.latest_debug_prompts = []
        system_state = self.db.get_system_state()
        state: NarrativeState = {
            'scene_id': system_state.get('current_scene_id', ''),
            'plot_id': system_state.get('current_plot_id', ''),
            'plot_progress': float(system_state.get('plot_progress', 0.0)),
            'scene_progress': float(system_state.get('scene_progress', 0.0)),
            'output_language': system_state.get('output_language', 'English'),
            'player_profile': self.db.get_player_profile(),
            'latest_user_input': '',
            'conversation_history': [],
            'retrieved_docs': [],
            'mandatory_events': [],
            'current_plot_raw_text': '',
            'scene_entry_turn': False,
            'next_plot_goal': '',
            'next_plot_excerpt': '',
            'next_scene_goal': '',
            'next_scene_plot_goal': '',
            'next_scene_plot_excerpt': '',
            'latest_turn_text': '',
            'dice_result': None,
            'skill_check_result': None,
            'need_check': False,
            'check_skill': '',
            'check_reason': '',
            'dice_type': '',
            'plot_advance_target': 'stay',
            'plot_advance_reason': '',
            'scene_completion_reason': '',
            'pre_response_transition_applied': False,
            'pre_response_transition_target': 'stay',
            'pre_response_transition_reason': '',
            'debug_prompts': [],
        }
        state = self.retrieve_memory(state)
        state = self.pre_response_transition(state)
        state = self.generate_retrieval_queries(state)
        state = self.vector_retrieve(state)
        state = self.generate_response(state)
        self.db.append_memory(state['scene_id'], state['plot_id'], '', state.get('response', ''))
        self.latest_debug_prompts = state.get('debug_prompts', [])
        return state

    def ensure_kp_opening(self, scene_id: str, plot_id: str) -> str | None:
        self.latest_debug_prompts = []
        if not scene_id or not plot_id:
            return None
        if not self._is_initial_story_position(scene_id, plot_id):
            return None
        if self.db.has_scene_opening(scene_id, KP_OPENING_MARKER):
            return None
        opening = self._generate_scene_opening(scene_id, plot_id)
        self.db.append_memory(scene_id, plot_id, KP_OPENING_MARKER, opening)
        return opening

    def set_debug_mode(self, enabled: bool) -> None:
        self.debug_mode = bool(enabled)

    def _record_prompt(self, state: NarrativeState | None, name: str, prompt: str) -> None:
        if not self.debug_mode:
            return
        entry = {'name': name, 'prompt': prompt}
        if state is not None:
            state.setdefault('debug_prompts', []).append(entry)
        self.latest_debug_prompts.append(entry)

    def _llm_call(self, prompt: str, *, step_name: str, model: str = "qwen/qwen3.5-397b-a17b") -> str:
        logger.info("LLM step=%s prompt_length=%s", step_name, len(prompt))
        return call_nvidia_llm(prompt, model=model, step_name=step_name).strip()

    def _get_output_language(self, state: NarrativeState | None = None) -> str:
        if state and state.get('output_language'):
            return str(state['output_language'])
        return str(self.db.get_system_state().get('output_language', 'English'))

    def _route_after_roll_check(self, state: NarrativeState) -> str:
        return 'roll_dice' if state.get('need_check') else 'generate_response'

    def _format_player_skill_list(self, state: NarrativeState) -> str:
        profile = state.get('player_profile', {}) or {}
        lines: list[str] = []
        for group_name in ('occupation', 'personal_interest'):
            chosen = profile.get('chosen_skill_allocations', {}).get(group_name, [])
            if chosen:
                lines.append(f"{group_name.title()} Skills:")
                lines.extend(str(item) for item in chosen)
        characteristics = profile.get('characteristics', {})
        if characteristics:
            lines.append('Characteristics:')
            lines.extend(f"{k}:{v}" for k, v in characteristics.items())
        derived = profile.get('derived_attributes', {})
        if derived:
            lines.append('Derived Attributes:')
            lines.extend(f"{k}:{v}" for k, v in derived.items())
        return '\n'.join(lines) or 'No player skills available.'

    def _format_recent_conversation(self, state: NarrativeState, rounds: int = 3) -> str:
        history = state.get('conversation_history', [])[-rounds:]
        if not history:
            return 'None'
        lines: list[str] = []
        for turn in history:
            user_text = str(turn.get('user', '')).strip() or '(no player input)'
            keeper_text = str(turn.get('agent', '')).strip() or '(no keeper response)'
            lines.append(f"Player: {user_text}")
            lines.append(f"Keeper: {keeper_text}")
        return '\n'.join(lines)

    def _plot_excerpt(self, raw_text: str, limit: int = 500) -> str:
        text = (raw_text or '').strip()
        if not text:
            return 'None'
        if len(text) <= limit:
            return text
        return f"{text[: limit - 3].rstrip()}..."

    def _is_initial_story_position(self, scene_id: str, plot_id: str) -> bool:
        scenes = self.db.list_scenes()
        if not scenes:
            return False
        first_scene = scenes[0]
        first_plot = (first_scene.get('plots') or [{}])[0]
        return scene_id == first_scene.get('scene_id') and plot_id == first_plot.get('plot_id')

    def _parse_roll_check_response(self, text: str) -> dict[str, Any]:
        match = re.search(r'\{.*\}', text, flags=re.DOTALL)
        payload = match.group(0) if match else text
        loader = json5.loads if json5 is not None else json.loads
        data = loader(payload)
        if not isinstance(data, dict):
            return {}
        return data

    def build_prompt(self, state: NarrativeState) -> NarrativeState:
        return state

    def retrieve_memory(self, state: NarrativeState) -> NarrativeState:
        state['conversation_history'] = self.db.get_recent_turns(state['scene_id'], state['plot_id'], limit=12)
        scene = self.db.get_scene(state['scene_id'])
        hints = story_position_context(self.db, state['scene_id'], state['plot_id'])
        state['current_plot_raw_text'] = hints.get('current_plot_raw_text', '')
        state['next_plot_goal'] = hints.get('next_plot_goal', '')
        state['next_plot_excerpt'] = hints.get('next_plot_excerpt', '')
        state['next_scene_goal'] = hints.get('next_scene_goal', '')
        state['next_scene_plot_goal'] = hints.get('next_scene_plot_goal', '')
        state['next_scene_plot_excerpt'] = hints.get('next_scene_plot_excerpt', '')
        if scene:
            state['scene_goal'] = scene.get('scene_goal', '')
            state['scene_description'] = scene.get('scene_description', '')
            state['current_scene_summary'] = scene.get('scene_summary', '')
            plots = scene.get('plots', [])
            for i, plot in enumerate(plots):
                if plot['plot_id'] == state['plot_id']:
                    state['plot_goal'] = plot.get('plot_goal', '')
                    state['mandatory_events'] = plot.get('mandatory_events', [])
                    if i > 0:
                        prev_plot = plots[i - 1]['plot_id']
                        state['previous_plot_summary'] = self.db.get_summary('plot', scene_id=state['scene_id'], plot_id=prev_plot)
                    else:
                        state['previous_plot_summary'] = ''
                    break
            first_plot = (plots or [{}])[0]
            state['scene_entry_turn'] = bool(
                not state.get('conversation_history')
                and state.get('plot_id')
                and state.get('plot_id') == first_plot.get('plot_id')
            )
        return state

    def pre_response_transition(self, state: NarrativeState) -> NarrativeState:
        if not (state.get('latest_user_input') or '').strip():
            return state

        evaluation = evaluate_pre_response_transition(
            state['latest_user_input'],
            plot_goal=state.get('plot_goal', ''),
            scene_goal=state.get('scene_goal', ''),
            next_plot_goal=state.get('next_plot_goal', ''),
            next_scene_goal=state.get('next_scene_goal', ''),
            current_plot_raw_text=state.get('current_plot_raw_text', ''),
            conversation_history=state.get('conversation_history', []),
            prompt_recorder=lambda prompt: self._record_prompt(state, 'pre_response_transition_prompt', prompt),
        )
        advance_target = str(evaluation.get('advance_target', 'stay'))
        reason = str(evaluation.get('reason', ''))
        state['pre_response_transition_target'] = advance_target
        state['pre_response_transition_reason'] = reason

        if advance_target == 'stay':
            return state

        origin_scene_id = state.get('scene_id', '')
        origin_plot_id = state.get('plot_id', '')
        if not origin_scene_id or not origin_plot_id:
            return state

        origin_state: NarrativeState = dict(state)
        origin_state['plot_completed'] = True
        origin_state['plot_progress'] = 1.0
        origin_state['plot_advance_target'] = advance_target
        origin_state['plot_advance_reason'] = reason
        origin_state['response'] = ''

        self.db.update_plot(origin_plot_id, status='completed', progress=1.0)
        self.db.save_summary(
            'plot',
            self._build_plot_summary(origin_state),
            scene_id=origin_scene_id,
            plot_id=origin_plot_id,
        )

        latest_turn_text = (
            f"User: {state.get('latest_user_input', '')}\n"
            "Agent: (transition pending)"
        )
        scene_evaluation = evaluate_scene_completion(
            self.db,
            origin_scene_id,
            conversation_history=state.get('conversation_history', []),
            latest_turn_text=latest_turn_text,
            next_scene_goal=state.get('next_scene_goal', ''),
            plot_advance_target=advance_target,
            plot_advance_reason=reason,
            prompt_recorder=lambda prompt: self._record_prompt(state, 'pre_response_scene_completion_prompt', prompt),
        )
        origin_scene_completed = bool(scene_evaluation.get('completed', False))
        origin_scene_progress = float(scene_evaluation.get('progress', state.get('scene_progress', 0.0)))

        if origin_scene_completed:
            origin_state['scene_completion_reason'] = str(scene_evaluation.get('reason', ''))
            scene_summary = self._build_scene_summary(origin_scene_id, state=origin_state)
            self.db.update_scene(origin_scene_id, {'status': 'completed', 'scene_summary': scene_summary})
            self.db.save_summary('scene', scene_summary, scene_id=origin_scene_id)
        else:
            self.db.update_scene(origin_scene_id, {'status': 'in_progress'})

        next_pos = next_story_position(self.db, origin_scene_id, origin_plot_id)
        state['scene_id'] = next_pos['current_scene_id']
        state['plot_id'] = next_pos['current_plot_id']
        state['plot_progress'] = float(next_pos.get('plot_progress', 0.0))
        state['scene_progress'] = origin_scene_progress if state['scene_id'] == origin_scene_id else float(next_pos.get('scene_progress', 0.0))
        state['plot_advance_target'] = advance_target
        state['plot_advance_reason'] = reason
        state['scene_completion_reason'] = str(scene_evaluation.get('reason', ''))
        state['pre_response_transition_applied'] = True
        state['plot_completed'] = False
        state['scene_completed'] = False
        state['latest_turn_text'] = latest_turn_text
        state['retrieved_docs'] = []
        state['prompt'] = ''
        state['roll_check_prompt'] = ''
        state['dice_result'] = None
        state['skill_check_result'] = None
        state['need_check'] = False
        state['check_skill'] = ''
        state['check_reason'] = ''
        state['dice_type'] = ''
        state['retrieval_queries'] = []
        state['response'] = ''
        return self.retrieve_memory(state)

    def generate_retrieval_queries(self, state: NarrativeState) -> NarrativeState:
        state['retrieval_queries'] = generate_retrieval_queries(
            state['latest_user_input'],
            state.get('plot_goal', ''),
            state.get('mandatory_events', []),
            state.get('conversation_history', []),
        )
        return state

    def vector_retrieve(self, state: NarrativeState) -> NarrativeState:
        docs = []
        for q in state.get('retrieval_queries', []):
            docs.extend(self.vector_store.search(q, k=3))
        state['retrieved_docs'] = docs[:8]
        return state

    def construct_context(self, state: NarrativeState) -> NarrativeState:
        categorized = categorize_docs(state.get('retrieved_docs', []))
        player_skill_list = self._format_player_skill_list(state)
        recent_conversation = self._format_recent_conversation(state, rounds=3)
        state['roll_check_prompt'] = ROLL_CHECK_PROMPT_TEMPLATE.format(
            user_input=state['latest_user_input'],
            scene_id=state.get('scene_id', ''),
            plot_id=state.get('plot_id', ''),
            current_scene_goal=state.get('scene_goal', '') or 'None',
            current_scene_description=state.get('scene_description', '') or 'None',
            current_plot_goal=state.get('plot_goal', '') or 'None',
            current_plot_excerpt=state.get('current_plot_raw_text', '') or 'None',
            mandatory_events=', '.join(state.get('mandatory_events', [])) or 'None',
            previous_plot_summary=state.get('previous_plot_summary', '') or 'None',
            current_scene_summary=state.get('current_scene_summary', '') or 'None',
            recent_conversation=recent_conversation,
            player_related_info=str(state.get('player_profile', {})),
            player_skill_list=player_skill_list,
        )
        self._record_prompt(state, 'roll_check_prompt', state['roll_check_prompt'])
        return state

    def check_whether_roll_dice(self, state: NarrativeState) -> NarrativeState:
        try:
            raw = self._llm_call(state['roll_check_prompt'], step_name='check_whether_roll_dice')
            parsed = self._parse_roll_check_response(raw)
            state['need_check'] = bool(parsed.get('need_check', False))
            state['check_skill'] = str(parsed.get('skill', '')).strip()
            state['check_reason'] = str(parsed.get('reason', '')).strip() or state['check_skill'] or 'skill check'
            state['dice_type'] = str(parsed.get('dice_type', '')).strip() or ('1d100' if state['need_check'] else '')
            logger.info(
                "Roll check decision need_check=%s skill=%s reason=%s dice_type=%s",
                state.get('need_check'),
                state.get('check_skill'),
                state.get('check_reason'),
                state.get('dice_type'),
            )
        except Exception as exc:
            logger.error("Roll check evaluation failed error=%s", exc)
            state['need_check'] = False
            state['check_skill'] = ''
            state['check_reason'] = ''
            state['dice_type'] = ''
        return state

    def roll_dice(self, state: NarrativeState) -> NarrativeState:
        dice_type = state.get('dice_type', '') or '1d100'
        dice_value = self._roll_dice_expr(dice_type)
        if dice_value:
            reason = state.get('check_reason', '') or 'skill check'
            state['dice_result'] = f"{dice_type}: {dice_value} (reason: {reason})"
            state['skill_check_result'] = self._build_skill_check_result(
                state,
                dice_type,
                state.get('check_skill', '') or state.get('check_reason', ''),
                dice_value,
            )
            logger.info(
                "Deterministic roll completed dice_result=%s skill_check_result=%s",
                state.get('dice_result'),
                state.get('skill_check_result'),
            )
        return state

    def generate_response(self, state: NarrativeState) -> NarrativeState:
        try:
            categorized = categorize_docs(state.get('retrieved_docs', []))
            recent_conversation = self._format_recent_conversation(state, rounds=3)
            state['prompt'] = RESPONSE_PROMPT_TEMPLATE.format(
                agent_role='Narrative Agent',
                game_system='TRPG',
                tone_style='Immersive and grounded',
                narrative_perspective='Second person',
                response_length='Concise',
                output_language=self._get_output_language(state),
                user_input=state['latest_user_input'],
                scene_id=state.get('scene_id', ''),
                plot_id=state.get('plot_id', ''),
                current_scene_goal=state.get('scene_goal', '') or 'None',
                current_scene_description=state.get('scene_description', '') or 'None',
                current_plot_goal=state.get('plot_goal', '') or 'None',
                current_plot_excerpt=state.get('current_plot_raw_text', '') or 'None',
                scene_entry_turn='true' if state.get('scene_entry_turn') else 'false',
                mandatory_events=', '.join(state.get('mandatory_events', [])) or 'None',
                previous_plot_summary=state.get('previous_plot_summary', '') or 'None',
                current_scene_summary=state.get('current_scene_summary', '') or 'None',
                recent_conversation=recent_conversation,
                npc_related_info=categorized['npc_related_info'],
                player_related_info=str(state.get('player_profile', {})),
                location_related_info=categorized['location_related_info'],
                game_rule_info=categorized['game_rule_info'],
                world_context_info=categorized['world_context_info'],
                truth_related_info=categorized['truth_related_info'],
                item_or_clue_info=categorized['item_or_clue_info'],
                plot_progress_percentage_or_state=f"{state.get('plot_progress', 0.0):.0%}",
                scene_progress_percentage_or_state=f"{state.get('scene_progress', 0.0):.0%}",
                dice_result=state.get('dice_result') or 'None',
                skill_check_result=state.get('skill_check_result') or 'None',
            )
            self._record_prompt(state, 'generate_response_prompt', state['prompt'])
            state['response'] = self._llm_call(state['prompt'], step_name='generate_response')
        except Exception as e:
            logger.error("LLM error in generate_response prompt_length=%s error=%s", len(state.get('prompt', '')), e)
            print("LLM error:", e)
            traceback.print_exc()
            state['response'] = self._fallback_response(state)
        return state

    def write_memory(self, state: NarrativeState) -> NarrativeState:
        self.db.append_memory(state['scene_id'], state['plot_id'], state['latest_user_input'], state['response'])
        return state

    def check_plot_completion(self, state: NarrativeState) -> NarrativeState:
        if state.get('pre_response_transition_applied'):
            return state
        evaluation = evaluate_plot_completion(
            state['latest_user_input'],
            state['response'],
            state.get('plot_progress', 0.0),
            state.get('mandatory_events', []),
            plot_goal=state.get('plot_goal', ''),
            scene_goal=state.get('scene_goal', ''),
            next_plot_goal=state.get('next_plot_goal', ''),
            next_scene_goal=state.get('next_scene_goal', ''),
            current_plot_raw_text=state.get('current_plot_raw_text', ''),
            conversation_history=state.get('conversation_history', []),
            prompt_recorder=lambda prompt: self._record_prompt(state, 'plot_completion_prompt', prompt),
        )
        done = bool(evaluation.get('completed', False))
        progress = float(evaluation.get('progress', state.get('plot_progress', 0.0)))
        state['plot_advance_target'] = str(evaluation.get('advance_target', 'stay'))
        state['plot_advance_reason'] = str(evaluation.get('reason', ''))
        state['plot_completed'] = done or state['plot_advance_target'] != 'stay'
        state['plot_progress'] = progress
        self.db.update_plot(state['plot_id'], progress=progress)
        if state['plot_completed']:
            self.db.update_plot(state['plot_id'], status='completed', progress=1.0)
            self.db.save_summary(
                'plot',
                self._build_plot_summary(state),
                scene_id=state['scene_id'],
                plot_id=state['plot_id'],
            )
            state['plot_progress'] = 1.0
        return state

    def check_scene_completion(self, state: NarrativeState) -> NarrativeState:
        if state.get('pre_response_transition_applied'):
            return state
        state['latest_turn_text'] = (
            f"User: {state.get('latest_user_input', '')}\n"
            f"Agent: {state.get('response', '')}"
        )
        evaluation = evaluate_scene_completion(
            self.db,
            state['scene_id'],
            conversation_history=state.get('conversation_history', []),
            latest_turn_text=state.get('latest_turn_text', ''),
            next_scene_goal=state.get('next_scene_goal', ''),
            plot_advance_target=state.get('plot_advance_target', 'stay'),
            plot_advance_reason=state.get('plot_advance_reason', ''),
            prompt_recorder=lambda prompt: self._record_prompt(state, 'scene_completion_prompt', prompt),
        )
        done = bool(evaluation.get('completed', False))
        progress = float(evaluation.get('progress', state.get('scene_progress', 0.0)))
        state['scene_completion_reason'] = str(evaluation.get('reason', ''))
        state['scene_completed'] = done
        state['scene_progress'] = progress
        if done:
            scene_summary = self._build_scene_summary(state['scene_id'], state=state)
            self.db.update_scene(state['scene_id'], {'status': 'completed', 'scene_summary': scene_summary})
            self.db.save_summary('scene', scene_summary, scene_id=state['scene_id'])
        else:
            self.db.update_scene(state['scene_id'], {'status': 'in_progress'})
        return state

    def update_state(self, state: NarrativeState) -> NarrativeState:
        if state.get('pre_response_transition_applied'):
            self.db.update_system_state(
                {
                    'current_scene_id': state.get('scene_id', ''),
                    'current_plot_id': state.get('plot_id', ''),
                    'plot_progress': state.get('plot_progress', 0.0),
                    'scene_progress': state.get('scene_progress', 0.0),
                    'current_scene_intro': '',
                }
            )
            return state
        if state.get('plot_completed') or state.get('scene_completed'):
            next_pos = next_story_position(self.db, state['scene_id'], state['plot_id'])
            self.db.update_system_state(
                {
                    'current_scene_id': next_pos['current_scene_id'],
                    'current_plot_id': next_pos['current_plot_id'],
                    'plot_progress': next_pos['plot_progress'],
                    'scene_progress': next_pos['scene_progress'],
                    'current_scene_intro': next_pos['current_scene_intro'],
                }
            )
            state['scene_id'] = next_pos['current_scene_id']
            state['plot_id'] = next_pos['current_plot_id']
        else:
            self.db.update_system_state(
                {
                    'plot_progress': state.get('plot_progress', 0.0),
                    'scene_progress': state.get('scene_progress', 0.0),
                }
            )
        return state

    def _roll_dice_expr(self, dice_expr: str) -> str | None:
        m = re.fullmatch(r'\s*(\d*)d(\d+)\s*', (dice_expr or '').lower())
        if not m:
            return None
        count = int(m.group(1) or '1')
        sides = int(m.group(2))
        count = max(1, min(count, 20))
        sides = max(2, min(sides, 1000))
        rolls = [random.randint(1, sides) for _ in range(count)]
        return f"{rolls} (sum={sum(rolls)})"

    def _normalize_skill_name(self, value: str) -> str:
        return re.sub(r'[^a-z0-9]+', '', (value or '').strip().lower())

    def _extract_named_value(self, text: str) -> tuple[str, int] | None:
        if ':' not in text:
            return None
        name, raw_value = text.split(':', 1)
        try:
            return name.strip(), int(float(raw_value.strip()))
        except ValueError:
            return None

    def _resolve_skill_value(self, state: NarrativeState, reason: str) -> tuple[str, int] | None:
        profile = state.get('player_profile', {}) or {}
        candidates: dict[str, tuple[str, int]] = {}

        for bucket_name in ('occupation', 'personal_interest'):
            for entry in profile.get('chosen_skill_allocations', {}).get(bucket_name, []):
                parsed = self._extract_named_value(str(entry))
                if parsed:
                    skill_name, skill_value = parsed
                    candidates[self._normalize_skill_name(skill_name)] = (skill_name, skill_value)

        for attr_group in ('characteristics', 'derived_attributes'):
            for skill_name, skill_value in profile.get(attr_group, {}).items():
                try:
                    candidates[self._normalize_skill_name(str(skill_name))] = (str(skill_name), int(float(skill_value)))
                except (TypeError, ValueError):
                    continue

        reason_norm = self._normalize_skill_name(reason)
        if not reason_norm:
            return None
        if 'sanity' in reason.lower():
            san_entry = candidates.get('san')
            if san_entry:
                return ('SAN', san_entry[1])
        for key, value in candidates.items():
            if key and (key in reason_norm or reason_norm in key):
                return value
        return None

    def _extract_roll_total(self, dice_text: str) -> int | None:
        m = re.search(r'sum=(\d+)', dice_text)
        if m:
            return int(m.group(1))
        return None

    def _evaluate_skill_check(self, roll_total: int, skill_value: int) -> str:
        if roll_total >= 96:
            return 'Worst Fail'
        if roll_total <= max(1, skill_value // 5):
            return 'Extreme Success'
        if roll_total <= max(1, skill_value // 2):
            return 'Hard Success'
        if roll_total <= skill_value:
            return 'Regular Success'
        return 'Fail'

    def _build_skill_check_result(
        self,
        state: NarrativeState,
        dice_type: str,
        reason: str,
        dice_text: str,
    ) -> str | None:
        if self._normalize_skill_name(dice_type) != '1d100':
            return None
        roll_total = self._extract_roll_total(dice_text)
        if roll_total is None:
            return None
        resolved = self._resolve_skill_value(state, reason)
        if resolved is None:
            return None
        skill_name, skill_value = resolved
        outcome = self._evaluate_skill_check(roll_total, skill_value)
        return f"{skill_name} {skill_value}: {outcome}"

    def _fallback_response(self, state: NarrativeState) -> str:
        output_language = self._get_output_language(state)
        user_input = state['latest_user_input']
        dice_hint = re.search(r'(\d*d\d+)', user_input.lower())
        if dice_hint:
            rolled = self._roll_dice_expr(dice_hint.group(1))
            if rolled:
                state['dice_result'] = f"{dice_hint.group(1)}: {rolled}"
                state['skill_check_result'] = None
        if output_language == 'Chinese':
            base = f"你的行动是：{user_input}。"
            if state.get('plot_goal'):
                base += f"剧情正朝着以下目标推进：{state['plot_goal']}。"
            if state.get('retrieved_docs'):
                base += f"相关线索：{state['retrieved_docs'][0]['content']}。"
            if state.get('dice_result'):
                base += f"已应用骰子结果（{state['dice_result']}）。"
            base += '接下来会发生什么？'
            return base
        base = f"You act: {user_input}. "
        if state.get('plot_goal'):
            base += f"The story advances toward: {state['plot_goal']}. "
        if state.get('retrieved_docs'):
            base += f"Relevant clue: {state['retrieved_docs'][0]['content']}. "
        if state.get('dice_result'):
            base += f"Dice result applied ({state['dice_result']}). "
        if state.get('skill_check_result'):
            base += f"Skill check result applied ({state['skill_check_result']}). "
        base += 'What do you do next?'
        return base

    def _get_previous_scene_summary(self, scene_id: str) -> str:
        scenes = self.db.list_scenes()
        current_idx = next((i for i, s in enumerate(scenes) if s.get('scene_id') == scene_id), -1)
        if current_idx <= 0:
            return ''
        previous_scene = scenes[current_idx - 1]
        previous_scene_id = previous_scene.get('scene_id', '')
        if not previous_scene_id:
            return ''
        return self.db.get_summary('scene', scene_id=previous_scene_id) or previous_scene.get('scene_summary', '') or ''

    def _generate_scene_opening(self, scene_id: str, plot_id: str) -> str:
        scene = self.db.get_scene(scene_id) or {}
        plot = self.db.get_plot(plot_id) or {}
        previous_scene_summary = self._get_previous_scene_summary(scene_id)
        output_language = self._get_output_language()
        plot_excerpt = self._plot_excerpt(str(plot.get('raw_text', '')), limit=500)
        prompt = f"""
You are a TRPG Keeper.

Write the opening narration for a new scene or plot.

Guidelines:
- Transition smoothly from the previous scene or plot into the current situation.
- Describe the immediate environment, situation, NPC dialogue, and NPC actions in a vivid and immersive way.
- Focus only on what is happening right now.
- Reveal information gradually. Do not provide too much information at once; encourage player exploration and role-play.
- Present the situation, then STOP and wait for the player to decide what to do next.
- Do NOT reveal future events or the full storyline.
- Do NOT decide the player character’s actions or thoughts.
- Do NOT ask hook questions.
- Write the opening entirely in {output_language}.


Scene ID: {scene_id}
Scene Goal: {scene.get('scene_goal', '')}
Scene Description: {scene.get('scene_description', '')}
Plot ID: {plot_id}
Plot Goal: {plot.get('plot_goal', '')}
Current Plot Excerpt: {plot_excerpt}
Mandatory Events: {plot.get('mandatory_events', [])}
Player Profile: {self.db.get_player_profile()}
Previous Scene Summary: {previous_scene_summary or 'None'}
"""
        self._record_prompt(None, 'scene_opening_prompt', prompt)
        try:
            result = self._llm_call(prompt, step_name='scene_opening_generation')
            if result:
                return result
        except Exception:
            pass
        if output_language == 'Chinese':
            return (
                f"夜色笼罩着{scene_id}。你感到命运的下一条线索正在将你向前牵引。\n"
                f"你当前的直接目标是：{plot.get('plot_goal', '推进剧情')}。\n"
                "你从哪里开始？"
            )
        return (
            f"Night settles over {scene_id}. You sense the next thread of fate pulling you forward.\n"
            f"Your immediate objective: {plot.get('plot_goal', 'advance the story')}.\n"
            "Where do you begin?"
        )

    def _build_plot_summary(self, state: NarrativeState) -> str:
        tail = state.get('conversation_history', [])[-8:]
        history_text = '\n'.join([f"User: {t.get('user', '')}\nAgent: {t.get('agent', '')}" for t in tail])
        plot_excerpt = self._plot_excerpt(state.get('current_plot_raw_text', ''), limit=650)
        prompt = f"""
Summarize the completed plot as durable narrative memory for future turns.

Return 5 to 7 bullet points.

Requirements:
- Preserve concrete, reusable information instead of vague recap.
- If the plot presented options or branches, list the exact options that mattered.
- State which option or branch the player actually entered or committed to.
- Preserve any branch logic or conditions that still matter later.
- Preserve key clues, NPC attitudes, revealed facts, and unresolved leads.
- Prefer exact named places, people, clues, options, and branch labels from the source when available.
- Do not write generic bullets like "the plot progressed" unless followed by specifics.

Scene: {state.get('scene_id', '')}
Plot: {state.get('plot_id', '')}
Plot Goal: {state.get('plot_goal', '')}
Mandatory Events: {state.get('mandatory_events', [])}
Plot Advance Target: {state.get('plot_advance_target', 'stay')}
Plot Advance Reason: {state.get('plot_advance_reason', '') or 'None'}
Next Plot Goal: {state.get('next_plot_goal', '') or 'None'}
Next Plot Excerpt: {state.get('next_plot_excerpt', '') or 'None'}
Next Scene Goal: {state.get('next_scene_goal', '') or 'None'}
Next Scene First Plot Goal: {state.get('next_scene_plot_goal', '') or 'None'}
Next Scene First Plot Excerpt: {state.get('next_scene_plot_excerpt', '') or 'None'}
Current Plot Excerpt:
{plot_excerpt}
Latest Turn User: {state.get('latest_user_input', '')}
Latest Turn Agent: {state.get('response', '')}
Recent History:
{history_text}
"""
        self._record_prompt(state, 'plot_summary_prompt', prompt)
        try:
            summary = call_nvidia_llm(prompt, step_name='plot_summary_generation').strip()
            if summary:
                return summary
        except Exception:
            pass
        bullets = [
            f"- Plot goal: {state.get('plot_goal', 'None')}",
            f"- Mandatory events or cues: {', '.join(state.get('mandatory_events', [])) or 'None recorded'}",
            f"- Entered branch / handoff: {state.get('plot_advance_target', 'stay')} ({state.get('plot_advance_reason', 'no explicit reason')})",
            f"- Latest player action: {state.get('latest_user_input', '') or 'None'}",
            f"- Latest keeper response: {state.get('response', '') or 'None'}",
        ]
        if state.get('next_plot_goal'):
            bullets.append(f"- Next plot in current scene: {state['next_plot_goal']}")
        if state.get('next_plot_excerpt'):
            bullets.append(f"- Next-plot detail: {state['next_plot_excerpt']}")
        if state.get('next_scene_goal'):
            bullets.append(f"- Next scene lead: {state['next_scene_goal']}")
        if state.get('next_scene_plot_goal'):
            bullets.append(f"- Next scene first plot: {state['next_scene_plot_goal']}")
        if state.get('next_scene_plot_excerpt'):
            bullets.append(f"- Next-scene plot detail: {state['next_scene_plot_excerpt']}")
        if plot_excerpt and plot_excerpt != 'None':
            bullets.append(f"- Plot excerpt / options context: {plot_excerpt}")
        return '\n'.join(bullets)

    def _build_scene_summary(self, scene_id: str, state: NarrativeState | None = None) -> str:
        scene = self.db.get_scene(scene_id) or {}
        plots = scene.get('plots', [])
        plot_lines = '\n'.join(
            [
                f"- {p.get('plot_id')}: goal={p.get('plot_goal', '')}, progress={p.get('progress', 0)}, status={p.get('status', '')}"
                for p in plots
            ]
        )
        prompt = f"""
Summarize a completed scene in 4 bullet points.
Include:
- core conflict
- emotional shift
- gained information
- narrative turning point

Scene: {scene_id}
Scene Goal: {scene.get('scene_goal', '')}
Plots:
{plot_lines}
"""
        self._record_prompt(state, 'scene_summary_prompt', prompt)
        try:
            summary = call_nvidia_llm(prompt, step_name='scene_summary_generation').strip()
            if summary:
                return summary
        except Exception:
            pass
        return f"Scene {scene_id} completed with major progress toward: {scene.get('scene_goal', '')}"
