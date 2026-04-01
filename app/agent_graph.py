from __future__ import annotations

import json5
import random
import re
from typing import Any, TypedDict
import traceback
import logging

from langgraph.graph import END, StateGraph

from app.database import Database
from app.llm_client import call_nvidia_llm
from app.rag import categorize_docs, generate_retrieval_queries
from app.state import evaluate_plot_completion, evaluate_scene_completion, next_story_position
from app.vector_store import ChromaStore

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """# SYSTEM PROMPT

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
- Immersion: {immersion_level}

## Constraints
- Use only information provided in Context.
- Do not fabricate missing knowledge.
- Avoid meta commentary.
- Hidden truth is keeper-only context. Use it for consistency, do not reveal it directly unless current plot already unlocks it.
- Final response language: {output_language}
- Write the response entirely in {output_language}.

## Player Interaction Rules

- The player controls the PC. Do NOT speak for the PC, extend their dialogue, or describe their internal thoughts.
- After the player acts or speaks, always advance the scene with NPC dialogue, NPC action, or environmental consequences.
- When a dice roll or player choice is required (e.g., selecting a branch), present it clearly in parentheses ().
- If the player's action significantly deviates from the main storyline, guide them back naturally through NPC dialogue, NPC actions, or environmental consequences.
- Focus on describing NPC reactions and changes in the scene.
- Reveal information gradually. Do not provide too much information at once; encourage player exploration and role-play.
- Do NOT ask rhetorical or leading questions about the PC’s beliefs, thoughts, or motivations.
- Do NOT ask questions to the player or suggest what they should do next.

## Game Mechanics

Use dice rolls for actions involving uncertainty.

### Skill Checks

Skill checks use **1d100** and produce one of the following outcomes:

- Extreme Success → major advantage or additional information  
- Hard Success → strong success with extra benefit  
- Regular Success → normal success  
- Fail → no meaningful progress  
- Worst Fail → severe negative consequence 

### Combat

Combat actions (attack, dodge, maneuver) require a roll.

Resolve the action by:
- determining hit, miss, or critical result
- applying damage
- updating HP
- describing the physical outcome

Higher success levels should produce stronger effects.

### Sanity

When the player encounters horror or supernatural events:

- perform a SAN check using a dice roll
- adjust SAN accordingly
- reflect psychological effects in the narrative

## Tool Use

Use tools when mechanical resolution is required.

### Dice Roll Tool

Use when:
- resolving skill checks
- resolving combat actions
- resolving sanity checks
- determining uncertain outcomes

### Call Format

TOOL_CALL: roll_dice
{{
  "dice_type": "{dice_type}",
  "reason": "{reason}"
}}

Example:

TOOL_CALL: roll_dice
{{
  "dice_type": "1d100",
  "reason": "Sanity check"
}}

### Rules
- Call when necessary.
- Do not fabricate dice results.
- Do not narrate outcomes before the tool returns.
- After receiving the result, incorporate it naturally into the story.

---

# INSTRUCTION

Generate the next narrative response.

Requirements:
- Move story toward Plot Goal.
- Maintain consistency with Memory.
- Use Retrieved Knowledge only if relevant.
- Preserve immersion.

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
    retrieval_queries: list[str]
    response: str
    dice_result: str | None
    plot_completed: bool
    scene_completed: bool
    scene_goal: str
    scene_description: str
    plot_goal: str
    mandatory_events: list[str]
    previous_plot_summary: str
    current_scene_summary: str
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
            'dice_result': None,
            'debug_prompts': [],
        }
        result = self.graph.invoke(state)
        self.latest_debug_prompts = result.get('debug_prompts', [])
        return result

    def ensure_kp_opening(self, scene_id: str, plot_id: str) -> str | None:
        self.latest_debug_prompts = []
        if not scene_id or not plot_id:
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

    def _build_tool_followup_prompt(self, state: NarrativeState) -> str:
        base_prompt = state.get('prompt', '')
        return (
            f"{base_prompt}\n\n"
            "# TOOL RESULT\n"
            f"roll_dice -> {state['dice_result']}\n\n"
            "Use the tool result above. Generate the final narrative response to the player only. "
            "Do not output TOOL_CALL."
        )

    def build_prompt(self, state: NarrativeState) -> NarrativeState:
        return state

    def retrieve_memory(self, state: NarrativeState) -> NarrativeState:
        state['conversation_history'] = self.db.get_recent_turns(state['scene_id'], state['plot_id'], limit=12)
        scene = self.db.get_scene(state['scene_id'])
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
        return state

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
        state['prompt'] = PROMPT_TEMPLATE.format(
            agent_role='Narrative Agent',
            game_system='TRPG',
            tone_style='Immersive and grounded',
            narrative_perspective='Second person',
            response_length='Concise',
            immersion_level='High',
            output_language=self._get_output_language(state),
            dice_type='{dice_type}',
            reason='{reason}',
            user_input=state['latest_user_input'],
            scene_id=state.get('scene_id', ''),
            plot_id=state.get('plot_id', ''),
            current_scene_goal=state.get('scene_goal', '') or 'None',
            current_scene_description=state.get('scene_description', '') or 'None',
            current_plot_goal=state.get('plot_goal', '') or 'None',
            mandatory_events=', '.join(state.get('mandatory_events', [])) or 'None',
            previous_plot_summary=state.get('previous_plot_summary', '') or 'None',
            current_scene_summary=state.get('current_scene_summary', '') or 'None',
            npc_related_info=categorized['npc_related_info'],
            player_related_info=str(state.get('player_profile', {})),
            location_related_info=categorized['location_related_info'],
            game_rule_info=categorized['game_rule_info'],
            world_context_info=categorized['world_context_info'],
            truth_related_info=categorized['truth_related_info'],
            item_or_clue_info=categorized['item_or_clue_info'],
            plot_progress_percentage_or_state=f"{state.get('plot_progress', 0.0):.0%}",
            scene_progress_percentage_or_state=f"{state.get('scene_progress', 0.0):.0%}",
        )
        self._record_prompt(state, 'turn_main_prompt', state['prompt'])
        return state

    def generate_response(self, state: NarrativeState) -> NarrativeState:
        try:
            first_pass = self._llm_call(state['prompt'], step_name='main_generation')
            tool_spec = self._extract_dice_tool_call(first_pass)
            if tool_spec:
                logger.info(
                    "Tool call detected step=tool_call_parse dice_type=%s reason=%s",
                    tool_spec['dice_type'],
                    tool_spec['reason'],
                )
                dice_value = self._roll_dice_expr(tool_spec['dice_type'])
                if dice_value:
                    state['dice_result'] = f"{tool_spec['dice_type']}: {dice_value} (reason: {tool_spec['reason']})"
                    logger.info("Tool execution step=roll_dice result=%s", state['dice_result'])
                    follow_prompt = (
                        "Dice Result:\n"
                        f"{state['dice_result']}\n\n"
                        f"Write the response entirely in {self._get_output_language(state)}.\n"
                        "Continue the narrative response to the player.\n"
                        "Do not output TOOL_CALL again."
                    )
                    self._record_prompt(state, 'turn_followup_tool_prompt', follow_prompt)
                    state['response'] = self._llm_call(follow_prompt, step_name='tool_followup_generation')
                else:
                    state['response'] = self._clean_tool_call_text(first_pass)
            else:
                state['response'] = first_pass.strip()
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
        done, progress = evaluate_plot_completion(
            state['latest_user_input'],
            state['response'],
            state.get('plot_progress', 0.0),
            state.get('mandatory_events', []),
            plot_goal=state.get('plot_goal', ''),
            scene_goal=state.get('scene_goal', ''),
            conversation_history=state.get('conversation_history', []),
        )
        state['plot_completed'] = done
        state['plot_progress'] = progress
        self.db.update_plot(state['plot_id'], progress=progress)
        if done:
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
        done, progress = evaluate_scene_completion(
            self.db,
            state['scene_id'],
            conversation_history=state.get('conversation_history', []),
        )
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

    def _extract_dice_tool_call(self, llm_output: str) -> dict[str, str] | None:
        if 'TOOL_CALL' not in llm_output:
            return None
        pattern = r'TOOL_CALL:\s*roll_dice\s*(\{.*?\})'
        m = re.search(pattern, llm_output, flags=re.DOTALL)
        if not m:
            return None
        try:
            payload = json5.loads(m.group(1))
        except Exception:
            return None
        dice_type = str(payload.get('dice_type', '')).strip()
        reason = str(payload.get('reason', 'skill check')).strip()
        if not dice_type:
            return None
        return {'dice_type': dice_type, 'reason': reason}

    def _clean_tool_call_text(self, text: str) -> str:
        cleaned = re.sub(r'TOOL_CALL:\s*roll_dice\s*\{.*?\}', '', text, flags=re.DOTALL)
        cleaned = cleaned.strip()
        return cleaned or 'You steady your breath as fate hangs in balance. What do you do next?'

    def _fallback_response(self, state: NarrativeState) -> str:
        output_language = self._get_output_language(state)
        user_input = state['latest_user_input']
        dice_hint = re.search(r'(\d*d\d+)', user_input.lower())
        if dice_hint:
            rolled = self._roll_dice_expr(dice_hint.group(1))
            if rolled:
                state['dice_result'] = f"{dice_hint.group(1)}: {rolled}"
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
Plot Goal: {plot.get('plot_goal', '')}
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
        prompt = f"""
Summarize the completed plot in 3 bullet points.
Focus on:
1) what happened
2) key clues
3) character changes

Scene: {state.get('scene_id', '')}
Plot: {state.get('plot_id', '')}
Plot Goal: {state.get('plot_goal', '')}
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
        return f"Plot {state.get('plot_id', '')} completed. Key progress was made toward {state.get('plot_goal', 'the plot goal')}."

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
