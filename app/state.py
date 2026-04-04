from __future__ import annotations

import json
import re
from typing import Any, Callable

from app.database import Database
from app.llm_client import call_nvidia_llm

STOPWORDS = {
    'about',
    'after',
    'again',
    'around',
    'begin',
    'being',
    'current',
    'details',
    'during',
    'from',
    'guide',
    'have',
    'into',
    'investigator',
    'learn',
    'more',
    'next',
    'plot',
    'scene',
    'some',
    'that',
    'their',
    'them',
    'then',
    'they',
    'this',
    'through',
    'with',
}

SETUP_MARKERS = (
    'set up',
    'setup',
    'prepare',
    'introduce',
    'initial contact',
    'begin the investigation',
    'ready to',
    'next steps',
    'guide the player',
    'choose',
    'choice',
    'option',
    'options',
    'can choose',
    'can ask',
    'presumably',
    'start',
)


def _extract_json_obj(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    text = text.strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except Exception:
        pass
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _normalize_text(text: str) -> str:
    return re.sub(r'\s+', ' ', (text or '').strip().lower())


def _truncate_text(text: str, limit: int = 500) -> str:
    text = (text or '').strip()
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3].rstrip()}..."


def _goal_keywords(text: str) -> list[str]:
    raw_tokens = re.findall(r'[a-z]{3,}|[\u4e00-\u9fff]{2,}', _normalize_text(text))
    keywords: list[str] = []
    for token in raw_tokens:
        if token in STOPWORDS:
            continue
        if token not in keywords:
            keywords.append(token)
    return keywords


def _goal_match_score(text: str, goal: str) -> float:
    norm_text = _normalize_text(text)
    norm_goal = _normalize_text(goal)
    if not norm_text or not norm_goal:
        return 0.0

    if len(norm_goal) >= 12 and norm_goal in norm_text:
        return 1.0

    keywords = _goal_keywords(goal)
    if not keywords:
        return 0.0

    hits = 0
    for token in keywords:
        if token in norm_text:
            hits += 1
    return hits / max(1, len(keywords))


def _is_setup_or_choice_like(plot_goal: str, current_plot_raw_text: str, scene_goal: str = '') -> bool:
    combined = _normalize_text(' '.join([plot_goal, scene_goal, _truncate_text(current_plot_raw_text, 320)]))
    return any(marker in combined for marker in SETUP_MARKERS)


def _infer_advance_target(
    combined_turn_text: str,
    next_plot_goal: str,
    next_scene_goal: str,
) -> tuple[str, float]:
    next_plot_score = _goal_match_score(combined_turn_text, next_plot_goal)
    next_scene_score = _goal_match_score(combined_turn_text, next_scene_goal)

    if next_scene_score >= 0.55 and next_scene_score > next_plot_score + 0.1:
        return 'next_scene', next_scene_score
    if next_plot_score >= 0.55:
        return 'next_plot', next_plot_score
    return 'stay', 0.0


def _count_event_hits(text: str, mandatory_events: list[str]) -> int:
    hits = 0
    norm_text = _normalize_text(text)
    for event in mandatory_events:
        event_norm = _normalize_text(event)
        if not event_norm:
            continue
        event_keywords = _goal_keywords(event_norm)
        if event_norm[:24] in norm_text:
            hits += 1
            continue
        if event_keywords and sum(1 for token in event_keywords if token in norm_text) >= max(1, len(event_keywords) // 2):
            hits += 1
    return hits


def _scene_is_setup_or_choice_like(scene_goal: str, plots: list[dict[str, Any]]) -> bool:
    combined = _normalize_text(scene_goal)
    if any(marker in combined for marker in SETUP_MARKERS):
        return True
    plot_goals = ' '.join(str(plot.get('plot_goal', '')) for plot in plots[:3])
    return any(marker in _normalize_text(plot_goals) for marker in SETUP_MARKERS)


def story_position_context(db: Database, current_scene_id: str, current_plot_id: str) -> dict[str, str]:
    current_plot = db.get_plot(current_plot_id) or {}
    current_scene = db.get_scene(current_scene_id) or {}
    scenes = db.list_scenes()

    next_plot_goal = ''
    next_plot_excerpt = ''
    plots = current_scene.get('plots', [])
    current_idx = next((i for i, plot in enumerate(plots) if plot.get('plot_id') == current_plot_id), -1)
    if current_idx >= 0:
        for plot in plots[current_idx + 1 :]:
            if plot.get('status') != 'completed':
                next_plot_goal = str(plot.get('plot_goal', ''))
                next_plot_excerpt = _truncate_text(str(plot.get('raw_text', '')), 500)
                break

    next_scene_goal = ''
    next_scene_plot_goal = ''
    next_scene_plot_excerpt = ''
    scene_idx = next((i for i, scene in enumerate(scenes) if scene.get('scene_id') == current_scene_id), -1)
    if scene_idx >= 0:
        for scene in scenes[scene_idx + 1 :]:
            if scene.get('status') != 'completed':
                next_scene_goal = str(scene.get('scene_goal', ''))
                next_scene_plots = scene.get('plots', [])
                next_scene_first_plot = next((plot for plot in next_scene_plots if plot.get('status') != 'completed'), next_scene_plots[0] if next_scene_plots else {})
                next_scene_plot_goal = str(next_scene_first_plot.get('plot_goal', ''))
                next_scene_plot_excerpt = _truncate_text(str(next_scene_first_plot.get('raw_text', '')), 500)
                break

    return {
        'current_plot_raw_text': _truncate_text(str(current_plot.get('raw_text', '')), 600),
        'next_plot_goal': next_plot_goal,
        'next_plot_excerpt': next_plot_excerpt,
        'next_scene_goal': next_scene_goal,
        'next_scene_plot_goal': next_scene_plot_goal,
        'next_scene_plot_excerpt': next_scene_plot_excerpt,
    }


def evaluate_plot_completion(
    user_input: str,
    response: str,
    current_progress: float,
    mandatory_events: list[str],
    *,
    plot_goal: str = '',
    scene_goal: str = '',
    next_plot_goal: str = '',
    next_scene_goal: str = '',
    current_plot_raw_text: str = '',
    conversation_history: list[dict[str, Any]] | None = None,
    prompt_recorder: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    history_tail = conversation_history[-8:] if conversation_history else []
    history_text = '\n'.join(
        [f"User: {t.get('user', '')}\nAgent: {t.get('agent', '')}" for t in history_tail]
    )
    current_plot_excerpt = _truncate_text(current_plot_raw_text, 500) or 'None'
    llm_prompt = f"""
You are a narrative-state evaluator for a script-driven investigation game.
Decide whether the CURRENT plot should stay active, complete and hand off to the next plot, or complete and move to the next scene.

Return JSON only:
{{
  "completed": true/false,
  "progress_delta": 0.0 to 0.6,
  "advance_target": "stay" | "next_plot" | "next_scene",
  "reason": "brief explanation"
}}

Rules:
- Use medium-aggressive progression.
- Mark completed=true when the current plot has fulfilled its narrative function, even if every detail was not exhaustively played out.
- `advance_target="next_plot"` means the player clearly shifted into the next unresolved plot inside the same scene.
- `advance_target="next_scene"` means the player clearly shifted into material that better matches the next unresolved scene.
- Mandatory events are strong evidence, but not hard gates.
- Setup / intro / choice-hub plots can complete once the player clearly commits to a downstream action.
- Do not advance just because the player is still digging deeper into the current plot.
- Keep output strict JSON.

Scene Goal:
{scene_goal or 'None'}

Current Plot Goal:
{plot_goal or 'None'}

Current Plot Excerpt:
{current_plot_excerpt}

Next Plot Goal:
{next_plot_goal or 'None'}

Next Scene Goal:
{next_scene_goal or 'None'}

Mandatory Events:
{mandatory_events}

Recent Conversation:
{history_text or 'None'}

Latest Turn:
User: {user_input}
Agent: {response}
"""
    if prompt_recorder:
        prompt_recorder(llm_prompt)

    try:
        llm_raw = call_nvidia_llm(llm_prompt, step_name='plot_completion_evaluation')
        parsed = _extract_json_obj(llm_raw)
        if parsed is not None:
            advance_target = str(parsed.get('advance_target', 'stay')).strip().lower()
            if advance_target not in {'stay', 'next_plot', 'next_scene'}:
                advance_target = 'stay'
            completed = bool(parsed.get('completed', False)) or advance_target != 'stay'
            delta = float(parsed.get('progress_delta', 0.0))
            delta = max(0.0, min(0.6, delta))
            reason = str(parsed.get('reason', '')).strip() or 'llm_evaluation'
            progress = 1.0 if completed else min(1.0, max(current_progress, current_progress + delta))
            return {
                'completed': completed,
                'progress': progress,
                'advance_target': advance_target,
                'reason': reason,
            }
    except Exception:
        pass

    combined_turn_text = _normalize_text(f'{user_input} {response}')
    setup_like = _is_setup_or_choice_like(plot_goal, current_plot_raw_text, scene_goal)
    advance_target, advance_score = _infer_advance_target(combined_turn_text, next_plot_goal, next_scene_goal)
    event_hits = _count_event_hits(combined_turn_text, mandatory_events)
    completion_cues = any(k in combined_turn_text for k in ['complete', 'resolved', 'finished', 'done', 'decide', 'choose'])

    progress = current_progress
    if event_hits:
        progress += min(0.45, 0.2 * event_hits)
    if completion_cues:
        progress += 0.2
    progress = min(1.0, max(progress, current_progress + 0.1))

    if advance_target != 'stay' and (setup_like or current_progress >= 0.45 or event_hits > 0 or not mandatory_events):
        return {
            'completed': True,
            'progress': 1.0,
            'advance_target': advance_target,
            'reason': f'fallback_shift_detected(score={advance_score:.2f})',
        }

    if progress >= 1.0:
        return {
            'completed': True,
            'progress': 1.0,
            'advance_target': 'stay',
            'reason': 'fallback_progress_threshold',
        }

    return {
        'completed': False,
        'progress': progress,
        'advance_target': 'stay',
        'reason': 'fallback_continue_current_plot',
    }


def evaluate_pre_response_transition(
    user_input: str,
    *,
    plot_goal: str = '',
    scene_goal: str = '',
    next_plot_goal: str = '',
    next_scene_goal: str = '',
    current_plot_raw_text: str = '',
    conversation_history: list[dict[str, Any]] | None = None,
    prompt_recorder: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    user_input = (user_input or '').strip()
    if not user_input:
        return {'advance_target': 'stay', 'reason': 'empty_player_input'}

    history_tail = conversation_history[-8:] if conversation_history else []
    history_text = '\n'.join(
        [f"User: {t.get('user', '')}\nAgent: {t.get('agent', '')}" for t in history_tail]
    )
    current_plot_excerpt = _truncate_text(current_plot_raw_text, 500) or 'None'
    llm_prompt = f"""
You are a narrative-state evaluator for a script-driven investigation game.
Before the Keeper responds, decide whether the player's latest action clearly moves play into the next plot or next scene.

Return JSON only:
{{
  "advance_target": "stay" | "next_plot" | "next_scene",
  "reason": "brief explanation"
}}

Rules:
- Use medium-aggressive progression.
- Setup / intro / choice-hub plots can hand off immediately when the player clearly commits to a downstream action.
- Only choose `next_plot` or `next_scene` if the player action fits that target better than the current plot.
- If the player is still engaging the current plot, return `stay`.
- Do not require every mandatory event to be exhausted before handing off from a setup plot.
- Keep output strict JSON.

Scene Goal:
{scene_goal or 'None'}

Current Plot Goal:
{plot_goal or 'None'}

Current Plot Excerpt:
{current_plot_excerpt}

Next Plot Goal:
{next_plot_goal or 'None'}

Next Scene Goal:
{next_scene_goal or 'None'}

Recent Conversation:
{history_text or 'None'}

Latest Player Input:
{user_input}
"""
    if prompt_recorder:
        prompt_recorder(llm_prompt)

    try:
        llm_raw = call_nvidia_llm(llm_prompt, step_name='pre_response_transition_evaluation')
        parsed = _extract_json_obj(llm_raw)
        if parsed is not None:
            advance_target = str(parsed.get('advance_target', 'stay')).strip().lower()
            if advance_target not in {'stay', 'next_plot', 'next_scene'}:
                advance_target = 'stay'
            reason = str(parsed.get('reason', '')).strip() or 'llm_evaluation'
            return {'advance_target': advance_target, 'reason': reason}
    except Exception:
        pass

    combined_text = _normalize_text(user_input)
    setup_like = _is_setup_or_choice_like(plot_goal, current_plot_raw_text, scene_goal)
    advance_target, advance_score = _infer_advance_target(combined_text, next_plot_goal, next_scene_goal)
    if advance_target != 'stay' and setup_like:
        return {
            'advance_target': advance_target,
            'reason': f'fallback_shift_detected(score={advance_score:.2f})',
        }
    return {'advance_target': 'stay', 'reason': 'fallback_continue_current_plot'}


def evaluate_scene_completion(
    db: Database,
    scene_id: str,
    *,
    conversation_history: list[dict[str, Any]] | None = None,
    latest_turn_text: str = '',
    next_scene_goal: str = '',
    plot_advance_target: str = 'stay',
    plot_advance_reason: str = '',
    prompt_recorder: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    scene = db.get_scene(scene_id)
    if not scene:
        return {'completed': False, 'progress': 0.0, 'reason': 'scene_missing'}

    plots = scene.get('plots', [])
    if not plots:
        return {'completed': True, 'progress': 1.0, 'reason': 'scene_without_plots'}

    completed_count = len([p for p in plots if p.get('status') == 'completed'])
    ratio_progress = completed_count / len(plots)

    plot_lines = '\n'.join(
        [
            f"- {p.get('plot_id')}: status={p.get('status')}, progress={p.get('progress')}, goal={p.get('plot_goal', '')}"
            for p in plots
        ]
    )
    history_tail = conversation_history[-8:] if conversation_history else []
    history_text = '\n'.join(
        [f"User: {t.get('user', '')}\nAgent: {t.get('agent', '')}" for t in history_tail]
    )
    llm_prompt = f"""
You are a narrative-state evaluator for a script-driven investigation game.
Estimate whether the CURRENT scene is complete after the latest turn.

Return JSON only:
{{
  "completed": true/false,
  "scene_progress": 0.0 to 1.0,
  "reason": "brief explanation"
}}

Rules:
- Use medium-aggressive progression.
- A scene can complete when its current dramatic purpose is fulfilled, even if not every optional sub-detail was explored.
- If the plot evaluator already decided to hand off to the next scene, treat that as strong evidence.
- Setup / intro / choice-hub scenes can complete once the player clearly commits to a downstream scene.
- Do not complete the scene if the player is still working within unresolved plots of the current scene.
- Keep output strict JSON.

Current Scene Goal:
{scene.get('scene_goal', '') or 'None'}

Plots:
{plot_lines}

Next Scene Goal:
{next_scene_goal or 'None'}

Recent Conversation:
{history_text or 'None'}

Latest Turn:
{latest_turn_text or 'None'}

Plot Evaluation Handoff:
target={plot_advance_target}
reason={plot_advance_reason or 'None'}
"""
    if prompt_recorder:
        prompt_recorder(llm_prompt)

    try:
        llm_raw = call_nvidia_llm(llm_prompt, step_name='scene_completion_evaluation')
        parsed = _extract_json_obj(llm_raw)
        if parsed is not None:
            completed = bool(parsed.get('completed', False))
            llm_progress = float(parsed.get('scene_progress', ratio_progress))
            llm_progress = max(0.0, min(1.0, llm_progress))
            progress = max(ratio_progress, llm_progress)
            if completed:
                progress = 1.0
            reason = str(parsed.get('reason', '')).strip() or 'llm_evaluation'
            return {
                'completed': completed or progress >= 1.0,
                'progress': progress,
                'reason': reason,
            }
    except Exception:
        pass

    setup_like = _scene_is_setup_or_choice_like(str(scene.get('scene_goal', '')), plots)
    if ratio_progress >= 1.0:
        return {'completed': True, 'progress': 1.0, 'reason': 'fallback_all_plots_completed'}
    if plot_advance_target == 'next_scene' and (len(plots) <= 1 or setup_like):
        return {'completed': True, 'progress': 1.0, 'reason': 'fallback_plot_handoff_to_next_scene'}
    return {'completed': False, 'progress': ratio_progress, 'reason': 'fallback_continue_current_scene'}


def next_story_position(db: Database, current_scene_id: str, current_plot_id: str) -> dict[str, Any]:
    scenes = db.list_scenes()
    current_scene = next((s for s in scenes if s['scene_id'] == current_scene_id), None)

    if current_scene:
        next_plot = next((p for p in current_scene.get('plots', []) if p.get('status') != 'completed'), None)
        if next_plot:
            return {
                'current_scene_id': current_scene_id,
                'current_plot_id': next_plot['plot_id'],
                'plot_progress': float(next_plot.get('progress', 0.0)),
                'scene_progress': 0.0,
                'current_scene_intro': '',
            }

    next_scene = next((s for s in scenes if s.get('status') != 'completed'), None)
    if not next_scene:
        return {
            'current_scene_id': current_scene_id,
            'current_plot_id': current_plot_id,
            'plot_progress': 1.0,
            'scene_progress': 1.0,
            'current_scene_intro': 'All scenes are complete.',
        }

    first_plot = next_scene.get('plots', [{}])[0]
    return {
        'current_scene_id': next_scene['scene_id'],
        'current_plot_id': first_plot.get('plot_id', ''),
        'plot_progress': float(first_plot.get('progress', 0.0)),
        'scene_progress': 0.0,
        'current_scene_intro': f"Scene {next_scene['scene_id']} begins: {next_scene.get('scene_goal', '')}",
    }
