from __future__ import annotations

import json
import re
from typing import Any

from app.database import Database
from app.llm_client import call_nvidia_llm


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


def evaluate_plot_completion(
    user_input: str,
    response: str,
    current_progress: float,
    mandatory_events: list[str],
    *,
    plot_goal: str = '',
    scene_goal: str = '',
    conversation_history: list[dict[str, Any]] | None = None,
) -> tuple[bool, float]:
    history_tail = conversation_history[-8:] if conversation_history else []
    history_text = '\n'.join(
        [f"User: {t.get('user', '')}\nAgent: {t.get('agent', '')}" for t in history_tail]
    )
    llm_prompt = f"""
You are a narrative-state evaluator.
Determine whether the CURRENT plot is completed after this turn.

Return JSON only:
{{
  "completed": true/false,
  "progress_delta": 0.0 to 0.5
}}

Rules:
- Mark completed=true only when current plot goal is substantially achieved.
- progress_delta is how much this turn advances current plot.
- Keep output strict JSON.

Scene Goal:
{scene_goal}

Plot Goal:
{plot_goal}

Mandatory Events:
{mandatory_events}

Recent Conversation:
{history_text}

Current Turn:
User: {user_input}
Agent: {response}
"""
    try:
        llm_raw = call_nvidia_llm(llm_prompt)
        parsed = _extract_json_obj(llm_raw)
        if parsed is not None:
            completed = bool(parsed.get('completed', False))
            delta = float(parsed.get('progress_delta', 0.0))
            delta = max(0.0, min(0.5, delta))
            progress = min(1.0, max(current_progress, current_progress + delta))
            if completed:
                progress = 1.0
            return progress >= 1.0, progress
    except Exception:
        pass

    progress = current_progress
    text = f'{user_input} {response}'.lower()
    for event in mandatory_events:
        if event and event.lower()[:20] in text:
            progress += 0.3
    if any(k in text for k in ['complete', 'resolved', 'finished', 'done']):
        progress += 0.4
    progress = min(1.0, max(progress, current_progress + 0.1))
    return progress >= 1.0, progress


def evaluate_scene_completion(
    db: Database,
    scene_id: str,
    *,
    conversation_history: list[dict[str, Any]] | None = None,
) -> tuple[bool, float]:
    scene = db.get_scene(scene_id)
    if not scene:
        return False, 0.0
    plots = scene.get('plots', [])
    if not plots:
        return True, 1.0

    completed_count = len([p for p in plots if p.get('status') == 'completed'])
    ratio_progress = completed_count / len(plots)

    plot_lines = '\n'.join(
        [f"- {p.get('plot_id')}: status={p.get('status')}, progress={p.get('progress')}, goal={p.get('plot_goal', '')}" for p in plots]
    )
    history_tail = conversation_history[-8:] if conversation_history else []
    history_text = '\n'.join(
        [f"User: {t.get('user', '')}\nAgent: {t.get('agent', '')}" for t in history_tail]
    )
    llm_prompt = f"""
You are a narrative-state evaluator.
Estimate current scene progress using plot status and recent dialogue.

Return JSON only:
{{
  "scene_progress": 0.0 to 1.0
}}

Scene Goal:
{scene.get('scene_goal', '')}

Plots:
{plot_lines}

Recent Conversation:
{history_text}
"""
    try:
        llm_raw = call_nvidia_llm(llm_prompt)
        parsed = _extract_json_obj(llm_raw)
        if parsed is not None:
            llm_progress = float(parsed.get('scene_progress', ratio_progress))
            llm_progress = max(0.0, min(1.0, llm_progress))
            progress = max(ratio_progress, llm_progress)
            return progress >= 1.0, progress
    except Exception:
        pass

    return ratio_progress >= 1.0, ratio_progress


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
