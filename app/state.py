from __future__ import annotations

from typing import Any

from app.database import Database


def evaluate_plot_completion(user_input: str, response: str, current_progress: float, mandatory_events: list[str]) -> tuple[bool, float]:
    progress = current_progress
    text = f'{user_input} {response}'.lower()
    for event in mandatory_events:
        if event and event.lower()[:20] in text:
            progress += 0.3
    if any(k in text for k in ['complete', 'resolved', 'finished', 'done']):
        progress += 0.4
    progress = min(1.0, max(progress, current_progress + 0.1))
    return progress >= 1.0, progress


def evaluate_scene_completion(db: Database, scene_id: str) -> tuple[bool, float]:
    scene = db.get_scene(scene_id)
    if not scene:
        return False, 0.0
    plots = scene.get('plots', [])
    if not plots:
        return True, 1.0
    completed = len([p for p in plots if p.get('status') == 'completed'])
    progress = completed / len(plots)
    return progress >= 1.0, progress


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
