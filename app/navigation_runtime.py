from __future__ import annotations

import json
from copy import deepcopy
from typing import Any


def _plot_visit_key(plot_id: str, visit_id: int) -> str:
    return f"{plot_id}::{int(visit_id)}"


def _current_plot_guidance(nav_state: dict[str, Any], plot_id: str, visit_id: int) -> dict[str, Any]:
    if not plot_id:
        return {"redirect_streak": 0, "last_alignment": "", "last_handoff_candidate": None}
    entry = nav_state.get("plot_guidance", {}).get(_plot_visit_key(plot_id, visit_id), {})
    if not isinstance(entry, dict):
        entry = {}
    return {
        "redirect_streak": max(0, int(entry.get("redirect_streak", 0) or 0)),
        "last_alignment": str(entry.get("last_alignment", "")).strip(),
        "last_handoff_candidate": deepcopy(entry.get("last_handoff_candidate")),
    }


def _normalize_navigation_state(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            value = {}
    if not isinstance(value, dict):
        value = {}

    visited_scene_ids = value.get("visited_scene_ids", [])
    visited_plot_ids = value.get("visited_plot_ids", [])
    return_stack = value.get("return_stack", [])
    hub_progress = value.get("hub_progress", {})
    plot_guidance = value.get("plot_guidance", {})
    latest_transition = value.get("latest_transition", {})

    if not isinstance(visited_scene_ids, list):
        visited_scene_ids = []
    if not isinstance(visited_plot_ids, list):
        visited_plot_ids = []
    if not isinstance(return_stack, list):
        return_stack = []
    if not isinstance(hub_progress, dict):
        hub_progress = {}
    if not isinstance(plot_guidance, dict):
        plot_guidance = {}
    if not isinstance(latest_transition, dict):
        latest_transition = {}

    normalized_stack: list[dict[str, Any]] = []
    for frame in return_stack:
        if not isinstance(frame, dict):
            continue
        target_kind = str(frame.get("target_kind", "")).strip().lower()
        target_id = str(frame.get("target_id", "")).strip()
        if not target_kind or not target_id:
            continue
        normalized_stack.append(
            {
                "target_kind": target_kind,
                "target_id": target_id,
                "visit_id": int(frame.get("visit_id", 0) or 0),
            }
        )

    normalized_plot_guidance: dict[str, dict[str, Any]] = {}
    for key, entry in plot_guidance.items():
        if not isinstance(entry, dict):
            continue
        candidate = entry.get("last_handoff_candidate")
        normalized_candidate: dict[str, Any] | None = None
        if isinstance(candidate, dict):
            target_kind = str(candidate.get("target_kind", "")).strip().lower()
            target_id = str(candidate.get("target_id", "")).strip()
            if target_kind and target_id:
                normalized_candidate = {"target_kind": target_kind, "target_id": target_id}
        normalized_plot_guidance[str(key)] = {
            "redirect_streak": max(0, int(entry.get("redirect_streak", 0) or 0)),
            "last_alignment": str(entry.get("last_alignment", "")).strip(),
            "last_handoff_candidate": normalized_candidate,
        }

    normalized_transition: dict[str, Any] = {}
    for field in (
        "source_scene_id",
        "source_plot_id",
        "source_scene_goal",
        "source_plot_goal",
        "target_scene_id",
        "target_plot_id",
        "target_scene_goal",
        "target_plot_goal",
        "transition_reason",
        "latest_user_input",
        "latest_agent_response",
        "relationship",
    ):
        normalized_transition[field] = str(latest_transition.get(field, "")).strip()

    return {
        "visited_scene_ids": [str(item).strip() for item in visited_scene_ids if str(item).strip()],
        "visited_plot_ids": [str(item).strip() for item in visited_plot_ids if str(item).strip()],
        "hub_progress": deepcopy(hub_progress),
        "return_stack": normalized_stack,
        "plot_guidance": normalized_plot_guidance,
        "latest_transition": normalized_transition,
    }


def update_plot_guidance_state(
    navigation_state: dict[str, Any] | None,
    *,
    plot_id: str,
    visit_id: int,
    alignment: str,
    handoff_candidate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    nav_state = _normalize_navigation_state(navigation_state or {})
    if not plot_id:
        return nav_state
    plot_guidance = nav_state.setdefault("plot_guidance", {})
    key = _plot_visit_key(plot_id, visit_id)
    current = _current_plot_guidance(nav_state, plot_id, visit_id)
    redirect_streak = int(current.get("redirect_streak", 0) or 0)
    if alignment == "off_topic":
        redirect_streak += 1
    else:
        redirect_streak = 0

    candidate_payload = None
    if handoff_candidate:
        target_kind = str(handoff_candidate.get("target_kind", "")).strip().lower()
        target_id = str(handoff_candidate.get("target_id", "")).strip()
        if target_kind and target_id:
            candidate_payload = {"target_kind": target_kind, "target_id": target_id}

    plot_guidance[key] = {
        "redirect_streak": redirect_streak,
        "last_alignment": str(alignment or "").strip(),
        "last_handoff_candidate": candidate_payload,
    }
    return nav_state


def set_latest_transition_context(
    navigation_state: dict[str, Any] | None,
    *,
    source_scene_id: str,
    source_plot_id: str,
    source_scene_goal: str,
    source_plot_goal: str,
    target_scene_id: str,
    target_plot_id: str,
    target_scene_goal: str,
    target_plot_goal: str,
    transition_reason: str,
    latest_user_input: str,
    latest_agent_response: str,
    relationship: str,
) -> dict[str, Any]:
    nav_state = _normalize_navigation_state(navigation_state or {})
    nav_state["latest_transition"] = {
        "source_scene_id": str(source_scene_id or "").strip(),
        "source_plot_id": str(source_plot_id or "").strip(),
        "source_scene_goal": str(source_scene_goal or "").strip(),
        "source_plot_goal": str(source_plot_goal or "").strip(),
        "target_scene_id": str(target_scene_id or "").strip(),
        "target_plot_id": str(target_plot_id or "").strip(),
        "target_scene_goal": str(target_scene_goal or "").strip(),
        "target_plot_goal": str(target_plot_goal or "").strip(),
        "transition_reason": str(transition_reason or "").strip(),
        "latest_user_input": str(latest_user_input or "").strip(),
        "latest_agent_response": str(latest_agent_response or "").strip(),
        "relationship": str(relationship or "").strip(),
    }
    return nav_state


def _push_return_frame(nav_state: dict[str, Any], target_kind: str, target_id: str, visit_id: int) -> None:
    if not target_kind or not target_id:
        return
    nav_state.setdefault("return_stack", []).append(
        {
            "target_kind": target_kind,
            "target_id": target_id,
            "visit_id": int(visit_id),
        }
    )


def _pop_return_frame(nav_state: dict[str, Any], target_kind: str, target_id: str) -> None:
    stack = nav_state.get("return_stack", [])
    for idx in range(len(stack) - 1, -1, -1):
        frame = stack[idx]
        if frame.get("target_kind") == target_kind and frame.get("target_id") == target_id:
            del stack[idx]
            break
