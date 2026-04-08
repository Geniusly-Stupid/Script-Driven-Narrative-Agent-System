from __future__ import annotations

import json
import re
from collections import deque
from copy import deepcopy
from typing import Any, Callable

try:
    import json5  # type: ignore[import-not-found]
except ImportError:
    json5 = None

from app.database import Database
from app.llm_client import call_nvidia_llm
from app.navigation import ensure_scene_navigation_defaults, normalize_navigation, normalize_target
from app.navigation_runtime import (
    _current_plot_guidance,
    _normalize_navigation_state,
    _pop_return_frame,
    _push_return_frame,
    set_latest_transition_context,
    update_plot_guidance_state,
)

PROGRESSION_MODEL = "qwen/qwen3.5-397b-a17b"
INTENT_ALIGNMENTS = {"target_choice", "current_plot", "off_topic"}
TURN_STATE_BEAT_STATUSES = {"open", "wrapped", "exhausted"}

def _extract_json_obj(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    text = text.strip()
    candidates: list[str] = [text]
    fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    for block in fenced_blocks:
        stripped = block.strip()
        if stripped:
            candidates.insert(0, stripped)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        candidates.append(match.group(0))

    loaders = [json.loads]
    if json5 is not None:
        loaders.append(json5.loads)

    for candidate in candidates:
        for loader in loaders:
            try:
                data = loader(candidate)
                if isinstance(data, dict):
                    return data
            except Exception:
                continue
    return None


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _truncate_text(text: str, limit: int = 500) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3].rstrip()}..."


def _normalize_transition_path(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"direct", "via_return", "stay"}:
        return normalized
    return "stay"


def _normalize_alignment(value: Any, *, default: str = "current_plot") -> str:
    alignment = str(value or "").strip().lower()
    if alignment in INTENT_ALIGNMENTS:
        return alignment
    return default


def _normalize_beat_status(value: Any) -> str:
    status = str(value or "").strip().lower()
    if status in TURN_STATE_BEAT_STATUSES:
        return status
    return "open"


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value or "").strip().lower()
    return normalized in {"1", "true", "yes", "y"}


def _story_graph(db: Database) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    scenes = db.list_scenes()
    ensure_scene_navigation_defaults(scenes)
    scene_map: dict[str, dict[str, Any]] = {}
    plot_map: dict[str, dict[str, Any]] = {}
    for scene in scenes:
        scene["navigation"] = normalize_navigation(scene.get("navigation"))
        scene_map[str(scene.get("scene_id", ""))] = scene
        for plot in scene.get("plots", []):
            plot["navigation"] = normalize_navigation(plot.get("navigation"))
            plot["scene_id"] = str(scene.get("scene_id", ""))
            plot_map[str(plot.get("plot_id", ""))] = plot
    return scenes, scene_map, plot_map


def _first_active_plot(scene: dict[str, Any]) -> dict[str, Any]:
    plots = scene.get("plots", [])
    if not isinstance(plots, list) or not plots:
        return {}
    return next(
        (plot for plot in plots if str(plot.get("status", "pending")) not in {"completed", "skipped"}),
        plots[0],
    )


def _is_target_resolved(target: dict[str, Any], scene_map: dict[str, dict[str, Any]], plot_map: dict[str, dict[str, Any]]) -> bool:
    target_kind = target.get("target_kind")
    target_id = target.get("target_id")
    if target_kind == "scene":
        scene = scene_map.get(target_id, {})
        return str(scene.get("status", "")) in {"completed", "skipped"}
    if target_kind == "plot":
        plot = plot_map.get(target_id, {})
        return str(plot.get("status", "")) in {"completed", "skipped"}
    return False


def _prerequisites_met(target: dict[str, Any], nav_state: dict[str, Any], scene_map: dict[str, dict[str, Any]], plot_map: dict[str, dict[str, Any]]) -> bool:
    required_ids = [str(item).strip() for item in target.get("prerequisites", []) if str(item).strip()]
    if not required_ids:
        return True

    visited_scene_ids = set(nav_state.get("visited_scene_ids", []))
    visited_plot_ids = set(nav_state.get("visited_plot_ids", []))

    for required_id in required_ids:
        if required_id in scene_map:
            scene = scene_map[required_id]
            if str(scene.get("status", "")) not in {"completed", "skipped"} and required_id not in visited_scene_ids:
                return False
        elif required_id in plot_map:
            plot = plot_map[required_id]
            if str(plot.get("status", "")) not in {"completed", "skipped"} and required_id not in visited_plot_ids:
                return False
        else:
            return False
    return True


def _scene_excerpt(scene: dict[str, Any]) -> str:
    plots = scene.get("plots", [])
    if not isinstance(plots, list):
        return ""
    return "\n".join(str(plot.get("raw_text", "")).strip() for plot in plots if str(plot.get("raw_text", "")).strip())


def _hydrate_target(
    target: dict[str, Any],
    nav_state: dict[str, Any],
    scene_map: dict[str, dict[str, Any]],
    plot_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    hydrated = normalize_target(target)
    target_kind = hydrated.get("target_kind")
    target_id = hydrated.get("target_id")
    scene_id = ""
    plot_id = ""
    status = ""

    if target_kind == "scene":
        scene = scene_map.get(target_id, {})
        scene_id = target_id
        plot = _first_active_plot(scene)
        plot_id = str(plot.get("plot_id", ""))
        status = str(scene.get("status", ""))
        hydrated["goal"] = hydrated.get("goal") or str(scene.get("scene_goal", ""))
        hydrated["excerpt"] = hydrated.get("excerpt") or _truncate_text(_scene_excerpt(scene), 500)
        hydrated["label"] = hydrated.get("label") or str(scene.get("scene_goal", ""))
    elif target_kind == "plot":
        plot = plot_map.get(target_id, {})
        plot_id = target_id
        scene_id = str(plot.get("scene_id", ""))
        status = str(plot.get("status", ""))
        hydrated["goal"] = hydrated.get("goal") or str(plot.get("plot_goal", ""))
        hydrated["excerpt"] = hydrated.get("excerpt") or _truncate_text(str(plot.get("raw_text", "")), 500)
        hydrated["label"] = hydrated.get("label") or str(plot.get("plot_goal", ""))

    hydrated["scene_id"] = scene_id
    hydrated["plot_id"] = plot_id
    hydrated["status"] = status
    hydrated["eligible"] = _prerequisites_met(hydrated, nav_state, scene_map, plot_map) and status not in {"completed", "skipped"}
    return hydrated


def _allowed_targets_for_node(
    node: dict[str, Any],
    nav_state: dict[str, Any],
    scene_map: dict[str, dict[str, Any]],
    plot_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    navigation = normalize_navigation(node.get("navigation"))
    return [_hydrate_target(target, nav_state, scene_map, plot_map) for target in navigation.get("allowed_targets", [])]


def _remaining_required_targets(
    node: dict[str, Any],
    nav_state: dict[str, Any],
    scene_map: dict[str, dict[str, Any]],
    plot_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    required_targets: list[dict[str, Any]] = []
    for target in _allowed_targets_for_node(node, nav_state, scene_map, plot_map):
        if not target.get("required"):
            continue
        if _is_target_resolved(target, scene_map, plot_map):
            continue
        required_targets.append(target)
    return required_targets


def _target_lines(targets: list[dict[str, Any]]) -> str:
    if not targets:
        return "None"
    lines = []
    for target in targets:
        excerpt = _truncate_text(str(target.get("excerpt", "")), 160) or "None"
        lines.append(
            f"- {target.get('target_kind')}:{target.get('target_id')} "
            f"label={target.get('label') or 'None'} "
            f"goal={target.get('goal') or 'None'} "
            f"role={target.get('role') or 'none'} "
            f"required={str(bool(target.get('required', False))).lower()} "
            f"eligible={str(bool(target.get('eligible', False))).lower()} "
            f"excerpt={excerpt}"
        )
    return "\n".join(lines)


def _eligible_targets(targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [target for target in targets if target.get("eligible")]


def _exhausted_targets(targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [target for target in targets if str(target.get("status", "")) in {"completed", "skipped"}]


def _blocked_targets(targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        target
        for target in targets
        if not target.get("eligible") and str(target.get("status", "")) not in {"completed", "skipped"}
    ]


def _first_eligible_target_by_role(targets: list[dict[str, Any]], role: str) -> dict[str, Any] | None:
    return next((target for target in targets if target.get("role") == role and target.get("eligible")), None)


def _first_target_of_kind(targets: list[dict[str, Any]], target_kind: str) -> dict[str, Any] | None:
    return next((target for target in targets if target.get("target_kind") == target_kind and target.get("eligible")), None)


def _node_progress(
    node: dict[str, Any],
    nav_state: dict[str, Any],
    scene_map: dict[str, dict[str, Any]],
    plot_map: dict[str, dict[str, Any]],
    fallback_progress: float = 0.0,
) -> float:
    node_kind = str(node.get("node_kind", "linear"))
    navigation = normalize_navigation(node.get("navigation"))
    node_id = str(node.get("plot_id") or node.get("scene_id") or "")
    policy = navigation.get("completion_policy", "terminal_on_resolve")
    allowed_targets = _allowed_targets_for_node(node, nav_state, scene_map, plot_map)
    branch_targets = [target for target in allowed_targets if target.get("role") == "branch"]

    if node_kind == "hub":
        if policy == "all_required_then_advance":
            required_targets = [target for target in allowed_targets if target.get("required")]
            total = len(required_targets)
            progress = fallback_progress if total == 0 else sum(1 for target in required_targets if _is_target_resolved(target, scene_map, plot_map)) / total
        elif policy == "exclusive_choice":
            progress = 1.0 if any(_is_target_resolved(target, scene_map, plot_map) for target in branch_targets) else fallback_progress
        else:
            total = len(branch_targets)
            progress = fallback_progress if total == 0 else min(0.9, sum(1 for target in branch_targets if _is_target_resolved(target, scene_map, plot_map)) / total)
        nav_state.setdefault("hub_progress", {})[node_id] = {
            "progress": round(float(progress), 4),
            "completion_policy": policy,
        }
        return float(progress)

    if str(node.get("status", "")) in {"completed", "skipped"}:
        return 1.0
    return float(fallback_progress)


def _legal_targets(
    allowed_targets: list[dict[str, Any]] | None,
    indirect_targets_via_return: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    legal_targets: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for transition_path, targets in (
        ("direct", _eligible_targets(allowed_targets or [])),
        ("via_return", _eligible_targets(indirect_targets_via_return or [])),
    ):
        for target in targets:
            target_key = (
                str(target.get("target_kind", "")).strip().lower(),
                str(target.get("target_id", "")).strip(),
            )
            if not target_key[0] or not target_key[1] or target_key in seen:
                continue
            seen.add(target_key)
            normalized_target = deepcopy(target)
            normalized_target["transition_path"] = transition_path
            legal_targets.append(normalized_target)
    return legal_targets


def _default_turn_state(
    *,
    choice_open: bool = False,
    offered_targets: list[dict[str, Any]] | None = None,
    beat_status: str = "open",
    summary: str = "",
) -> dict[str, Any]:
    return {
        "choice_open": bool(choice_open),
        "offered_targets": deepcopy(offered_targets or []),
        "beat_status": _normalize_beat_status(beat_status),
        "summary": str(summary or "").strip(),
    }


def _normalize_offered_targets(
    offered_targets: Any,
    legal_targets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(offered_targets, list):
        return []
    normalized_targets: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in offered_targets:
        if not isinstance(item, dict):
            continue
        validated_target, transition_path = _validate_selected_target(item, legal_targets, [])
        if validated_target is None:
            continue
        target_key = (
            str(validated_target.get("target_kind", "")).strip().lower(),
            str(validated_target.get("target_id", "")).strip(),
        )
        if target_key in seen:
            continue
        seen.add(target_key)
        normalized_target = deepcopy(validated_target)
        normalized_target["transition_path"] = transition_path
        normalized_targets.append(normalized_target)
    return normalized_targets


def _normalize_turn_state(
    value: Any,
    *,
    legal_targets: list[dict[str, Any]] | None = None,
    default_choice_open: bool = False,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        value = {}
    legal_targets = deepcopy(legal_targets or [])
    offered_targets = _normalize_offered_targets(value.get("offered_targets"), legal_targets)
    choice_open = _coerce_bool(value.get("choice_open", default_choice_open))
    if not offered_targets:
        choice_open = bool(choice_open and legal_targets)
    return _default_turn_state(
        choice_open=choice_open,
        offered_targets=offered_targets,
        beat_status=value.get("beat_status", "open"),
        summary=value.get("summary", ""),
    )


def _turn_state_summary(turn_state: dict[str, Any] | None) -> str:
    if not isinstance(turn_state, dict):
        turn_state = _default_turn_state()
    normalized_turn_state = {
        "choice_open": bool(turn_state.get("choice_open", False)),
        "offered_targets": deepcopy(turn_state.get("offered_targets", []))
        if isinstance(turn_state.get("offered_targets", []), list)
        else [],
        "beat_status": _normalize_beat_status(turn_state.get("beat_status", "open")),
        "summary": str(turn_state.get("summary", "") or "").strip(),
    }
    return (
        f"choice_open={str(bool(normalized_turn_state.get('choice_open', False))).lower()}; "
        f"beat_status={normalized_turn_state.get('beat_status', 'open')}; "
        f"offered_targets={_target_lines(normalized_turn_state.get('offered_targets', []))}; "
        f"summary={normalized_turn_state.get('summary', '') or 'None'}"
    )


def _build_transition_result(
    *,
    current_node_kind: str,
    target: dict[str, Any],
    transition_path: str,
    confidence: float,
    reason: str,
    close_current: bool | None = None,
) -> dict[str, Any]:
    close_current = _evaluate_close_current(current_node_kind, target) if close_current is None else bool(close_current)
    if transition_path == "via_return":
        close_current = True
    if current_node_kind == "hub" and transition_path == "direct" and str(target.get("role", "")) == "branch":
        close_current = False
    return {
        "action": "target",
        "target_kind": str(target.get("target_kind", "")),
        "target_id": str(target.get("target_id", "")),
        "transition_path": transition_path,
        "close_current": close_current,
        "confidence": max(0.0, min(1.0, confidence)),
        "reason": reason,
    }


def _build_stay_transition_result(reason: str, *, confidence: float = 0.0) -> dict[str, Any]:
    return {
        "action": "stay",
        "target_kind": "",
        "target_id": "",
        "transition_path": "stay",
        "close_current": False,
        "confidence": max(0.0, min(1.0, confidence)),
        "reason": reason,
    }


def _build_plot_completion_result(
    *,
    completed: bool,
    objective_satisfied: bool,
    progress: float,
    action: str,
    selected_target: dict[str, Any] | None,
    transition_path: str,
    close_current: bool,
    confidence: float,
    reason: str,
) -> dict[str, Any]:
    if action not in {"stay", "target"}:
        action = "stay"
    return {
        "completed": bool(completed),
        "objective_satisfied": bool(objective_satisfied),
        "progress": max(0.0, min(1.0, progress)),
        "action": action,
        "target_kind": str((selected_target or {}).get("target_kind", "")),
        "target_id": str((selected_target or {}).get("target_id", "")),
        "transition_path": _normalize_transition_path(transition_path),
        "close_current": bool(close_current),
        "confidence": max(0.0, min(1.0, confidence)),
        "reason": reason,
    }


def _resolve_selected_transition(
    parsed: dict[str, Any],
    *,
    current_node_kind: str,
    allowed_targets: list[dict[str, Any]],
    indirect_targets_via_return: list[dict[str, Any]] | None = None,
    default_reason: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    selected_target, selected_transition_path = _validate_selected_target(
        parsed,
        allowed_targets,
        indirect_targets_via_return,
    )
    if selected_target is None:
        return None, None
    close_current = parsed.get("close_current", _evaluate_close_current(current_node_kind, selected_target))
    transition_result = _build_transition_result(
        current_node_kind=current_node_kind,
        target=selected_target,
        transition_path=selected_transition_path,
        close_current=bool(close_current),
        confidence=max(0.0, min(1.0, float(parsed.get("confidence", 0.0) or 0.0))),
        reason=str(parsed.get("reason", "")).strip() or default_reason,
    )
    return selected_target, transition_result


def _validate_selected_target(
    parsed: dict[str, Any],
    allowed_targets: list[dict[str, Any]],
    indirect_targets_via_return: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any] | None, str]:
    target_kind = str(parsed.get("target_kind", "")).strip().lower()
    target_id = str(parsed.get("target_id", "")).strip()
    if not target_kind or not target_id:
        return None, "stay"
    for target in allowed_targets:
        if target.get("target_kind") == target_kind and target.get("target_id") == target_id and target.get("eligible"):
            return target, _normalize_transition_path(target.get("transition_path", "direct"))
    for target in indirect_targets_via_return or []:
        if target.get("target_kind") == target_kind and target.get("target_id") == target_id and target.get("eligible"):
            return target, _normalize_transition_path(target.get("transition_path", "via_return"))
    return None, "stay"


def _evaluate_close_current(current_node_kind: str, target: dict[str, Any] | None) -> bool:
    if target is None:
        return current_node_kind not in {"hub"}
    role = str(target.get("role", "")).strip().lower()
    if current_node_kind == "hub" and role == "branch":
        return False
    return True


def _latest_memory_turn(db: Database, scene_id: str, plot_id: str, visit_id: int) -> dict[str, Any] | None:
    recent_turns = db.get_recent_turns(scene_id, plot_id, limit=1, visit_id=visit_id)
    if not recent_turns:
        return None
    return recent_turns[-1]


def _latest_agent_turn_excerpt(db: Database, scene_id: str, plot_id: str, visit_id: int) -> str:
    latest_turn = _latest_memory_turn(db, scene_id, plot_id, visit_id)
    if not latest_turn:
        return ""
    return _truncate_text(str(latest_turn.get("agent", "")), 320)


def _node_from_target(
    target: dict[str, Any] | None,
    scene_map: dict[str, dict[str, Any]],
    plot_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    normalized_target = normalize_target(target or {})
    if normalized_target.get("target_kind") == "scene":
        return scene_map.get(str(normalized_target.get("target_id", "")), {})
    if normalized_target.get("target_kind") == "plot":
        return plot_map.get(str(normalized_target.get("target_id", "")), {})
    return {}


def _target_identity(target: dict[str, Any] | None) -> tuple[str, str]:
    normalized_target = normalize_target(target or {})
    return (
        str(normalized_target.get("target_kind", "")).strip().lower(),
        str(normalized_target.get("target_id", "")).strip(),
    )


def _node_identity(node: dict[str, Any] | None) -> tuple[str, str]:
    if not isinstance(node, dict):
        return ("", "")
    plot_id = str(node.get("plot_id", "")).strip()
    if plot_id:
        return ("plot", plot_id)
    scene_id = str(node.get("scene_id", "")).strip()
    if scene_id:
        return ("scene", scene_id)
    return ("", "")


def _target_matches_identity(target: dict[str, Any] | None, target_kind: str, target_id: str) -> bool:
    kind, identity = _target_identity(target)
    return kind == str(target_kind or "").strip().lower() and identity == str(target_id or "").strip()


def _via_return_reachable_targets(
    node: dict[str, Any],
    nav_state: dict[str, Any],
    scene_map: dict[str, dict[str, Any]],
    plot_map: dict[str, dict[str, Any]],
    *,
    current_scene_id: str,
    current_plot_id: str,
    max_depth: int = 4,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    seen_targets: set[tuple[str, str]] = set()
    visited_states: set[tuple[tuple[str, str], bool]] = set()
    queue: deque[tuple[dict[str, Any], bool, int]] = deque([(node, False, 0)])

    while queue:
        current_node, connectors_used, depth = queue.popleft()
        node_key = _node_identity(current_node)
        state_key = (node_key, connectors_used)
        if state_key in visited_states:
            continue
        visited_states.add(state_key)
        if depth > max_depth:
            continue

        navigation = normalize_navigation(current_node.get("navigation"))
        implicit_return = normalize_target(navigation.get("return_target"))
        if implicit_return.get("target_kind") and implicit_return.get("target_id"):
            hydrated_return = _hydrate_target(implicit_return, nav_state, scene_map, plot_map)
            next_node = _node_from_target(hydrated_return, scene_map, plot_map)
            if next_node and depth < max_depth:
                queue.append((next_node, True, depth + 1))

        for target in _allowed_targets_for_node(current_node, nav_state, scene_map, plot_map):
            if not target.get("eligible"):
                continue
            if _target_matches_identity(target, "scene", current_scene_id) or _target_matches_identity(target, "plot", current_plot_id):
                continue

            role = str(target.get("role", "")).strip().lower()
            target_key = _target_identity(target)
            if role == "return":
                next_node = _node_from_target(target, scene_map, plot_map)
                if next_node and depth < max_depth:
                    queue.append((next_node, True, depth + 1))
                continue

            if connectors_used and target_key not in seen_targets:
                seen_targets.add(target_key)
                results.append(target)

    return results


def _find_via_return_resolution(
    node: dict[str, Any],
    nav_state: dict[str, Any],
    scene_map: dict[str, dict[str, Any]],
    plot_map: dict[str, dict[str, Any]],
    *,
    target_kind: str,
    target_id: str,
    current_scene_id: str,
    current_plot_id: str,
    max_depth: int = 4,
) -> dict[str, Any] | None:
    normalized_target_kind = str(target_kind or "").strip().lower()
    normalized_target_id = str(target_id or "").strip()
    if not normalized_target_kind or not normalized_target_id:
        return None

    queue: deque[tuple[dict[str, Any], bool, int, list[dict[str, Any]]]] = deque([(node, False, 0, [])])
    visited_states: set[tuple[tuple[str, str], bool]] = set()

    while queue:
        current_node, connectors_used, depth, connectors = queue.popleft()
        node_key = _node_identity(current_node)
        state_key = (node_key, connectors_used)
        if state_key in visited_states:
            continue
        visited_states.add(state_key)
        if depth > max_depth:
            continue

        navigation = normalize_navigation(current_node.get("navigation"))
        implicit_return = normalize_target(navigation.get("return_target"))
        if implicit_return.get("target_kind") and implicit_return.get("target_id"):
            hydrated_return = _hydrate_target(implicit_return, nav_state, scene_map, plot_map)
            if (
                hydrated_return.get("eligible")
                and not _target_matches_identity(hydrated_return, "scene", current_scene_id)
                and not _target_matches_identity(hydrated_return, "plot", current_plot_id)
            ):
                if _target_matches_identity(hydrated_return, normalized_target_kind, normalized_target_id):
                    return {
                        "target": hydrated_return,
                        "parent_node": current_node,
                        "connectors": connectors,
                    }
                next_node = _node_from_target(hydrated_return, scene_map, plot_map)
                if next_node and depth < max_depth:
                    queue.append((next_node, True, depth + 1, connectors + [hydrated_return]))

        for target in _allowed_targets_for_node(current_node, nav_state, scene_map, plot_map):
            if not target.get("eligible"):
                continue
            if _target_matches_identity(target, "scene", current_scene_id) or _target_matches_identity(target, "plot", current_plot_id):
                continue

            role = str(target.get("role", "")).strip().lower()
            if role == "return":
                if _target_matches_identity(target, normalized_target_kind, normalized_target_id):
                    return {
                        "target": target,
                        "parent_node": current_node,
                        "connectors": connectors,
                    }
                next_node = _node_from_target(target, scene_map, plot_map)
                if next_node and depth < max_depth:
                    queue.append((next_node, True, depth + 1, connectors + [target]))
                continue

            if connectors_used and _target_matches_identity(target, normalized_target_kind, normalized_target_id):
                return {
                    "target": target,
                    "parent_node": current_node,
                    "connectors": connectors,
                }

    return None


def _indirect_targets_via_return(
    node: dict[str, Any],
    nav_state: dict[str, Any],
    scene_map: dict[str, dict[str, Any]],
    plot_map: dict[str, dict[str, Any]],
    *,
    current_scene_id: str,
    current_plot_id: str,
) -> list[dict[str, Any]]:
    return _via_return_reachable_targets(
        node,
        nav_state,
        scene_map,
        plot_map,
        current_scene_id=current_scene_id,
        current_plot_id=current_plot_id,
    )


def _transition_context_summary(latest_transition: dict[str, Any]) -> str:
    if not isinstance(latest_transition, dict):
        return "None"
    if not any(str(latest_transition.get(key, "")).strip() for key in latest_transition):
        return "None"
    return (
        f"from {latest_transition.get('source_plot_id', '') or 'unknown'} "
        f"({latest_transition.get('source_plot_goal', '') or 'unknown'}) "
        f"to {latest_transition.get('target_plot_id', '') or 'unknown'} "
        f"({latest_transition.get('target_plot_goal', '') or 'unknown'}); "
        f"reason={latest_transition.get('transition_reason', '') or 'none'}; "
        f"latest_user_input={latest_transition.get('latest_user_input', '') or 'none'}; "
        f"relationship={latest_transition.get('relationship', '') or 'none'}"
    )


def _current_scene_boundary_summary(scene_goal: str, scene_description: str, current_plot_raw_text: str) -> str:
    parts: list[str] = []
    if scene_goal:
        parts.append(f"Scene Goal: {scene_goal}")
    if scene_description:
        parts.append(f"Scene Scope: {_truncate_text(scene_description, 220)}")
    if current_plot_raw_text:
        parts.append(f"Current Plot Context: {_truncate_text(current_plot_raw_text, 220)}")
    return "\n".join(parts) or "Use the current scene goal and plot excerpt as the scope boundary."


def _objective_checklist_summary(plot_goal: str, mandatory_events: list[str]) -> str:
    if mandatory_events:
        lines = [f"- {event}" for event in mandatory_events if str(event).strip()]
        return "\n".join(lines) or f"- Advance the current plot goal: {plot_goal or 'current plot'}"
    return f"- Advance the current plot goal: {plot_goal or 'current plot'}"


def _plot_exit_conditions_summary(
    *,
    current_node_kind: str,
    allowed_targets: list[dict[str, Any]],
    return_target: dict[str, Any] | None,
) -> str:
    lines = [
        "- End only if the current plot objective is clearly achieved.",
        "- End if the investigator explicitly gives up or explicitly leaves the current scene.",
    ]
    eligible_targets = _eligible_targets(allowed_targets)
    if eligible_targets:
        lines.append(f"- Legal immediate handoff targets: {_target_lines(eligible_targets)}")
    if return_target:
        lines.append(f"- If the plot closes without an explicit target, resolve toward: {_target_lines([return_target])}")
    if current_node_kind == "hub":
        lines.append("- A hub stays open until the player actually chooses a branch or an exit.")
    return "\n".join(lines)


def _completion_signals_summary(plot_goal: str, mandatory_events: list[str]) -> str:
    lines = [f"- The plot should advance toward: {plot_goal or 'the current objective'}"]
    if mandatory_events:
        lines.append("- Strong completion signals include covering these beats:")
        lines.extend([f"  - {event}" for event in mandatory_events if str(event).strip()])
    else:
        lines.append("- Strong completion signals include resolving the current investigation beat in-scene.")
    return "\n".join(lines)


def _alignment_candidate_summary(candidate: dict[str, Any] | None) -> str:
    if not candidate:
        return "None"
    return _target_lines([candidate])


def _conversation_history_text(conversation_history: list[dict[str, Any]] | None, *, limit: int = 8) -> str:
    history_tail = conversation_history[-limit:] if conversation_history else []
    return "\n".join([f"User: {turn.get('user', '')}\nAgent: {turn.get('agent', '')}" for turn in history_tail])


def _intent_summary(result: dict[str, Any]) -> str:
    return (
        f"alignment={result.get('alignment', 'current_plot')}; "
        f"action={result.get('action', 'stay')}; "
        f"target={result.get('target_kind', '')}:{result.get('target_id', '') or 'none'}; "
        f"transition_path={result.get('transition_path', 'stay')}; "
        f"close_current={str(bool(result.get('close_current', False))).lower()}; "
        f"confidence={float(result.get('confidence', 0.0) or 0.0):.2f}; "
        f"reason={result.get('reason', '') or 'None'}"
    )


def _stay_intent_result(
    reason: str,
    *,
    alignment: str = "current_plot",
    confidence: float = 0.0,
) -> dict[str, Any]:
    result = _build_stay_transition_result(reason, confidence=confidence)
    result["alignment"] = _normalize_alignment(alignment)
    result["candidate_target"] = None
    result["candidate_score"] = 0.0
    result["summary"] = _intent_summary(result)
    return result


def _target_intent_result(
    *,
    alignment: str,
    current_node_kind: str,
    target: dict[str, Any],
    transition_path: str,
    confidence: float,
    reason: str,
    close_current: bool | None = None,
) -> dict[str, Any]:
    result = _build_transition_result(
        current_node_kind=current_node_kind,
        target=target,
        transition_path=transition_path,
        confidence=confidence,
        reason=reason,
        close_current=close_current,
    )
    candidate_target = deepcopy(target)
    candidate_target["transition_path"] = transition_path
    result["alignment"] = _normalize_alignment(alignment, default="target_choice")
    result["candidate_target"] = candidate_target
    result["candidate_score"] = max(0.0, min(1.0, confidence))
    result["summary"] = _intent_summary(result)
    return result


def _build_eval_context(
    user_input: str,
    *,
    plot_goal: str = "",
    scene_goal: str = "",
    scene_description: str = "",
    current_plot_raw_text: str = "",
    current_node_kind: str = "linear",
    allowed_targets: list[dict[str, Any]] | None = None,
    indirect_targets_via_return: list[dict[str, Any]] | None = None,
    remaining_required_targets: list[dict[str, Any]] | None = None,
    return_target: dict[str, Any] | None = None,
    mandatory_events: list[str] | None = None,
    redirect_streak: int = 0,
    latest_agent_turn_excerpt: str = "",
    latest_turn_state: dict[str, Any] | None = None,
    choice_prompt_active: bool = False,
    conversation_history: list[dict[str, Any]] | None = None,
    alignment_prompt_recorder: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    allowed_targets = deepcopy(allowed_targets or [])
    indirect_targets_via_return = deepcopy(indirect_targets_via_return or [])
    remaining_required_targets = deepcopy(remaining_required_targets or [])
    return_target = deepcopy(return_target or {})
    mandatory_events = list(mandatory_events or [])
    legal_targets = _legal_targets(allowed_targets, indirect_targets_via_return)
    latest_turn_state = _normalize_turn_state(
        latest_turn_state,
        legal_targets=legal_targets,
        default_choice_open=choice_prompt_active,
    )
    eligible_targets = _eligible_targets(allowed_targets)
    eligible_indirect_targets = _eligible_targets(indirect_targets_via_return)
    alignment = classify_player_alignment(
        user_input,
        plot_goal=plot_goal,
        scene_goal=scene_goal,
        scene_description=scene_description,
        current_plot_raw_text=current_plot_raw_text,
        mandatory_events=mandatory_events,
        allowed_targets=allowed_targets,
        indirect_targets_via_return=indirect_targets_via_return,
        remaining_required_targets=remaining_required_targets,
        redirect_streak=redirect_streak,
        latest_agent_turn_excerpt=latest_agent_turn_excerpt,
        latest_turn_state=latest_turn_state,
        choice_prompt_active=choice_prompt_active,
        current_node_kind=current_node_kind,
        conversation_history=conversation_history,
        prompt_recorder=alignment_prompt_recorder,
    )
    selected_target = alignment.get("candidate_target")
    plot_exit_conditions_summary = _plot_exit_conditions_summary(
        current_node_kind=current_node_kind,
        allowed_targets=allowed_targets,
        return_target=return_target,
    )
    if eligible_indirect_targets:
        plot_exit_conditions_summary += (
            "\n- Legal handoff targets reachable after closing and returning first: "
            f"{_target_lines(eligible_indirect_targets)}"
        )
    return {
        "plot_goal": plot_goal,
        "scene_goal": scene_goal,
        "scene_description": scene_description,
        "current_plot_raw_text": current_plot_raw_text,
        "current_node_kind": current_node_kind,
        "allowed_targets": allowed_targets,
        "indirect_targets_via_return": indirect_targets_via_return,
        "remaining_required_targets": remaining_required_targets,
        "return_target": return_target,
        "mandatory_events": mandatory_events,
        "redirect_streak": int(redirect_streak),
        "latest_agent_turn_excerpt": latest_agent_turn_excerpt,
        "latest_turn_state": latest_turn_state,
        "latest_turn_state_summary": _turn_state_summary(latest_turn_state),
        "choice_prompt_active": bool(choice_prompt_active or latest_turn_state.get("choice_open", False)),
        "conversation_history": conversation_history or [],
        "history_text": _conversation_history_text(conversation_history),
        "alignment": alignment,
        "selected_target": selected_target,
        "selected_confidence": float(alignment.get("candidate_score", 0.0)),
        "selected_transition_path": _normalize_transition_path(alignment.get("transition_path", "stay")),
        "legal_targets": legal_targets,
        "eligible_targets": eligible_targets,
        "eligible_indirect_targets": eligible_indirect_targets,
        "current_plot_excerpt": _truncate_text(current_plot_raw_text, 500) or "None",
        "current_scene_boundary_summary": _current_scene_boundary_summary(
            scene_goal,
            scene_description,
            current_plot_raw_text,
        ),
        "objective_checklist_summary": _objective_checklist_summary(plot_goal, mandatory_events),
        "completion_signals_summary": _completion_signals_summary(plot_goal, mandatory_events),
        "plot_exit_conditions_summary": plot_exit_conditions_summary,
    }


def _build_pre_response_transition_prompt(ctx: dict[str, Any], user_input: str) -> str:
    return f"""
You are a narrative-state evaluator for a script-driven investigation game.
Classify the player's latest intent using the legal navigation graph only.

Return JSON only:
{{
  "alignment": "target_choice" | "current_plot" | "off_topic",
  "action": "stay" | "target",
  "target_kind": "scene" | "plot" | "",
  "target_id": "id or empty string",
  "transition_path": "stay" | "direct" | "via_return",
  "close_current": true/false,
  "confidence": 0.0 to 1.0,
  "reason": "brief explanation"
}}

Rules:
- Only select a target from Eligible Targets or Indirect Targets Via Return.
- Use action=target only when the player clearly commits to a legal target.
- If the player is continuing the current beat, return alignment=current_plot and action=stay.
- If the player is unrelated to the active narrative and legal targets, return alignment=off_topic and action=stay.
- If Latest Turn State shows an open choice, prefer the offered targets as the valid answer space.
- If the current node is a hub and the player picks a branch, prefer close_current=false.
- If the chosen target is only reachable after closing the current branch first, use transition_path=via_return and close_current=true.
- Do not invent targets. Do not select exhausted or blocked targets.
- Keep output strict JSON.

Current Node Kind:
{ctx['current_node_kind']}

Scene Goal:
{ctx['scene_goal'] or 'None'}

Current Plot Goal:
{ctx['plot_goal'] or 'None'}

Current Plot Excerpt:
{ctx['current_plot_excerpt']}

Current Scene Boundary:
{ctx['current_scene_boundary_summary']}

Allowed Targets:
{_target_lines(ctx['allowed_targets'])}

Eligible Targets:
{_target_lines(ctx['eligible_targets'])}

Indirect Targets Via Return:
{_target_lines(ctx['eligible_indirect_targets'])}

Remaining Required Targets:
{_target_lines(ctx['remaining_required_targets'])}

Mandatory Events:
{ctx['mandatory_events']}

Redirect State:
redirect_streak={ctx['redirect_streak']}

Latest Turn State:
{ctx['latest_turn_state_summary']}

Latest Agent Turn Excerpt:
{ctx['latest_agent_turn_excerpt'] or 'None'}

Recent Conversation:
{ctx['history_text'] or 'None'}

Latest Player Input:
{user_input}
""".strip()


def _build_plot_completion_prompt(ctx: dict[str, Any], user_input: str, response: str) -> str:
    return f"""
You are a narrative-state evaluator for a script-driven investigation game.
Decide whether the current plot stays active, closes, or hands off to a legal target.

Return JSON only:
{{
  "completed": true/false,
  "objective_satisfied": true/false,
  "progress_delta": 0.0 to 0.6,
  "action": "stay" | "target",
  "target_kind": "scene" | "plot" | "",
  "target_id": "id or empty string",
  "transition_path": "stay" | "direct" | "via_return",
  "close_current": true/false,
  "confidence": 0.0 to 1.0,
  "reason": "brief explanation"
}}

Rules:
- Only select a target from Eligible Targets or Indirect Targets Via Return.
- If the player is still working the current beat, keep action=stay and close_current=false.
- If the current beat is resolved or intentionally wrapped, you may set close_current=true.
- For hubs, do not close the current node without an explicit legal branch or exit choice.
- If the player is off-topic, keep action=stay and close_current=false.
- Use Mandatory Events as evidence, not as hard requirements.
- Keep output strict JSON.

Current Node Kind:
{ctx['current_node_kind']}

Scene Goal:
{ctx['scene_goal'] or 'None'}

Current Plot Goal:
{ctx['plot_goal'] or 'None'}

Current Plot Excerpt:
{ctx['current_plot_excerpt']}

Current Scene Boundary:
{ctx['current_scene_boundary_summary']}

Active Plot Objective:
{ctx['plot_goal'] or 'None'}

Objective Checklist:
{ctx['objective_checklist_summary']}

Completion Signals:
{ctx['completion_signals_summary']}

Plot Exit Conditions:
{ctx['plot_exit_conditions_summary']}

Allowed Targets:
{_target_lines(ctx['allowed_targets'])}

Eligible Targets:
{_target_lines(ctx['eligible_targets'])}

Indirect Targets Via Return:
{_target_lines(ctx['eligible_indirect_targets'])}

Remaining Required Targets:
{_target_lines(ctx['remaining_required_targets'])}

Mandatory Events:
{ctx['mandatory_events']}

Latest Turn State:
{ctx['latest_turn_state_summary']}

Player Intent:
{ctx['alignment'].get('summary', 'None')}

Recent Conversation:
{ctx['history_text'] or 'None'}

Latest Turn:
User: {user_input}
Agent: {response}
""".strip()


def classify_player_alignment(
    user_input: str,
    *,
    plot_goal: str = "",
    scene_goal: str = "",
    scene_description: str = "",
    current_plot_raw_text: str = "",
    mandatory_events: list[str] | None = None,
    allowed_targets: list[dict[str, Any]] | None = None,
    indirect_targets_via_return: list[dict[str, Any]] | None = None,
    remaining_required_targets: list[dict[str, Any]] | None = None,
    redirect_streak: int = 0,
    latest_agent_turn_excerpt: str = "",
    latest_turn_state: dict[str, Any] | None = None,
    choice_prompt_active: bool = False,
    current_node_kind: str = "linear",
    conversation_history: list[dict[str, Any]] | None = None,
    prompt_recorder: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    user_input = (user_input or "").strip()
    allowed_targets = deepcopy(allowed_targets or [])
    indirect_targets_via_return = deepcopy(indirect_targets_via_return or [])
    legal_targets = _legal_targets(allowed_targets, indirect_targets_via_return)
    latest_turn_state = _normalize_turn_state(
        latest_turn_state,
        legal_targets=legal_targets,
        default_choice_open=choice_prompt_active,
    )
    if not user_input:
        return _stay_intent_result("empty_player_input")

    ctx = {
        "plot_goal": plot_goal,
        "scene_goal": scene_goal,
        "scene_description": scene_description,
        "current_plot_excerpt": _truncate_text(current_plot_raw_text, 2000) or "None",
        "current_scene_boundary_summary": _current_scene_boundary_summary(
            scene_goal,
            scene_description,
            current_plot_raw_text,
        ),
        "current_node_kind": current_node_kind,
        "allowed_targets": allowed_targets,
        "eligible_targets": _eligible_targets(allowed_targets),
        "eligible_indirect_targets": _eligible_targets(indirect_targets_via_return),
        "remaining_required_targets": deepcopy(remaining_required_targets or []),
        "mandatory_events": list(mandatory_events or []),
        "redirect_streak": int(redirect_streak),
        "latest_turn_state_summary": _turn_state_summary(latest_turn_state),
        "latest_agent_turn_excerpt": latest_agent_turn_excerpt,
        "history_text": _conversation_history_text(conversation_history),
    }
    prompt = _build_pre_response_transition_prompt(ctx, user_input)
    if prompt_recorder:
        prompt_recorder(prompt)

    try:
        llm_raw = call_nvidia_llm(
            prompt,
            model=PROGRESSION_MODEL,
            step_name="player_alignment_classification",
            allow_env_override=False,
        )
        parsed = _extract_json_obj(llm_raw)
        if parsed is None:
            return _stay_intent_result("llm_unavailable_stay")
        action = str(parsed.get("action", "stay")).strip().lower()
        if action not in {"stay", "target"}:
            action = "stay"
        confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.0) or 0.0)))
        alignment = _normalize_alignment(
            parsed.get("alignment"),
            default="target_choice" if action == "target" else "current_plot",
        )
        if action == "target":
            selected_target, transition_path = _validate_selected_target(
                parsed,
                allowed_targets,
                indirect_targets_via_return,
            )
            if selected_target is None:
                fallback_alignment = alignment if alignment != "target_choice" else "current_plot"
                return _stay_intent_result("llm_invalid_target_stay", alignment=fallback_alignment, confidence=confidence)
            return _target_intent_result(
                alignment=alignment,
                current_node_kind=current_node_kind,
                target=selected_target,
                transition_path=transition_path,
                confidence=confidence,
                reason=str(parsed.get("reason", "")).strip() or "llm_target_choice",
                close_current=parsed.get("close_current"),
            )
        return _stay_intent_result(
            str(parsed.get("reason", "")).strip() or "llm_stay_current_plot",
            alignment=alignment,
            confidence=confidence,
        )
    except Exception:
        return _stay_intent_result("llm_unavailable_stay")


def _build_turn_state_prompt(
    *,
    response: str,
    current_node_kind: str,
    allowed_targets: list[dict[str, Any]] | None = None,
    indirect_targets_via_return: list[dict[str, Any]] | None = None,
) -> str:
    legal_targets = _legal_targets(allowed_targets, indirect_targets_via_return)
    return f"""
You are extracting structured turn state for a script-driven investigation game.

Return JSON only:
{{
  "choice_open": true/false,
  "offered_targets": [{{"target_kind": "scene|plot", "target_id": "id"}}],
  "beat_status": "open" | "wrapped" | "exhausted",
  "summary": "brief summary"
}}

Rules:
- `offered_targets` must be a subset of the legal targets listed below.
- Set `choice_open=true` only if the response clearly leaves the player at an active choice point.
- Use `beat_status=wrapped` when the response closes the current beat but still points onward.
- Use `beat_status=exhausted` when the response makes it clear this beat has nothing further to offer.
- Keep output strict JSON.

Current Node Kind:
{current_node_kind}

Legal Targets:
{_target_lines(legal_targets)}

Keeper Response:
{response or 'None'}
""".strip()


def extract_turn_state(
    response: str,
    *,
    current_node_kind: str = "linear",
    allowed_targets: list[dict[str, Any]] | None = None,
    indirect_targets_via_return: list[dict[str, Any]] | None = None,
    prompt_recorder: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    legal_targets = _legal_targets(allowed_targets, indirect_targets_via_return)
    default_state = _default_turn_state(summary="llm_unavailable")
    prompt = _build_turn_state_prompt(
        response=response,
        current_node_kind=current_node_kind,
        allowed_targets=allowed_targets,
        indirect_targets_via_return=indirect_targets_via_return,
    )
    if prompt_recorder:
        prompt_recorder(prompt)

    try:
        llm_raw = call_nvidia_llm(
            prompt,
            model=PROGRESSION_MODEL,
            step_name="turn_state_extraction",
            allow_env_override=False,
        )
        parsed = _extract_json_obj(llm_raw)
        if parsed is None:
            return default_state
        return _normalize_turn_state(parsed, legal_targets=legal_targets)
    except Exception:
        return default_state


def story_position_context(
    db: Database,
    current_scene_id: str,
    current_plot_id: str,
    navigation_state: dict[str, Any] | None = None,
    *,
    current_visit_id: int = 0,
) -> dict[str, Any]:
    nav_state = _normalize_navigation_state(navigation_state or {})
    scenes, scene_map, plot_map = _story_graph(db)
    current_scene = scene_map.get(current_scene_id, {})
    current_plot = plot_map.get(current_plot_id, {})
    mandatory_events = list(current_plot.get("mandatory_events", []) or [])

    current_node = current_plot or current_scene
    current_navigation = normalize_navigation(current_node.get("navigation"))
    scene_navigation = normalize_navigation(current_scene.get("navigation"))

    allowed_targets = _allowed_targets_for_node(current_node, nav_state, scene_map, plot_map)
    eligible_targets = _eligible_targets(allowed_targets)
    exhausted_targets = _exhausted_targets(allowed_targets)
    blocked_targets = _blocked_targets(allowed_targets)
    eligible_branch_targets = [target for target in eligible_targets if target.get("role") == "branch"]
    eligible_exit_targets = [target for target in eligible_targets if target.get("role") == "exit"]
    remaining_required_targets = _remaining_required_targets(current_node, nav_state, scene_map, plot_map)
    return_target = normalize_target(current_navigation.get("return_target"))
    use_scene_level_fallback = bool(
        current_plot
        and not (return_target.get("target_kind") and return_target.get("target_id"))
        and not eligible_targets
        and scene_navigation.get("return_target")
    )
    if use_scene_level_fallback:
        return_target = normalize_target(scene_navigation.get("return_target"))
    return_target_hydrated = None
    if return_target.get("target_kind") and return_target.get("target_id"):
        return_target_hydrated = _hydrate_target(return_target, nav_state, scene_map, plot_map)
    indirect_targets_via_return = _indirect_targets_via_return(
        current_node,
        nav_state,
        scene_map,
        plot_map,
        current_scene_id=current_scene_id,
        current_plot_id=current_plot_id,
    )
    if use_scene_level_fallback and not indirect_targets_via_return:
        indirect_targets_via_return = _indirect_targets_via_return(
            current_scene,
            nav_state,
            scene_map,
            plot_map,
            current_scene_id=current_scene_id,
            current_plot_id=current_plot_id,
        )

    latest_turn = _latest_memory_turn(db, current_scene_id, current_plot_id, int(current_visit_id)) or {}
    legal_targets = _legal_targets(allowed_targets, indirect_targets_via_return)
    latest_turn_state = _normalize_turn_state(latest_turn.get("turn_state"), legal_targets=legal_targets)
    latest_agent_turn_excerpt = _truncate_text(str(latest_turn.get("agent", "")), 320)
    choice_prompt_active = bool(latest_turn_state.get("choice_open", False))

    first_plot_target = _first_target_of_kind(allowed_targets, "plot")
    first_scene_target = _first_target_of_kind(allowed_targets, "scene")
    next_scene_plot_target = None
    if first_scene_target and first_scene_target.get("scene_id"):
        target_scene = scene_map.get(str(first_scene_target.get("scene_id", "")), {})
        next_scene_plot = _first_active_plot(target_scene)
        if next_scene_plot:
            next_scene_plot_target = {
                "goal": str(next_scene_plot.get("plot_goal", "")),
                "excerpt": _truncate_text(str(next_scene_plot.get("raw_text", "")), 500),
            }

    scene_progress = _node_progress(
        current_scene,
        nav_state,
        scene_map,
        plot_map,
        fallback_progress=float(db.get_system_state().get("scene_progress", 0.0)),
    )
    plot_progress = _node_progress(
        current_plot,
        nav_state,
        scene_map,
        plot_map,
        fallback_progress=float(current_plot.get("progress", 0.0) or 0.0),
    )
    current_node_id = str((current_plot or current_scene).get("plot_id") or (current_plot or current_scene).get("scene_id") or "")
    current_hub_meta = nav_state.get("hub_progress", {}).get(current_node_id, {})
    current_hub_status = "No active hub."
    if current_hub_meta:
        current_hub_status = (
            f"policy={current_hub_meta.get('completion_policy', 'terminal_on_resolve')}; "
            f"progress={float(current_hub_meta.get('progress', 0.0) or 0.0):.0%}; "
            f"remaining_required={len(remaining_required_targets)}; "
            f"eligible_branches={len(eligible_branch_targets)}; "
            f"exhausted_branches={len([target for target in exhausted_targets if target.get('role') == 'branch'])}; "
            f"exit_available={str(bool(eligible_exit_targets)).lower()}"
        )
    redirect_state = _current_plot_guidance(nav_state, str(current_plot.get("plot_id", "")), int(current_visit_id))
    redirect_state_summary = (
        f"redirect_streak={int(redirect_state.get('redirect_streak', 0) or 0)}; "
        f"last_alignment={redirect_state.get('last_alignment', '') or 'none'}; "
        f"last_handoff_candidate={_alignment_candidate_summary(redirect_state.get('last_handoff_candidate'))}"
    )
    current_scene_boundary_summary = _current_scene_boundary_summary(
        str(current_scene.get("scene_goal", "")),
        str(current_scene.get("scene_description", "")),
        str(current_plot.get("raw_text", "")),
    )
    opening_choice_allowed = bool(
        str(current_node.get("node_kind", "linear")) == "hub"
        and bool(eligible_targets)
    )
    active_plot_objective = str(current_plot.get("plot_goal", "")) or str(current_scene.get("scene_goal", ""))
    objective_checklist_summary = _objective_checklist_summary(active_plot_objective, mandatory_events)
    completion_signals_summary = _completion_signals_summary(active_plot_objective, mandatory_events)
    plot_exit_conditions_summary = _plot_exit_conditions_summary(
        current_node_kind=str(current_node.get("node_kind", "linear")),
        allowed_targets=allowed_targets,
        return_target=return_target_hydrated,
    )
    latest_transition = deepcopy(nav_state.get("latest_transition", {}))

    return {
        "current_plot_raw_text": _truncate_text(str(current_plot.get("raw_text", "")), 3000),
        "current_navigation": current_navigation,
        "allowed_targets": allowed_targets,
        "allowed_targets_summary": _target_lines(allowed_targets),
        "eligible_targets": eligible_targets,
        "eligible_targets_summary": _target_lines(eligible_targets),
        "indirect_targets_via_return": indirect_targets_via_return,
        "indirect_targets_summary": _target_lines(indirect_targets_via_return),
        "eligible_branch_targets_summary": _target_lines(eligible_branch_targets),
        "eligible_exit_targets_summary": _target_lines(eligible_exit_targets),
        "exhausted_targets": exhausted_targets,
        "exhausted_targets_summary": _target_lines(exhausted_targets),
        "blocked_targets": blocked_targets,
        "blocked_targets_summary": _target_lines(blocked_targets),
        "remaining_required_targets": remaining_required_targets,
        "remaining_required_targets_summary": _target_lines(remaining_required_targets),
        "return_target": return_target_hydrated,
        "return_target_summary": _target_lines([return_target_hydrated] if return_target_hydrated else []),
        "current_node_kind": str(current_node.get("node_kind", "linear")),
        "scene_node_kind": str(current_scene.get("node_kind", "linear")),
        "plot_node_kind": str(current_plot.get("node_kind", "linear")),
        "hub_progress": deepcopy(nav_state.get("hub_progress", {})),
        "current_hub_status": current_hub_status,
        "redirect_streak": int(redirect_state.get("redirect_streak", 0) or 0),
        "last_alignment": str(redirect_state.get("last_alignment", "")),
        "last_handoff_candidate_summary": _alignment_candidate_summary(redirect_state.get("last_handoff_candidate")),
        "redirect_state_summary": redirect_state_summary,
        "active_plot_objective": active_plot_objective,
        "objective_checklist_summary": objective_checklist_summary,
        "completion_signals_summary": completion_signals_summary,
        "plot_exit_conditions_summary": plot_exit_conditions_summary,
        "current_scene_boundary_summary": current_scene_boundary_summary,
        "latest_agent_turn_excerpt": latest_agent_turn_excerpt or "None",
        "latest_turn_state": latest_turn_state,
        "latest_turn_state_summary": _turn_state_summary(latest_turn_state),
        "choice_prompt_active": choice_prompt_active,
        "opening_choice_allowed": opening_choice_allowed,
        "latest_transition": latest_transition,
        "transition_context_summary": _transition_context_summary(latest_transition),
        "visited_scene_ids": list(nav_state.get("visited_scene_ids", [])),
        "visited_plot_ids": list(nav_state.get("visited_plot_ids", [])),
        "plot_progress": plot_progress,
        "scene_progress": scene_progress,
        "next_plot_goal": str(first_plot_target.get("goal", "")) if first_plot_target else "",
        "next_plot_excerpt": str(first_plot_target.get("excerpt", "")) if first_plot_target else "",
        "next_scene_goal": str(first_scene_target.get("goal", "")) if first_scene_target else "",
        "next_scene_plot_goal": str((next_scene_plot_target or {}).get("goal", "")),
        "next_scene_plot_excerpt": str((next_scene_plot_target or {}).get("excerpt", "")),
        "return_target_goal": str((return_target_hydrated or {}).get("goal", "")),
        "hub_exit_available": bool(eligible_exit_targets),
    }


def evaluate_plot_completion(
    user_input: str,
    response: str,
    current_progress: float,
    mandatory_events: list[str],
    *,
    plot_goal: str = "",
    scene_goal: str = "",
    scene_description: str = "",
    current_plot_raw_text: str = "",
    current_node_kind: str = "linear",
    allowed_targets: list[dict[str, Any]] | None = None,
    indirect_targets_via_return: list[dict[str, Any]] | None = None,
    remaining_required_targets: list[dict[str, Any]] | None = None,
    redirect_streak: int = 0,
    latest_agent_turn_excerpt: str = "",
    latest_turn_state: dict[str, Any] | None = None,
    choice_prompt_active: bool = False,
    conversation_history: list[dict[str, Any]] | None = None,
    prompt_recorder: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    ctx = _build_eval_context(
        user_input,
        plot_goal=plot_goal,
        scene_goal=scene_goal,
        scene_description=scene_description,
        current_plot_raw_text=current_plot_raw_text,
        current_node_kind=current_node_kind,
        allowed_targets=allowed_targets,
        indirect_targets_via_return=indirect_targets_via_return,
        remaining_required_targets=remaining_required_targets,
        mandatory_events=mandatory_events,
        redirect_streak=redirect_streak,
        latest_agent_turn_excerpt=latest_agent_turn_excerpt,
        latest_turn_state=latest_turn_state,
        choice_prompt_active=choice_prompt_active,
        conversation_history=conversation_history,
    )
    llm_prompt = _build_plot_completion_prompt(ctx, user_input, response)
    if prompt_recorder:
        prompt_recorder(llm_prompt)

    try:
        llm_raw = call_nvidia_llm(
            llm_prompt,
            model=PROGRESSION_MODEL,
            step_name="plot_completion_evaluation",
            allow_env_override=False,
        )
        parsed = _extract_json_obj(llm_raw)
        if parsed is None:
            raise ValueError("missing completion json")

        action = str(parsed.get("action", "stay")).strip().lower()
        if action not in {"stay", "target"}:
            action = "stay"
        selected_target = None
        transition_path = "stay"
        close_current = bool(parsed.get("close_current", False))
        confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.0) or 0.0)))
        if action == "target":
            selected_target, transition_result = _resolve_selected_transition(
                parsed,
                current_node_kind=current_node_kind,
                allowed_targets=ctx["allowed_targets"],
                indirect_targets_via_return=ctx["indirect_targets_via_return"],
                default_reason="llm_plot_completion_target",
            )
            if transition_result is None:
                return _build_plot_completion_result(
                    completed=False,
                    objective_satisfied=False,
                    progress=current_progress,
                    action="stay",
                    selected_target=None,
                    transition_path="stay",
                    close_current=False,
                    confidence=confidence,
                    reason="llm_invalid_target_stay",
                )
            transition_path = str(transition_result.get("transition_path", "stay"))
            close_current = bool(transition_result.get("close_current", False))
            confidence = float(transition_result.get("confidence", confidence))
            reason = str(transition_result.get("reason", "")).strip() or "llm_plot_completion_target"
        else:
            reason = str(parsed.get("reason", "")).strip() or "llm_plot_completion_stay"
            if current_node_kind == "hub":
                close_current = False

        if ctx["alignment"].get("alignment") == "off_topic" and action != "target":
            return _build_plot_completion_result(
                completed=False,
                objective_satisfied=False,
                progress=current_progress,
                action="stay",
                selected_target=None,
                transition_path="stay",
                close_current=False,
                confidence=confidence,
                reason=reason,
            )

        completed = bool(parsed.get("completed", False))
        objective_satisfied = bool(parsed.get("objective_satisfied", False))
        progress_delta = max(0.0, min(0.6, float(parsed.get("progress_delta", 0.0) or 0.0)))
        progress = min(1.0, max(current_progress, current_progress + progress_delta))
        if close_current:
            progress = 1.0
            completed = True
        if current_node_kind == "hub" and action != "target":
            close_current = False
            completed = False
            progress = current_progress

        return _build_plot_completion_result(
            completed=completed,
            objective_satisfied=objective_satisfied,
            progress=progress,
            action=action,
            selected_target=selected_target,
            transition_path=transition_path,
            close_current=close_current,
            confidence=confidence,
            reason=reason,
        )
    except Exception:
        return _build_plot_completion_result(
            completed=False,
            objective_satisfied=False,
            progress=current_progress,
            action="stay",
            selected_target=None,
            transition_path="stay",
            close_current=False,
            confidence=0.0,
            reason="llm_unavailable_stay",
        )


def evaluate_pre_response_transition(
    user_input: str,
    *,
    plot_goal: str = "",
    scene_goal: str = "",
    scene_description: str = "",
    current_plot_raw_text: str = "",
    current_node_kind: str = "linear",
    allowed_targets: list[dict[str, Any]] | None = None,
    indirect_targets_via_return: list[dict[str, Any]] | None = None,
    remaining_required_targets: list[dict[str, Any]] | None = None,
    return_target: dict[str, Any] | None = None,
    mandatory_events: list[str] | None = None,
    redirect_streak: int = 0,
    latest_agent_turn_excerpt: str = "",
    latest_turn_state: dict[str, Any] | None = None,
    choice_prompt_active: bool = False,
    conversation_history: list[dict[str, Any]] | None = None,
    prompt_recorder: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    user_input = (user_input or "").strip()
    if not user_input:
        return _build_stay_transition_result("empty_player_input")

    ctx = _build_eval_context(
        user_input,
        plot_goal=plot_goal,
        scene_goal=scene_goal,
        scene_description=scene_description,
        current_plot_raw_text=current_plot_raw_text,
        current_node_kind=current_node_kind,
        allowed_targets=allowed_targets,
        indirect_targets_via_return=indirect_targets_via_return,
        remaining_required_targets=remaining_required_targets,
        return_target=return_target,
        mandatory_events=mandatory_events,
        redirect_streak=redirect_streak,
        latest_agent_turn_excerpt=latest_agent_turn_excerpt,
        latest_turn_state=latest_turn_state,
        choice_prompt_active=choice_prompt_active,
        conversation_history=conversation_history,
        alignment_prompt_recorder=prompt_recorder,
    )
    alignment = ctx["alignment"]
    if alignment.get("action") == "target" and ctx.get("selected_target"):
        return {
            "action": "target",
            "target_kind": str(alignment.get("target_kind", "")),
            "target_id": str(alignment.get("target_id", "")),
            "transition_path": _normalize_transition_path(alignment.get("transition_path", "stay")),
            "close_current": bool(alignment.get("close_current", False)),
            "confidence": max(0.0, min(1.0, float(alignment.get("confidence", 0.0) or 0.0))),
            "reason": str(alignment.get("reason", "")).strip() or "llm_target_choice",
        }
    return _build_stay_transition_result(
        str(alignment.get("reason", "")).strip() or "llm_unavailable_stay",
        confidence=max(0.0, min(1.0, float(alignment.get("confidence", 0.0) or 0.0))),
    )


def evaluate_scene_completion(
    db: Database,
    scene_id: str,
    *,
    current_plot_id: str = "",
    plot_transition: dict[str, Any] | None = None,
    navigation_state: dict[str, Any] | None = None,
    conversation_history: list[dict[str, Any]] | None = None,
    latest_turn_text: str = "",
    prompt_recorder: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    def _result(completed: bool, progress: float, reason: str) -> dict[str, Any]:
        if prompt_recorder:
            prompt_recorder(
                f"deterministic_scene_completion scene_id={scene_id} "
                f"target_kind={target_kind or 'none'} target_id={target_id or 'none'} "
                f"reason={reason} latest_turn={latest_turn_text or 'None'}"
            )
        return {"completed": completed, "progress": progress, "reason": reason}

    nav_state = _normalize_navigation_state(navigation_state or {})
    scenes, scene_map, plot_map = _story_graph(db)
    scene = scene_map.get(scene_id)
    if not scene:
        target_kind = ""
        target_id = ""
        return _result(False, 0.0, "scene_missing")

    current_plot = plot_map.get(current_plot_id or "", _first_active_plot(scene))
    scene_navigation = normalize_navigation(scene.get("navigation"))
    scene_progress = _node_progress(scene, nav_state, scene_map, plot_map, fallback_progress=0.0)

    transition = plot_transition or {}
    target_kind = str(transition.get("target_kind", "")).strip().lower()
    target_id = str(transition.get("target_id", "")).strip()
    close_current = bool(transition.get("close_current", False))
    target = None
    if target_kind and target_id:
        target = normalize_target({"target_kind": target_kind, "target_id": target_id})
        target = _hydrate_target(target, nav_state, scene_map, plot_map)

    if str(scene.get("node_kind", "")) == "terminal":
        if close_current or str(current_plot.get("status", "")) in {"completed", "skipped"}:
            return _result(True, 1.0, "terminal_scene_resolved")
        return _result(False, scene_progress, "terminal_scene_pending")

    if str(scene.get("node_kind", "")) == "hub":
        if target and target.get("role") == "branch" and str(target.get("scene_id", "")) != scene_id:
            return _result(False, scene_progress, "hub_entered_branch")
        if target and target.get("role") == "exit":
            return _result(True, 1.0, "hub_exit_selected")
        return _result(False, scene_progress, "hub_still_open")

    if target:
        if target.get("target_kind") == "scene" and str(target.get("scene_id", "")) and str(target.get("scene_id", "")) != scene_id:
            return _result(True, 1.0, "scene_handoff_to_other_scene")
        if target.get("target_kind") == "plot" and str(target.get("scene_id", "")) == scene_id:
            return _result(False, scene_progress, "scene_handoff_to_internal_plot")

    if close_current and scene_navigation.get("return_target"):
        return _result(True, 1.0, "scene_closed_before_return")
    return _result(False, scene_progress, "scene_continues")


def _target_to_position(
    target: dict[str, Any],
    scene_map: dict[str, dict[str, Any]],
    plot_map: dict[str, dict[str, Any]],
) -> tuple[str, str]:
    target_kind = target.get("target_kind")
    target_id = target.get("target_id")
    if target_kind == "scene":
        scene = scene_map.get(target_id, {})
        plot = _first_active_plot(scene)
        return str(scene.get("scene_id", "")), str(plot.get("plot_id", ""))
    if target_kind == "plot":
        plot = plot_map.get(target_id, {})
        return str(plot.get("scene_id", "")), str(plot.get("plot_id", ""))
    return "", ""


def _find_valid_target(
    node: dict[str, Any],
    target_kind: str,
    target_id: str,
    nav_state: dict[str, Any],
    scene_map: dict[str, dict[str, Any]],
    plot_map: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    if not target_kind or not target_id:
        return None
    for target in _allowed_targets_for_node(node, nav_state, scene_map, plot_map):
        if target.get("target_kind") == target_kind and target.get("target_id") == target_id and target.get("eligible"):
            return target
    return None


def _default_transition_target(
    node: dict[str, Any],
    nav_state: dict[str, Any],
    scene_map: dict[str, dict[str, Any]],
    plot_map: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    navigation = normalize_navigation(node.get("navigation"))
    node_kind = str(node.get("node_kind", "linear"))
    allowed_targets = _allowed_targets_for_node(node, nav_state, scene_map, plot_map)

    if node_kind == "hub" and navigation.get("completion_policy") == "all_required_then_advance":
        remaining_required = _remaining_required_targets(node, nav_state, scene_map, plot_map)
        if not remaining_required:
            exit_target = next((target for target in allowed_targets if target.get("role") == "exit" and target.get("eligible")), None)
            if exit_target:
                return exit_target

    return_target = normalize_target(navigation.get("return_target"))
    if return_target.get("target_kind") and return_target.get("target_id"):
        return _hydrate_target(return_target, nav_state, scene_map, plot_map)

    eligible_targets = [target for target in allowed_targets if target.get("eligible")]
    if len(eligible_targets) == 1:
        return eligible_targets[0]

    linear_target = next((target for target in eligible_targets if target.get("role") in {"", "exit"}), None)
    if linear_target and node_kind != "hub":
        return linear_target
    return None


def _resolve_via_return_target(
    db: Database,
    current_node: dict[str, Any],
    *,
    target_kind: str,
    target_id: str,
    nav_state: dict[str, Any],
    scene_map: dict[str, dict[str, Any]],
    plot_map: dict[str, dict[str, Any]],
    current_visit_id: int,
) -> dict[str, Any] | None:
    current_scene_id = str(current_node.get("scene_id", "") or "")
    current_plot_id = str(current_node.get("plot_id", "") or "")
    resolution = _find_via_return_resolution(
        current_node,
        nav_state,
        scene_map,
        plot_map,
        target_kind=target_kind,
        target_id=target_id,
        current_scene_id=current_scene_id,
        current_plot_id=current_plot_id,
    )
    if resolution is None:
        return None

    selected_target = resolution["target"]
    parent_node = resolution["parent_node"]
    connectors = resolution.get("connectors", [])
    for connector in connectors:
        connector_kind, connector_id = _target_identity(connector)
        _pop_return_frame(nav_state, connector_kind, connector_id)

    parent_navigation = normalize_navigation(parent_node.get("navigation"))
    parent_return_target = normalize_target(parent_navigation.get("return_target"))
    if str(selected_target.get("role", "")).strip().lower() == "return" or _target_matches_identity(
        parent_return_target,
        str(selected_target.get("target_kind", "")),
        str(selected_target.get("target_id", "")),
    ):
        _pop_return_frame(
            nav_state,
            str(selected_target.get("target_kind", "")),
            str(selected_target.get("target_id", "")),
        )

    parent_node_kind = str(parent_node.get("node_kind", "linear"))
    if parent_node_kind == "hub" and selected_target.get("role") == "branch":
        if selected_target.get("target_kind") == "scene":
            _push_return_frame(nav_state, "scene", str(parent_node.get("scene_id", "")), int(current_visit_id))
        else:
            _push_return_frame(nav_state, "plot", str(parent_node.get("plot_id", "")), int(current_visit_id))
        if bool(parent_navigation.get("close_unselected_on_advance")):
            _mark_sibling_targets_skipped(db, parent_node, selected_target, scene_map, plot_map)

    if parent_node_kind == "hub" and selected_target.get("role") == "exit":
        if str(parent_navigation.get("completion_policy", "")) == "optional_until_exit":
            _mark_sibling_targets_skipped(db, parent_node, selected_target, scene_map, plot_map)

    return selected_target


def _mark_visited(nav_state: dict[str, Any], scene_id: str, plot_id: str) -> None:
    if scene_id and scene_id not in nav_state.setdefault("visited_scene_ids", []):
        nav_state["visited_scene_ids"].append(scene_id)
    if plot_id and plot_id not in nav_state.setdefault("visited_plot_ids", []):
        nav_state["visited_plot_ids"].append(plot_id)


def _mark_sibling_targets_skipped(
    db: Database,
    hub_node: dict[str, Any],
    selected_target: dict[str, Any],
    scene_map: dict[str, dict[str, Any]],
    plot_map: dict[str, dict[str, Any]],
) -> None:
    empty_nav_state = {"visited_scene_ids": [], "visited_plot_ids": [], "hub_progress": {}, "return_stack": []}
    for target in _allowed_targets_for_node(hub_node, empty_nav_state, scene_map, plot_map):
        if target.get("role") != "branch":
            continue
        if target.get("target_id") == selected_target.get("target_id") and target.get("target_kind") == selected_target.get("target_kind"):
            continue
        if target.get("target_kind") == "scene":
            scene_id = str(target.get("target_id", ""))
            db.update_scene(scene_id, {"status": "skipped"})
            scene = scene_map.get(scene_id, {})
            for plot in scene.get("plots", []):
                db.update_plot(str(plot.get("plot_id", "")), status="skipped", progress=1.0)
        elif target.get("target_kind") == "plot":
            db.update_plot(str(target.get("target_id", "")), status="skipped", progress=1.0)


def next_story_position(
    db: Database,
    current_scene_id: str,
    current_plot_id: str,
    navigation_state: dict[str, Any] | None = None,
    advance_decision: dict[str, Any] | None = None,
    *,
    current_visit_id: int = 0,
) -> dict[str, Any]:
    nav_state = _normalize_navigation_state(navigation_state or {})
    scenes, scene_map, plot_map = _story_graph(db)
    current_scene = scene_map.get(current_scene_id, {})
    current_plot = plot_map.get(current_plot_id, {})
    current_node = current_plot or current_scene
    current_navigation = normalize_navigation(current_node.get("navigation"))
    advance = advance_decision or {}
    requested_target_kind = str(advance.get("target_kind", ""))
    requested_target_id = str(advance.get("target_id", ""))
    requested_transition_path = _normalize_transition_path(advance.get("transition_path", "stay"))

    selected_target = None
    if requested_transition_path == "via_return":
        selected_target = _resolve_via_return_target(
            db,
            current_node,
            target_kind=requested_target_kind,
            target_id=requested_target_id,
            nav_state=nav_state,
            scene_map=scene_map,
            plot_map=plot_map,
            current_visit_id=int(current_visit_id),
        )
    if selected_target is None:
        selected_target = _find_valid_target(
            current_node,
            requested_target_kind,
            requested_target_id,
            nav_state,
            scene_map,
            plot_map,
        )
    if selected_target is None and requested_target_kind and requested_target_id:
        selected_target = _resolve_via_return_target(
            db,
            current_node,
            target_kind=requested_target_kind,
            target_id=requested_target_id,
            nav_state=nav_state,
            scene_map=scene_map,
            plot_map=plot_map,
            current_visit_id=int(current_visit_id),
        )
    if selected_target is None:
        selected_target = _default_transition_target(current_node, nav_state, scene_map, plot_map)

    if selected_target is None and bool(advance.get("close_current", False)) and current_plot:
        if requested_transition_path == "via_return" and requested_target_kind and requested_target_id:
            selected_target = _resolve_via_return_target(
                db,
                current_scene,
                target_kind=requested_target_kind,
                target_id=requested_target_id,
                nav_state=nav_state,
                scene_map=scene_map,
                plot_map=plot_map,
                current_visit_id=int(current_visit_id),
            )
        if selected_target is None and requested_target_kind and requested_target_id:
            selected_target = _find_valid_target(
                current_scene,
                requested_target_kind,
                requested_target_id,
                nav_state,
                scene_map,
                plot_map,
            )
        if selected_target is None:
            selected_target = _default_transition_target(current_scene, nav_state, scene_map, plot_map)

    if selected_target is None:
        current_scene_progress = _node_progress(
            current_scene,
            nav_state,
            scene_map,
            plot_map,
            fallback_progress=float(db.get_system_state().get("scene_progress", 0.0)),
        )
        current_plot_progress = _node_progress(
            current_plot,
            nav_state,
            scene_map,
            plot_map,
            fallback_progress=float(current_plot.get("progress", 0.0) or 0.0),
        )
        return {
            "current_scene_id": current_scene_id,
            "current_plot_id": current_plot_id,
            "plot_progress": float(current_plot_progress),
            "scene_progress": float(current_scene_progress),
            "current_scene_intro": "",
            "navigation_state": nav_state,
            "current_visit_id": int(current_visit_id),
        }

    current_node_kind = str(current_node.get("node_kind", "linear"))
    if current_node_kind == "hub" and selected_target.get("role") == "branch":
        if str(current_scene.get("node_kind", "")) == "hub" and selected_target.get("target_kind") == "scene":
            _push_return_frame(nav_state, "scene", str(current_scene.get("scene_id", "")), int(current_visit_id))
        else:
            _push_return_frame(nav_state, "plot", str(current_plot.get("plot_id", "")), int(current_visit_id))
        if bool(current_navigation.get("close_unselected_on_advance")):
            _mark_sibling_targets_skipped(db, current_node, selected_target, scene_map, plot_map)

    if current_node_kind == "hub" and selected_target.get("role") == "exit":
        if str(current_navigation.get("completion_policy", "")) == "optional_until_exit":
            _mark_sibling_targets_skipped(db, current_node, selected_target, scene_map, plot_map)

    return_target = normalize_target(current_navigation.get("return_target"))
    if return_target.get("target_kind") and return_target.get("target_id") and selected_target.get("role") == "return":
        _pop_return_frame(nav_state, str(return_target.get("target_kind", "")), str(return_target.get("target_id", "")))

    if selected_target.get("target_kind") == "scene":
        target_scene = scene_map.get(str(selected_target.get("target_id", "")), {})
        if target_scene and str(target_scene.get("node_kind", "")) == "terminal":
            parent_scene_return = normalize_target(current_scene.get("navigation", {}).get("return_target"))
            if parent_scene_return.get("target_kind") == "scene" and parent_scene_return.get("target_id"):
                hub_scene = scene_map.get(str(parent_scene_return.get("target_id", "")), {})
                if hub_scene:
                    _mark_sibling_targets_skipped(
                        db,
                        hub_scene,
                        {"target_kind": "scene", "target_id": current_scene_id},
                        scene_map,
                        plot_map,
                    )

    next_scene_id, next_plot_id = _target_to_position(selected_target, scene_map, plot_map)
    if not next_scene_id or not next_plot_id:
        current_scene_progress = _node_progress(
            current_scene,
            nav_state,
            scene_map,
            plot_map,
            fallback_progress=float(db.get_system_state().get("scene_progress", 0.0)),
        )
        current_plot_progress = _node_progress(
            current_plot,
            nav_state,
            scene_map,
            plot_map,
            fallback_progress=float(current_plot.get("progress", 0.0) or 0.0),
        )
        return {
            "current_scene_id": current_scene_id,
            "current_plot_id": current_plot_id,
            "plot_progress": float(current_plot_progress),
            "scene_progress": float(current_scene_progress),
            "current_scene_intro": "",
            "navigation_state": nav_state,
            "current_visit_id": int(current_visit_id),
        }

    next_scene = scene_map.get(next_scene_id, {})
    next_plot = plot_map.get(next_plot_id, {})
    _mark_visited(nav_state, current_scene_id, current_plot_id)
    next_visit_id = int(current_visit_id) + 1 if (next_scene_id != current_scene_id or next_plot_id != current_plot_id) else int(current_visit_id)

    scene_progress = _node_progress(next_scene, nav_state, scene_map, plot_map, fallback_progress=0.0)
    plot_progress = _node_progress(next_plot, nav_state, scene_map, plot_map, fallback_progress=float(next_plot.get("progress", 0.0) or 0.0))

    if str(next_scene.get("node_kind", "")) == "terminal" and str(next_scene.get("status", "")) in {"completed", "skipped"}:
        current_scene_intro = "All scenes are complete."
    elif next_scene_id != current_scene_id:
        current_scene_intro = f"Scene {next_scene_id} begins: {next_scene.get('scene_goal', '')}"
    else:
        current_scene_intro = ""

    return {
        "current_scene_id": next_scene_id,
        "current_plot_id": next_plot_id,
        "plot_progress": float(plot_progress),
        "scene_progress": float(scene_progress),
        "current_scene_intro": current_scene_intro,
        "navigation_state": nav_state,
        "current_visit_id": next_visit_id,
    }
