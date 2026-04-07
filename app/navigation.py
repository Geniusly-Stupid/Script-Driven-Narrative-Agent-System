from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any

NODE_KINDS = {"linear", "hub", "branch", "terminal"}
COMPLETION_POLICIES = {
    "all_required_then_advance",
    "optional_until_exit",
    "exclusive_choice",
    "terminal_on_resolve",
}
TARGET_KINDS = {"scene", "plot"}


def natural_sort_key(value: str) -> tuple[Any, ...]:
    parts = re.split(r"(\d+)", (value or "").lower())
    key: list[Any] = []
    for part in parts:
        if not part:
            continue
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part)
    return tuple(key)


def make_target(
    target_kind: str,
    target_id: str,
    *,
    label: str = "",
    required: bool = False,
    role: str = "",
    prerequisites: list[str] | None = None,
    goal: str = "",
    excerpt: str = "",
) -> dict[str, Any]:
    return normalize_target(
        {
            "target_kind": target_kind,
            "target_id": target_id,
            "label": label,
            "required": required,
            "role": role,
            "prerequisites": prerequisites or [],
            "goal": goal,
            "excerpt": excerpt,
        }
    )


def default_navigation(
    *,
    completion_policy: str = "terminal_on_resolve",
    close_unselected_on_advance: bool = False,
) -> dict[str, Any]:
    return {
        "allowed_targets": [],
        "return_target": None,
        "completion_policy": completion_policy,
        "prerequisites": [],
        "close_unselected_on_advance": bool(close_unselected_on_advance),
    }


def normalize_target(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {
            "target_kind": "",
            "target_id": "",
            "label": "",
            "required": False,
            "role": "",
            "prerequisites": [],
            "goal": "",
            "excerpt": "",
        }

    target_kind = str(value.get("target_kind", "")).strip().lower()
    if target_kind not in TARGET_KINDS:
        target_kind = ""

    target_id = str(value.get("target_id", "")).strip()
    prerequisites = value.get("prerequisites", [])
    if not isinstance(prerequisites, list):
        prerequisites = []

    return {
        "target_kind": target_kind,
        "target_id": target_id,
        "label": str(value.get("label", "")).strip(),
        "required": bool(value.get("required", False)),
        "role": str(value.get("role", "")).strip().lower(),
        "prerequisites": [str(item).strip() for item in prerequisites if str(item).strip()],
        "goal": str(value.get("goal", "")).strip(),
        "excerpt": str(value.get("excerpt", "")).strip(),
    }


def normalize_navigation(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            value = {}
    if not isinstance(value, dict):
        value = {}

    allowed_targets = value.get("allowed_targets", [])
    if not isinstance(allowed_targets, list):
        allowed_targets = []

    prerequisites = value.get("prerequisites", [])
    if not isinstance(prerequisites, list):
        prerequisites = []

    completion_policy = str(value.get("completion_policy", "terminal_on_resolve")).strip().lower()
    if completion_policy not in COMPLETION_POLICIES:
        completion_policy = "terminal_on_resolve"

    return {
        "allowed_targets": [
            target
            for target in (normalize_target(item) for item in allowed_targets)
            if target.get("target_kind") and target.get("target_id")
        ],
        "return_target": _normalize_optional_target(value.get("return_target")),
        "completion_policy": completion_policy,
        "prerequisites": [str(item).strip() for item in prerequisites if str(item).strip()],
        "close_unselected_on_advance": bool(value.get("close_unselected_on_advance", False)),
    }


def serialize_navigation(value: Any) -> str:
    return json.dumps(normalize_navigation(value), ensure_ascii=False)


def clone_navigation(value: Any) -> dict[str, Any]:
    return deepcopy(normalize_navigation(value))


def ensure_scene_navigation_defaults(scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not scenes:
        return scenes

    ordered_scenes = sorted(scenes, key=lambda scene: natural_sort_key(str(scene.get("scene_id", ""))))
    scene_to_index = {str(scene.get("scene_id", "")): idx for idx, scene in enumerate(ordered_scenes)}

    for scene_idx, scene in enumerate(ordered_scenes):
        scene.setdefault("node_kind", "")
        scene["node_kind"] = _normalize_node_kind(scene.get("node_kind"), default="linear")
        scene["navigation"] = normalize_navigation(scene.get("navigation"))

        plots = scene.get("plots", [])
        if not isinstance(plots, list):
            plots = []
            scene["plots"] = plots

        ordered_plots = sorted(plots, key=lambda plot: natural_sort_key(str(plot.get("plot_id", ""))))
        for plot_idx, plot in enumerate(ordered_plots):
            plot.setdefault("node_kind", "")
            plot["node_kind"] = _normalize_node_kind(plot.get("node_kind"), default="linear")
            plot["navigation"] = normalize_navigation(plot.get("navigation"))

            if plot["navigation"]["allowed_targets"] or plot["navigation"]["return_target"]:
                if not plot["node_kind"]:
                    plot["node_kind"] = "terminal" if plot_idx == len(ordered_plots) - 1 and scene_idx == len(ordered_scenes) - 1 else "linear"
                continue

            next_plot = ordered_plots[plot_idx + 1] if plot_idx + 1 < len(ordered_plots) else None
            next_scene = ordered_scenes[scene_idx + 1] if scene_idx + 1 < len(ordered_scenes) else None

            if next_plot:
                plot["navigation"] = default_navigation(completion_policy="all_required_then_advance")
                plot["navigation"]["allowed_targets"] = [
                    make_target(
                        "plot",
                        str(next_plot.get("plot_id", "")),
                        label=str(next_plot.get("plot_goal", "")),
                    )
                ]
                if plot["node_kind"] == "terminal":
                    plot["node_kind"] = "linear"
            elif next_scene:
                plot["navigation"] = default_navigation(completion_policy="all_required_then_advance")
                plot["navigation"]["allowed_targets"] = [
                    make_target(
                        "scene",
                        str(next_scene.get("scene_id", "")),
                        label=str(next_scene.get("scene_goal", "")),
                    )
                ]
                if plot["node_kind"] == "terminal":
                    plot["node_kind"] = "linear"
            else:
                plot["navigation"] = default_navigation(completion_policy="terminal_on_resolve")
                plot["node_kind"] = "terminal"

        if scene["navigation"]["allowed_targets"] or scene["navigation"]["return_target"]:
            continue

        next_scene = ordered_scenes[scene_idx + 1] if scene_idx + 1 < len(ordered_scenes) else None
        if next_scene:
            scene["navigation"] = default_navigation(completion_policy="all_required_then_advance")
            scene["navigation"]["allowed_targets"] = [
                make_target(
                    "scene",
                    str(next_scene.get("scene_id", "")),
                    label=str(next_scene.get("scene_goal", "")),
                )
            ]
            if scene["node_kind"] == "terminal":
                scene["node_kind"] = "linear"
        else:
            scene["navigation"] = default_navigation(completion_policy="terminal_on_resolve")
            scene["node_kind"] = "terminal"

    return scenes


def get_scene_plot_ids(scene: dict[str, Any]) -> list[str]:
    plots = scene.get("plots", [])
    if not isinstance(plots, list):
        return []
    return [str(plot.get("plot_id", "")).strip() for plot in plots if str(plot.get("plot_id", "")).strip()]


def parse_json_object(value: Any, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if isinstance(value, dict):
        return deepcopy(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return deepcopy(default or {})


def _normalize_optional_target(value: Any) -> dict[str, Any] | None:
    target = normalize_target(value)
    if not target.get("target_kind") or not target.get("target_id"):
        return None
    return target


def _normalize_node_kind(value: Any, *, default: str = "linear") -> str:
    raw = str(value or "").strip().lower()
    if raw in NODE_KINDS:
        return raw
    return default
