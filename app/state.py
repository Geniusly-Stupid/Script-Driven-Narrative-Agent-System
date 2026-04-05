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

STOPWORDS = {
    "about",
    "after",
    "again",
    "around",
    "begin",
    "being",
    "current",
    "details",
    "during",
    "from",
    "guide",
    "have",
    "into",
    "investigator",
    "learn",
    "more",
    "next",
    "plot",
    "scene",
    "some",
    "that",
    "their",
    "them",
    "then",
    "they",
    "this",
    "through",
    "with",
}

SETUP_MARKERS = (
    "set up",
    "setup",
    "prepare",
    "introduce",
    "initial contact",
    "begin the investigation",
    "ready to",
    "next steps",
    "guide the player",
    "choose",
    "choice",
    "option",
    "options",
    "can choose",
    "can ask",
    "presumably",
    "start",
)

ABANDONMENT_MARKERS = (
    "give up",
    "stop looking",
    "stop investigating",
    "nothing more here",
    "nothing else here",
    "no more to find",
    "forget it",
    "i'm done here",
    "we're done here",
    "let's leave",
    "head back",
    "go back",
    "back out",
    "move on from here",
    "算了",
    "放弃",
    "不查了",
    "不搜了",
    "不继续了",
    "不想查了",
    "没什么可查",
    "没什么可搜",
    "查不到了",
    "先这样吧",
    "回去吧",
    "离开这里",
    "去别处",
)

EXIT_INTENT_MARKERS = (
    "move on",
    "next step",
    "next lead",
    "press on",
    "continue on",
    "continue the investigation",
    "head out",
    "leave now",
    "go to the conclusion",
    "go to the ending",
    "继续",
    "下一步",
    "继续吧",
    "往下",
    "推进",
    "去下一步",
    "去结局",
    "去结尾",
)

LEAVE_SCENE_MARKERS = (
    "leave this place",
    "leave the room",
    "leave the scene",
    "leave the area",
    "walk away",
    "step outside",
    "go outside",
    "head outside",
    "leave here",
    "get out of here",
    "go somewhere else",
    "check somewhere else",
    "move to another place",
    "move to another lead",
    "let's go somewhere else",
    "离开这里",
    "离开这个地方",
    "出去",
    "去别处",
    "换个地方",
    "去下一个地方",
)

PROGRESSION_MODEL = "qwen/qwen3.5-397b-a17b"
HIGH_CONFIDENCE_TARGET_THRESHOLD = 0.78
CURRENT_PLOT_RELEVANCE_THRESHOLD = 0.34

PLOT_CLOSED_RESPONSE_MARKERS = (
    "no more useful information",
    "nothing more can be found here",
    "nothing more here",
    "nothing else here",
    "no further clue",
    "no further clues",
    "no further information",
    "the lead is exhausted",
    "the clue has been secured",
    "you have what you came for",
    "time to move on",
    "hardly anything more to uncover",
    "\u5df2\u96be\u6709\u66f4\u591a\u4fe1\u606f\u53ef\u6316",
    "\u6682\u65f6\u5df2\u65e0\u66f4\u591a\u79d8\u5bc6\u53ef\u6316",
    "\u8fd9\u91cc\u5df2\u65e0\u66f4\u591a\u4fe1\u606f",
    "\u8fd9\u91cc\u8c03\u67e5\u4e0d\u5230\u66f4\u591a\u4fe1\u606f",
    "\u5df2\u7ecf\u62ff\u5230\u7ebf\u7d22",
    "\u5df2\u83b7\u53d6\u5173\u952e\u7ebf\u7d22",
    "\u53ef\u4ee5\u9009\u62e9\u524d\u5f80\u5176\u4ed6\u5730\u70b9",
    "\u53ef\u4ee5\u524d\u5f80\u5176\u4ed6\u5730\u70b9",
    "\u5728\u6b64\u5904\u7ed3\u675f\u5f53\u524d\u7684\u884c\u52a8",
)

CHOICE_PROMPT_MARKERS = (
    "choose one option",
    "choose one",
    "choose an option",
    "which do you choose",
    "what would you like to do next",
    "what topic",
    "which topic",
    "\u8bf7\u9009\u62e9",
    "\u53ef\u4ee5\u9009\u62e9",
    "\u9009\u9879",
    "\u4f60\u53ef\u4ee5",
    "\u4f60\u6253\u7b97",
    "\u4f60\u66f4\u503e\u5411\u4e8e\u54ea\u4e00\u79cd",
)

CASE_ACCEPTANCE_MARKERS = (
    "i can help",
    "i want to help",
    "i'll help",
    "i will help",
    "tell me more",
    "go on",
    "let's begin",
    "let us begin",
    "take the case",
    "accept the case",
    "hear the details",
    "start from the beginning",
    "what happened",
    "how can i help",
    "\u6211\u53ef\u4ee5\u600e\u4e48\u5e2e\u52a9\u4f60",
    "\u6211\u60f3\u5e2e\u52a9\u4f60",
    "\u6211\u613f\u610f\u5e2e\u4f60",
    "\u6211\u4f1a\u5e2e\u4f60",
    "\u8bf7\u7ee7\u7eed\u8bf4",
    "\u8bf7\u8bf4\u4e0b\u53bb",
    "\u544a\u8bc9\u6211\u8be6\u60c5",
    "\u5148\u8bf4\u8bf4\u60c5\u51b5",
    "\u5148\u628a\u60c5\u51b5\u544a\u8bc9\u6211",
    "\u53d1\u751f\u4e86\u4ec0\u4e48",
    "\u6211\u63a5\u4e0b\u8fd9\u4e2a\u6848\u5b50",
    "\u6211\u613f\u610f\u63a5\u624b",
)

INTRO_PLOT_MARKERS = (
    "introduce the case",
    "introduce the client",
    "initial setup",
    "set up the case",
    "contact from",
    "task given to the investigator",
    "player information",
    "briefed on the mystery",
    "\u4ecb\u7ecd\u6848\u4ef6",
    "\u4ecb\u7ecd\u59d4\u6258\u4eba",
    "\u521d\u59cb\u94fa\u57ab",
    "\u521d\u59cb\u8bbe\u5b9a",
    "\u6848\u4ef6\u8bf4\u660e",
    "\u59d4\u6258\u4eba",
    "\u5931\u7a83\u4e66\u7c4d",
    "\u53d4\u53d4\u5931\u8e2a",
)


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


def _goal_keywords(text: str) -> list[str]:
    raw_tokens = re.findall(r"[a-z]{3,}|[\u4e00-\u9fff]{2,}", _normalize_text(text))
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


def _is_setup_or_choice_like(plot_goal: str, current_plot_raw_text: str, scene_goal: str = "") -> bool:
    combined = _normalize_text(" ".join([plot_goal, scene_goal, _truncate_text(current_plot_raw_text, 320)]))
    return any(marker in combined for marker in SETUP_MARKERS)


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


def _target_match_score(text: str, target: dict[str, Any]) -> float:
    label = str(target.get("label", ""))
    goal = str(target.get("goal", ""))
    excerpt = str(target.get("excerpt", ""))[:300]

    label_score = _goal_match_score(text, label)
    goal_score = _goal_match_score(text, goal)
    excerpt_score = _goal_match_score(text, excerpt)
    combined_focus = _goal_match_score(text, " ".join(part for part in (label, goal) if part))

    best = max(label_score, goal_score, combined_focus, min(excerpt_score, 0.6))
    if label_score >= 0.5 and goal_score >= 0.34:
        best = max(best, min(1.0, max(label_score, goal_score) + 0.12))
    if label_score >= 0.9 or goal_score >= 0.9:
        best = max(best, 0.97)
    return min(1.0, best)


def _same_target(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
    if not a or not b:
        return False
    return (
        str(a.get("target_kind", "")).strip().lower() == str(b.get("target_kind", "")).strip().lower()
        and str(a.get("target_id", "")).strip() == str(b.get("target_id", "")).strip()
    )


def _extract_choice_index(user_input: str) -> int | None:
    normalized = _normalize_text(user_input)
    if not normalized:
        return None

    choice_map = {
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "first": 1,
        "second": 2,
        "third": 3,
        "fourth": 4,
        "fifth": 5,
        "一": 1,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "第一": 1,
        "第二": 2,
        "第三": 3,
        "第四": 4,
        "第五": 5,
    }
    explicit_choice_markers = ("option", "choice", "choose", "pick", "select", "选", "第")
    if len(normalized) > 32 and not any(marker in normalized for marker in explicit_choice_markers):
        return None

    digit_match = re.search(r"(?<!\d)([1-5])(?!\d)", normalized)
    if digit_match:
        return int(digit_match.group(1))

    for token, index in choice_map.items():
        if normalized == token or f" {token} " in f" {normalized} ":
            return index
    return None


def _build_transition_result(
    *,
    current_node_kind: str,
    target: dict[str, Any],
    transition_path: str,
    confidence: float,
    reason: str,
) -> dict[str, Any]:
    close_current = _evaluate_close_current(current_node_kind, target)
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


def _deterministic_pre_response_shortcut(
    *,
    user_input: str,
    current_node_kind: str,
    choice_prompt_active: bool,
    alignment: dict[str, Any],
    best_target: dict[str, Any] | None,
    best_score: float,
    candidate_transition_path: str,
    eligible_targets: list[dict[str, Any]],
    eligible_indirect_targets: list[dict[str, Any]],
) -> dict[str, Any] | None:
    ordered_targets = list(eligible_targets) + list(eligible_indirect_targets)
    choice_index = _extract_choice_index(user_input)
    if choice_index and 1 <= choice_index <= len(ordered_targets):
        selected = ordered_targets[choice_index - 1]
        transition_path = "direct" if choice_index <= len(eligible_targets) else "via_return"
        reason_prefix = "choice_prompt_numeric_match" if choice_prompt_active else "explicit_numeric_choice_match"
        return _build_transition_result(
            current_node_kind=current_node_kind,
            target=selected,
            transition_path=transition_path,
            confidence=0.98,
            reason=f"{reason_prefix}(index={choice_index})",
        )

    if choice_prompt_active:

        if best_target and (
            bool(alignment.get("high_confidence_target_choice")) or best_score >= HIGH_CONFIDENCE_TARGET_THRESHOLD
        ):
            if _same_target(best_target, next((target for target in eligible_targets if _same_target(target, best_target)), None)):
                return _build_transition_result(
                    current_node_kind=current_node_kind,
                    target=best_target,
                    transition_path="direct",
                    confidence=max(best_score, 0.9),
                    reason=f"choice_prompt_target_match(score={best_score:.2f})",
                )
            if _same_target(
                best_target,
                next((target for target in eligible_indirect_targets if _same_target(target, best_target)), None),
            ):
                return _build_transition_result(
                    current_node_kind=current_node_kind,
                    target=best_target,
                    transition_path="via_return",
                    confidence=max(best_score, 0.9),
                    reason=f"choice_prompt_target_match(score={best_score:.2f})",
                )

    if current_node_kind == "hub" and best_target and candidate_transition_path == "direct":
        if bool(alignment.get("high_confidence_target_choice")) and any(
            _same_target(best_target, target) for target in eligible_targets
        ):
            return _build_transition_result(
                current_node_kind=current_node_kind,
                target=best_target,
                transition_path="direct",
                confidence=max(best_score, 0.88),
                reason=f"hub_target_match(score={best_score:.2f})",
            )

    if current_node_kind == "hub" and best_target and candidate_transition_path == "via_return":
        if bool(alignment.get("high_confidence_target_choice")) and any(
            _same_target(best_target, target) for target in eligible_indirect_targets
        ):
            return _build_transition_result(
                current_node_kind=current_node_kind,
                target=best_target,
                transition_path="via_return",
                confidence=max(best_score, 0.9),
                reason=f"hub_via_return_target_match(score={best_score:.2f})",
            )

    if current_node_kind != "hub" and best_target and candidate_transition_path == "via_return":
        if bool(alignment.get("high_confidence_target_choice")) and any(
            _same_target(best_target, target) for target in eligible_indirect_targets
        ):
            return _build_transition_result(
                current_node_kind=current_node_kind,
                target=best_target,
                transition_path="via_return",
                confidence=max(best_score, 0.9),
                reason=f"non_hub_via_return_target_match(score={best_score:.2f})",
            )

    return None


def _best_target_match(
    text: str,
    allowed_targets: list[dict[str, Any]],
    indirect_targets_via_return: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any] | None, float, str]:
    best_target = None
    best_score = 0.0
    best_path = "stay"

    for transition_path, targets in (
        ("direct", _eligible_targets(allowed_targets)),
        ("via_return", _eligible_targets(indirect_targets_via_return or [])),
    ):
        for target in targets:
            score = _target_match_score(text, target)
            if score > best_score:
                best_target = target
                best_score = score
                best_path = transition_path
    return best_target, best_score, best_path


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
            return target, "direct"
    for target in indirect_targets_via_return or []:
        if target.get("target_kind") == target_kind and target.get("target_id") == target_id and target.get("eligible"):
            return target, "via_return"
    return None, "stay"


def _evaluate_close_current(current_node_kind: str, target: dict[str, Any] | None) -> bool:
    if target is None:
        return current_node_kind not in {"hub"}
    role = str(target.get("role", "")).strip().lower()
    if current_node_kind == "hub" and role == "branch":
        return False
    return True


def _contains_any_marker(text: str, markers: tuple[str, ...]) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    return any(marker in normalized for marker in markers)


def _is_explicit_abandonment(user_input: str) -> bool:
    return _contains_any_marker(user_input, ABANDONMENT_MARKERS)


def _is_explicit_leave_scene(user_input: str) -> bool:
    return _contains_any_marker(user_input, LEAVE_SCENE_MARKERS)


def _is_exit_intent(user_input: str) -> bool:
    return _contains_any_marker(user_input, EXIT_INTENT_MARKERS)


def _is_case_acceptance(user_input: str) -> bool:
    return _contains_any_marker(user_input, CASE_ACCEPTANCE_MARKERS)


def _looks_like_intro_plot(plot_goal: str, scene_goal: str, current_plot_raw_text: str) -> bool:
    combined = " ".join(
        part for part in (plot_goal or "", scene_goal or "", _truncate_text(current_plot_raw_text, 260) or "") if part
    )
    return _contains_any_marker(combined, INTRO_PLOT_MARKERS)


def _should_auto_advance_intro_plot(
    *,
    user_input: str,
    current_node_kind: str,
    plot_goal: str,
    scene_goal: str,
    current_plot_raw_text: str,
    eligible_targets: list[dict[str, Any]] | None = None,
) -> bool:
    if str(current_node_kind or "").strip().lower() != "linear":
        return False
    if len(list(eligible_targets or [])) != 1:
        return False
    if not (user_input or "").strip():
        return False
    return _looks_like_intro_plot(plot_goal, scene_goal, current_plot_raw_text)


def _response_suggests_plot_closed(text: str) -> bool:
    return _contains_any_marker(text, PLOT_CLOSED_RESPONSE_MARKERS)


def _is_choice_prompt_active(text: str) -> bool:
    raw_text = (text or "").strip()
    if not raw_text:
        return False
    normalized = _normalize_text(raw_text)
    if any(marker in normalized for marker in CHOICE_PROMPT_MARKERS):
        return True
    return bool(re.search(r"(^|\n)\s*\d+\s*[\.\u3001\)]", raw_text))


def _latest_agent_turn_excerpt(db: Database, scene_id: str, plot_id: str, visit_id: int) -> str:
    recent_turns = db.get_recent_turns(scene_id, plot_id, limit=1, visit_id=visit_id)
    if not recent_turns:
        return ""
    return _truncate_text(str(recent_turns[-1].get("agent", "")), 320)


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
    if alignment == "off_topic_unrelated":
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
) -> dict[str, Any]:
    user_input = (user_input or "").strip()
    mandatory_events = list(mandatory_events or [])
    allowed_targets = deepcopy(allowed_targets or [])
    indirect_targets_via_return = deepcopy(indirect_targets_via_return or [])
    normalized_input = _normalize_text(user_input)
    candidate_target, candidate_score, candidate_transition_path = _best_target_match(
        normalized_input,
        allowed_targets,
        indirect_targets_via_return,
    )
    explicit_abandon = _is_explicit_abandonment(user_input)
    explicit_leave_scene = _is_explicit_leave_scene(user_input)
    explicit_exit_intent = _is_exit_intent(user_input)

    plot_relevance_score = max(
        _goal_match_score(user_input, plot_goal),
        _goal_match_score(user_input, scene_goal) * 0.75,
        _goal_match_score(user_input, _truncate_text(current_plot_raw_text, 280)) * 0.6,
        _goal_match_score(user_input, _truncate_text(scene_description, 220)) * 0.45,
    )
    event_hits = _count_event_hits(user_input, mandatory_events)
    current_plot_relevant = bool(event_hits) or plot_relevance_score >= CURRENT_PLOT_RELEVANCE_THRESHOLD

    explicit_label_match = False
    if candidate_target:
        candidate_target = deepcopy(candidate_target)
        candidate_target["transition_path"] = candidate_transition_path
        label_candidates = [
            str(candidate_target.get("label", "")),
            str(candidate_target.get("goal", "")),
        ]
        explicit_label_match = any(
            candidate and len(_normalize_text(candidate)) >= 4 and _normalize_text(candidate) in normalized_input
            for candidate in label_candidates
        )

    high_confidence_target_choice = bool(candidate_target) and (
        explicit_label_match or candidate_score >= HIGH_CONFIDENCE_TARGET_THRESHOLD
    )

    if explicit_abandon:
        alignment = "explicit_abandon"
    elif explicit_leave_scene:
        alignment = "explicit_leave_scene"
    elif high_confidence_target_choice:
        alignment = "high_confidence_target_choice"
    elif current_plot_relevant:
        alignment = "current_plot_relevant"
    else:
        alignment = "off_topic_unrelated"

    return {
        "alignment": alignment,
        "explicit_abandon": explicit_abandon,
        "explicit_leave_scene": explicit_leave_scene,
        "explicit_exit_intent": explicit_exit_intent,
        "high_confidence_target_choice": high_confidence_target_choice,
        "current_plot_relevant": current_plot_relevant,
        "off_topic_unrelated": alignment == "off_topic_unrelated",
        "candidate_target": candidate_target,
        "candidate_score": float(candidate_score),
        "plot_relevance_score": float(plot_relevance_score),
        "event_hits": int(event_hits),
        "summary": (
            f"alignment={alignment}; "
            f"plot_relevance={plot_relevance_score:.2f}; "
            f"event_hits={event_hits}; "
            f"candidate={_alignment_candidate_summary(candidate_target)}; "
            f"candidate_score={candidate_score:.2f}; "
            f"transition_path={candidate_transition_path}"
        ),
    }


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
    latest_agent_turn_excerpt = _latest_agent_turn_excerpt(db, current_scene_id, current_plot_id, int(current_visit_id))
    choice_prompt_active = _is_choice_prompt_active(latest_agent_turn_excerpt)

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
        and _is_choice_prompt_active(str(current_plot.get("raw_text", "")))
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
        "current_plot_raw_text": _truncate_text(str(current_plot.get("raw_text", "")), 600),
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
        "choice_prompt_active": bool(choice_prompt_active),
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
    choice_prompt_active: bool = False,
    conversation_history: list[dict[str, Any]] | None = None,
    prompt_recorder: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    allowed_targets = deepcopy(allowed_targets or [])
    indirect_targets_via_return = deepcopy(indirect_targets_via_return or [])
    remaining_required_targets = deepcopy(remaining_required_targets or [])
    eligible_targets = _eligible_targets(allowed_targets)
    eligible_indirect_targets = _eligible_targets(indirect_targets_via_return)
    exit_target = _first_eligible_target_by_role(allowed_targets, "exit")
    indirect_exit_target = _first_eligible_target_by_role(eligible_indirect_targets, "exit")
    alignment = classify_player_alignment(
        user_input,
        plot_goal=plot_goal,
        scene_goal=scene_goal,
        scene_description=scene_description,
        current_plot_raw_text=current_plot_raw_text,
        mandatory_events=mandatory_events,
        allowed_targets=allowed_targets,
        indirect_targets_via_return=indirect_targets_via_return,
    )
    history_tail = conversation_history[-8:] if conversation_history else []
    history_text = "\n".join([f"User: {turn.get('user', '')}\nAgent: {turn.get('agent', '')}" for turn in history_tail])
    current_plot_excerpt = _truncate_text(current_plot_raw_text, 500) or "None"
    explicit_abandonment = bool(alignment.get("explicit_abandon"))
    explicit_leave_scene = bool(alignment.get("explicit_leave_scene"))
    explicit_exit_intent = bool(alignment.get("explicit_exit_intent"))
    high_confidence_target_choice = bool(alignment.get("high_confidence_target_choice"))
    candidate_target = alignment.get("candidate_target")
    candidate_transition_path = _normalize_transition_path((candidate_target or {}).get("transition_path", "stay"))
    plot_exit_conditions_summary = _plot_exit_conditions_summary(
        current_node_kind=current_node_kind,
        allowed_targets=allowed_targets,
        return_target=None,
    )
    if eligible_indirect_targets:
        plot_exit_conditions_summary += (
            "\n- Legal handoff targets reachable after closing and returning first: "
            f"{_target_lines(eligible_indirect_targets)}"
        )
    completion_signals_summary = _completion_signals_summary(plot_goal, mandatory_events)
    objective_checklist_summary = _objective_checklist_summary(plot_goal, mandatory_events)
    current_scene_boundary_summary = _current_scene_boundary_summary(scene_goal, scene_description, current_plot_raw_text)
    response_suggests_close = _response_suggests_plot_closed(response)
    intro_auto_advance = _should_auto_advance_intro_plot(
        user_input=user_input,
        current_node_kind=current_node_kind,
        plot_goal=plot_goal,
        scene_goal=scene_goal,
        current_plot_raw_text=current_plot_raw_text,
        eligible_targets=eligible_targets,
    )

    llm_prompt = f"""
You are a narrative-state evaluator for a script-driven investigation game.
Decide whether the CURRENT plot should stay active, close while remaining in the current hub, or hand off to one of the explicitly allowed targets.

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
- You may only choose targets from the Allowed Targets list.
- Only targets from the Eligible Targets list are currently enterable.
- You may choose a target from Indirect Targets Via Return only if the current plot closes first and the return hub can legally route there in the same turn.
- If the player input is off-topic and unrelated to both the current plot and all eligible targets, you must return action=stay and close_current=false.
- Only allow action=target when the player clearly and confidently commits to a legal target.
- If the current node is a hub and the player simply selects one branch, prefer close_current=false.
- If the player clearly commits to an exit target, clearly leaves the current scene, clearly gives up, or the current dramatic beat is finished, close_current=true.
- For non-hub plots, if the objective is already satisfied or the latest response clearly states that nothing more useful remains here, set objective_satisfied=true and close_current=true.
- If Latest Agent Turn already presented a clear choice and the player is answering that choice, prefer transitioning now instead of repeating the same choice point.
- The first priority is to keep the current plot active until it is either completed, explicitly abandoned, explicitly left, or confidently handed off.
- Do NOT auto-close a hub just because it is structurally ready to exit. Keep hubs open unless the player selected a branch or explicitly leaves through an eligible exit.
- Mandatory events are evidence, not hard gates.
- Keep output strict JSON.

Current Node Kind:
{current_node_kind}

Scene Goal:
{scene_goal or 'None'}

Current Plot Goal:
{plot_goal or 'None'}

Current Plot Excerpt:
{current_plot_excerpt}

Current Scene Boundary:
{current_scene_boundary_summary}

Active Plot Objective:
{plot_goal or 'None'}

Objective Checklist:
{objective_checklist_summary}

Completion Signals:
{completion_signals_summary}

Plot Exit Conditions:
{plot_exit_conditions_summary}

Allowed Targets:
{_target_lines(allowed_targets)}

Eligible Targets:
{_target_lines(eligible_targets)}

Indirect Targets Via Return:
{_target_lines(eligible_indirect_targets)}

Remaining Required Targets:
{_target_lines(remaining_required_targets)}

Mandatory Events:
{mandatory_events}

Redirect State:
redirect_streak={redirect_streak}

Latest Agent Turn Excerpt:
{latest_agent_turn_excerpt or 'None'}

Choice Prompt Active:
{str(bool(choice_prompt_active)).lower()}

Player Alignment Precheck:
{alignment.get('summary', 'None')}

Recent Conversation:
{history_text or 'None'}

Latest Turn:
User: {user_input}
Agent: {response}
"""
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
        if parsed is not None:
            selected_target, selected_transition_path = _validate_selected_target(
                parsed,
                allowed_targets,
                indirect_targets_via_return,
            )
            action = str(parsed.get("action", "stay")).strip().lower()
            if action not in {"stay", "target"}:
                action = "stay"
            completed = bool(parsed.get("completed", False))
            objective_satisfied = bool(parsed.get("objective_satisfied", False))
            delta = max(0.0, min(0.6, float(parsed.get("progress_delta", 0.0))))
            close_current = bool(parsed.get("close_current", completed))
            confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.0) or 0.0)))
            transition_path = _normalize_transition_path(parsed.get("transition_path", selected_transition_path))
            if selected_target is None:
                action = "stay"
                transition_path = "stay"
                close_current = bool(parsed.get("close_current", completed))
                if current_node_kind == "hub":
                    completed = False
                    close_current = False
                if alignment.get("off_topic_unrelated"):
                    completed = False
                    close_current = False
                    delta = 0.0
            else:
                transition_path = selected_transition_path
                close_current = bool(parsed.get("close_current", _evaluate_close_current(current_node_kind, selected_target)))
                if transition_path == "via_return":
                    close_current = True
                if current_node_kind == "hub" and transition_path == "direct" and str(selected_target.get("role", "")) == "branch":
                    close_current = False
                completed = completed or close_current or (objective_satisfied and current_node_kind != "hub")

            if current_node_kind != "hub" and (objective_satisfied or response_suggests_close):
                close_current = True
                completed = True

            if alignment.get("off_topic_unrelated"):
                progress = current_progress
            else:
                progress = 1.0 if close_current else min(1.0, max(current_progress, current_progress + delta))
            if current_node_kind != "hub" and progress >= 1.0:
                close_current = True
                completed = True
                progress = 1.0
            return {
                "completed": completed,
                "objective_satisfied": objective_satisfied or (current_node_kind != "hub" and response_suggests_close),
                "progress": progress,
                "action": action,
                "target_kind": str((selected_target or {}).get("target_kind", "")),
                "target_id": str((selected_target or {}).get("target_id", "")),
                "transition_path": transition_path,
                "close_current": close_current,
                "confidence": confidence,
                "reason": str(parsed.get("reason", "")).strip() or "llm_evaluation",
            }
    except Exception:
        pass

    combined_turn_text = _normalize_text(f"{user_input} {response}")
    setup_like = _is_setup_or_choice_like(plot_goal, current_plot_raw_text, scene_goal)
    case_acceptance = _is_case_acceptance(user_input)
    best_target = candidate_target
    best_score = float(alignment.get("candidate_score", 0.0))
    event_hits = max(_count_event_hits(combined_turn_text, mandatory_events), int(alignment.get("event_hits", 0)))
    completion_cues = any(
        cue in combined_turn_text
        for cue in ["complete", "resolved", "finished", "done", "decide", "choose", "leave", "head to", "go to"]
    )

    progress = current_progress
    if event_hits:
        progress += min(0.45, 0.2 * event_hits)
    if completion_cues and bool(alignment.get("current_plot_relevant")):
        progress += 0.2
    if current_node_kind != "hub" and response_suggests_close:
        progress = 1.0
    progress = min(1.0, max(progress, current_progress))

    if current_node_kind == "hub" and exit_target and (explicit_abandonment or explicit_leave_scene or explicit_exit_intent):
        return {
            "completed": True,
            "objective_satisfied": False,
            "progress": 1.0,
            "action": "target",
            "target_kind": str(exit_target.get("target_kind", "")),
            "target_id": str(exit_target.get("target_id", "")),
            "transition_path": "direct",
            "close_current": True,
            "confidence": 1.0,
            "reason": "explicit_exit_intent",
        }

    if intro_auto_advance:
        sole_target = eligible_targets[0]
        return {
            "completed": True,
            "objective_satisfied": True,
            "progress": 1.0,
            "action": "target",
            "target_kind": str(sole_target.get("target_kind", "")),
            "target_id": str(sole_target.get("target_id", "")),
            "transition_path": "direct",
            "close_current": True,
            "confidence": 0.95 if case_acceptance else 0.88,
            "reason": "intro_case_acceptance_transition" if case_acceptance else "intro_generic_reply_transition",
        }

    if explicit_abandonment:
        if current_node_kind == "hub":
            return {
                "completed": False,
                "objective_satisfied": False,
                "progress": current_progress,
                "action": "stay",
                "target_kind": "",
                "target_id": "",
                "transition_path": "stay",
                "close_current": False,
                "confidence": 1.0,
                "reason": "explicit_abandonment_but_hub_stays_open",
            }
        return {
            "completed": True,
            "objective_satisfied": False,
            "progress": 1.0,
            "action": "stay",
            "target_kind": "",
            "target_id": "",
            "transition_path": "stay",
            "close_current": True,
            "confidence": 1.0,
            "reason": "explicit_abandonment",
        }

    if explicit_leave_scene and current_node_kind != "hub":
        return {
            "completed": True,
            "objective_satisfied": False,
            "progress": 1.0,
            "action": "stay",
            "target_kind": "",
            "target_id": "",
            "transition_path": "stay",
            "close_current": True,
            "confidence": 1.0,
            "reason": "explicit_leave_scene",
        }

    if alignment.get("off_topic_unrelated"):
        return {
            "completed": False,
            "objective_satisfied": False,
            "progress": current_progress,
            "action": "stay",
            "target_kind": "",
            "target_id": "",
            "transition_path": "stay",
            "close_current": False,
            "confidence": 0.0,
            "reason": "off_topic_unrelated",
        }

    if (
        (high_confidence_target_choice or choice_prompt_active)
        and best_target
        and (best_score >= HIGH_CONFIDENCE_TARGET_THRESHOLD or choice_prompt_active)
        and (setup_like or current_node_kind in {"hub", "branch"} or event_hits > 0 or completion_cues)
    ):
        close_current = _evaluate_close_current(current_node_kind, best_target)
        if candidate_transition_path == "via_return":
            close_current = True
        if current_node_kind == "hub" and candidate_transition_path == "direct" and str(best_target.get("role", "")) == "branch":
            close_current = False
        return {
            "completed": close_current,
            "objective_satisfied": False,
            "progress": 1.0 if close_current else progress,
            "action": "target",
            "target_kind": str(best_target.get("target_kind", "")),
            "target_id": str(best_target.get("target_id", "")),
            "transition_path": candidate_transition_path,
            "close_current": close_current,
            "confidence": max(best_score, 0.85 if choice_prompt_active else 0.0),
            "reason": f"fallback_target_match(score={best_score:.2f})",
        }

    if progress >= 1.0 and current_node_kind != "hub":
        return {
            "completed": True,
            "objective_satisfied": bool(alignment.get("current_plot_relevant")) or response_suggests_close,
            "progress": 1.0,
            "action": "stay",
            "target_kind": "",
            "target_id": "",
            "transition_path": "stay",
            "close_current": True,
            "confidence": 0.9 if response_suggests_close else 0.7,
            "reason": "fallback_progress_threshold",
        }

    return {
        "completed": False,
        "objective_satisfied": False,
        "progress": progress,
        "action": "stay",
        "target_kind": "",
        "target_id": "",
        "transition_path": "stay",
        "close_current": False,
        "confidence": 0.0,
        "reason": "fallback_continue_current_plot",
    }


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
    choice_prompt_active: bool = False,
    conversation_history: list[dict[str, Any]] | None = None,
    prompt_recorder: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    user_input = (user_input or "").strip()
    if not user_input:
        return {
            "action": "stay",
            "target_kind": "",
            "target_id": "",
            "transition_path": "stay",
            "close_current": False,
            "confidence": 0.0,
            "reason": "empty_player_input",
        }

    allowed_targets = deepcopy(allowed_targets or [])
    indirect_targets_via_return = deepcopy(indirect_targets_via_return or [])
    remaining_required_targets = deepcopy(remaining_required_targets or [])
    return_target = deepcopy(return_target or {})
    eligible_targets = _eligible_targets(allowed_targets)
    eligible_indirect_targets = _eligible_targets(indirect_targets_via_return)
    exit_target = _first_eligible_target_by_role(allowed_targets, "exit")
    indirect_exit_target = _first_eligible_target_by_role(eligible_indirect_targets, "exit")
    mandatory_events = list(mandatory_events or [])
    alignment = classify_player_alignment(
        user_input,
        plot_goal=plot_goal,
        scene_goal=scene_goal,
        scene_description=scene_description,
        current_plot_raw_text=current_plot_raw_text,
        mandatory_events=mandatory_events,
        allowed_targets=allowed_targets,
        indirect_targets_via_return=indirect_targets_via_return,
    )
    history_tail = conversation_history[-8:] if conversation_history else []
    history_text = "\n".join([f"User: {turn.get('user', '')}\nAgent: {turn.get('agent', '')}" for turn in history_tail])
    current_plot_excerpt = _truncate_text(current_plot_raw_text, 500) or "None"
    current_scene_boundary_summary = _current_scene_boundary_summary(scene_goal, scene_description, current_plot_raw_text)
    best_target = alignment.get("candidate_target")
    best_score = float(alignment.get("candidate_score", 0.0))
    candidate_transition_path = _normalize_transition_path((best_target or {}).get("transition_path", "stay"))
    case_acceptance = _is_case_acceptance(user_input)
    intro_auto_advance = _should_auto_advance_intro_plot(
        user_input=user_input,
        current_node_kind=current_node_kind,
        plot_goal=plot_goal,
        scene_goal=scene_goal,
        current_plot_raw_text=current_plot_raw_text,
        eligible_targets=eligible_targets,
    )
    deterministic_shortcut = _deterministic_pre_response_shortcut(
        user_input=user_input,
        current_node_kind=current_node_kind,
        choice_prompt_active=choice_prompt_active,
        alignment=alignment,
        best_target=best_target,
        best_score=best_score,
        candidate_transition_path=candidate_transition_path,
        eligible_targets=eligible_targets,
        eligible_indirect_targets=eligible_indirect_targets,
    )
    if deterministic_shortcut is not None:
        if prompt_recorder is not None:
            prompt_recorder(
                f"""
Deterministic pre-response transition shortcut applied.

Current Node Kind:
{current_node_kind}

Choice Prompt Active:
{str(bool(choice_prompt_active)).lower()}

Latest Player Input:
{user_input}

Latest Player Alignment:
{alignment.get('summary', 'None')}

Latest Agent Turn Excerpt:
{latest_agent_turn_excerpt or 'None'}

Eligible Targets:
{_target_lines(eligible_targets)}

Indirect Targets Via Return:
{_target_lines(eligible_indirect_targets)}

Selected Transition:
{json.dumps(deterministic_shortcut, ensure_ascii=False, indent=2)}
""".strip()
            )
        return deterministic_shortcut

    if exit_target and _is_exit_intent(user_input):
        explicit_exit_shortcut = {
            "action": "target",
            "target_kind": str(exit_target.get("target_kind", "")),
            "target_id": str(exit_target.get("target_id", "")),
            "transition_path": "direct",
            "close_current": True,
            "confidence": 1.0,
            "reason": "explicit_exit_intent",
        }
        if prompt_recorder is not None:
            prompt_recorder(
                f"""
Deterministic explicit-exit shortcut applied.

Latest Player Input:
{user_input}

Current Node Kind:
{current_node_kind}

Eligible Direct Exit:
{_target_lines([exit_target])}

Selected Transition:
{json.dumps(explicit_exit_shortcut, ensure_ascii=False, indent=2)}
""".strip()
            )
        return explicit_exit_shortcut

    if indirect_exit_target and _is_exit_intent(user_input):
        explicit_exit_shortcut = {
            "action": "target",
            "target_kind": str(indirect_exit_target.get("target_kind", "")),
            "target_id": str(indirect_exit_target.get("target_id", "")),
            "transition_path": "via_return",
            "close_current": True,
            "confidence": 1.0,
            "reason": "explicit_exit_intent_via_return",
        }
        if prompt_recorder is not None:
            prompt_recorder(
                f"""
Deterministic explicit-exit shortcut applied.

Latest Player Input:
{user_input}

Current Node Kind:
{current_node_kind}

Eligible Indirect Exit:
{_target_lines([indirect_exit_target])}

Selected Transition:
{json.dumps(explicit_exit_shortcut, ensure_ascii=False, indent=2)}
""".strip()
            )
        return explicit_exit_shortcut

    if (
        current_node_kind != "hub"
        and _is_exit_intent(user_input)
        and str(return_target.get("target_kind", "")).strip()
        and str(return_target.get("target_id", "")).strip()
        and bool(return_target.get("eligible", True))
    ):
        explicit_exit_shortcut = {
            "action": "target",
            "target_kind": str(return_target.get("target_kind", "")),
            "target_id": str(return_target.get("target_id", "")),
            "transition_path": "via_return",
            "close_current": True,
            "confidence": 1.0,
            "reason": "explicit_exit_to_return_target",
        }
        if prompt_recorder is not None:
            prompt_recorder(
                f"""
Deterministic explicit-exit shortcut applied.

Latest Player Input:
{user_input}

Current Node Kind:
{current_node_kind}

Legal Return Target:
{_target_lines([return_target])}

Selected Transition:
{json.dumps(explicit_exit_shortcut, ensure_ascii=False, indent=2)}
""".strip()
            )
        return explicit_exit_shortcut

    llm_prompt = f"""
You are a narrative-state evaluator for a script-driven investigation game.
Before the Keeper responds, decide whether the player's latest action clearly enters one of the explicitly allowed targets.

Return JSON only:
{{
  "action": "stay" | "target",
  "target_kind": "scene" | "plot" | "",
  "target_id": "id or empty string",
  "transition_path": "stay" | "direct" | "via_return",
  "close_current": true/false,
  "confidence": 0.0 to 1.0,
  "reason": "brief explanation"
}}

Rules:
- You may only choose targets from the Allowed Targets list.
- Only targets from the Eligible Targets list are currently enterable.
- You may choose a target from Indirect Targets Via Return only if the current plot can legally close first and the return hub can route there immediately.
- If the input is off-topic and unrelated to both the current plot and all eligible targets, you must return action=stay.
- Only choose action=target for a legal target choice that is clearly supported by the player's input.
- If Latest Agent Turn presented several options and the player is answering with a short phrase, map that answer to the closest legal target by label, goal, or excerpt even if the wording is shorter or in a different language.
- If the current node is a hub and the player chooses a branch, prefer close_current=false.
- If the current node is not a hub and the chosen target is only reachable via return, set transition_path=via_return and close_current=true.
- If the player names an exhausted or unavailable lead, return action=stay.
- If the current node is a hub and the player clearly wants to move on, you may select an eligible exit target.
- If Latest Agent Turn already presented a clear choice and the player is answering that choice, prefer transitioning now instead of repeating the same choice point.
- If the player is still working the current node, return action=stay.
- Keep output strict JSON.

Current Node Kind:
{current_node_kind}

Scene Goal:
{scene_goal or 'None'}

Current Plot Goal:
{plot_goal or 'None'}

Current Plot Excerpt:
{current_plot_excerpt}

Current Scene Boundary:
{current_scene_boundary_summary}

Allowed Targets:
{_target_lines(allowed_targets)}

Eligible Targets:
{_target_lines(eligible_targets)}

Indirect Targets Via Return:
{_target_lines(eligible_indirect_targets)}

Remaining Required Targets:
{_target_lines(remaining_required_targets)}

Mandatory Events:
{mandatory_events}

Redirect State:
redirect_streak={redirect_streak}

Latest Agent Turn Excerpt:
{latest_agent_turn_excerpt or 'None'}

Choice Prompt Active:
{str(bool(choice_prompt_active)).lower()}

Player Alignment Precheck:
{alignment.get('summary', 'None')}

Recent Conversation:
{history_text or 'None'}

Latest Player Input:
{user_input}
"""
    if prompt_recorder:
        prompt_recorder(llm_prompt)

    try:
        llm_raw = call_nvidia_llm(
            llm_prompt,
            model=PROGRESSION_MODEL,
            step_name="pre_response_transition_evaluation",
            allow_env_override=False,
        )
        parsed = _extract_json_obj(llm_raw)
        if parsed is not None:
            selected_target, selected_transition_path = _validate_selected_target(
                parsed,
                allowed_targets,
                indirect_targets_via_return,
            )
            if selected_target is not None:
                transition_path = selected_transition_path
                close_current = bool(parsed.get("close_current", _evaluate_close_current(current_node_kind, selected_target)))
                if transition_path == "via_return":
                    close_current = True
                if current_node_kind == "hub" and transition_path == "direct" and str(selected_target.get("role", "")) == "branch":
                    close_current = False
                return {
                    "action": "target",
                    "target_kind": str(selected_target.get("target_kind", "")),
                    "target_id": str(selected_target.get("target_id", "")),
                    "transition_path": transition_path,
                    "close_current": close_current,
                    "confidence": max(0.0, min(1.0, float(parsed.get("confidence", 0.0) or 0.0))),
                    "reason": str(parsed.get("reason", "")).strip() or "llm_evaluation",
                }
    except Exception:
        pass

    if choice_prompt_active and (eligible_targets or eligible_indirect_targets):
        choice_resolution_prompt = f"""
You are resolving the player's answer to the immediately previous choice prompt in a script-driven investigation game.

Return JSON only:
{{
  "action": "stay" | "target",
  "target_kind": "scene" | "plot" | "",
  "target_id": "id or empty string",
  "transition_path": "stay" | "direct" | "via_return",
  "close_current": true/false,
  "confidence": 0.0 to 1.0,
  "reason": "brief explanation"
}}

Rules:
- The player is most likely answering the immediately previous choice prompt.
- Map short answers like a noun phrase, location, clue name, topic, or action phrase to the closest legal target.
- You may only choose targets from Eligible Targets or Indirect Targets Via Return.
- If a target is only reachable after closing the current branch first, use transition_path=via_return and close_current=true.
- If the answer still does not match any legal target, return action=stay.
- Keep output strict JSON.

Latest Agent Turn Excerpt:
{latest_agent_turn_excerpt or 'None'}

Eligible Targets:
{_target_lines(eligible_targets)}

Indirect Targets Via Return:
{_target_lines(eligible_indirect_targets)}

Latest Player Input:
{user_input}
"""
        if prompt_recorder:
            prompt_recorder(choice_resolution_prompt)
        try:
            llm_raw = call_nvidia_llm(
                choice_resolution_prompt,
                model=PROGRESSION_MODEL,
                step_name="pre_response_transition_evaluation",
                allow_env_override=False,
            )
            parsed = _extract_json_obj(llm_raw)
            if parsed is not None:
                selected_target, selected_transition_path = _validate_selected_target(
                    parsed,
                    allowed_targets,
                    indirect_targets_via_return,
                )
                if selected_target is not None:
                    close_current = bool(parsed.get("close_current", _evaluate_close_current(current_node_kind, selected_target)))
                    if selected_transition_path == "via_return":
                        close_current = True
                    if current_node_kind == "hub" and selected_transition_path == "direct" and str(selected_target.get("role", "")) == "branch":
                        close_current = False
                    return {
                        "action": "target",
                        "target_kind": str(selected_target.get("target_kind", "")),
                        "target_id": str(selected_target.get("target_id", "")),
                        "transition_path": selected_transition_path,
                        "close_current": close_current,
                        "confidence": max(0.0, min(1.0, float(parsed.get("confidence", 0.0) or 0.0))),
                        "reason": str(parsed.get("reason", "")).strip() or "choice_prompt_resolution",
                    }
        except Exception:
            pass

    if intro_auto_advance:
        sole_target = eligible_targets[0]
        return {
            "action": "target",
            "target_kind": str(sole_target.get("target_kind", "")),
            "target_id": str(sole_target.get("target_id", "")),
            "transition_path": "direct",
            "close_current": True,
            "confidence": 0.95 if case_acceptance else 0.88,
            "reason": "intro_case_acceptance_transition" if case_acceptance else "intro_generic_reply_transition",
        }

    if exit_target and _is_exit_intent(user_input):
        return {
            "action": "target",
            "target_kind": str(exit_target.get("target_kind", "")),
            "target_id": str(exit_target.get("target_id", "")),
            "transition_path": "direct",
            "close_current": True,
            "confidence": 1.0,
            "reason": "explicit_exit_intent",
        }
    if indirect_exit_target and _is_exit_intent(user_input):
        return {
            "action": "target",
            "target_kind": str(indirect_exit_target.get("target_kind", "")),
            "target_id": str(indirect_exit_target.get("target_id", "")),
            "transition_path": "via_return",
            "close_current": True,
            "confidence": 1.0,
            "reason": "explicit_exit_intent_via_return",
        }
    if (
        current_node_kind != "hub"
        and _is_exit_intent(user_input)
        and str(return_target.get("target_kind", "")).strip()
        and str(return_target.get("target_id", "")).strip()
        and bool(return_target.get("eligible", True))
    ):
        return {
            "action": "target",
            "target_kind": str(return_target.get("target_kind", "")),
            "target_id": str(return_target.get("target_id", "")),
            "transition_path": "via_return",
            "close_current": True,
            "confidence": 1.0,
            "reason": "explicit_exit_to_return_target",
        }
    if bool(alignment.get("off_topic_unrelated")):
        return {
            "action": "stay",
            "target_kind": "",
            "target_id": "",
            "transition_path": "stay",
            "close_current": False,
            "confidence": 0.0,
            "reason": "off_topic_unrelated",
        }
    setup_like = _is_setup_or_choice_like(plot_goal, current_plot_raw_text, scene_goal)
    if (
        (bool(alignment.get("high_confidence_target_choice")) or choice_prompt_active)
        and best_target
        and (best_score >= HIGH_CONFIDENCE_TARGET_THRESHOLD or choice_prompt_active)
        and (setup_like or current_node_kind in {"hub", "branch"})
    ):
        close_current = _evaluate_close_current(current_node_kind, best_target)
        if candidate_transition_path == "via_return":
            close_current = True
        if current_node_kind == "hub" and candidate_transition_path == "direct" and str(best_target.get("role", "")) == "branch":
            close_current = False
        return {
            "action": "target",
            "target_kind": str(best_target.get("target_kind", "")),
            "target_id": str(best_target.get("target_id", "")),
            "transition_path": candidate_transition_path,
            "close_current": close_current,
            "confidence": max(best_score, 0.85 if choice_prompt_active else 0.0),
            "reason": f"fallback_target_match(score={best_score:.2f})",
        }

    return {
        "action": "stay",
        "target_kind": "",
        "target_id": "",
        "transition_path": "stay",
        "close_current": False,
        "confidence": 0.0,
        "reason": "fallback_continue_current_plot",
    }


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
