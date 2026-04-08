from __future__ import annotations

import json

from app.llm_client import call_nvidia_llm


def generate_retrieval_queries(
    user_input: str,
    plot_goal: str,
    mandatory_events: list[str],
    conversation_history: list[dict],
) -> list[str]:
    history_tail = "\n".join(turn.get("user", "") for turn in conversation_history[-4:])
    events_text = "\n".join(mandatory_events[:5])

    prompt = f"""
You are a narrative retrieval planner.

Your task is to generate retrieval queries for a story-driven RAG system.

Given:
- The current user input
- The current plot goal
- Mandatory upcoming events
- Recent conversation history

You must output a JSON object in the following format:

{{
  "queries": ["query1", "query2", "query3"]
}}

Guidelines:
- Only include queries that help maintain narrative consistency.
- Focus on relationships, world rules, setting details, unresolved plot threads.
- Do NOT explain anything.
- Output valid JSON only.

------------------------------------

User Input:
{user_input}

Plot Goal:
{plot_goal}

Mandatory Events:
{events_text}

Recent Conversation:
{history_tail}
"""

    try:
        response = call_nvidia_llm(prompt, step_name="generate_retrieval_queries")
    except Exception:
        response = ""

    try:
        data = json.loads(response)
        queries = data.get("queries", [])
        if isinstance(queries, list):
            return [q for q in queries if isinstance(q, str) and q.strip()]
    except Exception:
        pass

    fallback_queries = [user_input, plot_goal]
    return [q for q in fallback_queries if q]


def categorize_docs(docs: list[dict]) -> dict[str, str]:
    buckets: dict[str, list[str]] = {
        'npc': [],
        'player': [],
        'location': [],
        'rule': [],
        'item_or_clue': [],
        'world_context': [],
        'truth': [],
    }
    for d in docs:
        meta = d.get('metadata', {}) or {}
        t = meta.get('type', 'event')
        content = d.get('content', '')
        if t == 'npc':
            buckets['npc'].append(content)
        elif t == 'location':
            buckets['location'].append(content)
        elif t == 'rule':
            buckets['rule'].append(content)
        elif t in {'item', 'event', 'clue'}:
            buckets['item_or_clue'].append(content)
        elif t == 'world_context':
            knowledge_type = str(meta.get('knowledge_type', 'other')).strip().lower()
            if knowledge_type == 'truth':
                buckets['truth'].append(content)
            else:
                buckets['world_context'].append(f"[{knowledge_type}] {content}" if knowledge_type else content)

    return {
        'npc_related_info': '\n'.join(buckets['npc']) or 'None',
        'player_related_info': '\n'.join(buckets['player']) or 'None',
        'location_related_info': '\n'.join(buckets['location']) or 'None',
        'game_rule_info': '\n'.join(buckets['rule']) or 'None',
        'item_or_clue_info': '\n'.join(buckets['item_or_clue']) or 'None',
        'world_context_info': '\n'.join(buckets['world_context']) or 'None',
        'truth_related_info': '\n'.join(buckets['truth']) or 'None',
    }
