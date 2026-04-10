from __future__ import annotations

import json

from app.llm_client import call_nvidia_llm


def generate_retrieval_queries(
    user_input: str,
    plot_goal: str,
    conversation_history: list[dict],
) -> list[str]:
    history_tail = "\n".join(turn.get("user", "") for turn in conversation_history[-4:])

    prompt = f"""
You are a narrative retrieval planner.

Your task is to generate retrieval queries for a story-driven RAG system.

Given:
- The current user input
- The current plot goal
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
        'setting': [],
        'clue': [],
    }
    for doc in docs:
        meta = doc.get('metadata', {}) or {}
        raw_type = str(meta.get('type') or meta.get('knowledge_type') or 'other').strip().lower()
        content = str(doc.get('content', '')).strip()
        if not content:
            continue
        if raw_type == 'npc':
            buckets['npc'].append(content)
            continue
        if raw_type in {'clue', 'item', 'event'}:
            buckets['clue'].append(content)
            continue
        # The current schema emits setting|npc|clue|other. Any legacy knowledge payloads are
        # folded into setting so the prompt surface stays on the current schema.
        buckets['setting'].append(content)

    return {
        'npc_related_info': '\n'.join(buckets['npc']) or 'None',
        'setting': '\n'.join(buckets['setting']) or 'None',
        'clue': '\n'.join(buckets['clue']) or 'None',
    }
