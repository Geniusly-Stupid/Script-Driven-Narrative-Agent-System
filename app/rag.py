from __future__ import annotations
import json
from app.llm_client import call_nvidia_llm


def generate_retrieval_queries(
    user_input: str,
    plot_goal: str,
    mandatory_events: list[str],
    conversation_history: list[dict],
) -> list[str]:

    # ===== 1. 构建输入信息 =====

    history_tail = "\n".join(
        turn.get("user", "") for turn in conversation_history[-4:]
    )

    events_text = "\n".join(mandatory_events[:5])

    # ===== 2. Prompt 模板 =====

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

    # ===== 3. 调用 LLM =====

    try:
        response = call_nvidia_llm(prompt)
    except Exception:
        response = ""
    # print("response: \n", response)

    # ===== 4. 解析 JSON =====

    try:
        data = json.loads(response)
        queries = data.get("queries", [])
        if isinstance(queries, list):
            return [q for q in queries if isinstance(q, str) and q.strip()]
    except Exception:
        pass

    # ===== 5. fallback（极重要） =====
    # 防止模型输出不规范导致系统崩溃

    fallback_queries = [user_input, plot_goal]
    return [q for q in fallback_queries if q]


def categorize_docs(docs: list[dict]) -> dict[str, str]:
    buckets: dict[str, list[str]] = {
        'npc': [],
        'player': [],
        'location': [],
        'rule': [],
        'item_or_clue': [],
    }
    for d in docs:
        t = d.get('metadata', {}).get('type', 'event')
        content = d.get('content', '')
        if t == 'npc':
            buckets['npc'].append(content)
        elif t == 'location':
            buckets['location'].append(content)
        elif t == 'rule':
            buckets['rule'].append(content)
        elif t in ['item', 'event']:
            buckets['item_or_clue'].append(content)

    return {
        'npc_related_info': '\n'.join(buckets['npc']) or 'None',
        'player_related_info': '\n'.join(buckets['player']) or 'None',
        'location_related_info': '\n'.join(buckets['location']) or 'None',
        'game_rule_info': '\n'.join(buckets['rule']) or 'None',
        'item_or_clue_info': '\n'.join(buckets['item_or_clue']) or 'None',
    }
