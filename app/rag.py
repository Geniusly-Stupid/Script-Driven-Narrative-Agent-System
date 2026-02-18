from __future__ import annotations


def generate_retrieval_queries(
    user_input: str,
    plot_goal: str,
    mandatory_events: list[str],
    conversation_history: list[dict],
) -> list[str]:
    history_tail = ' '.join(turn.get('user', '') for turn in conversation_history[-4:])
    events_text = '; '.join(mandatory_events[:3])
    queries = [user_input, plot_goal, events_text, history_tail]
    return [q for q in queries if q]


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
