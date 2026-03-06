import argparse
import json
import sqlite3
from collections import defaultdict


def _safe_json_loads(value: str):
    try:
        return json.loads(value)
    except Exception:
        return value


def _scene_sort_key(scene_id: str):
    if scene_id.startswith('scene_'):
        tail = scene_id.split('_')[-1]
        if tail.isdigit():
            return (0, int(tail))
    return (1, scene_id)


def _truncate(text: str, max_len: int = 140) -> str:
    text = (text or '').replace('\n', ' ').strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + '...'


def main() -> int:
    parser = argparse.ArgumentParser(description='Inspect narrative SQLite database content.')
    parser.add_argument('--db', default='narrative.db', help='Path to SQLite database file.')
    parser.add_argument('--memory-limit', type=int, default=20, help='How many recent memory rows to show.')
    parser.add_argument('--summary-limit', type=int, default=20, help='How many summary rows to show.')
    parser.add_argument('--full-text', action='store_true', help='Show full text fields without truncation.')
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    print(f'DB: {args.db}')

    tables = ['scenes', 'plots', 'knowledge_base', 'memory', 'summaries', 'system_state']
    counts = {}
    for t in tables:
        try:
            counts[t] = cur.execute(f'SELECT COUNT(*) AS c FROM {t}').fetchone()['c']
        except Exception:
            counts[t] = 'N/A'

    print('\n=== TABLE COUNTS ===')
    for t in tables:
        print(f'- {t}: {counts[t]}')

    state = cur.execute('SELECT * FROM system_state WHERE id = 1').fetchone()
    print('\n=== SYSTEM STATE ===')
    if state:
        s = dict(state)
        s['player_profile'] = _safe_json_loads(s.get('player_profile', '{}') or '{}')
        print(s)
    else:
        print('No system_state row found (id=1).')

    parse_structure = cur.execute(
        "SELECT content FROM summaries WHERE summary_type='parse_structure' AND scene_id='' AND plot_id=''"
    ).fetchone()
    parse_warnings = cur.execute(
        "SELECT content FROM summaries WHERE summary_type='parse_warnings' AND scene_id='' AND plot_id=''"
    ).fetchone()
    parse_mode = cur.execute(
        "SELECT content FROM summaries WHERE summary_type='parse_mode' AND scene_id='' AND plot_id=''"
    ).fetchone()

    print('\n=== PARSE METADATA ===')
    if parse_mode:
        print('parse_mode:', parse_mode['content'])
    if parse_structure:
        print('structure:', _safe_json_loads(parse_structure['content']))
    if parse_warnings:
        print('warnings:', _safe_json_loads(parse_warnings['content']))

    scene_rows = [dict(r) for r in cur.execute('SELECT * FROM scenes').fetchall()]
    plot_rows = [dict(r) for r in cur.execute('SELECT * FROM plots').fetchall()]

    for p in plot_rows:
        p['mandatory_events'] = _safe_json_loads(p.get('mandatory_events', '[]') or '[]')
        p['npc'] = _safe_json_loads(p.get('npc', '[]') or '[]')
        p['locations'] = _safe_json_loads(p.get('locations', '[]') or '[]')

    plot_map = defaultdict(list)
    for p in plot_rows:
        plot_map[p['scene_id']].append(p)

    for sid in plot_map:
        plot_map[sid].sort(key=lambda x: x.get('plot_id', ''))

    scene_rows.sort(key=lambda x: _scene_sort_key(x.get('scene_id', '')))

    print('\n=== SCENES + PLOTS ===')
    if not scene_rows:
        print('No scenes.')
    for s in scene_rows:
        scene_description = s.get('scene_description', '')
        scene_summary = s.get('scene_summary', '')
        print(
            {
                'scene_id': s.get('scene_id', ''),
                'scene_goal': s.get('scene_goal', ''),
                'source_page_start': s.get('source_page_start', ''),
                'source_page_end': s.get('source_page_end', ''),
                'scene_description': scene_description if args.full_text else _truncate(scene_description),
                'scene_summary': scene_summary if args.full_text else _truncate(scene_summary),
                'status': s.get('status', ''),
                'plots': len(plot_map.get(s.get('scene_id', ''), [])),
            }
        )

        for p in plot_map.get(s.get('scene_id', ''), []):
            print(
                {
                    'plot_id': p.get('plot_id', ''),
                    'plot_goal': p.get('plot_goal', ''),
                    'source_page_start': p.get('source_page_start', ''),
                    'source_page_end': p.get('source_page_end', ''),
                    'mandatory_events': p.get('mandatory_events', []),
                    'npc': p.get('npc', []),
                    'locations': p.get('locations', []),
                    'status': p.get('status', ''),
                    'progress': p.get('progress', 0.0),
                }
            )

    print('\n=== KNOWLEDGE BASE ===')
    knowledge_rows = [
        dict(r)
        for r in cur.execute(
            'SELECT * FROM knowledge_base ORDER BY source_page_start, source_page_end, knowledge_id'
        ).fetchall()
    ]
    if not knowledge_rows:
        print('No knowledge records.')
    for k in knowledge_rows:
        metadata = _safe_json_loads(k.get('metadata', '{}') or '{}')
        print(
            {
                'knowledge_id': k.get('knowledge_id', ''),
                'knowledge_type': k.get('knowledge_type', ''),
                'title': k.get('title', ''),
                'content': k.get('content', '') if args.full_text else _truncate(k.get('content', '')),
                'source_page_start': k.get('source_page_start', ''),
                'source_page_end': k.get('source_page_end', ''),
                'metadata': metadata,
            }
        )

    print('\n=== RECENT MEMORY ===')
    memory_rows = [
        dict(r)
        for r in cur.execute(
            'SELECT id, scene_id, plot_id, user, agent, timestamp FROM memory ORDER BY id DESC LIMIT ?',
            (max(0, args.memory_limit),),
        ).fetchall()
    ]
    if not memory_rows:
        print('No memory rows.')
    for m in reversed(memory_rows):
        print(
            {
                'id': m.get('id', ''),
                'scene_id': m.get('scene_id', ''),
                'plot_id': m.get('plot_id', ''),
                'user': m.get('user', '') if args.full_text else _truncate(m.get('user', '')),
                'agent': m.get('agent', '') if args.full_text else _truncate(m.get('agent', '')),
                'timestamp': m.get('timestamp', ''),
            }
        )

    print('\n=== SUMMARIES ===')
    summary_rows = [
        dict(r)
        for r in cur.execute(
            'SELECT id, summary_type, scene_id, plot_id, content FROM summaries ORDER BY id DESC LIMIT ?',
            (max(0, args.summary_limit),),
        ).fetchall()
    ]
    if not summary_rows:
        print('No summary rows.')
    for sm in reversed(summary_rows):
        print(
            {
                'id': sm.get('id', ''),
                'summary_type': sm.get('summary_type', ''),
                'scene_id': sm.get('scene_id', ''),
                'plot_id': sm.get('plot_id', ''),
                'content': sm.get('content', '') if args.full_text else _truncate(sm.get('content', '')),
            }
        )

    conn.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
