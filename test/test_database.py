import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.database import Database


def _create_legacy_db(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        '''
        CREATE TABLE IF NOT EXISTS scenes (
            scene_id TEXT PRIMARY KEY,
            scene_goal TEXT NOT NULL,
            status TEXT NOT NULL,
            scene_summary TEXT NOT NULL DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS plots (
            plot_id TEXT PRIMARY KEY,
            scene_id TEXT NOT NULL,
            plot_goal TEXT NOT NULL,
            mandatory_events TEXT NOT NULL,
            npc TEXT NOT NULL,
            locations TEXT NOT NULL,
            status TEXT NOT NULL,
            progress REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scene_id TEXT NOT NULL,
            plot_id TEXT NOT NULL,
            user TEXT NOT NULL,
            agent TEXT NOT NULL,
            timestamp TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            summary_type TEXT NOT NULL,
            scene_id TEXT,
            plot_id TEXT,
            content TEXT NOT NULL,
            UNIQUE(summary_type, scene_id, plot_id)
        );

        CREATE TABLE IF NOT EXISTS system_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            stage TEXT NOT NULL,
            current_scene_id TEXT NOT NULL,
            current_plot_id TEXT NOT NULL,
            plot_progress REAL NOT NULL,
            scene_progress REAL NOT NULL,
            player_profile TEXT NOT NULL,
            current_scene_intro TEXT NOT NULL
        );
        '''
    )
    conn.execute(
        "INSERT OR REPLACE INTO system_state (id, stage, current_scene_id, current_plot_id, plot_progress, scene_progress, player_profile, current_scene_intro) VALUES (1, 'upload', '', '', 0.0, 0.0, '{}', '')"
    )
    conn.commit()
    conn.close()


def main() -> int:
    db_path = ROOT / 'test' / 'debug_narrative.db'
    print(f'[test_database] input: db_path={db_path}')

    try:
        if db_path.exists():
            os.remove(db_path)

        _create_legacy_db(db_path)
        db = Database(str(db_path))
        print('[test_database] output: initialized and migrated SQLite successfully')

        scene_cols = [r['name'] for r in db.conn.execute('PRAGMA table_info(scenes)').fetchall()]
        plot_cols = [r['name'] for r in db.conn.execute('PRAGMA table_info(plots)').fetchall()]
        print('[test_database] output: scenes columns ->', scene_cols)
        print('[test_database] output: plots columns ->', plot_cols)

        assert 'scene_description' in scene_cols, 'scene_description column migration failed'
        assert 'source_page_start' in scene_cols and 'source_page_end' in scene_cols, 'scene span columns migration failed'
        assert 'source_page_start' in plot_cols and 'source_page_end' in plot_cols, 'plot span columns migration failed'

        knowledge_table = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='knowledge_base'"
        ).fetchone()
        assert knowledge_table is not None, 'knowledge_base table migration failed'

        scenes = [
            {
                'scene_id': 'scene_t1',
                'scene_goal': 'Test scene write',
                'scene_description': 'A detailed paragraph that guides pacing and choices.',
                'source_page_start': 3,
                'source_page_end': 5,
                'status': 'pending',
                'scene_summary': '',
                'plots': [
                    {
                        'plot_id': 'scene_t1_plot_1',
                        'plot_goal': 'Test plot write',
                        'mandatory_events': ['event_a', 'event_b'],
                        'npc': ['Alice'],
                        'locations': ['Hall'],
                        'source_page_start': 3,
                        'source_page_end': 4,
                        'status': 'pending',
                        'progress': 0.0,
                    },
                    {
                        'plot_id': 'scene_t1_plot_2',
                        'plot_goal': 'Test plot write 2',
                        'mandatory_events': ['event_c'],
                        'npc': ['Bob'],
                        'locations': ['Street'],
                        'source_page_start': 5,
                        'source_page_end': 5,
                        'status': 'pending',
                        'progress': 0.0,
                    },
                ],
            }
        ]
        print('[test_database] input: insert scenes')
        db.insert_scenes(scenes)

        knowledge_items = [
            {
                'knowledge_id': 'knowledge_1',
                'knowledge_type': 'setting',
                'title': 'Town Rule',
                'content': 'No one leaves after dusk.',
                'source_page_start': 1,
                'source_page_end': 1,
                'metadata': {'source': 'preface'},
            }
        ]
        print('[test_database] input: insert knowledge')
        db.insert_knowledge(knowledge_items)

        fetched = db.get_scene('scene_t1')
        print('[test_database] output: fetched scene ->', fetched)
        assert fetched is not None and fetched.get('scene_description'), 'scene_description persistence failed'
        assert fetched.get('source_page_start') == 3 and fetched.get('source_page_end') == 5, 'scene span persistence failed'
        assert len(fetched.get('plots', [])) == 2
        assert fetched['plots'][0].get('source_page_start') == 3

        fetched_knowledge = db.get_knowledge_by_type('setting')
        print('[test_database] output: fetched knowledge ->', fetched_knowledge)
        assert len(fetched_knowledge) == 1, 'knowledge retrieval failed'

        print('[test_database] input: reset story data')
        db.reset_story_data()
        scene_count = db.conn.execute('SELECT COUNT(*) AS c FROM scenes').fetchone()['c']
        knowledge_count = db.conn.execute('SELECT COUNT(*) AS c FROM knowledge_base').fetchone()['c']
        print('[test_database] output: counts after reset ->', {'scenes': scene_count, 'knowledge': knowledge_count})
        assert scene_count == 0 and knowledge_count == 0, 'reset_story_data should clear scenes and knowledge'

        db.close()
        print('[test_database] result: PASS')
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f'[test_database] result: FAIL -> {exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
