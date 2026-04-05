import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.database import Database
from app.navigation import default_navigation, make_target


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
    snapshot_path = db_path.with_name(f'{db_path.stem}.initial_story_snapshot{db_path.suffix}')
    print(f'[test_database] input: db_path={db_path}')

    try:
        if db_path.exists():
            os.remove(db_path)
        if snapshot_path.exists():
            os.remove(snapshot_path)

        _create_legacy_db(db_path)
        db = Database(str(db_path))
        print('[test_database] output: initialized and migrated SQLite successfully')

        scene_cols = [r['name'] for r in db.conn.execute('PRAGMA table_info(scenes)').fetchall()]
        plot_cols = [r['name'] for r in db.conn.execute('PRAGMA table_info(plots)').fetchall()]
        memory_cols = [r['name'] for r in db.conn.execute('PRAGMA table_info(memory)').fetchall()]
        system_cols = [r['name'] for r in db.conn.execute('PRAGMA table_info(system_state)').fetchall()]
        print('[test_database] output: scenes columns ->', scene_cols)
        print('[test_database] output: plots columns ->', plot_cols)

        assert 'scene_description' in scene_cols, 'scene_description column migration failed'
        assert 'node_kind' in scene_cols and 'navigation_json' in scene_cols, 'scene navigation migration failed'
        assert 'source_page_start' in scene_cols and 'source_page_end' in scene_cols, 'scene span columns migration failed'
        assert 'node_kind' in plot_cols and 'navigation_json' in plot_cols, 'plot navigation migration failed'
        assert 'source_page_start' in plot_cols and 'source_page_end' in plot_cols, 'plot span columns migration failed'
        assert 'raw_text' in plot_cols, 'plot raw_text column migration failed'
        assert 'visit_id' in memory_cols, 'memory visit_id migration failed'
        assert 'navigation_state_json' in system_cols and 'current_visit_id' in system_cols, 'system navigation migration failed'

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
                'node_kind': 'hub',
                'navigation': {
                    **default_navigation(completion_policy='all_required_then_advance'),
                    'allowed_targets': [
                        make_target('plot', 'scene_t1_plot_2', label='Test plot write 2', role='branch', required=True)
                    ],
                },
                'plots': [
                    {
                        'plot_id': 'scene_t1_plot_1',
                        'plot_goal': 'Test plot write',
                        'mandatory_events': ['event_a', 'event_b'],
                        'npc': ['Alice'],
                        'locations': ['Hall'],
                        'raw_text': 'Line 3\nLine 4',
                        'source_page_start': 3,
                        'source_page_end': 4,
                        'status': 'pending',
                        'progress': 0.0,
                        'node_kind': 'hub',
                        'navigation': {
                            **default_navigation(completion_policy='all_required_then_advance'),
                            'allowed_targets': [
                                make_target('plot', 'scene_t1_plot_2', label='Test plot write 2', role='branch', required=True)
                            ],
                        },
                    },
                    {
                        'plot_id': 'scene_t1_plot_2',
                        'plot_goal': 'Test plot write 2',
                        'mandatory_events': ['event_c'],
                        'npc': ['Bob'],
                        'locations': ['Street'],
                        'raw_text': 'Line 5',
                        'source_page_start': 5,
                        'source_page_end': 5,
                        'status': 'pending',
                        'progress': 0.0,
                        'node_kind': 'branch',
                        'navigation': {
                            **default_navigation(completion_policy='terminal_on_resolve'),
                            'return_target': make_target('plot', 'scene_t1_plot_1', label='Test plot write', role='return'),
                        },
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
        assert fetched.get('node_kind') == 'hub', 'scene node_kind persistence failed'
        assert fetched.get('navigation', {}).get('completion_policy') == 'all_required_then_advance', 'scene navigation persistence failed'
        assert len(fetched.get('plots', [])) == 2
        assert fetched['plots'][0].get('source_page_start') == 3
        assert fetched['plots'][0].get('raw_text') == 'Line 3\nLine 4', 'plot raw_text persistence failed'
        assert fetched['plots'][0].get('node_kind') == 'hub', 'plot node_kind persistence failed'
        assert fetched['plots'][1].get('navigation', {}).get('return_target', {}).get('target_id') == 'scene_t1_plot_1', 'plot navigation persistence failed'

        db.append_memory('scene_t1', 'scene_t1_plot_1', 'hello', 'world', visit_id=7)
        turns = db.get_recent_turns('scene_t1', 'scene_t1_plot_1', limit=5, visit_id=7)
        assert turns and turns[0]['visit_id'] == 7, 'memory visit_id persistence failed'

        db.update_system_state({'navigation_state': {'return_stack': [{'target_kind': 'plot', 'target_id': 'scene_t1_plot_1', 'visit_id': 7}]}, 'current_visit_id': 7})
        system_state = db.get_system_state()
        assert system_state['current_visit_id'] == 7, 'current_visit_id persistence failed'
        assert system_state['navigation_state']['return_stack'][0]['target_id'] == 'scene_t1_plot_1', 'navigation_state persistence failed'

        fetched_knowledge = db.get_knowledge_by_type('setting')
        print('[test_database] output: fetched knowledge ->', fetched_knowledge)
        assert len(fetched_knowledge) == 1, 'knowledge retrieval failed'

        print('[test_database] input: save initial story snapshot')
        db.update_scene('scene_t1', {'status': 'in_progress'})
        db.update_system_state(
            {
                'stage': 'parse',
                'current_scene_id': 'scene_t1',
                'current_plot_id': 'scene_t1_plot_1',
                'plot_progress': 0.0,
                'scene_progress': 0.0,
                'player_profile': {},
                'current_scene_intro': '',
                'navigation_state': {},
                'current_visit_id': 0,
            }
        )
        db.save_summary('parse_structure', '{"story":"initial"}')
        pre_snapshot_turns = db.get_recent_turns('scene_t1', 'scene_t1_plot_1', limit=20)
        saved_snapshot = Path(db.save_initial_story_snapshot())
        assert saved_snapshot == snapshot_path and snapshot_path.exists(), 'initial story snapshot save failed'

        print('[test_database] input: mutate runtime state after snapshot')
        db.append_memory('scene_t1', 'scene_t1_plot_1', 'new run user', 'new run agent', visit_id=9)
        db.save_summary('plot', 'runtime summary', scene_id='scene_t1', plot_id='scene_t1_plot_1')
        db.update_plot('scene_t1_plot_1', status='completed', progress=1.0)
        db.update_scene('scene_t1', {'status': 'completed'})
        db.update_system_state(
            {
                'stage': 'session',
                'plot_progress': 1.0,
                'scene_progress': 1.0,
                'player_profile': {'name': 'Tester'},
                'navigation_state': {'visited_scene_ids': ['scene_t1']},
                'current_visit_id': 9,
            }
        )

        print('[test_database] input: restore initial story snapshot')
        restored_snapshot = Path(db.restore_initial_story_snapshot())
        restored_state = db.get_system_state()
        restored_scene = db.get_scene('scene_t1')
        restored_plot = db.get_plot('scene_t1_plot_1')
        restored_turns = db.get_recent_turns('scene_t1', 'scene_t1_plot_1', limit=20)
        assert restored_snapshot == snapshot_path, 'initial story snapshot restore returned wrong path'
        assert restored_state['stage'] == 'parse', 'snapshot restore should recover parse stage'
        assert restored_state['current_scene_id'] == 'scene_t1', 'snapshot restore should recover current scene'
        assert restored_state['current_plot_id'] == 'scene_t1_plot_1', 'snapshot restore should recover current plot'
        assert restored_state['player_profile'] == {}, 'snapshot restore should discard runtime player data'
        assert restored_state['current_visit_id'] == 0, 'snapshot restore should reset runtime visit id'
        assert restored_state['navigation_state'] == {}, 'snapshot restore should reset runtime navigation state'
        assert restored_scene is not None and restored_scene.get('status') == 'in_progress', 'snapshot restore should recover scene status'
        assert restored_plot is not None and restored_plot.get('status') == 'pending', 'snapshot restore should recover plot status'
        assert restored_plot is not None and float(restored_plot.get('progress', 0.0)) == 0.0, 'snapshot restore should recover plot progress'
        assert restored_turns == pre_snapshot_turns, 'snapshot restore should discard only post-snapshot memory'
        assert db.get_summary('plot', scene_id='scene_t1', plot_id='scene_t1_plot_1') == '', 'snapshot restore should discard runtime summaries'
        assert db.get_summary('parse_structure') == '{"story":"initial"}', 'snapshot restore should keep parse summaries'

        print('[test_database] input: reset story data')
        db.reset_story_data()
        scene_count = db.conn.execute('SELECT COUNT(*) AS c FROM scenes').fetchone()['c']
        knowledge_count = db.conn.execute('SELECT COUNT(*) AS c FROM knowledge_base').fetchone()['c']
        print('[test_database] output: counts after reset ->', {'scenes': scene_count, 'knowledge': knowledge_count})
        assert scene_count == 0 and knowledge_count == 0, 'reset_story_data should clear scenes and knowledge'

        db.close()
        db.delete_initial_story_snapshot()
        print('[test_database] result: PASS')
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f'[test_database] result: FAIL -> {exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
