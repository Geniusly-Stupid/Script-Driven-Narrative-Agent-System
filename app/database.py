from __future__ import annotations

import json
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from app.navigation import ensure_scene_navigation_defaults, parse_json_object, serialize_navigation


class Database:
    def __init__(self, db_path: str = 'narrative.db') -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.executescript(
            '''
            CREATE TABLE IF NOT EXISTS scenes (
                scene_id TEXT PRIMARY KEY,
                scene_goal TEXT NOT NULL,
                status TEXT NOT NULL,
                scene_summary TEXT NOT NULL DEFAULT '',
                scene_description TEXT NOT NULL DEFAULT '',
                node_kind TEXT NOT NULL DEFAULT 'linear',
                navigation_json TEXT NOT NULL DEFAULT '{}',
                source_page_start INTEGER NOT NULL DEFAULT 1,
                source_page_end INTEGER NOT NULL DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS plots (
                plot_id TEXT PRIMARY KEY,
                scene_id TEXT NOT NULL,
                plot_goal TEXT NOT NULL,
                mandatory_events TEXT NOT NULL,
                npc TEXT NOT NULL,
                locations TEXT NOT NULL,
                raw_text TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL,
                progress REAL NOT NULL,
                node_kind TEXT NOT NULL DEFAULT 'linear',
                navigation_json TEXT NOT NULL DEFAULT '{}',
                source_page_start INTEGER NOT NULL DEFAULT 1,
                source_page_end INTEGER NOT NULL DEFAULT 1,
                FOREIGN KEY(scene_id) REFERENCES scenes(scene_id)
            );

            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scene_id TEXT NOT NULL,
                plot_id TEXT NOT NULL,
                user TEXT NOT NULL,
                agent TEXT NOT NULL,
                turn_state_json TEXT NOT NULL DEFAULT '{}',
                visit_id INTEGER NOT NULL DEFAULT 0,
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
                current_scene_intro TEXT NOT NULL,
                output_language TEXT NOT NULL DEFAULT 'English',
                navigation_state_json TEXT NOT NULL DEFAULT '{}',
                current_visit_id INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS knowledge_base (
                knowledge_id TEXT PRIMARY KEY,
                knowledge_type TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                source_page_start INTEGER NOT NULL,
                source_page_end INTEGER NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}'
            );
            '''
        )
        self.conn.commit()
        self._migrate_schema()

        if not self.conn.execute('SELECT 1 FROM system_state WHERE id = 1').fetchone():
            self.conn.execute(
                '''
                INSERT INTO system_state (
                    id,
                    stage,
                    current_scene_id,
                    current_plot_id,
                    plot_progress,
                    scene_progress,
                    player_profile,
                    current_scene_intro,
                    output_language,
                    navigation_state_json,
                    current_visit_id
                )
                VALUES (1, 'upload', '', '', 0.0, 0.0, '{}', '', 'English', '{}', 0)
                '''
            )
            self.conn.commit()
    def _migrate_schema(self) -> None:
        self._ensure_column('scenes', "scene_description TEXT NOT NULL DEFAULT ''")
        self._ensure_column('scenes', "node_kind TEXT NOT NULL DEFAULT 'linear'")
        self._ensure_column('scenes', "navigation_json TEXT NOT NULL DEFAULT '{}'")
        self._ensure_column('scenes', 'source_page_start INTEGER NOT NULL DEFAULT 1')
        self._ensure_column('scenes', 'source_page_end INTEGER NOT NULL DEFAULT 1')
        self._ensure_column('plots', "node_kind TEXT NOT NULL DEFAULT 'linear'")
        self._ensure_column('plots', "navigation_json TEXT NOT NULL DEFAULT '{}'")
        self._ensure_column('plots', 'source_page_start INTEGER NOT NULL DEFAULT 1')
        self._ensure_column('plots', 'source_page_end INTEGER NOT NULL DEFAULT 1')
        self._ensure_column('plots', "raw_text TEXT NOT NULL DEFAULT ''")
        self._ensure_column('memory', 'visit_id INTEGER NOT NULL DEFAULT 0')
        self._ensure_column('memory', "turn_state_json TEXT NOT NULL DEFAULT '{}'")
        self._ensure_column('system_state', "output_language TEXT NOT NULL DEFAULT 'English'")
        self._ensure_column('system_state', "navigation_state_json TEXT NOT NULL DEFAULT '{}'")
        self._ensure_column('system_state', 'current_visit_id INTEGER NOT NULL DEFAULT 0')

        self.conn.execute(
            '''
            CREATE TABLE IF NOT EXISTS knowledge_base (
                knowledge_id TEXT PRIMARY KEY,
                knowledge_type TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                source_page_start INTEGER NOT NULL,
                source_page_end INTEGER NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}'
            )
            '''
        )
        self.conn.commit()
        self._backfill_linear_navigation_metadata()

    def _ensure_column(self, table: str, column_def: str) -> None:
        col_name = column_def.split()[0]
        rows = self.conn.execute(f'PRAGMA table_info({table})').fetchall()
        existing_cols = {r['name'] for r in rows}
        if col_name not in existing_cols:
            self.conn.execute(f'ALTER TABLE {table} ADD COLUMN {column_def}')

    def close(self) -> None:
        self.conn.close()

    def initial_story_snapshot_path(self) -> Path:
        db_file = Path(self.db_path)
        suffix = db_file.suffix or '.db'
        stem = db_file.stem if db_file.suffix else db_file.name
        return db_file.with_name(f'{stem}.initial_story_snapshot{suffix}')

    def has_initial_story_snapshot(self) -> bool:
        return self.initial_story_snapshot_path().exists()

    def delete_initial_story_snapshot(self) -> None:
        snapshot_path = self.initial_story_snapshot_path()
        if not snapshot_path.exists():
            return
        for _ in range(10):
            try:
                snapshot_path.unlink()
                return
            except FileNotFoundError:
                return
            except PermissionError:
                time.sleep(0.1)

    def save_initial_story_snapshot(self) -> str:
        snapshot_path = self.initial_story_snapshot_path()
        self.conn.commit()
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(snapshot_path)) as snapshot_conn:
            self.conn.backup(snapshot_conn)
        return str(snapshot_path)

    def restore_initial_story_snapshot(self) -> str:
        snapshot_path = self.initial_story_snapshot_path()
        if not snapshot_path.exists():
            raise FileNotFoundError(f'Initial story snapshot not found: {snapshot_path}')
        self.conn.commit()
        with sqlite3.connect(str(snapshot_path)) as snapshot_conn:
            snapshot_conn.backup(self.conn)
        self.conn.commit()
        return str(snapshot_path)

    def reset_story_data(self) -> None:
        cur = self.conn.cursor()
        cur.execute('DELETE FROM scenes')
        cur.execute('DELETE FROM plots')
        cur.execute('DELETE FROM memory')
        cur.execute('DELETE FROM summaries')
        cur.execute('DELETE FROM knowledge_base')
        cur.execute(
            '''
            UPDATE system_state
            SET current_scene_id = '',
                current_plot_id = '',
                plot_progress = 0.0,
                scene_progress = 0.0,
                current_scene_intro = '',
                navigation_state_json = '{}',
                current_visit_id = 0
            WHERE id = 1
            '''
        )
        self.conn.commit()

    def insert_scenes(self, scenes: list[dict[str, Any]]) -> None:
        cur = self.conn.cursor()
        for scene in scenes:
            cur.execute(
                '''
                INSERT OR REPLACE INTO scenes(
                    scene_id,
                    scene_goal,
                    status,
                    scene_summary,
                    scene_description,
                    node_kind,
                    navigation_json,
                    source_page_start,
                    source_page_end
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                (
                    scene['scene_id'],
                    scene['scene_goal'],
                    scene.get('status', 'pending'),
                    scene.get('scene_summary', ''),
                    scene.get('scene_description', ''),
                    scene.get('node_kind', 'linear'),
                    serialize_navigation(scene.get('navigation')),
                    int(scene.get('source_page_start', 1)),
                    int(scene.get('source_page_end', scene.get('source_page_start', 1))),
                ),
            )
            for plot in scene.get('plots', []):
                cur.execute(
                    '''
                    INSERT OR REPLACE INTO plots(
                        plot_id,
                        scene_id,
                        plot_goal,
                        mandatory_events,
                        npc,
                        locations,
                        raw_text,
                        status,
                        progress,
                        node_kind,
                        navigation_json,
                        source_page_start,
                        source_page_end
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''',
                    (
                        plot['plot_id'],
                        scene['scene_id'],
                        plot['plot_goal'],
                        json.dumps(plot.get('mandatory_events', []), ensure_ascii=False),
                        json.dumps(plot.get('npc', []), ensure_ascii=False),
                        json.dumps(plot.get('locations', []), ensure_ascii=False),
                        plot.get('raw_text', ''),
                        plot.get('status', 'pending'),
                        float(plot.get('progress', 0.0)),
                        plot.get('node_kind', 'linear'),
                        serialize_navigation(plot.get('navigation')),
                        int(plot.get('source_page_start', scene.get('source_page_start', 1))),
                        int(plot.get('source_page_end', plot.get('source_page_start', scene.get('source_page_end', 1)))),
                    ),
                )
        self.conn.commit()
        self._backfill_linear_navigation_metadata()

    def insert_knowledge(self, knowledge_items: list[dict[str, Any]]) -> None:
        cur = self.conn.cursor()
        for item in knowledge_items:
            metadata = item.get('metadata', {})
            if not isinstance(metadata, dict):
                metadata = {}
            cur.execute(
                '''
                INSERT OR REPLACE INTO knowledge_base(
                    knowledge_id,
                    knowledge_type,
                    title,
                    content,
                    source_page_start,
                    source_page_end,
                    metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''',
                (
                    item['knowledge_id'],
                    item.get('knowledge_type', 'other'),
                    item.get('title', ''),
                    item.get('content', ''),
                    int(item.get('source_page_start', 1)),
                    int(item.get('source_page_end', item.get('source_page_start', 1))),
                    json.dumps(metadata, ensure_ascii=False),
                ),
            )
        self.conn.commit()

    def list_knowledge(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            'SELECT * FROM knowledge_base ORDER BY source_page_start, source_page_end, knowledge_id'
        ).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item['metadata'] = json.loads(item.get('metadata', '{}') or '{}')
            out.append(item)
        return out

    def get_knowledge_by_type(self, knowledge_type: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            'SELECT * FROM knowledge_base WHERE knowledge_type = ? ORDER BY source_page_start, source_page_end, knowledge_id',
            (knowledge_type,),
        ).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item['metadata'] = json.loads(item.get('metadata', '{}') or '{}')
            out.append(item)
        return out

    def list_scenes(self) -> list[dict[str, Any]]:
        scenes = []
        for row in self.conn.execute('SELECT * FROM scenes').fetchall():
            scene = self._hydrate_scene_row(dict(row))
            scene['plots'] = self._plots_for_scene(scene['scene_id'])
            scenes.append(scene)
        scenes.sort(key=lambda s: self._natural_id_key(str(s.get('scene_id', ''))))
        return scenes

    def get_scene(self, scene_id: str) -> dict[str, Any] | None:
        row = self.conn.execute('SELECT * FROM scenes WHERE scene_id = ?', (scene_id,)).fetchone()
        if not row:
            return None
        scene = self._hydrate_scene_row(dict(row))
        scene['plots'] = self._plots_for_scene(scene_id)
        return scene

    def _plots_for_scene(self, scene_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute('SELECT * FROM plots WHERE scene_id = ?', (scene_id,)).fetchall()
        plots = []
        for row in rows:
            p = self._hydrate_plot_row(dict(row))
            p['mandatory_events'] = json.loads(p['mandatory_events'])
            p['npc'] = json.loads(p['npc'])
            p['locations'] = json.loads(p['locations'])
            plots.append(p)
        plots.sort(key=lambda p: self._natural_id_key(str(p.get('plot_id', ''))))
        return plots

    def get_plot(self, plot_id: str) -> dict[str, Any] | None:
        row = self.conn.execute('SELECT * FROM plots WHERE plot_id = ?', (plot_id,)).fetchone()
        if not row:
            return None
        p = self._hydrate_plot_row(dict(row))
        p['mandatory_events'] = json.loads(p['mandatory_events'])
        p['npc'] = json.loads(p['npc'])
        p['locations'] = json.loads(p['locations'])
        return p

    def update_plot(self, plot_id: str, status: str | None = None, progress: float | None = None) -> None:
        if status is not None:
            self.conn.execute('UPDATE plots SET status = ? WHERE plot_id = ?', (status, plot_id))
        if progress is not None:
            self.conn.execute('UPDATE plots SET progress = ? WHERE plot_id = ?', (progress, plot_id))
        self.conn.commit()

    def update_scene(self, scene_id: str, updates: dict[str, Any]) -> None:
        for key, value in updates.items():
            self.conn.execute(f'UPDATE scenes SET {key} = ? WHERE scene_id = ?', (value, scene_id))
        self.conn.commit()

    def append_memory(
        self,
        scene_id: str,
        plot_id: str,
        user: str,
        agent: str,
        visit_id: int = 0,
        *,
        turn_state: dict[str, Any] | None = None,
    ) -> None:
        self.conn.execute(
            'INSERT INTO memory(scene_id, plot_id, user, agent, turn_state_json, visit_id, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)',
            (
                scene_id,
                plot_id,
                user,
                agent,
                json.dumps(turn_state or {}, ensure_ascii=False),
                int(visit_id),
                datetime.utcnow().isoformat(),
            ),
        )
        self.conn.commit()

    def get_recent_turns(
        self,
        scene_id: str,
        plot_id: str,
        limit: int = 12,
        *,
        visit_id: int | None = None,
    ) -> list[dict[str, Any]]:
        if visit_id is None:
            rows = self.conn.execute(
                'SELECT user, agent, turn_state_json, timestamp, visit_id FROM memory WHERE scene_id = ? AND plot_id = ? ORDER BY id DESC LIMIT ?',
                (scene_id, plot_id, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                'SELECT user, agent, turn_state_json, timestamp, visit_id FROM memory WHERE scene_id = ? AND plot_id = ? AND visit_id = ? ORDER BY id DESC LIMIT ?',
                (scene_id, plot_id, int(visit_id), limit),
            ).fetchall()
        hydrated_rows: list[dict[str, Any]] = []
        for row in reversed(rows):
            record = dict(row)
            record['turn_state'] = parse_json_object(record.get('turn_state_json', '{}'))
            hydrated_rows.append(record)
        return hydrated_rows

    def get_global_recent_turns(self, limit: int = 12) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            'SELECT user, agent, turn_state_json, timestamp, visit_id FROM memory ORDER BY id DESC LIMIT ?',
            (limit,),
        ).fetchall()
        hydrated_rows: list[dict[str, Any]] = []
        for row in reversed(rows):
            record = dict(row)
            record['turn_state'] = parse_json_object(record.get('turn_state_json', '{}'))
            hydrated_rows.append(record)
        return hydrated_rows

    def has_scene_opening(self, scene_id: str, marker: str, visit_id: int | None = None) -> bool:
        if visit_id is None:
            row = self.conn.execute(
                'SELECT 1 FROM memory WHERE scene_id = ? AND user = ? LIMIT 1',
                (scene_id, marker),
            ).fetchone()
        else:
            row = self.conn.execute(
                'SELECT 1 FROM memory WHERE scene_id = ? AND user = ? AND visit_id = ? LIMIT 1',
                (scene_id, marker, int(visit_id)),
            ).fetchone()
        return row is not None

    def save_summary(self, summary_type: str, content: str, scene_id: str = '', plot_id: str = '') -> None:
        self.conn.execute(
            '''
            INSERT INTO summaries(summary_type, scene_id, plot_id, content)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(summary_type, scene_id, plot_id) DO UPDATE SET content = excluded.content
            ''',
            (summary_type, scene_id, plot_id, content),
        )
        self.conn.commit()

    def get_summary(self, summary_type: str, scene_id: str = '', plot_id: str = '') -> str:
        row = self.conn.execute(
            'SELECT content FROM summaries WHERE summary_type = ? AND scene_id = ? AND plot_id = ?',
            (summary_type, scene_id, plot_id),
        ).fetchone()
        return row['content'] if row else ''

    def get_system_state(self) -> dict[str, Any]:
        row = self.conn.execute('SELECT * FROM system_state WHERE id = 1').fetchone()
        state = dict(row)
        state['player_profile'] = json.loads(state.get('player_profile', '{}') or '{}')
        state['navigation_state_json'] = state.get('navigation_state_json', '{}')
        state['navigation_state'] = parse_json_object(state.get('navigation_state_json', '{}'))
        return state

    def update_system_state(self, updates: dict[str, Any]) -> None:
        current = self.get_system_state()
        if 'player_profile' in updates and isinstance(updates['player_profile'], dict):
            updates = updates.copy()
            updates['player_profile'] = json.dumps(updates['player_profile'], ensure_ascii=False)
        elif 'player_profile' not in updates:
            updates = updates.copy()
            updates['player_profile'] = json.dumps(current.get('player_profile', {}), ensure_ascii=False)

        if 'navigation_state' in updates and isinstance(updates['navigation_state'], dict):
            updates = updates.copy()
            updates['navigation_state_json'] = json.dumps(updates.pop('navigation_state'), ensure_ascii=False)
        elif 'navigation_state_json' not in updates:
            updates = updates.copy()
            updates['navigation_state_json'] = json.dumps(current.get('navigation_state', {}), ensure_ascii=False)

        for k, v in updates.items():
            self.conn.execute(f'UPDATE system_state SET {k} = ? WHERE id = 1', (v,))
        self.conn.commit()

    def get_player_profile(self) -> dict[str, Any]:
        return self.get_system_state().get('player_profile', {})

    def save_player_profile(self, profile: dict[str, Any]) -> None:
        self.update_system_state({'player_profile': profile})

    def _hydrate_scene_row(self, scene: dict[str, Any]) -> dict[str, Any]:
        scene['navigation'] = parse_json_object(scene.get('navigation_json', '{}'))
        scene['node_kind'] = str(scene.get('node_kind', 'linear') or 'linear')
        return scene

    def _hydrate_plot_row(self, plot: dict[str, Any]) -> dict[str, Any]:
        plot['navigation'] = parse_json_object(plot.get('navigation_json', '{}'))
        plot['node_kind'] = str(plot.get('node_kind', 'linear') or 'linear')
        return plot

    def _backfill_linear_navigation_metadata(self) -> None:
        scene_rows = [dict(row) for row in self.conn.execute('SELECT * FROM scenes').fetchall()]
        if not scene_rows:
            return

        plot_rows_by_scene: dict[str, list[dict[str, Any]]] = {}
        for row in self.conn.execute('SELECT * FROM plots').fetchall():
            plot = dict(row)
            plot_rows_by_scene.setdefault(str(plot.get('scene_id', '')), []).append(plot)

        scenes: list[dict[str, Any]] = []
        for scene in scene_rows:
            hydrated_scene = self._hydrate_scene_row(scene)
            plots = [self._hydrate_plot_row(plot) for plot in plot_rows_by_scene.get(str(scene.get('scene_id', '')), [])]
            plots.sort(key=lambda plot: self._natural_id_key(str(plot.get('plot_id', ''))))
            hydrated_scene['plots'] = plots
            scenes.append(hydrated_scene)

        scenes.sort(key=lambda scene: self._natural_id_key(str(scene.get('scene_id', ''))))
        ensure_scene_navigation_defaults(scenes)

        cur = self.conn.cursor()
        for scene in scenes:
            cur.execute(
                'UPDATE scenes SET node_kind = ?, navigation_json = ? WHERE scene_id = ?',
                (
                    scene.get('node_kind', 'linear'),
                    serialize_navigation(scene.get('navigation')),
                    scene.get('scene_id', ''),
                ),
            )
            for plot in scene.get('plots', []):
                cur.execute(
                    'UPDATE plots SET node_kind = ?, navigation_json = ? WHERE plot_id = ?',
                    (
                        plot.get('node_kind', 'linear'),
                        serialize_navigation(plot.get('navigation')),
                        plot.get('plot_id', ''),
                    ),
                )
        self.conn.commit()

    @staticmethod
    def _natural_id_key(value: str) -> tuple[Any, ...]:
        parts = re.split(r'(\d+)', (value or '').lower())
        key: list[Any] = []
        for p in parts:
            if not p:
                continue
            if p.isdigit():
                key.append(int(p))
            else:
                key.append(p)
        return tuple(key)
