import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.database import Database
from app.navigation import default_navigation


def _build_db(db_path: Path) -> Database:
    if db_path.exists():
        os.remove(db_path)
    db = Database(str(db_path))
    nav = default_navigation()
    db.insert_scenes(
        [
            {
                'scene_id': 'scene_t1',
                'scene_goal': 'Follow the first lead.',
                'scene_description': 'A small test scene.',
                'status': 'in_progress',
                'scene_summary': '',
                'node_kind': 'linear',
                'navigation': nav,
                'plots': [
                    {
                        'plot_id': 'scene_t1_plot_1',
                        'plot_goal': 'Inspect the room.',
                        'npc': [],
                        'locations': [],
                        'raw_text': 'A quiet room with scattered notes.',
                        'status': 'in_progress',
                        'progress': 0.0,
                        'node_kind': 'linear',
                        'navigation': nav,
                    }
                ],
            }
        ]
    )
    return db


def main() -> int:
    db_path = ROOT / 'test' / 'debug_database_turn_state.db'
    db = None
    reopened = None
    try:
        print('[test_database] input: create database and persist structured turn state')
        db = _build_db(db_path)
        turn_state = {
            'choice_open': True,
            'offered_targets': [{'target_kind': 'scene', 'target_id': 'scene_t1'}],
            'beat_status': 'wrapped',
            'summary': 'keeper offered a follow-up lead',
        }
        db.append_memory(
            'scene_t1',
            'scene_t1_plot_1',
            'hello',
            'world',
            visit_id=7,
            turn_state=turn_state,
        )
        turns = db.get_recent_turns('scene_t1', 'scene_t1_plot_1', limit=5, visit_id=7)
        assert len(turns) == 1, 'expected one persisted memory turn'
        assert turns[0]['visit_id'] == 7, 'visit_id should round-trip'
        assert turns[0]['turn_state'] == turn_state, 'turn_state_json should hydrate back into dict form'

        print('[test_database] input: reopen database and confirm turn_state_json migration persists')
        db.close()
        db = None
        reopened = Database(str(db_path))
        reopened_turns = reopened.get_recent_turns('scene_t1', 'scene_t1_plot_1', limit=5, visit_id=7)
        assert reopened_turns[0]['turn_state'] == turn_state, 'reopened database should preserve turn_state_json'

        print('[test_database] input: check runtime story-specific patches are gone')
        for source_path in (
            ROOT / 'app' / 'database.py',
            ROOT / 'app' / 'state.py',
            ROOT / 'app' / 'agent_graph.py',
        ):
            source = source_path.read_text(encoding='utf-8')
            for token in (
                '_apply_bundled_story_repairs',
                '_repair_',
                '_invalidate_',
                'Kimball',
                'scene_11',
                'scene_12',
            ):
                assert token not in source, f'{token} should not remain in {source_path.name}'

        print('[test_database] result: PASS')
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f'[test_database] result: FAIL -> {exc}')
        return 1
    finally:
        if db is not None:
            db.close()
        if reopened is not None:
            reopened.close()
        if db_path.exists():
            os.remove(db_path)


if __name__ == '__main__':
    raise SystemExit(main())
