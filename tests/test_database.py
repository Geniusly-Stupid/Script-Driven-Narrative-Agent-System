import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.database.mongo import mongo_manager
from backend.database.repository import repo


async def main() -> int:
    print('[test_database] START')
    try:
        print('[test_database] Connecting to MongoDB...')
        await mongo_manager.connect()

        print('[test_database] Ensuring indexes...')
        await repo.ensure_indexes()

        print('[test_database] Inserting mock scene...')
        await repo.db.scenes.delete_many({'scene_id': {'$regex': '^test_scene_'}})
        scene_doc = {
            'scene_id': 'test_scene_1',
            'scene_goal': 'Validate MongoDB scene CRUD',
            'plots': [
                {
                    'plot_id': 'test_scene_1_plot_1',
                    'plot_goal': 'Confirm insertion and retrieval',
                    'mandatory_events': ['insert', 'fetch'],
                    'npc': ['Tester'],
                    'locations': ['Lab'],
                    'status': 'pending',
                    'progress': 0.0,
                }
            ],
            'status': 'pending',
            'scene_summary': '',
        }
        await repo.db.scenes.insert_one(scene_doc)

        print('[test_database] Retrieving mock scene...')
        fetched = await repo.get_scene('test_scene_1')
        print('[test_database] Retrieved:', {k: fetched.get(k) for k in ['scene_id', 'scene_goal', 'status']} if fetched else None)

        if not fetched:
            raise RuntimeError('Failed to retrieve inserted scene.')

        print('[test_database] Cleaning up test data...')
        await repo.db.scenes.delete_many({'scene_id': {'$regex': '^test_scene_'}})

        print('[test_database] SUCCESS')
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f'[test_database] FAILED: {exc}')
        return 1
    finally:
        try:
            await mongo_manager.disconnect()
        except Exception:
            pass


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
