import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.database import Database


def main() -> int:
    db_path = ROOT / 'test' / 'debug_narrative.db'
    print(f'[test_database] 输入: db_path={db_path}')

    try:
        if db_path.exists():
            os.remove(db_path)

        db = Database(str(db_path))
        print('[test_database] 输出: 初始化 SQLite 成功')

        scenes = [
            {
                'scene_id': 'scene_t1',
                'scene_goal': '测试场景写入',
                'status': 'pending',
                'scene_summary': '',
                'plots': [
                    {
                        'plot_id': 'scene_t1_plot_1',
                        'plot_goal': '测试剧情写入',
                        'mandatory_events': ['event_a', 'event_b'],
                        'npc': ['Alice'],
                        'locations': ['Hall'],
                        'status': 'pending',
                        'progress': 0.0,
                    }
                ],
            }
        ]
        print('[test_database] 输入: 插入场景数据')
        db.insert_scenes(scenes)

        fetched = db.get_scene('scene_t1')
        print('[test_database] 输出: 查询场景结果 ->', fetched)

        print('[test_database] 输入: 写入 memory')
        db.append_memory('scene_t1', 'scene_t1_plot_1', '用户说话', 'Agent回复')
        turns = db.get_recent_turns('scene_t1', 'scene_t1_plot_1')
        print('[test_database] 输出: memory ->', turns)

        print('[test_database] 输入: 更新 system_state')
        db.update_system_state({'stage': 'session', 'current_scene_id': 'scene_t1', 'current_plot_id': 'scene_t1_plot_1'})
        state = db.get_system_state()
        print('[test_database] 输出: system_state ->', state)

        db.close()

        check = sqlite3.connect(str(db_path))
        count = check.execute('SELECT COUNT(*) FROM scenes').fetchone()[0]
        check.close()
        print(f'[test_database] 输出: scenes 表记录数 = {count}')

        print('[test_database] 结果: PASS')
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f'[test_database] 结果: FAIL -> {exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
