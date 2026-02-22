import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.database import Database
from app.state import evaluate_plot_completion, evaluate_scene_completion, next_story_position


def main() -> int:
    db_path = ROOT / 'test' / 'debug_state.db'
    try:
        if db_path.exists():
            db_path.unlink()

        db = Database(str(db_path))
        db.insert_scenes(
            [
                {
                    'scene_id': 's1',
                    'scene_goal': '完成序章',
                    'status': 'in_progress',
                    'scene_summary': '',
                    'plots': [
                        {
                            'plot_id': 's1p1',
                            'plot_goal': '击败守卫',
                            'mandatory_events': ['击败守卫'],
                            'npc': ['Guard'],
                            'locations': ['Gate'],
                            'status': 'pending',
                            'progress': 0.0,
                        }
                    ],
                },
                {
                    'scene_id': 's2',
                    'scene_goal': '开启调查',
                    'status': 'pending',
                    'scene_summary': '',
                    'plots': [
                        {
                            'plot_id': 's2p1',
                            'plot_goal': '收集线索',
                            'mandatory_events': ['检查房间'],
                            'npc': ['Witness'],
                            'locations': ['House'],
                            'status': 'pending',
                            'progress': 0.0,
                        }
                    ],
                },
            ]
        )

        print('[test_state] 输入: user_input/response/current_progress/events')
        done, progress = evaluate_plot_completion('我击败守卫并完成任务', '剧情完成', 0.4, ['击败守卫'])
        print('[test_state] 输出: plot completion ->', {'done': done, 'progress': progress})

        db.update_plot('s1p1', status='completed', progress=1.0)
        scene_done, scene_progress = evaluate_scene_completion(db, 's1')
        print('[test_state] 输出: scene completion ->', {'done': scene_done, 'progress': scene_progress})

        nxt = next_story_position(db, 's1', 's1p1')
        print('[test_state] 输出: next_story_position ->', nxt)

        db.close()
        print('[test_state] 结果: PASS')
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f'[test_state] 结果: FAIL -> {exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
