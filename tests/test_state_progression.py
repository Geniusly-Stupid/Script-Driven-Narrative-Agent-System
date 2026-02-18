import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.agents.narrative_graph import narrative_agent
from backend.database.mongo import mongo_manager
from backend.database.repository import repo
from backend.models.schemas import PlotModel, SceneModel


async def main() -> int:
    print('[test_state_progression] START')
    try:
        print('[test_state_progression] Preparing scenes for progression simulation...')
        await mongo_manager.connect()
        await repo.ensure_indexes()
        await repo.clear_story_data()
        await repo.db.system_state.delete_many({})

        scenes = [
            SceneModel(
                scene_id='prog_scene_1',
                scene_goal='Finish opening conflict',
                plots=[
                    PlotModel(
                        plot_id='prog_scene_1_plot_1',
                        plot_goal='Resolve first confrontation',
                        mandatory_events=['confront enemy'],
                        npc=['Enemy'],
                        locations=['Square'],
                        status='in_progress',
                        progress=0.95,
                    )
                ],
                status='in_progress',
                scene_summary='',
            ),
            SceneModel(
                scene_id='prog_scene_2',
                scene_goal='Begin investigation',
                plots=[
                    PlotModel(
                        plot_id='prog_scene_2_plot_1',
                        plot_goal='Collect first clue',
                        mandatory_events=['search room'],
                        npc=['Guard'],
                        locations=['Manor'],
                        status='pending',
                        progress=0.0,
                    )
                ],
                status='pending',
                scene_summary='',
            ),
        ]
        await repo.insert_scenes(scenes)
        await repo.update_system_state(
            {
                'stage': 'session',
                'current_scene_id': 'prog_scene_1',
                'current_plot_id': 'prog_scene_1_plot_1',
                'plot_progress': 0.95,
                'scene_progress': 0.0,
            }
        )

        state = {
            'scene_id': 'prog_scene_1',
            'plot_id': 'prog_scene_1_plot_1',
            'plot_progress': 0.95,
            'scene_progress': 0.0,
            'latest_user_input': 'I defeat the enemy and secure the square.',
            'response': 'You finish the confrontation and stabilize the area.',
        }

        print('[test_state_progression] Simulating plot completion...')
        state = await narrative_agent.check_plot_completion(state)
        print(f"  plot_completed={state.get('plot_completed')} plot_progress={state.get('plot_progress')}")

        print('[test_state_progression] Simulating scene completion...')
        state = await narrative_agent.check_scene_completion(state)
        print(f"  scene_completed={state.get('scene_completed')} scene_progress={state.get('scene_progress')}")

        print('[test_state_progression] Updating global system state...')
        state = await narrative_agent.update_state(state)

        system_state = await repo.get_system_state()
        print('[test_state_progression] Updated system_state:')
        print({
            'current_scene_id': system_state.get('current_scene_id'),
            'current_plot_id': system_state.get('current_plot_id'),
            'plot_progress': system_state.get('plot_progress'),
            'scene_progress': system_state.get('scene_progress'),
            'stage': system_state.get('stage'),
        })

        print('[test_state_progression] SUCCESS')
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f'[test_state_progression] FAILED: {exc}')
        return 1
    finally:
        try:
            await mongo_manager.disconnect()
        except Exception:
            pass


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
