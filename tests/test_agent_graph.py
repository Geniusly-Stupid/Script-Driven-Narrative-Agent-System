import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.agents.narrative_graph import narrative_agent
from backend.database.mongo import mongo_manager
from backend.database.repository import repo
from backend.models.schemas import PlayerProfileModel, SceneModel, PlotModel
from backend.vector_store.chroma_store import chroma_store


async def main() -> int:
    print('[test_agent_graph] START')
    try:
        print('[test_agent_graph] Connecting MongoDB and preparing state...')
        await mongo_manager.connect()
        await repo.ensure_indexes()

        await repo.clear_story_data()
        await repo.db.player_profiles.delete_many({})
        await repo.db.system_state.delete_many({})

        scene = SceneModel(
            scene_id='agent_scene_1',
            scene_goal='Find the hidden relic',
            plots=[
                PlotModel(
                    plot_id='agent_scene_1_plot_1',
                    plot_goal='Investigate clues in the old temple',
                    mandatory_events=['inspect altar', 'decipher inscription'],
                    npc=['Archivist'],
                    locations=['Old Temple'],
                    status='pending',
                    progress=0.0,
                )
            ],
            status='in_progress',
            scene_summary='Arrival at the temple entrance.',
        )
        await repo.insert_scenes([scene])
        await repo.create_player_profile(
            PlayerProfileModel(
                name='TestHero',
                background='Scholar adventurer',
                traits=['curious', 'careful'],
                stats={'int': 10, 'agi': 7},
                special_skills=['lore'],
            )
        )
        await repo.update_system_state(
            {
                'stage': 'session',
                'current_scene_id': 'agent_scene_1',
                'current_plot_id': 'agent_scene_1_plot_1',
                'plot_progress': 0.0,
                'scene_progress': 0.0,
            }
        )

        chroma_store.reset()
        chroma_store.add_documents(
            [
                {'type': 'npc', 'name': 'Archivist', 'description': 'The Archivist knows temple legends.', 'metadata': {'scene_id': 'agent_scene_1', 'plot_id': 'agent_scene_1_plot_1'}},
                {'type': 'location', 'name': 'Old Temple', 'description': 'The temple altar has old glyphs.', 'metadata': {'scene_id': 'agent_scene_1', 'plot_id': 'agent_scene_1_plot_1'}},
            ]
        )

        print('[test_agent_graph] Running one narrative turn...')
        final_state = await narrative_agent.run_turn('I inspect the altar and ask Archivist about symbols.')

        print('[test_agent_graph] Prompt:')
        print(final_state.get('prompt', '')[:1200])
        print('[test_agent_graph] Retrieved docs:')
        for d in final_state.get('retrieved_docs', []):
            print(f"  - type={d.get('metadata', {}).get('type')} name={d.get('metadata', {}).get('name')} content={d.get('content')}")
        print('[test_agent_graph] Agent response:')
        print(final_state.get('response', ''))

        updated_state = await repo.get_system_state()
        print('[test_agent_graph] Updated system_state:')
        print({k: updated_state.get(k) for k in ['current_scene_id', 'current_plot_id', 'plot_progress', 'scene_progress', 'stage']})

        print('[test_agent_graph] SUCCESS')
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f'[test_agent_graph] FAILED: {exc}')
        return 1
    finally:
        try:
            await mongo_manager.disconnect()
        except Exception:
            pass


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
