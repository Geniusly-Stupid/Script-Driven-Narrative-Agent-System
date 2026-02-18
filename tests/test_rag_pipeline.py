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
from backend.vector_store.chroma_store import chroma_store


async def main() -> int:
    print('[test_rag_pipeline] START')
    try:
        print('[test_rag_pipeline] Preparing MongoDB + Chroma fixtures...')
        await mongo_manager.connect()
        await repo.ensure_indexes()
        await repo.clear_story_data()

        scene = SceneModel(
            scene_id='rag_scene_1',
            scene_goal='Track suspects through city districts',
            plots=[
                PlotModel(
                    plot_id='rag_scene_1_plot_1',
                    plot_goal='Identify suspect hideout from clues',
                    mandatory_events=['collect witness reports', 'cross-check maps'],
                    npc=['Witness'],
                    locations=['Harbor', 'Old Market'],
                    status='in_progress',
                    progress=0.2,
                )
            ],
            status='in_progress',
            scene_summary='The chase begins at dusk.',
        )
        await repo.insert_scenes([scene])
        await repo.update_system_state(
            {
                'stage': 'session',
                'current_scene_id': 'rag_scene_1',
                'current_plot_id': 'rag_scene_1_plot_1',
                'plot_progress': 0.2,
                'scene_progress': 0.0,
            }
        )

        chroma_store.reset()
        chroma_store.add_documents(
            [
                {'type': 'npc', 'name': 'Witness', 'description': 'Witness saw cloaked figure near Old Market.', 'metadata': {'scene_id': 'rag_scene_1', 'plot_id': 'rag_scene_1_plot_1'}},
                {'type': 'location', 'name': 'Harbor', 'description': 'Harbor smugglers use hidden warehouses.', 'metadata': {'scene_id': 'rag_scene_1', 'plot_id': 'rag_scene_1_plot_1'}},
                {'type': 'event', 'name': 'Map Clue', 'description': 'A map marks tunnels connecting market to harbor.', 'metadata': {'scene_id': 'rag_scene_1', 'plot_id': 'rag_scene_1_plot_1'}},
            ]
        )

        state = {
            'scene_id': 'rag_scene_1',
            'plot_id': 'rag_scene_1_plot_1',
            'plot_progress': 0.2,
            'scene_progress': 0.0,
            'latest_user_input': 'I compare witness reports with the city map.',
            'conversation_history': [],
            'retrieved_docs': [],
            'mandatory_events': ['collect witness reports', 'cross-check maps'],
            'plot_goal': 'Identify suspect hideout from clues',
        }

        print('[test_rag_pipeline] Generating retrieval queries...')
        state = await narrative_agent.generate_retrieval_queries(state)
        for q in state.get('retrieval_queries', []):
            print(f'  - {q}')

        print('[test_rag_pipeline] Running vector retrieval...')
        state = await narrative_agent.vector_retrieve(state)
        for i, d in enumerate(state.get('retrieved_docs', []), start=1):
            print(f"  {i}. type={d.get('metadata', {}).get('type')} name={d.get('metadata', {}).get('name')} content={d.get('content')}")

        print('[test_rag_pipeline] SUCCESS')
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f'[test_rag_pipeline] FAILED: {exc}')
        return 1
    finally:
        try:
            await mongo_manager.disconnect()
        except Exception:
            pass


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
