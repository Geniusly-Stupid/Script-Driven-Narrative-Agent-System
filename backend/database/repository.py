from datetime import datetime
from typing import Any

from pymongo import ASCENDING

from backend.database.mongo import mongo_manager
from backend.models.schemas import PlayerProfileModel, SceneModel, SystemStateModel


class NarrativeRepository:
    @property
    def db(self):
        if mongo_manager.db is None:
            raise RuntimeError('Database is not initialized')
        return mongo_manager.db

    async def ensure_indexes(self) -> None:
        await self.db.scenes.create_index([('scene_id', ASCENDING)], unique=True)
        await self.db.scenes.create_index([('status', ASCENDING)])
        await self.db.knowledge_base.create_index([('type', ASCENDING), ('name', ASCENDING)])
        await self.db.player_profiles.create_index([('name', ASCENDING)], unique=True)
        await self.db.conversation_memory.create_index([('scene_id', ASCENDING), ('plot_id', ASCENDING)], unique=True)
        await self.db.plot_summaries.create_index([('plot_id', ASCENDING)], unique=True)
        await self.db.scene_summaries.create_index([('scene_id', ASCENDING)], unique=True)
        await self.db.system_state.create_index([('stage', ASCENDING)])

    async def clear_story_data(self) -> None:
        await self.db.scenes.delete_many({})
        await self.db.knowledge_base.delete_many({})
        await self.db.conversation_memory.delete_many({})
        await self.db.plot_summaries.delete_many({})
        await self.db.scene_summaries.delete_many({})

    async def get_system_state(self) -> dict[str, Any]:
        state = await self.db.system_state.find_one({})
        if state:
            return state
        default_state = SystemStateModel().model_dump()
        await self.db.system_state.insert_one(default_state)
        return await self.db.system_state.find_one({})

    async def update_system_state(self, updates: dict[str, Any]) -> None:
        await self.db.system_state.update_one({}, {'$set': updates}, upsert=True)

    async def insert_scenes(self, scenes: list[SceneModel]) -> None:
        if not scenes:
            return
        await self.db.scenes.insert_many([scene.model_dump() for scene in scenes])

    async def list_scenes(self) -> list[dict[str, Any]]:
        return [doc async for doc in self.db.scenes.find({}, {'_id': 0})]

    async def get_scene(self, scene_id: str) -> dict[str, Any] | None:
        return await self.db.scenes.find_one({'scene_id': scene_id})

    async def update_scene(self, scene_id: str, updates: dict[str, Any]) -> None:
        await self.db.scenes.update_one({'scene_id': scene_id}, {'$set': updates})

    async def create_player_profile(self, profile: PlayerProfileModel) -> None:
        await self.db.player_profiles.delete_many({})
        await self.db.player_profiles.insert_one(profile.model_dump())

    async def get_player_profile(self) -> dict[str, Any] | None:
        return await self.db.player_profiles.find_one({}, {'_id': 0})

    async def append_turn(self, scene_id: str, plot_id: str, user: str, agent: str) -> None:
        await self.db.conversation_memory.update_one(
            {'scene_id': scene_id, 'plot_id': plot_id},
            {'$push': {'turns': {'user': user, 'agent': agent, 'timestamp': datetime.utcnow()}}},
            upsert=True,
        )

    async def get_turns(self, scene_id: str, plot_id: str, limit: int = 20) -> list[dict[str, Any]]:
        memory = await self.db.conversation_memory.find_one({'scene_id': scene_id, 'plot_id': plot_id}, {'_id': 0})
        if not memory:
            return []
        return memory.get('turns', [])[-limit:]

    async def save_plot_summary(self, scene_id: str, plot_id: str, summary: str) -> None:
        await self.db.plot_summaries.update_one(
            {'plot_id': plot_id},
            {'$set': {'scene_id': scene_id, 'plot_id': plot_id, 'summary': summary}},
            upsert=True,
        )

    async def get_plot_summary(self, plot_id: str) -> str:
        doc = await self.db.plot_summaries.find_one({'plot_id': plot_id}, {'_id': 0, 'summary': 1})
        if not doc:
            return ''
        return str(doc.get('summary', ''))

    async def save_scene_summary(self, scene_id: str, summary: str) -> None:
        await self.db.scene_summaries.update_one(
            {'scene_id': scene_id},
            {'$set': {'scene_id': scene_id, 'summary': summary}},
            upsert=True,
        )

    async def create_knowledge_docs(self, docs: list[dict[str, Any]]) -> None:
        if docs:
            await self.db.knowledge_base.insert_many(docs)


repo = NarrativeRepository()
