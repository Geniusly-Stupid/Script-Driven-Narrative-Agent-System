from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class PlotModel(BaseModel):
    plot_id: str
    plot_goal: str
    mandatory_events: list[str] = Field(default_factory=list)
    npc: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    status: Literal['pending', 'in_progress', 'completed'] = 'pending'
    progress: float = 0.0


class SceneModel(BaseModel):
    scene_id: str
    scene_goal: str
    plots: list[PlotModel] = Field(default_factory=list)
    status: Literal['pending', 'in_progress', 'completed'] = 'pending'
    scene_summary: str = ''


class KnowledgeBaseModel(BaseModel):
    type: Literal['npc', 'location', 'item', 'event', 'rule']
    name: str
    description: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class PlayerProfileModel(BaseModel):
    name: str
    background: str
    traits: list[str] = Field(default_factory=list)
    stats: dict[str, int | float] = Field(default_factory=dict)
    special_skills: list[str] = Field(default_factory=list)


class TurnModel(BaseModel):
    user: str
    agent: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ConversationMemoryModel(BaseModel):
    scene_id: str
    plot_id: str
    turns: list[TurnModel] = Field(default_factory=list)


class SystemStateModel(BaseModel):
    current_scene_id: str = ''
    current_plot_id: str = ''
    plot_progress: float = 0.0
    scene_progress: float = 0.0
    stage: Literal['upload', 'parse', 'character', 'session'] = 'upload'


class UploadResponse(BaseModel):
    message: str
    stage: str
    scenes: list[SceneModel]


class CharacterRequest(PlayerProfileModel):
    pass


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    dice_result: str | None = None
    stage: str
    current_scene_id: str
    current_plot_id: str
    plot_progress: float
    scene_progress: float
