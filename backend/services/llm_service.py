from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from backend.config import settings


class LLMService:
    def __init__(self) -> None:
        self.enabled = bool(settings.openai_api_key)
        self.client = None
        if self.enabled:
            self.client = ChatOpenAI(model=settings.openai_model, api_key=settings.openai_api_key, temperature=0.3)

    async def complete_json(self, system_prompt: str, user_prompt: str, fallback: dict) -> dict:
        if not self.client:
            return fallback
        response = await self.client.ainvoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt + '\nReturn valid JSON only.'),
            ]
        )
        try:
            return json.loads(response.content)
        except Exception:
            return fallback

    async def complete_text(self, prompt: str, fallback: str) -> str:
        if not self.client:
            return fallback
        response = await self.client.ainvoke([HumanMessage(content=prompt)])
        if isinstance(response.content, str):
            return response.content
        return fallback


llm_service = LLMService()
