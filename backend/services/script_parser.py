from __future__ import annotations

import re
from collections import defaultdict

from backend.models.schemas import PlotModel, SceneModel
from backend.services.llm_service import llm_service


class ScriptParserService:
    async def parse_script(self, text: str) -> list[SceneModel]:
        fallback = self._heuristic_parse(text)
        fallback_payload = {'scenes': [scene.model_dump() for scene in fallback]}

        system_prompt = (
            'You segment scripts into scenes and plots. '
            'Each scene must include: scene_id, scene_goal, plots[], status, scene_summary. '
            'Each plot must include: plot_id, plot_goal, mandatory_events, npc, locations, status, progress.'
        )
        user_prompt = f'Script:\n{text[:20000]}'
        structured = await llm_service.complete_json(system_prompt, user_prompt, fallback=fallback_payload)

        scenes = []
        for scene in structured.get('scenes', []):
            try:
                parsed_scene = SceneModel(**scene)
                if not parsed_scene.plots:
                    parsed_scene.plots = fallback[0].plots if fallback else []
                scenes.append(parsed_scene)
            except Exception:
                continue
        return scenes or fallback

    def build_knowledge_docs(self, scenes: list[SceneModel]) -> list[dict]:
        docs = []
        seen = set()
        for scene in scenes:
            for plot in scene.plots:
                for npc in plot.npc:
                    key = ('npc', npc)
                    if key not in seen:
                        seen.add(key)
                        docs.append(
                            {
                                'type': 'npc',
                                'name': npc,
                                'description': f'NPC {npc} appears in {scene.scene_id}/{plot.plot_id}.',
                                'metadata': {'scene_id': scene.scene_id, 'plot_id': plot.plot_id},
                            }
                        )
                for location in plot.locations:
                    key = ('location', location)
                    if key not in seen:
                        seen.add(key)
                        docs.append(
                            {
                                'type': 'location',
                                'name': location,
                                'description': f'Location {location} is relevant to {scene.scene_id}/{plot.plot_id}.',
                                'metadata': {'scene_id': scene.scene_id, 'plot_id': plot.plot_id},
                            }
                        )
                for event in plot.mandatory_events:
                    key = ('event', event)
                    if key not in seen:
                        seen.add(key)
                        docs.append(
                            {
                                'type': 'event',
                                'name': event[:80],
                                'description': event,
                                'metadata': {'scene_id': scene.scene_id, 'plot_id': plot.plot_id},
                            }
                        )
        return docs

    def _heuristic_parse(self, text: str) -> list[SceneModel]:
        chunks = [c.strip() for c in re.split(r'(?=SCENE\s+\d+|Scene\s+\d+)', text) if c.strip()]
        if not chunks:
            chunks = [text[:6000]]

        scenes: list[SceneModel] = []
        for idx, chunk in enumerate(chunks[:10], start=1):
            scene_id = f'scene_{idx}'
            lines = [line.strip() for line in chunk.splitlines() if line.strip()]
            scene_goal = lines[0][:180] if lines else f'Advance story in {scene_id}'

            words = re.findall(r'[A-Z][a-zA-Z]{2,}', chunk)
            freq = defaultdict(int)
            for word in words:
                freq[word] += 1
            top_tokens = [k for k, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:6]]
            npc = top_tokens[:3] if top_tokens else ['Guide']
            locations = top_tokens[3:5] if len(top_tokens) >= 5 else ['Unknown Site']

            plot = PlotModel(
                plot_id=f'{scene_id}_plot_1',
                plot_goal=f'Resolve central conflict of {scene_id}',
                mandatory_events=[
                    f'Key event derived from script fragment {idx}',
                    f'Character decision point in {scene_id}',
                ],
                npc=npc,
                locations=locations,
                status='pending',
                progress=0.0,
            )
            scenes.append(
                SceneModel(
                    scene_id=scene_id,
                    scene_goal=scene_goal,
                    plots=[plot],
                    status='pending',
                    scene_summary='',
                )
            )
        return scenes


script_parser_service = ScriptParserService()
