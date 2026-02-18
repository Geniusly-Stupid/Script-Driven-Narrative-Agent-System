from __future__ import annotations

import os
import tempfile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.agents.narrative_graph import narrative_agent
from backend.database.repository import repo
from backend.models.schemas import CharacterRequest, ChatRequest, ChatResponse, SceneModel, UploadResponse
from backend.services.pdf_parser import PDFParserService
from backend.services.script_parser import script_parser_service
from backend.vector_store.chroma_store import chroma_store

router = APIRouter(prefix='/api/workflow', tags=['workflow'])


def _assert_stage(current: str, expected: str) -> None:
    if current != expected:
        raise HTTPException(status_code=409, detail=f'Current stage is {current}; expected {expected}.')


@router.get('/status')
async def get_status():
    state = await repo.get_system_state()
    scenes = await repo.list_scenes()
    return {
        'system_state': {
            'current_scene_id': state.get('current_scene_id', ''),
            'current_plot_id': state.get('current_plot_id', ''),
            'plot_progress': state.get('plot_progress', 0.0),
            'scene_progress': state.get('scene_progress', 0.0),
            'stage': state.get('stage', 'upload'),
        },
        'scenes': scenes,
    }


@router.post('/upload-script', response_model=UploadResponse)
async def upload_script(
    file: UploadFile = File(...),
    start_page: int = Form(1),
    end_page: int | None = Form(None),
):
    state = await repo.get_system_state()
    _assert_stage(state.get('stage', 'upload'), 'upload')

    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail='Only PDF files are supported.')

    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    script_text = PDFParserService.extract_text(tmp_path, start_page=start_page, end_page=end_page)

    await repo.clear_story_data()
    chroma_store.reset()

    scenes = await script_parser_service.parse_script(script_text)
    if not scenes:
        raise HTTPException(status_code=400, detail='No scenes were parsed from the script.')

    await repo.insert_scenes(scenes)
    knowledge_docs = script_parser_service.build_knowledge_docs(scenes)
    await repo.create_knowledge_docs(knowledge_docs)
    chroma_store.add_documents(knowledge_docs)

    first_scene = scenes[0]
    first_plot = first_scene.plots[0] if first_scene.plots else None

    await repo.update_scene(first_scene.scene_id, {'status': 'in_progress'})
    await repo.update_system_state(
        {
            'stage': 'parse',
            'current_scene_id': first_scene.scene_id,
            'current_plot_id': first_plot.plot_id if first_plot else '',
            'plot_progress': 0.0,
            'scene_progress': 0.0,
        }
    )

    return UploadResponse(message='Script uploaded and parsed successfully.', stage='parse', scenes=scenes)


@router.get('/scenes')
async def list_scenes():
    state = await repo.get_system_state()
    if state.get('stage') not in ['parse', 'character', 'session']:
        raise HTTPException(status_code=409, detail='Scenes are available after upload and parse stage.')
    return {'scenes': await repo.list_scenes()}


@router.post('/confirm-structure')
async def confirm_structure():
    state = await repo.get_system_state()
    _assert_stage(state.get('stage', 'upload'), 'parse')
    await repo.update_system_state({'stage': 'character'})
    return {'message': 'Structure confirmed.', 'stage': 'character'}


@router.post('/character')
async def create_character(payload: CharacterRequest):
    state = await repo.get_system_state()
    _assert_stage(state.get('stage', 'upload'), 'character')

    await repo.create_player_profile(payload)
    await repo.update_system_state({'stage': 'session'})
    return {'message': 'Character created.', 'stage': 'session'}


@router.post('/session/message', response_model=ChatResponse)
async def session_message(payload: ChatRequest):
    state = await repo.get_system_state()
    _assert_stage(state.get('stage', 'upload'), 'session')

    result = await narrative_agent.run_turn(payload.message)
    state = await repo.get_system_state()

    return ChatResponse(
        response=result.get('response', ''),
        dice_result=result.get('dice_result'),
        stage=state.get('stage', 'session'),
        current_scene_id=state.get('current_scene_id', ''),
        current_plot_id=state.get('current_plot_id', ''),
        plot_progress=state.get('plot_progress', 0.0),
        scene_progress=state.get('scene_progress', 0.0),
    )
