from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.database.mongo import mongo_manager
from backend.database.repository import repo
from backend.routers.workflow import router as workflow_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    await mongo_manager.connect()
    await repo.ensure_indexes()
    await repo.get_system_state()
    yield
    await mongo_manager.disconnect()


app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/health')
async def health_check():
    return {'status': 'ok'}


app.include_router(workflow_router)
