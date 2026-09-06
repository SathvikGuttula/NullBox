from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.calls import router as calls_router
from app.api.health import router as health_router
from app.config import settings
from app.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):

    await init_db()

    yield


app = FastAPI(
    title="VoxShield API",
    description=(
        "AI-powered real-time voice integrity, "
        "deepfake detection and impersonation "
        "prevention platform."
    ),
    version="0.2.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,

    allow_origins=settings.cors_origin_list,

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


app.include_router(
    health_router
)

app.include_router(
    calls_router
)


@app.get("/")
async def root():

    return {
        "name": "VoxShield",
        "status": "online",
        "version": "0.2.0",
    }