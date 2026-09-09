import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.analyze import router as analyze_router
from app.api.calls import router as calls_router
from app.api.health import router as health_router
from app.api.speakers import router as speakers_router
from app.config import settings
from app.database import init_db


logger = logging.getLogger("voxshield")


@asynccontextmanager
async def lifespan(app: FastAPI):

    # A missing database must not stop the API from booting. The audio
    # analysis path does not touch Postgres at all, so refusing to start
    # without Docker running would take the live demo down over a dependency
    # it does not use. Persistence degrades; detection keeps working.
    try:
        await init_db()
        app.state.database_available = True
    except Exception as exc:
        app.state.database_available = False
        logger.warning(
            "database unavailable, continuing without persistence: %s: %s",
            type(exc).__name__,
            exc,
        )

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

app.include_router(
    speakers_router
)

app.include_router(
    analyze_router
)


@app.get("/")
async def root():

    return {
        "name": "VoxShield",
        "status": "online",
        "version": "0.2.0",
    }