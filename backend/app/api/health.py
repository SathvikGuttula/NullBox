from fastapi import APIRouter

from app.config import settings

router = APIRouter(prefix="/api/v1", tags=["System"])


@router.get("/health")
async def health_check() -> dict:
    return {
        "status": "healthy",
        "service": settings.app_name,
        "environment": settings.app_env,
    }