import uuid

from fastapi import APIRouter, Query, WebSocket

from app.websocket.audio_stream import (
    audio_stream_manager,
)


router = APIRouter(
    prefix="/api/v1/calls",
    tags=["Calls"],
)


@router.post("/demo")
async def create_demo_call():

    call_id = str(uuid.uuid4())

    return {
        "call_id": call_id,
        "status": "created",
        "mode": "demo",
    }


@router.websocket("/{call_id}/stream")
async def audio_stream(
    websocket: WebSocket,
    call_id: str,
    claimed_identity: str | None = Query(
        None,
        description="Who the caller says they are. Enables the speaker "
                    "verification branch; without it the stream is anti-spoof "
                    "only and says so.",
    ),
):

    await audio_stream_manager.connect(
        call_id,
        websocket,
        claimed_identity=claimed_identity,
    )

    await audio_stream_manager.process_stream(
        call_id,
        websocket,
    )