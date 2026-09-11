"""
Call lifecycle and the live audio websocket.

    POST /api/v1/calls/demo                     -> a call id
    WS   /api/v1/calls/{call_id}/stream         PCM16 in, analysis out

The websocket takes two optional query parameters, and both matter for what
comes back:

``claimed_identity``
    Who the caller says they are. Without it the speaker branch is skipped and
    the response says so - a low risk score from anti-spoof alone must never be
    read as "this is who they claim to be".

``context``
    Call metadata as a JSON object, the same shape ``POST /analyze`` accepts.
    Without it the live path can only ever see two of the three branches, and a
    genuine-human impersonator - real speech, wrong person, urgent unusual
    request - is exactly the case those two branches are worst at.

Both are fixed for the life of the call. Letting them change mid-stream would
let a caller re-aim the identity check at whichever profile happened to match.
"""

import json
import uuid

from fastapi import APIRouter, Query, WebSocket, status

from app.ml.context import CallContext
from app.websocket.audio_stream import audio_stream_manager

router = APIRouter(prefix="/api/v1/calls", tags=["Calls"])


def parse_context(raw: str | None) -> CallContext | None:
    """
    JSON string -> CallContext, or ``ValueError`` with a usable message.

    Unknown keys are rejected rather than ignored. A caller who sends
    ``{"caller_is_known": false}`` and gets a 200 back would reasonably assume
    the signal was counted, and it was not.
    """

    if not raw:
        return None

    return CallContext.from_dict(json.loads(raw))


@router.post("/demo")
async def create_demo_call() -> dict:
    return {
        "call_id": str(uuid.uuid4()),
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
    context: str | None = Query(
        None,
        description='Call metadata as a JSON object, e.g. {"caller_known": '
                    'false, "transaction_amount": 50000}. Any subset of '
                    "CallContext's fields; anything omitted is treated as "
                    '"not collected" rather than as safe.',
    ),
):
    try:
        call_context = parse_context(context)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        # Reject before accepting. A websocket that opens and then dies with a
        # generic 1011 gives the browser nothing to show; closing with a
        # policy-violation code and the reason attached is readable in the
        # client's onclose handler.
        await websocket.close(
            code=status.WS_1008_POLICY_VIOLATION,
            reason=f"bad context: {exc}"[:120],
        )
        return

    await audio_stream_manager.connect(
        call_id,
        websocket,
        claimed_identity=claimed_identity,
        call_context=call_context,
    )

    await audio_stream_manager.process_stream(call_id, websocket)
