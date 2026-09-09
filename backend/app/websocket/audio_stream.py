import json

from fastapi import WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool

from app.audio.feature_pipeline import (
    AudioFeaturePipeline,
)


class AudioStreamManager:

    def __init__(self):

        self.active_connections = {}

        self.pipelines = {}

    async def connect(
        self,
        call_id: str,
        websocket: WebSocket,
        claimed_identity: str | None = None,
    ):

        await websocket.accept()

        self.active_connections[
            call_id
        ] = websocket

        # The claimed identity is fixed for the life of the call. Letting it
        # change mid-stream would let a caller re-aim the identity check at
        # whichever profile happened to match.
        self.pipelines[
            call_id
        ] = AudioFeaturePipeline(
            sample_rate=16000,
            claimed_identity=claimed_identity,
        )

    async def disconnect(
        self,
        call_id: str,
    ):

        self.active_connections.pop(
            call_id,
            None,
        )

        self.pipelines.pop(
            call_id,
            None,
        )

    async def process_stream(
        self,
        call_id: str,
        websocket: WebSocket,
    ):

        pipeline = self.pipelines[
            call_id
        ]

        segment_id = 0

        try:

            while True:

                message = (
                    await websocket.receive()
                )

                # websocket.receive() returns the raw ASGI message and does
                # NOT raise on disconnect - it yields a disconnect message.
                # Relying on the WebSocketDisconnect except-branch alone left
                # this loop spinning on a dead socket forever.
                if message.get("type") == "websocket.disconnect":
                    await self.disconnect(call_id)
                    return

                if "bytes" in message:

                    audio_bytes = (
                        message["bytes"]
                    )

                    # Feature extraction and model inference are synchronous
                    # CPU/GPU work measured in tens of milliseconds. Running
                    # them inline would block the event loop and stall every
                    # other concurrent call on this worker.
                    result = await run_in_threadpool(
                        pipeline.process,
                        audio_bytes,
                    )

                    segment_id += 1

                    response = {
                        "type": "voice_analysis",
                        "call_id": call_id,
                        "segment_id": segment_id,
                        "analysis": result,
                    }

                    await websocket.send_text(
                        json.dumps(response)
                    )

                elif "text" in message:

                    try:

                        data = json.loads(
                            message["text"]
                        )

                        if (
                            data.get("type")
                            == "ping"
                        ):

                            await websocket.send_text(
                                json.dumps(
                                    {
                                        "type": "pong"
                                    }
                                )
                            )

                    except json.JSONDecodeError:

                        pass

        except WebSocketDisconnect:

            await self.disconnect(
                call_id
            )

        except Exception:

            await self.disconnect(
                call_id
            )

            raise


audio_stream_manager = (
    AudioStreamManager()
)