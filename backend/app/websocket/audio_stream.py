import json

from fastapi import WebSocket, WebSocketDisconnect

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
    ):

        await websocket.accept()

        self.active_connections[
            call_id
        ] = websocket

        self.pipelines[
            call_id
        ] = AudioFeaturePipeline(
            sample_rate=16000,
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

                if "bytes" in message:

                    audio_bytes = (
                        message["bytes"]
                    )

                    result = pipeline.process(
                        audio_bytes
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