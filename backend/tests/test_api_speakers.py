"""
Speaker and analysis endpoints.

The encoder is stubbed so these run without an 80 MB download: what is being
tested is the HTTP boundary — validation, error codes, and the guarantee that
audio never reaches disk — not the quality of ECAPA embeddings.
"""

import io
import json

import numpy as np
import pytest
import soundfile as sf
from httpx import ASGITransport, AsyncClient

from app.ml.speaker import SpeakerEncoder, l2_normalize


DIM = 192


class StubEncoder:
    """Deterministic embeddings keyed on the audio, so identity is stable."""

    model_name = "stub-encoder"

    def embed(self, audio, sample_rate=None):
        seed = int(abs(float(np.asarray(audio, dtype=np.float64).sum())) * 1000) % 9973
        rng = np.random.default_rng(seed)
        return l2_normalize(rng.normal(size=DIM))


class SameSpeakerEncoder(StubEncoder):
    """Every clip maps near one identity direction."""

    def __init__(self, base_seed=7, spread=0.15):
        self.anchor = l2_normalize(np.random.default_rng(base_seed).normal(size=DIM))
        self.spread = spread
        self.calls = 0

    def embed(self, audio, sample_rate=None):
        self.calls += 1
        rng = np.random.default_rng(self.calls)
        return l2_normalize(self.anchor + self.spread * l2_normalize(rng.normal(size=DIM)))


def wav_bytes(seconds=3.0, frequency=220.0, sample_rate=16000):
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    return buffer.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch):
    """App wired to a temporary store and a stub encoder."""

    from app.api import speakers as speakers_api
    from app.ml.config import resolve_path  # noqa: F401

    speakers_api.reset_registry()

    store = tmp_path / "speakers.json"
    monkeypatch.setattr(
        speakers_api, "resolve_path", lambda p: store if "speakers" in str(p) else tmp_path / p
    )

    encoder = SameSpeakerEncoder()
    monkeypatch.setattr(SpeakerEncoder, "get_shared", classmethod(lambda cls, **k: encoder))

    from app.main import app

    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test"), encoder, store


# ---------------------------------------------------------------------------
# enrollment
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_enroll_then_verify(client):
    http, _, _ = client

    async with http as c:
        files = [("samples", (f"s{i}.wav", wav_bytes(frequency=200 + i * 10), "audio/wav"))
                 for i in range(5)]
        response = await c.post("/api/v1/speakers/enroll",
                                data={"identity": "alice"}, files=files)

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["identity"] == "alice"
        assert body["samples"] == 5

        response = await c.post(
            "/api/v1/speakers/verify",
            data={"identity": "alice"},
            files={"sample": ("probe.wav", wav_bytes(), "audio/wav")},
        )

        assert response.status_code == 200
        assert response.json()["decision"] == "MATCH"


@pytest.mark.anyio
async def test_enroll_rejects_too_few_samples(client):
    http, _, _ = client

    async with http as c:
        files = [("samples", (f"s{i}.wav", wav_bytes(), "audio/wav")) for i in range(2)]
        response = await c.post("/api/v1/speakers/enroll",
                                data={"identity": "bob"}, files=files)

    assert response.status_code == 400
    assert "at least 3" in response.json()["detail"]


@pytest.mark.anyio
async def test_enroll_rejects_empty_identity(client):
    http, _, _ = client

    async with http as c:
        files = [("samples", (f"s{i}.wav", wav_bytes(), "audio/wav")) for i in range(3)]
        response = await c.post("/api/v1/speakers/enroll",
                                data={"identity": "   "}, files=files)

    assert response.status_code == 400


@pytest.mark.anyio
async def test_audio_is_not_persisted(client):
    """
    The privacy guarantee. Only the embedding centroid is stored - the store
    must contain no waveform, and nothing that could reconstruct one.
    """

    http, _, store = client

    async with http as c:
        files = [("samples", (f"s{i}.wav", wav_bytes(frequency=200 + i * 10), "audio/wav"))
                 for i in range(3)]
        await c.post("/api/v1/speakers/enroll", data={"identity": "alice"}, files=files)

    assert store.exists()
    data = json.loads(store.read_text())
    entry = data["profiles"][0]

    assert set(entry) == {
        "identity", "centroid", "sample_count", "consistency",
        "encoder", "created_at", "threshold_version",
    }
    assert len(entry["centroid"]) == DIM


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_non_audio_upload_is_rejected(client):
    http, _, _ = client

    async with http as c:
        response = await c.post(
            "/api/v1/speakers/verify",
            data={"identity": "alice"},
            files={"sample": ("evil.wav", b"this is not audio at all", "audio/wav")},
        )

    assert response.status_code == 400
    assert "could not decode" in response.json()["detail"]


@pytest.mark.anyio
async def test_too_short_audio_is_rejected(client):
    http, _, _ = client

    async with http as c:
        response = await c.post(
            "/api/v1/speakers/verify",
            data={"identity": "alice"},
            files={"sample": ("blip.wav", wav_bytes(seconds=0.2), "audio/wav")},
        )

    assert response.status_code == 400
    assert "at least" in response.json()["detail"]


@pytest.mark.anyio
async def test_empty_upload_is_rejected(client):
    http, _, _ = client

    async with http as c:
        response = await c.post(
            "/api/v1/speakers/verify",
            data={"identity": "alice"},
            files={"sample": ("empty.wav", b"", "audio/wav")},
        )

    assert response.status_code == 400


# ---------------------------------------------------------------------------
# registry management
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_unknown_identity_is_no_match_not_404(client):
    """
    Verification against an unknown identity is a decision, not an error - the
    caller asked "is this Alice?" and the answer is no.
    """

    http, _, _ = client

    async with http as c:
        response = await c.post(
            "/api/v1/speakers/verify",
            data={"identity": "nobody"},
            files={"sample": ("probe.wav", wav_bytes(), "audio/wav")},
        )

    assert response.status_code == 200
    assert response.json()["decision"] == "NO_MATCH"


@pytest.mark.anyio
async def test_list_and_delete(client):
    http, _, _ = client

    async with http as c:
        files = [("samples", (f"s{i}.wav", wav_bytes(frequency=200 + i * 10), "audio/wav"))
                 for i in range(3)]
        await c.post("/api/v1/speakers/enroll", data={"identity": "alice"}, files=files)

        listing = (await c.get("/api/v1/speakers")).json()
        assert listing["count"] == 1
        assert listing["speakers"][0]["identity"] == "alice"
        assert "calibrated" in listing["thresholds"]

        assert (await c.delete("/api/v1/speakers/alice")).status_code == 200
        assert (await c.get("/api/v1/speakers")).json()["count"] == 0
        assert (await c.delete("/api/v1/speakers/alice")).status_code == 404


# ---------------------------------------------------------------------------
# the analyze endpoint
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_analyze_without_identity_says_so(client):
    """
    A low risk score from anti-spoof alone must not be mistaken for "this is
    who they claim to be". The response has to say the identity branch was
    skipped.
    """

    http, _, _ = client

    async with http as c:
        response = await c.post(
            "/api/v1/analyze",
            files={"sample": ("probe.wav", wav_bytes(), "audio/wav")},
        )

    assert response.status_code == 200
    body = response.json()

    assert body["speaker"] is None
    assert any("No identity was claimed" in n for n in body["notes"])
    assert "identity_mismatch" in body["risk"]["missing_signals"]


@pytest.mark.anyio
async def test_analyze_with_identity_includes_both_branches(client):
    http, _, _ = client

    async with http as c:
        files = [("samples", (f"s{i}.wav", wav_bytes(frequency=200 + i * 10), "audio/wav"))
                 for i in range(4)]
        await c.post("/api/v1/speakers/enroll", data={"identity": "alice"}, files=files)

        response = await c.post(
            "/api/v1/analyze",
            data={"claimed_identity": "alice"},
            files={"sample": ("probe.wav", wav_bytes(), "audio/wav")},
        )

    body = response.json()

    assert body["speaker"] is not None
    assert body["speaker"]["decision"] == "MATCH"
    assert "risk_score" in body["risk"]
    assert "decision" in body["risk"]


@pytest.mark.anyio
async def test_analyze_reports_an_untrained_detector(client):
    """
    An untrained anti-spoof model returns 0.5 as a placeholder. Feeding that
    into the fusion would manufacture a 50% risk contribution out of nothing,
    so it must be excluded and declared.
    """

    http, _, _ = client

    async with http as c:
        response = await c.post(
            "/api/v1/analyze",
            files={"sample": ("probe.wav", wav_bytes(), "audio/wav")},
        )

    body = response.json()
    status = body["anti_spoof"]["model_status"]

    if status != "neural":
        assert body["anti_spoof"]["synthetic_probability"] is None
        assert any("Anti-spoof branch did not contribute" in n for n in body["notes"])
