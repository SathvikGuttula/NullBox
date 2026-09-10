"""
Audit harness: simulate the EXACT browser stream (4096-sample PCM16 chunks of
16 kHz audio) through AudioFeaturePipeline, the same object the websocket uses.

Run:  cd backend && ../.venv/Scripts/python.exe audit_stream_sim.py
"""
from __future__ import annotations

import json
import statistics
import sys
import time

import numpy as np

from app.audio.feature_pipeline import AudioFeaturePipeline
from app.audio.vad import VoiceActivityDetector

SR = 16000
CHUNK = 4096  # samples -> 256 ms, exactly what LiveCall.tsx sends


def speechlike(seconds: float, rms_target: float, seed: int = 0,
               pause_ratio: float = 0.25) -> np.ndarray:
    """
    Voiced speech: harmonic source at a wandering F0 shaped by three formants,
    amplitude-modulated at a ~4 Hz syllable rate, with occasional word pauses.
    Scaled so the overall RMS equals rms_target (typical mic speech is
    0.02-0.12 depending on gain).
    """
    rng = np.random.default_rng(seed)
    n = int(SR * seconds)
    t = np.arange(n) / SR

    f0 = 120.0 + 25.0 * np.sin(2 * np.pi * 0.7 * t) + 8.0 * rng.standard_normal(n).cumsum() / np.sqrt(n)
    phase = 2 * np.pi * np.cumsum(f0) / SR

    sig = np.zeros(n)
    for k in range(1, 26):
        # crude formant envelope at 600 / 1400 / 2600 Hz
        fk = f0.mean() * k
        gain = sum(1.0 / (1.0 + ((fk - f) / 250.0) ** 2) for f in (600.0, 1400.0, 2600.0))
        sig += gain / k * np.sin(k * phase + rng.uniform(0, 2 * np.pi))

    syllable = 0.55 + 0.45 * np.sin(2 * np.pi * 4.0 * t)
    sig *= syllable

    # word pauses: silence pause_ratio of the time in ~350 ms blocks
    if pause_ratio > 0:
        block = int(0.35 * SR)
        for start in range(0, n, block):
            if rng.random() < pause_ratio:
                sig[start:start + block] *= 0.02

    sig += 0.002 * rng.standard_normal(n)  # room noise
    cur = float(np.sqrt(np.mean(sig ** 2)))
    return (sig / cur * rms_target).astype(np.float32)


def to_pcm16(x: np.ndarray) -> bytes:
    return np.clip(x, -1.0, 1.0).astype(np.float32).__mul__(32767).astype("<i2").tobytes()


# ---------------------------------------------------------------------------
# 1. VAD characterisation on its own (no models, fast)
# ---------------------------------------------------------------------------
def vad_sweep():
    print("=" * 78)
    print("1. VAD gate: fraction of 256 ms chunks marked speech, 30 s continuous talk")
    print("=" * 78)
    print(f"{'speech RMS':>11} {'dBFS':>7} {'speech chunks':>14} {'ratio':>7} "
          f"{'final floor':>12} {'final thr':>10}")
    for rms in (0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3):
        vad = VoiceActivityDetector(sample_rate=SR)
        audio = speechlike(30.0, rms, seed=7)
        n_sp = 0
        total = 0
        last = {}
        for i in range(0, len(audio) - CHUNK + 1, CHUNK):
            last = vad.process(audio[i:i + CHUNK])
            total += 1
            n_sp += int(last["speech"])
        db = 20 * np.log10(rms)
        print(f"{rms:>11.3f} {db:>7.1f} {n_sp:>8}/{total:<5} {n_sp / total:>7.2f} "
              f"{last.get('noise_floor', 0):>12.5f} {last.get('threshold', 0):>10.5f}")
    print()

    # the realistic case: 2 s of room silence first, then talking
    print("   with 2 s of genuine room silence before speech starts:")
    for rms in (0.02, 0.05, 0.12):
        vad = VoiceActivityDetector(sample_rate=SR)
        rng = np.random.default_rng(3)
        lead = (0.002 * rng.standard_normal(2 * SR)).astype(np.float32)
        audio = np.concatenate([lead, speechlike(28.0, rms, seed=7)])
        n_sp = tot = 0
        for i in range(0, len(audio) - CHUNK + 1, CHUNK):
            r = vad.process(audio[i:i + CHUNK])
            tot += 1
            n_sp += int(r["speech"])
        print(f"      RMS {rms:.3f}: {n_sp}/{tot} chunks speech ({n_sp / tot:.2f})")
    print()


# ---------------------------------------------------------------------------
# 2. Full pipeline, chunk by chunk, timed
# ---------------------------------------------------------------------------
def full_run(rms_target: float, seconds: float = 40.0, label: str = ""):
    print("=" * 78)
    print(f"2. Full pipeline @ speech RMS {rms_target:.3f} {label}")
    print("=" * 78)

    t0 = time.perf_counter()
    pipe = AudioFeaturePipeline(sample_rate=SR, claimed_identity=None)
    print(f"   pipeline construction (loads wav2vec2 + checkpoint): "
          f"{time.perf_counter() - t0:.2f}s")

    audio = speechlike(seconds, rms_target, seed=11)
    chunks = [audio[i:i + CHUNK] for i in range(0, len(audio) - CHUNK + 1, CHUNK)]

    times = []
    first_df = None
    first_risk = None
    speech_chunks = 0
    rows = []

    for idx, c in enumerate(chunks, start=1):
        b = to_pcm16(c)
        t = time.perf_counter()
        out = pipe.process(b)
        dt = (time.perf_counter() - t) * 1000.0
        times.append(dt)

        speech_chunks += int(out["vad"].get("speech", False))

        if out["deepfake"]["available"] and first_df is None:
            first_df = idx
        if out["risk"] is not None and first_risk is None:
            first_risk = idx

        rows.append((idx, dt, out))

    print(f"   chunks sent                : {len(chunks)}  "
          f"({len(chunks) * CHUNK / SR:.1f}s of audio)")
    print(f"   chunks VAD called speech   : {speech_chunks} "
          f"({speech_chunks / len(chunks):.2%})")
    print(f"   first deepfake result at   : chunk {first_df} "
          f"({'never' if first_df is None else f'{first_df * CHUNK / SR:.2f}s of stream'})")
    print(f"   first fused risk at        : chunk {first_risk} "
          f"({'never' if first_risk is None else f'{first_risk * CHUNK / SR:.2f}s of stream'})")
    print(f"   windows scored total       : {rows[-1][2]['deepfake']['windows_scored']}")
    print()
    print(f"   per-chunk wall clock (ms)  min {min(times):7.1f}  "
          f"med {statistics.median(times):7.1f}  "
          f"mean {statistics.mean(times):7.1f}  "
          f"p95 {sorted(times)[int(0.95 * len(times))]:7.1f}  "
          f"max {max(times):7.1f}")
    print(f"   real-time budget per chunk : 256.0 ms")
    over = sum(1 for x in times if x > 256.0)
    print(f"   chunks over budget         : {over}/{len(times)} ({over / len(times):.1%})")
    backlog = sum(times) - len(times) * 256.0
    print(f"   cumulative drift over {len(times)} chunks: {backlog / 1000.0:+.2f}s "
          f"(positive = the server falls behind the mic and never catches up)")
    print()

    # split the cost: heavy-feature chunks vs light
    heavy = [t for i, t in enumerate(times, 1) if i % 3 == 1]
    light = [t for i, t in enumerate(times, 1) if i % 3 != 1]
    print(f"   prosody (librosa.pyin) chunks : med {statistics.median(heavy):7.1f} ms")
    print(f"   cached-feature chunks         : med {statistics.median(light):7.1f} ms")
    print()

    if first_risk:
        r = rows[first_risk - 1][2]["risk"]
        print("   first risk payload:")
        print("     " + json.dumps(
            {k: r[k] for k in ("risk_score", "risk_level", "decision",
                               "available_signals", "missing_signals", "calibrated")}))
        print("     reasons:", r["reasons"])
    last = rows[-1][2]
    if last["risk"]:
        print("   final risk:", json.dumps(
            {k: last["risk"][k] for k in ("risk_score", "risk_level", "decision")}))
    if last["deepfake"]["result"]:
        d = last["deepfake"]["result"]
        print("   final deepfake:", json.dumps(
            {k: d[k] for k in ("raw_probability", "smoothed_probability",
                               "model_status", "window_seconds", "window_index")}))
    print("   final stream block:", json.dumps(last["stream"]))
    print()
    return rows, times


# ---------------------------------------------------------------------------
# 3. silence-only stream: does the UI ever get a risk?
# ---------------------------------------------------------------------------
def silence_run():
    print("=" * 78)
    print("3. Quiet-room stream (no speech): 20 s")
    print("=" * 78)
    pipe = AudioFeaturePipeline(sample_rate=SR)
    rng = np.random.default_rng(5)
    audio = (0.002 * rng.standard_normal(20 * SR)).astype(np.float32)
    got_risk = False
    for i in range(0, len(audio) - CHUNK + 1, CHUNK):
        out = pipe.process(to_pcm16(audio[i:i + CHUNK]))
        if out["risk"] is not None:
            got_risk = True
    print(f"   risk ever produced: {got_risk}")
    print(f"   deepfake available: {out['deepfake']['available']}, "
          f"windows_scored={out['deepfake']['windows_scored']}")
    print(f"   stream: {json.dumps(out['stream'])}")
    print()


# ---------------------------------------------------------------------------
# 4. discontinuity: what the model actually sees when VAD drops chunks
# ---------------------------------------------------------------------------
def splice_check():
    print("=" * 78)
    print("4. Does the anti-spoof window get spliced audio when VAD drops chunks?")
    print("=" * 78)
    from app.ml.inference import StreamingInferenceEngine

    class Spy:
        def __init__(self):
            self.windows = []

        def predict(self, w, sr=None):
            self.windows.append(np.array(w))
            return {"synthetic_probability": 0.5, "confidence": 0.5,
                    "model_status": "neural"}

    spy = Spy()
    eng = StreamingInferenceEngine(detector=spy, sample_rate=SR)
    # feed 4 s of a ramp so any splice shows up as a discontinuity
    ramp = np.arange(SR * 6, dtype=np.float32) / (SR * 6)
    kept = []
    for i in range(0, len(ramp) - CHUNK + 1, CHUNK):
        c = ramp[i:i + CHUNK]
        if (i // CHUNK) % 3 != 1:  # pretend VAD drops every 3rd chunk
            kept.append(c)
            eng.add_audio(c)
    if spy.windows:
        w = spy.windows[0]
        jumps = np.abs(np.diff(w))
        print(f"   window len {w.size} samples; max sample-to-sample jump "
              f"{jumps.max():.5f} vs smooth-ramp step {1/(SR*6):.7f}")
        print(f"   number of jumps > 10x the smooth step: "
              f"{int((jumps > 10 / (SR * 6)).sum())}")
    print()


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "vad"):
        vad_sweep()
    if which in ("all", "splice"):
        splice_check()
    if which in ("all", "full"):
        full_run(0.12, 40.0, "(loud / close mic)")
    if which in ("all", "quiet"):
        full_run(0.04, 40.0, "(normal laptop mic level)")
    if which in ("all", "silence"):
        silence_run()
