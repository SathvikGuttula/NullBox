"""
End-to-end verification against a running VoxShield API.

    python scripts/verify_system.py                    # against localhost:8000
    python scripts/verify_system.py --url http://...   # somewhere else
    python scripts/verify_system.py --quick            # skip the websocket

Every check states what it expects and why, hits the real HTTP or websocket
endpoint, and prints PASS or FAIL with the number it actually got. The exit
code is the number of failures, so this is usable in CI.

This is not a substitute for the unit suite (``pytest``), which tests the
pieces in isolation with the models stubbed. This exercises the assembled
system with real audio and a real checkpoint, which is where the interesting
failures have all been: a detector that silently reported itself untrained
because a path resolved against the wrong directory, a registry whose
calibration file was gitignored, an endpoint that scored four seconds of hold
music as 99.7% synthetic.

Speaker verification without a human
------------------------------------
The speaker checks enrol one Windows TTS voice and verify against another.
They are different speakers as far as ECAPA is concerned - measured cross-voice
similarity is around 0.1 against an in-voice similarity above 0.7 - so the full
enrol / MATCH / NO_MATCH path can be exercised on a laptop with no recording
and no dataset. It is a smoke test of the plumbing, not a measurement of
speaker-verification accuracy; that number (0.373% EER on 67 unseen speakers)
comes from the calibration run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.config import resolve_path  # noqa: E402


class Report:
    """Accumulates results and prints them as they happen."""

    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.skipped = 0

    def section(self, title: str) -> None:
        print(f"\n{title}")
        print("-" * len(title))

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        if ok:
            self.passed += 1
            print(f"  PASS  {name}" + (f"   {detail}" if detail else ""))
        else:
            self.failed += 1
            print(f"  FAIL  {name}" + (f"   {detail}" if detail else ""))
        return ok

    def skip(self, name: str, why: str) -> None:
        self.skipped += 1
        print(f"  SKIP  {name}   {why}")

    def summary(self) -> int:
        print()
        print("=" * 62)
        print(f"  {self.passed} passed, {self.failed} failed, {self.skipped} skipped")
        print("=" * 62)
        return self.failed


def post_analyze(url: str, path: Path, **fields) -> tuple[int, dict]:
    with path.open("rb") as handle:
        response = requests.post(
            f"{url}/api/v1/analyze",
            files={"sample": (path.name, handle, "audio/wav")},
            data={k: v for k, v in fields.items() if v is not None},
            timeout=180,
        )
    try:
        return response.status_code, response.json()
    except ValueError:
        return response.status_code, {"raw": response.text[:200]}


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


def check_system(url: str, report: Report) -> dict:
    report.section("System")

    try:
        health = requests.get(f"{url}/api/v1/health", timeout=10).json()
    except Exception as exc:
        report.check("API is reachable", False, str(exc)[:80])
        print(f"\n  Nothing else can run. Start it with:\n"
              f"    cd backend && python -m uvicorn app.main:app --port 8000\n")
        raise SystemExit(1)

    report.check("API is reachable", health.get("status") == "healthy")

    status = requests.get(f"{url}/api/v1/status", timeout=60).json()

    anti = status["anti_spoof"]
    report.check(
        "anti-spoof checkpoint present",
        anti["checkpoint_present"],
        f"{anti.get('checkpoint_mb')} MB",
    )
    report.check(
        "detector scores are calibrated",
        anti["calibration"].get("present") and not anti["calibration"].get("error"),
        f"scale {anti['calibration'].get('scale')}, bias {anti['calibration'].get('bias')}",
    )

    detection = anti["calibration"].get("detection_rate_at_0.5")
    report.check(
        "calibration records its held-out performance",
        detection is not None and detection > 0.90,
        f"detects {detection:.1%} at {anti['calibration'].get('false_alarm_at_0.5'):.2%} "
        f"false alarm" if detection else "",
    )

    speaker = status["speaker"]
    report.check(
        "speaker encoder available",
        speaker.get("speechbrain_installed", False),
    )
    report.check(
        "speaker thresholds are calibrated",
        speaker.get("thresholds", {}).get("calibrated", False),
        speaker.get("thresholds", {}).get("version", ""),
    )

    report.check(
        "fusion declares itself uncalibrated",
        status["fusion"]["calibrated"] is False,
        "correct - the weights are placeholders and must say so",
    )

    if status.get("warnings"):
        print("\n  Declared warnings (these are the system being honest, not failures):")
        for warning in status["warnings"]:
            print(f"    - {warning[:100]}")

    return status


def check_spoof_detection(url: str, corpus: Path, report: Report) -> None:
    report.section("Anti-spoof: synthetic speech must be detected")

    files = sorted((corpus / "spoof").glob("*.wav"))

    if not files:
        report.skip(
            "synthetic speech detection",
            "no spoof samples - run scripts/make_test_audio.py",
        )
        return

    for path in files:
        code, body = post_analyze(url, path)

        if code != 200:
            report.check(path.name, False, f"HTTP {code}")
            continue

        anti = body["anti_spoof"]
        probability = anti["synthetic_probability"]
        risk = body["risk"]

        ok = (
            anti["model_status"] == "neural"
            and probability is not None
            and probability >= 0.5
        )
        report.check(
            path.name,
            ok,
            f"p={probability:.4f} (raw {anti['raw_probability']:.4f})  "
            f"risk {risk['risk_score']:.0f} {risk['decision']}"
            if probability is not None
            else f"status={anti['model_status']}",
        )


def check_edge_cases(url: str, corpus: Path, report: Report) -> None:
    report.section("Edge cases: degenerate input must be refused, not guessed")

    # name -> (expected http status, expected model_status or None)
    expectations = {
        "silence.wav": (200, "no_speech"),
        "white_noise.wav": (200, "no_speech"),
        "pure_tone.wav": (200, "no_speech"),
        "dial_tone.wav": (200, "no_speech"),
        "mains_hum.wav": (200, "no_speech"),
        "dc_offset.wav": (200, "no_speech"),
        "too_short.wav": (400, None),
        "not_audio.wav": (400, None),
        "empty.wav": (400, None),
        "speech_stereo.wav": (200, "neural"),
        "speech_8k.wav": (200, "neural"),
        "speech_48k.wav": (200, "neural"),
        "speech_clipped.wav": (200, "neural"),
        "speech_quiet.wav": (200, "neural"),
        "speech_muted.wav": (200, "no_speech"),
    }

    for name, (expected_code, expected_status) in expectations.items():
        path = corpus / "edge" / name

        if not path.exists():
            report.skip(name, "not generated")
            continue

        code, body = post_analyze(url, path)

        if code != expected_code:
            report.check(name, False, f"HTTP {code}, expected {expected_code}")
            continue

        if expected_status is None:
            report.check(name, True, f"HTTP {code} - {str(body.get('detail', ''))[:52]}")
            continue

        actual = body["anti_spoof"]["model_status"]
        ok = actual == expected_status
        detail = f"status={actual}"

        if expected_status == "no_speech":
            # The critical property: a refused clip must NOT come back as ALLOW.
            # Reporting "cleared" for audio nobody could assess is the failure
            # this whole gate exists to prevent.
            decision = body["risk"]["decision"]
            ok = ok and decision == "INSUFFICIENT_EVIDENCE"
            detail += f", decision={decision}"

        report.check(name, ok, detail)


def check_speaker(url: str, corpus: Path, report: Report) -> None:
    report.section("Speaker verification: enrol one voice, reject another")

    voices: dict[str, list[Path]] = {}
    for path in sorted((corpus / "spoof").glob("tts_*.wav")):
        voice = path.stem.split("_")[1]
        voices.setdefault(voice, []).append(path)

    usable = {v: p for v, p in voices.items() if len(p) >= 3}

    if len(usable) < 2:
        report.skip(
            "speaker enrol / verify",
            f"need two voices with 3+ clips each, found {len(usable)}",
        )
        return

    names = sorted(usable)
    enrolled_voice, other_voice = names[0], names[1]
    identity = f"verify.{enrolled_voice}"

    requests.delete(f"{url}/api/v1/speakers/{identity}", timeout=30)

    files = [
        ("samples", (p.name, p.open("rb"), "audio/wav")) for p in usable[enrolled_voice]
    ]
    try:
        response = requests.post(
            f"{url}/api/v1/speakers/enroll",
            files=files,
            data={"identity": identity},
            timeout=300,
        )
    finally:
        for _, (_, handle, _) in files:
            handle.close()

    if response.status_code != 200:
        report.check("enrol a voice", False, f"HTTP {response.status_code} {response.text[:80]}")
        return

    profile = response.json()
    report.check(
        "enrol a voice",
        profile["samples"] >= 3,
        f"{profile['samples']} samples, consistency {profile['consistency']:.3f}",
    )
    report.check(
        "enrolment stores no audio",
        "discarded" in profile.get("note", "").lower(),
        profile.get("note", "")[:56],
    )

    def verify(path: Path) -> dict:
        with path.open("rb") as handle:
            return requests.post(
                f"{url}/api/v1/speakers/verify",
                files={"sample": (path.name, handle, "audio/wav")},
                data={"identity": identity},
                timeout=180,
            ).json()

    same = verify(usable[enrolled_voice][0])
    report.check(
        "same voice -> MATCH",
        same["decision"] == "MATCH",
        f"similarity {same['similarity']:.3f}, decision {same['decision']}",
    )

    different = verify(usable[other_voice][0])
    report.check(
        "different voice -> NO_MATCH",
        different["decision"] == "NO_MATCH",
        f"similarity {different['similarity']:.3f}, decision {different['decision']}",
    )

    report.check(
        "the two are clearly separated",
        same["similarity"] - different["similarity"] > 0.3,
        f"gap {same['similarity'] - different['similarity']:.3f}",
    )

    unknown = requests.post(
        f"{url}/api/v1/speakers/verify",
        files={"sample": (usable[enrolled_voice][0].name,
                          usable[enrolled_voice][0].open("rb"), "audio/wav")},
        data={"identity": "nobody.enrolled.as.this"},
        timeout=180,
    ).json()
    report.check(
        "unenrolled identity -> NO_MATCH, not an error",
        unknown["decision"] == "NO_MATCH",
        unknown["reasons"][0][:60] if unknown.get("reasons") else "",
    )

    # The impersonation case: right claim, wrong voice, both branches needed.
    code, body = post_analyze(
        url, usable[other_voice][0], claimed_identity=identity
    )
    if code == 200 and body.get("speaker"):
        report.check(
            "impersonator escalates through the fused risk",
            body["risk"]["decision"] in ("SECONDARY_VERIFICATION", "HIGH_RISK_WORKFLOW"),
            f"risk {body['risk']['risk_score']:.0f} {body['risk']['decision']}",
        )
    else:
        report.skip("impersonator escalates", f"HTTP {code}")

    requests.delete(f"{url}/api/v1/speakers/{identity}", timeout=30)


def check_identity_validation(url: str, corpus: Path, report: Report) -> None:
    report.section("Input validation")

    sample = next((corpus / "spoof").glob("*.wav"), None)
    if sample is None:
        report.skip("identity validation", "no sample audio")
        return

    for hostile in ("../../../../tmp/pwned", "a/b", "X" * 500, ""):
        with sample.open("rb") as handle:
            response = requests.post(
                f"{url}/api/v1/speakers/verify",
                files={"sample": (sample.name, handle, "audio/wav")},
                data={"identity": hostile},
                timeout=120,
            )
        label = (hostile[:24] + "...") if len(hostile) > 24 else (hostile or "<empty>")
        report.check(
            f"identity {label!r} rejected",
            response.status_code in (400, 422),
            f"HTTP {response.status_code}",
        )

    for bad_context, why in [
        ('{"caller_known": false', "malformed JSON"),
        ("[1, 2, 3]", "an array, not an object"),
        ('{"not_a_field": 1}', "unknown field"),
        ('{"transaction_amount": "lots"}', "wrong type"),
    ]:
        code, body = post_analyze(url, sample, context=bad_context)
        report.check(
            f"context rejected: {why}",
            code == 400,
            f"HTTP {code} - {str(body.get('detail',''))[:48]}",
        )


def check_context_fusion(url: str, corpus: Path, report: Report) -> None:
    report.section("Context fusion")

    sample = next((corpus / "spoof").glob("*.wav"), None)
    if sample is None:
        report.skip("context fusion", "no sample audio")
        return

    clean = json.dumps(
        {
            "caller_known": True,
            "trusted_contact": True,
            "device_known": True,
            "beneficiary_known": True,
            "hour_of_day": 14,
        }
    )
    fraudulent = json.dumps(
        {
            "caller_known": False,
            "trusted_contact": False,
            "device_known": False,
            "hour_of_day": 23,
            "transaction_amount": 250000,
            "typical_transaction_amount": 2000,
            "beneficiary_known": False,
            "beneficiary_age_days": 0,
        }
    )

    _, low = post_analyze(url, sample, context=clean)
    _, high = post_analyze(url, sample, context=fraudulent)

    report.check(
        "benign context scores 0 risk",
        low["context"]["risk"] == 0.0,
        f"{len(low['context']['signals_used'])} signals used",
    )
    report.check(
        "fraud-shaped context scores high",
        high["context"]["risk"] > 0.9,
        f"risk {high['context']['risk']:.2f}, {len(high['context']['reasons'])} reasons",
    )
    report.check(
        "context raises the fused score",
        high["risk"]["risk_score"] >= low["risk"]["risk_score"],
        f"{low['risk']['risk_score']:.0f} -> {high['risk']['risk_score']:.0f}",
    )
    report.check(
        "high value plus risk requires a human",
        high["policy"]["requires_human"],
        ", ".join(a["code"] for a in high["policy"]["actions"])[:70],
    )
    report.check(
        "no action ever rejects the transaction",
        all(
            a["code"] != "REJECT" and "reject the" not in a["description"].lower()
            for a in high["policy"]["actions"]
        ),
    )
    report.check(
        "the response declares its inputs uncalibrated",
        high["policy"]["calibrated"] is False
        and any("uncalibrated" in r.lower() for r in high["policy"]["rationale"]),
    )


def check_live_stream(url: str, corpus: Path, report: Report) -> None:
    report.section("Live websocket stream")

    try:
        import asyncio

        import numpy as np
        import soundfile as sf
        import websockets
    except ImportError as exc:
        report.skip("websocket stream", f"missing dependency: {exc}")
        return

    sample = next((corpus / "spoof").glob("*_long.wav"), None) or next(
        (corpus / "spoof").glob("*.wav"), None
    )
    if sample is None:
        report.skip("websocket stream", "no sample audio")
        return

    audio, rate = sf.read(sample, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    ws_url = url.replace("http://", "ws://").replace("https://", "wss://")

    async def run() -> dict:
        call_id = requests.post(f"{url}/api/v1/calls/demo", timeout=30).json()["call_id"]

        results: list[dict] = []

        # 4096 samples at 16 kHz = 256 ms, exactly what the browser's
        # ScriptProcessorNode sends.
        chunk = 4096

        async with websockets.connect(
            f"{ws_url}/api/v1/calls/{call_id}/stream", max_size=None
        ) as socket:
            for start in range(0, len(audio) - chunk, chunk):
                pcm = (np.clip(audio[start : start + chunk], -1, 1) * 32767).astype("<i2")
                await socket.send(pcm.tobytes())
                message = json.loads(await asyncio.wait_for(socket.recv(), timeout=60))
                if message.get("type") == "voice_analysis":
                    results.append(message["analysis"])

        return {"results": results, "chunks": len(results)}

    try:
        outcome = asyncio.run(run())
    except Exception as exc:
        report.check("websocket stream completes", False, str(exc)[:90])
        return

    results = outcome["results"]

    report.check(
        "websocket accepts browser-sized chunks",
        outcome["chunks"] > 0,
        f"{outcome['chunks']} chunks of 256 ms",
    )

    scored = [
        r for r in results
        if r.get("deepfake", {}).get("result")
        and r["deepfake"]["result"].get("model_status") == "neural"
    ]
    report.check(
        "the detector scores windows during the call",
        len(scored) > 0,
        f"{len(scored)} windows scored",
    )

    with_risk = [r for r in results if r.get("risk")]
    report.check("a fused risk is produced", len(with_risk) > 0)

    if with_risk:
        final = with_risk[-1]["risk"]
        report.check(
            "synthetic speech drives the live risk up",
            final["risk_score"] >= 50,
            f"final risk {final['risk_score']:.0f} {final['decision']}",
        )
        report.check(
            "the live risk carries reasons",
            bool(final.get("reasons")),
            final["reasons"][0][:60] if final.get("reasons") else "none",
        )

    policies = [r for r in results if r.get("policy")]
    report.check(
        "the live stream emits policy actions",
        len(policies) > 0,
        ", ".join(a["code"] for a in policies[-1]["policy"]["actions"])[:60]
        if policies else "",
    )

    # A bad context must be refused at connect time with a readable reason,
    # not accepted and then silently ignored.
    async def bad_context() -> str:
        call_id = requests.post(f"{url}/api/v1/calls/demo", timeout=30).json()["call_id"]
        try:
            async with websockets.connect(
                f"{ws_url}/api/v1/calls/{call_id}/stream?context=NOTJSON"
            ):
                return "accepted"
        except Exception as exc:
            return str(exc)

    outcome = asyncio.run(bad_context())
    report.check(
        "websocket rejects a malformed context",
        outcome != "accepted",
        outcome[:70],
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify a running VoxShield API end to end.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--corpus", default="datasets/test-audio")
    parser.add_argument("--quick", action="store_true", help="skip the websocket check")
    args = parser.parse_args()

    url = args.url.rstrip("/")
    corpus = resolve_path(args.corpus)

    print("=" * 62)
    print("  VoxShield end-to-end verification")
    print(f"  API    {url}")
    print(f"  corpus {corpus}")
    print("=" * 62)

    if not corpus.exists():
        print(f"\nNo test corpus at {corpus}.")
        print("Generate it first:  python scripts/make_test_audio.py")
        return 1

    report = Report()

    check_system(url, report)
    check_spoof_detection(url, corpus, report)
    check_edge_cases(url, corpus, report)
    check_speaker(url, corpus, report)
    check_identity_validation(url, corpus, report)
    check_context_fusion(url, corpus, report)

    if args.quick:
        report.skip("websocket stream", "--quick")
    else:
        check_live_stream(url, corpus, report)

    return report.summary()


if __name__ == "__main__":
    raise SystemExit(main())
