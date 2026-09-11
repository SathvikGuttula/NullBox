"""
Build a local test corpus, with no dataset download.

    python scripts/make_test_audio.py

Why this exists
---------------
The ASVspoof corpus is 7.6 GB and the local copy is a 50 MB fragment, so there
is no real audio on this machine to test against. Without it every check is
either synthetic tones - which the system now correctly refuses to score - or
nothing at all.

Windows ships three text-to-speech voices through SAPI. They are a completely
different synthesis family from anything in ASVspoof 2019 LA (which is neural
TTS and voice conversion), so they are a genuine out-of-distribution test, and
the shipped checkpoint flags them at 0.956 - 0.977 raw, 1.000 calibrated. That
makes them a real spoof sample that costs nothing to produce.

What it writes
--------------
``datasets/test-audio/``

    spoof/      SAPI text-to-speech, one file per installed voice, plus a
                short and a long variant. These SHOULD be detected.

    edge/       The degenerate inputs a tester reaches for in the first two
                minutes: silence, white noise, a pure tone, a DC offset, a
                clipped signal, 8 kHz, 48 kHz, stereo, too-short, and a text
                file with a .wav extension. Each has a documented expected
                outcome - most should be REFUSED, not scored.

    bonafide/   Empty, with a README. Genuine human speech has to come from a
                microphone; nothing here can fabricate it, and a synthetic
                stand-in would be a spoof sample mislabelled as bonafide,
                which is worse than an empty folder.

On the missing half
-------------------
A detector evaluated only on spoof samples cannot tell you anything about its
false-alarm rate. Record yourself through the UI to supply the other half - see
TESTING.md. The published false-alarm number (0.79%) comes from 3,678 held-out
bonafide utterances in the evaluation set, not from anything generated here.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.config import resolve_path  # noqa: E402

SAMPLE_RATE = 16000

SCRIPTS = {
    "transfer": (
        "Hello, this is the finance director speaking. I need you to authorise "
        "an urgent wire transfer of two hundred and fifty thousand rupees to a "
        "new supplier account before the end of the day today. Please treat "
        "this as confidential until the deal is announced."
    ),
    "short": "This is a short verification sample for the voice security system.",
    "long": (
        "Good afternoon. I am calling about the account ending in four four two "
        "one. There has been an unusual pattern of activity and I need to walk "
        "you through the verification steps. First I will need to confirm a few "
        "details, and then we can look at the transactions together. This will "
        "only take a few minutes of your time. I appreciate your patience while "
        "we sort this out, and I want to reassure you that the account is "
        "secure while we are speaking."
    ),
}


POWERSHELL_TEMPLATE = """
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo({rate}, `
    [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, `
    [System.Speech.AudioFormat.AudioChannel]::Mono)
$index = 0
foreach ($voice in $synth.GetInstalledVoices()) {{
    $name = $voice.VoiceInfo.Name
    if (-not $voice.Enabled) {{ continue }}
    try {{
        $synth.SelectVoice($name)
        $safe = ($name -replace '[^A-Za-z0-9]', '')
        $path = Join-Path '{outdir}' ("tts_" + $safe + "_{label}.wav")
        $synth.SetOutputToWaveFile($path, $format)
        $synth.Speak('{text}')
        $synth.SetOutputToNull()
        Write-Output ("OK|" + $path + "|" + $name)
        $index++
    }} catch {{
        Write-Output ("FAIL|" + $name + "|" + $_.Exception.Message)
    }}
}}
$synth.Dispose()
"""


def generate_tts(out_dir: Path) -> list[tuple[Path, str]]:
    """
    Synthesise each script with every installed Windows voice.

    Returns the files written. On a non-Windows machine, or one with no voices
    installed, this returns an empty list rather than failing - the edge-case
    corpus is still worth having on its own.
    """

    if sys.platform != "win32":
        print("  not Windows - skipping text-to-speech generation")
        return []

    written: list[tuple[Path, str]] = []

    for label, text in SCRIPTS.items():
        # Single quotes are the string delimiter in the generated PowerShell,
        # and PowerShell escapes them by doubling.
        script = POWERSHELL_TEMPLATE.format(
            rate=SAMPLE_RATE,
            outdir=str(out_dir).replace("'", "''"),
            label=label,
            text=text.replace("'", "''"),
        )

        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                capture_output=True,
                text=True,
                timeout=180,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(f"  text-to-speech failed for {label!r}: {exc}")
            continue

        for line in result.stdout.splitlines():
            parts = line.strip().split("|")
            if parts[0] == "OK" and len(parts) >= 3:
                written.append((Path(parts[1]), parts[2]))
            elif parts[0] == "FAIL":
                print(f"  voice {parts[1]} failed: {parts[2] if len(parts) > 2 else ''}")

    return written


def generate_edge_cases(out_dir: Path, reference: Path | None) -> list[tuple[str, str]]:
    """
    The degenerate inputs, each with the outcome it should produce.

    Returns ``(filename, expectation)`` pairs so the README and TESTING.md do
    not have to restate them and drift apart.
    """

    rng = np.random.default_rng(1234)
    seconds = 4.0
    count = int(SAMPLE_RATE * seconds)
    t = np.arange(count) / SAMPLE_RATE

    cases: list[tuple[str, np.ndarray, int, str]] = [
        (
            "silence.wav",
            np.zeros(count, dtype=np.float32),
            SAMPLE_RATE,
            "REFUSED - no_speech. The model has never seen silence; uncaught it "
            "scores 0.999 synthetic.",
        ),
        (
            "white_noise.wav",
            rng.normal(0, 0.1, count).astype(np.float32),
            SAMPLE_RATE,
            "REFUSED - no_speech. The energy VAD passes this; the speech gate "
            "is what stops it.",
        ),
        (
            "pure_tone.wav",
            (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32),
            SAMPLE_RATE,
            "REFUSED - no_speech. Steady signal, no gaps between words.",
        ),
        (
            "dial_tone.wav",
            (0.3 * (np.sin(2 * np.pi * 350 * t) + np.sin(2 * np.pi * 440 * t)) / 2).astype(
                np.float32
            ),
            SAMPLE_RATE,
            "REFUSED - no_speech. A real dial tone is 350 Hz + 440 Hz.",
        ),
        (
            "mains_hum.wav",
            (
                0.2 * sum(np.sin(2 * np.pi * 50 * (k + 1) * t) / (k + 1) for k in range(6))
            ).astype(np.float32),
            SAMPLE_RATE,
            "REFUSED - no_speech. 50 Hz hum with harmonics.",
        ),
        (
            "dc_offset.wav",
            np.full(count, 0.4, dtype=np.float32),
            SAMPLE_RATE,
            "REFUSED - no_speech. Constant signal.",
        ),
        (
            "too_short.wav",
            rng.normal(0, 0.1, int(SAMPLE_RATE * 0.4)).astype(np.float32),
            SAMPLE_RATE,
            "HTTP 400 - under the one-second minimum.",
        ),
    ]

    written: list[tuple[str, str]] = []

    for name, audio, rate, expectation in cases:
        sf.write(out_dir / name, audio, rate)
        written.append((name, expectation))

    # Format variants, which need a real speech file to derive from.
    if reference is not None and reference.exists():
        speech, rate = sf.read(reference, dtype="float32", always_2d=False)
        if speech.ndim > 1:
            speech = speech.mean(axis=1)

        variants: list[tuple[str, np.ndarray, int, str]] = [
            (
                "speech_stereo.wav",
                np.stack([speech, speech], axis=1),
                rate,
                "DETECTED. Channels are averaged to mono on the way in.",
            ),
            (
                "speech_8k.wav",
                speech[::2],
                rate // 2,
                "DETECTED. Resampled to 16 kHz on the way in - this is the "
                "telephone-bandwidth case.",
            ),
            (
                "speech_48k.wav",
                np.repeat(speech, 3),
                rate * 3,
                "DETECTED. Resampled down to 16 kHz.",
            ),
            (
                "speech_clipped.wav",
                np.clip(speech * 20.0, -1.0, 1.0).astype(np.float32),
                rate,
                "DETECTED. Heavy clipping does not rescue a synthetic sample.",
            ),
            (
                "speech_quiet.wav",
                (speech * 0.05).astype(np.float32),
                rate,
                "DETECTED. Quiet but well above the silence floor; the "
                "detector peak-normalises before scoring.",
            ),
            (
                "speech_muted.wav",
                (speech * 0.001).astype(np.float32),
                rate,
                "REFUSED - no_speech. Below -60 dBFS is a muted microphone, "
                "and amplifying it would score the noise floor.",
            ),
        ]

        for name, audio, rate_out, expectation in variants:
            sf.write(out_dir / name, audio, rate_out)
            written.append((name, expectation))

    # Not audio at all.
    (out_dir / "not_audio.wav").write_bytes(
        b"This is a plain text file with a .wav extension. It is not audio.\n"
    )
    written.append(
        ("not_audio.wav", "HTTP 400 - cannot be decoded, with a message saying so.")
    )

    (out_dir / "empty.wav").write_bytes(b"")
    written.append(("empty.wav", "HTTP 400 - the file is empty."))

    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a local test corpus with no dataset download.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out", default="datasets/test-audio")
    args = parser.parse_args()

    root = resolve_path(args.out)
    spoof_dir = root / "spoof"
    edge_dir = root / "edge"
    bonafide_dir = root / "bonafide"

    for directory in (spoof_dir, edge_dir, bonafide_dir):
        directory.mkdir(parents=True, exist_ok=True)

    print(f"writing to {root}\n")

    print("synthetic speech (Windows SAPI)")
    tts = generate_tts(spoof_dir)

    for path, voice in tts:
        info = sf.info(str(path))
        print(f"  {path.name:<44} {info.duration:5.1f}s  [{voice}]")

    if not tts:
        print("  no voices available - the spoof folder is empty")
        print("  On Windows: Settings > Time & Language > Speech > Manage voices")

    reference = tts[0][0] if tts else None

    print("\nedge cases")
    edges = generate_edge_cases(edge_dir, reference)
    for name, expectation in edges:
        print(f"  {name:<24} {expectation[:66]}")

    readme = root / "README.md"
    lines = [
        "# Local test corpus",
        "",
        "Generated by `python backend/scripts/make_test_audio.py`. Not committed",
        "(`*.wav` is gitignored) - regenerate it on any machine in a few seconds.",
        "",
        "## spoof/",
        "",
        "Windows SAPI text-to-speech. A different synthesis family from anything",
        "in ASVspoof 2019 LA, so this is genuinely out-of-distribution for the",
        "checkpoint - and it is still detected, at 0.956-0.977 raw / 1.000",
        "calibrated. Every file here **should** be flagged.",
        "",
    ]

    if tts:
        lines.append("| file | voice |")
        lines.append("| --- | --- |")
        for path, voice in tts:
            lines.append(f"| `{path.name}` | {voice} |")
        lines.append("")

    lines += [
        "## edge/",
        "",
        "The inputs a tester tries first. Most of these **should be refused**,",
        "not scored: the detector is a speech model, and asked about silence or",
        "noise it answers confidently and wrongly (0.999 and 0.997 synthetic",
        "respectively, measured). A refusal reports `model_status: no_speech`",
        "and drops the branch from the fusion.",
        "",
        "| file | expected |",
        "| --- | --- |",
    ]
    for name, expectation in edges:
        lines.append(f"| `{name}` | {expectation} |")

    lines += [
        "",
        "## bonafide/",
        "",
        "Deliberately empty. Genuine human speech has to come from a microphone,",
        "and nothing in this script can fabricate it - a synthetic stand-in would",
        "be a spoof sample mislabelled as bonafide, which is worse than an empty",
        "folder. Record your own through the UI (see TESTING.md).",
        "",
        "Without bonafide samples you can measure detection but **not** false",
        "alarms. The published 0.79% false-alarm rate comes from 3,678 held-out",
        "bonafide utterances in the evaluation set, not from anything here.",
        "",
    ]

    readme.write_text("\n".join(lines), encoding="utf-8")

    (bonafide_dir / "README.md").write_text(
        "Put recordings of real people here.\n\n"
        "Record through the VoxShield UI, or with Windows Voice Recorder and\n"
        "export as WAV. At least one second, ideally five to ten. These are the\n"
        "samples that tell you the false-alarm rate; the spoof folder cannot.\n",
        encoding="utf-8",
    )

    print(f"\nwrote {readme}")
    print(f"\n{len(tts)} spoof file(s), {len(edges)} edge case(s), 0 bonafide")
    print("Record bonafide samples through the UI - see TESTING.md.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
