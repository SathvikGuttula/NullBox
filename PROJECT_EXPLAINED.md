# VoxShield — Complete Technical Explanation

**SIH 2026 · Problem Statement 26104 · Theme: Blockchain & Cybersecurity**

Written by tracing the actual code at commit `23d1da0`. Every claim below points
at a file. Where something is missing, partial, or unverifiable, it says so.

> **Read this first — the single most important correction.**
> **There is no speech-to-text in this project.** No Whisper, no Vosk, no
> transcription API, no `Wav2Vec2ForCTC`. Searching the entire repository
> (Python, TypeScript, requirements, config) for `whisper|transcri|vosk|stt|
> speech.to.text|deepgram|assemblyai` returns **zero matches**. The system never
> converts speech to text and never looks at *what was said*. It analyses **how
> the voice sounds** — the waveform and its spectrum. Section D explains exactly
> what is analysed instead. If a teammate says "our system transcribes the call",
> that is wrong and a judge will catch it.

---

## Table of contents

- [A. Project overview](#a-project-overview)
- [B. End-to-end system flow](#b-end-to-end-system-flow)
- [C. Every AI/ML component](#c-every-aiml-component)
- [D. How voice becomes a decision (there is no text)](#d-how-voice-becomes-a-decision-there-is-no-text)
- [E. The detection logic, step by step](#e-the-detection-logic-step-by-step)
- [F. Datasets](#f-datasets)
- [G. Training pipeline](#g-training-pipeline)
- [H. Metrics](#h-metrics)
- [I. Real-time processing](#i-real-time-processing)
- [J. Frontend](#j-frontend)
- [K. Backend](#k-backend)
- [L. Database / storage](#l-database--storage)
- [M. Security](#m-security)
- [N. Prevention and response](#n-prevention-and-response)
- [O. Project phases](#o-project-phases)
- [P. File-by-file architecture](#p-file-by-file-architecture)
- [Q. Technology stack](#q-technology-stack)
- [R. What is actually working right now](#r-what-is-actually-working-right-now)
- [S. Testing](#s-testing)
- [T. Questions judges are likely to ask](#t-questions-judges-are-likely-to-ask)

---

# A. Project overview

## What the project actually does

VoxShield takes an audio recording (uploaded, or streamed live from a
microphone) plus optional information about the call, and returns a **0–100 risk
score with reason codes and a list of actions a bank should take**.

It answers four separate questions and combines them:

| Question | Which component answers it | Status |
| --- | --- | --- |
| Is this speech machine-generated? | Trained anti-spoof neural network | **Working** |
| Is this the person they claim to be? | Pretrained ECAPA-TDNN speaker embeddings | **Working** |
| Is the situation around the call suspicious? | Rule-based context scorer | **Working (weights are placeholders)** |
| What should be done about it? | Rule-based policy engine | **Working (bands are placeholders)** |

## Why the problem exists

Text-to-speech and voice-conversion tools have become good enough that a
convincing clone of a specific person can be produced from a short sample of
their voice. Meanwhile, a very large amount of financial and enterprise
authorisation still runs over the phone, where the only identity check is "this
sounds like the person I know".

## What voice cloning / impersonation attacks are

Three distinct attacks, which matter because they fail in different ways:

1. **Text-to-speech (TTS)** — the attacker types text and a model speaks it in
   the target's voice. Fully synthetic audio.
2. **Voice conversion (VC)** — the attacker speaks, and a model rewrites their
   voice to sound like the target. The words and timing are human; the timbre is
   generated.
3. **Human impersonation** — a real person simply pretends to be someone else.
   **Nothing about this audio is synthetic.**

Attack 3 is the one people forget. An anti-spoof model looking for synthesis
artefacts will correctly say "this is real speech" and wave the attacker
straight through. This is why VoxShield is not only a deepfake detector — see
the measured demonstration in [Section E](#e-the-detection-logic-step-by-step).

## The problem we are solving

Deciding whether a voice on a call should be trusted enough to authorise an
action — and, when it should not, producing a **specific, reversible,
human-checkable next step** rather than a number on a screen.

## Intended users

Bank and enterprise **contact-centre fraud systems**. The API is designed to sit
behind a call platform, not in front of a consumer. The frontend in this
repository is a **testing and demonstration console**, not a product UI.

## What makes the solution useful

- **Fusion, not one model.** Three independent branches, each of which catches
  attacks the others miss.
- **Explanations, not scores.** Every decision carries reason codes and the
  point contribution of each branch.
- **Honest about uncertainty.** Branches that could not run are reported as
  *not assessed* and excluded from the maths — never scored as zero risk. A call
  where nothing could be assessed returns `INSUFFICIENT_EVIDENCE`, never
  `ALLOW`.
- **Calibrated.** The model's raw output is not a probability; the system
  measures and corrects that (see [Section H](#h-metrics)).

## Input the system receives

| Input | Where | Required |
| --- | --- | --- |
| Audio (WAV / FLAC / OGG, any rate, mono or stereo) | `sample` file upload | Yes |
| Claimed identity | `claimed_identity` form field / query param | No |
| Call context as a JSON object | `context` form field / query param | No |
| Live PCM stream (16-bit, 16 kHz, mono) | WebSocket binary frames | For live mode |

MP3 and M4A are **rejected** — the backend decodes with `libsndfile` via
`soundfile`, which does not read them (`app/api/speakers.py::decode_upload`).

## Output it produces

A JSON object (`app/api/analyze.py`) containing:

- `anti_spoof` — calibrated synthetic probability, raw model score, model
  status, and the speech-presence check result
- `speaker` — cosine similarity, MATCH / UNCERTAIN / NO_MATCH, reasons
- `context` — risk 0–1, points scored, signals used, signals not collected
- `risk` — fused 0–100 score, level, decision, per-branch contributions,
  available and missing signals
- `policy` — the list of actions, whether a human is required, rationale
- `notes` — plain-English explanations of anything that did not contribute

## Does it work in real time?

**Yes, with a specific definition.** "Real time" here means:

- The browser sends **4096-sample chunks of 16 kHz PCM = 256 ms** of audio per
  message over a WebSocket.
- The server buffers until it has a **3-second window**, scores it, then slides
  forward by a **1-second hop** — so a new score arrives roughly **once per
  second** after the first ~3 seconds of speech.
- Measured inference cost: **~30 ms per window on an RTX 3050**, after a
  one-time ~6 s model load.

So it is real-time in the sense of *continuous streaming analysis with about
one second of latency*, not sub-100 ms per utterance. This is measured, not
estimated — `scripts/verify_system.py` replays exactly this chunk size and the
verification run reports 111 chunks and 24 scored windows for a ~28 s clip.

## Prevention vs detection

| Part | Which | Where |
| --- | --- | --- |
| Anti-spoof model | Detection | `app/ml/detector.py` |
| Speaker verification | Detection | `app/ml/speaker.py` |
| Context scoring | Detection | `app/ml/context.py` |
| Policy actions | **Prevention** | `app/ml/policy.py` |

The prevention half is the policy engine: it converts a risk level into
concrete actions (callback on a number already on file, MFA, transaction hold,
manager approval, security escalation). **See the honesty note in
[Section N](#n-prevention-and-response): these actions are returned as data.
VoxShield does not execute them — there is no telephony or banking integration.**

## What happens when a suspicious voice is detected

Traced through `app/ml/fusion.py` → `app/ml/policy.py`:

1. The fused risk score is computed and mapped to a decision band
   (`ALLOW` < 40, `WARN` ≥ 40, `SECONDARY_VERIFICATION` ≥ 60,
   `HIGH_RISK_WORKFLOW` ≥ 80).
2. A confirmed identity mismatch (`NO_MATCH`) **floors the score at 60**
   regardless of the other branches.
3. The policy engine attaches actions, escalating with risk.
4. **No action ever rejects the transaction.** The heaviest available action is
   `TRANSACTION_HOLD`, whose description literally says *"do not reject it"*.
   This is asserted structurally in
   `tests/test_context_policy.py::test_no_action_ever_rejects_a_transaction`.

---

# B. End-to-end system flow

There are **two distinct pipelines**. They share the models but not the code
path.

## B.1 File analysis pipeline (`POST /api/v1/analyze`)

| # | Stage | File / function | In | Out | Fails how |
| --- | --- | --- | --- | --- | --- |
| 1 | Receive upload | `app/api/analyze.py::analyze` | multipart form | bytes | FastAPI 422 if `sample` missing |
| 2 | Decode & validate | `app/api/speakers.py::decode_upload` | bytes | float32 mono @16 kHz | HTTP 400 empty / undecodable / <1 s; 413 if >25 MB |
| 3 | **Speech-presence gate** | `app/audio/speech_check.py::assess_speech` | waveform | verdict + measurements | Never raises. Fail ⇒ branch skipped, `model_status: "no_speech"` |
| 4 | Anti-spoof inference | `app/ml/detector.py::VoiceSpoofDetector.predict` | waveform | raw + calibrated probability | Caught, logged; branch reported unavailable |
| 5 | Calibration | `app/ml/calibration.py::ProbabilityCalibrator.calibrate` | raw p | calibrated p | Identity map if no calibration file |
| 6 | Speaker embedding | `app/ml/speaker.py::SpeakerEncoder.embed` | waveform | 192-d vector | `ImportError` ⇒ branch skipped |
| 7 | Speaker verification | `app/ml/speaker.py::SpeakerRegistry.verify` | vector + identity | similarity + decision | Unknown identity ⇒ `NO_MATCH` with a reason, not an error |
| 8 | Context validation | `app/ml/context.py::CallContext.from_dict` | JSON | validated object | HTTP 400 with the exact field and why |
| 9 | Context scoring | `app/ml/context.py::ContextAnalyzer.assess` | context | risk 0–1 + reasons | — |
| 10 | Fusion | `app/ml/fusion.py::RiskFusion.fuse` | 3 optional risks | 0–100 + contributions | Zero branches ⇒ `INSUFFICIENT_EVIDENCE` |
| 11 | Policy | `app/ml/policy.py::PolicyEngine.decide` | decision + amount | actions + rationale | — |
| 12 | Respond | `app/api/analyze.py` | all of the above | JSON | — |

Stages 4 and 6 run inside `starlette.concurrency.run_in_threadpool` so a
transformer forward pass does not block the event loop and stall other calls.

## B.2 Live streaming pipeline (WebSocket)

| # | Stage | File / function | Notes |
| --- | --- | --- | --- |
| 1 | Open socket | `app/api/calls.py::audio_stream` | Parses `claimed_identity` and `context` from the query string **before** accepting; a bad context closes with code 1008 and a readable reason |
| 2 | Build per-call pipeline | `app/websocket/audio_stream.py::AudioStreamManager.connect` | One `AudioFeaturePipeline` per call, stored in a dict keyed by `call_id` |
| 3 | Receive PCM | `AudioStreamManager.process_stream` | Handles `websocket.disconnect` explicitly — relying on the exception alone left the loop spinning on a dead socket |
| 4 | Decode PCM16 | `app/audio/feature_pipeline.py::_decode_pcm16` | Odd byte count (chunk split mid-sample) drops the trailing byte instead of raising |
| 5 | Voice activity | `app/audio/vad.py::VoiceActivityDetector.process` | Energy + percentile noise floor + hangover |
| 6 | Acoustic features | `app/audio/features.py::AcousticFeatureExtractor.extract` | **Display only** — see the note below |
| 7 | Buffer & window | `app/ml/inference.py::StreamingInferenceEngine.add_audio` | 3 s window, 1 s hop |
| 8 | Speech gate + score | `StreamingInferenceEngine._infer` | Refused windows do **not** update the risk engine |
| 9 | Temporal smoothing | `app/ml/risk.py::TemporalRiskEngine.update` | Exponential smoothing, α = 0.35, with hysteresis |
| 10 | Rolling speaker | `app/ml/speaker.py::LiveSpeakerTracker.add_speech` | Accumulates a running centroid; estimate improves over the call |
| 11 | Fuse + policy | `feature_pipeline.py::process` | Same fusion and policy objects as the file path |
| 12 | Send JSON | `audio_stream.py` | One `voice_analysis` message per chunk |

> **Important:** the acoustic features in stage 6 (MFCC, pitch, spectral
> centroid/bandwidth/rolloff/flatness, ZCR, voiced ratio) are computed and sent
> to the dashboard, but **they are not inputs to any detection decision**. They
> are displayed to make the stream look alive and to help a human eyeball the
> audio. The neural model computes its own log-Mel spectrogram internally from
> the raw waveform. This is a real and important distinction — do not claim the
> MFCCs feed the classifier.

## B.3 Text flowchart

```
                       ┌──────────────────────────────────┐
                       │  AUDIO IN                        │
                       │  file upload  OR  mic → WebSocket│
                       └──────────────┬───────────────────┘
                                      │
                        ┌─────────────┴─────────────┐
                        │                           │
              [FILE PATH]                    [LIVE PATH]
                        │                           │
            decode_upload()                 _decode_pcm16()
            WAV/FLAC/OGG → float32          PCM16 → float32
            resample → 16 kHz mono          (already 16 kHz mono)
                        │                           │
                        │                    VoiceActivityDetector
                        │                    energy + noise floor
                        │                           │ (speech only)
                        │                    StreamingInferenceEngine
                        │                    buffer → 3 s window, 1 s hop
                        │                           │
                        └─────────────┬─────────────┘
                                      │
                        ╔═════════════▼═══════════════╗
                        ║  assess_speech()            ║
                        ║  frame-energy p95/p10       ║
                        ║  < 3.0  → NOT SPEECH        ║
                        ╚═════════════╤═══════════════╝
                             fail ────┴──── pass
                               │              │
                     branch withheld    AntiSpoofModel
                     model_status =      ┌────┴────┐
                     "no_speech"    wav2vec2   log-Mel CNN
                               │     (768-d)    (128-d)
                               │         └────┬────┘
                               │     attentive stats pooling
                               │       concat → MLP → 2 logits
                               │              │ softmax
                               │        raw p(spoof)
                               │              │
                               │     ProbabilityCalibrator
                               │     σ(3.7545·logit(p) + 9.4999)
                               │              │
                               │      calibrated p(spoof)
                               └──────────────┤
                                              │
   ┌──────────────────┐   ┌──────────────────┐│┌──────────────────┐
   │ SPEAKER BRANCH   │   │ CONTEXT BRANCH   │││ (prosody, NLP,   │
   │ ECAPA-TDNN 192-d │   │ rule-based       │││  forensics —     │
   │ cosine vs enrol  │   │ weighted signals │││  NOT IMPLEMENTED)│
   └────────┬─────────┘   └────────┬─────────┘│└──────────────────┘
            │                      │          │
            └──────────┬───────────┴──────────┘
                       │
              ╔════════▼═════════════════════════════╗
              ║  RiskFusion.fuse()                   ║
              ║  weighted mean over AVAILABLE        ║
              ║  branches only, renormalised         ║
              ║  NO_MATCH floors the score at 60     ║
              ║  zero branches → INSUFFICIENT_EVIDENCE║
              ╚════════╤═════════════════════════════╝
                       │  0–100 + reason codes
              ╔════════▼═════════════════════════════╗
              ║  PolicyEngine.decide()               ║
              ║  MONITOR → NOTIFY → MFA → CALLBACK   ║
              ║  → TRANSACTION_HOLD → ESCALATE       ║
              ║  never a rejection                   ║
              ╚════════╤═════════════════════════════╝
                       │
                  JSON response / WebSocket frame
                       │
                  Next.js dashboard
```

---

# C. Every AI/ML component

There are **two neural networks** and **three rule-based components**. Nothing
else. No external AI API is called anywhere.

## C.1 Anti-spoof classifier — the only model trained in this project

| | |
| --- | --- |
| **Name** | `AntiSpoofModel` (project-specific), built on `facebook/wav2vec2-base` |
| **Type** | Two-branch neural classifier: self-supervised speech transformer + log-Mel CNN, fused |
| **Task** | Binary classification — bonafide vs spoof |
| **Input** | Raw waveform, 16 kHz mono, exactly 64,000 samples (4.0 s) |
| **Output** | 2 logits → softmax → `spoof_probability`, `bonafide_probability` |
| **Defined in** | `backend/app/ml/model.py` |
| **Loaded in** | `backend/app/ml/detector.py::VoiceSpoofDetector.__init__` |
| **Status** | **Fine-tuned in this project** from a pretrained encoder |
| **Checkpoint** | `models/voxshield_antispoof.pt`, 362.6 MB |

### Architecture, traced from `model.py`

Every shape below was **verified by running the model**, not read off the code.

```
waveform (1, 64000)                        4.0 s @ 16 kHz
   │
   ├── SSL branch:  wav2vec2-base  →  (1, 199, 768)      [verified]
   │                AttentiveStatsPooling  →  (1, 1536)
   │                   learned attention over the 199 frames,
   │                   returns weighted mean ‖ weighted std
   │
   └── Spectral branch:  MelSpectrogram(n_fft=512, hop=160,
   │                        n_mels=80, f_min=20, f_max=8000)
   │                     →  (1, 80, 401)                 [verified]
   │                     → AmplitudeToDB(top_db=80)
   │                     → per-utterance standardisation
   │                     → Conv2d(1→32)   BN ReLU MaxPool(2)
   │                     → Conv2d(32→64)  BN ReLU MaxPool(2)
   │                     → Conv2d(64→128) BN ReLU AdaptiveAvgPool(1)
   │                     → Dropout → Linear(128→128) → ReLU
   │                                              (1, 128)
   │
   concat → (1, 1664)                                    [verified]
        → LayerNorm → Dropout(0.2)
        → Linear(1664 → 256) → ReLU → Dropout(0.2)
        → Linear(256 → 2)                                [verified]
```

**Parameter count (measured):** 95.0 M total, of which 94.4 M is the wav2vec2
encoder. The spectral branch, pooling and classifier together are ~0.6 M — the
overwhelming majority of the model is the pretrained encoder being fine-tuned.

**One inaccuracy found while writing this document, worth knowing:** the comment
in `model.py` says *"wav2vec2-base was pretrained on raw waveforms
(do_normalize=False)"*. That is **wrong for this checkpoint** — the actual
`Wav2Vec2FeatureExtractor` for `facebook/wav2vec2-base` reports
`do_normalize = True`, verified by loading it. The *mechanism* the comment
describes is correct and important (read the flag off the checkpoint rather than
assume it), and because the value is read the same way at training and at
inference there is **no behavioural bug** — the input is normalised in both.
Only the comment's example is inaccurate. Left unchanged, since this
documentation pass was not authorised to modify code.

### Three design decisions worth defending

1. **The mel filterbank spans the full 0–8 kHz Nyquist band** (`f_max =
   sample_rate / 2`). Neural vocoders leave their most reliable fingerprints in
   the top octave; ASR recipes commonly cap at 4 kHz and would throw that
   evidence away.
2. **Attentive statistics pooling, not mean pooling.** Vocoder artefacts are
   *bursty* — they live in a few frames. A plain mean dilutes exactly the
   evidence the model needs. The weighted standard deviation additionally
   captures frame-to-frame inconsistency, itself a synthesis giveaway.
3. **The spectral branch is computed inside `forward()` on-device**, in forced
   fp32 (`torch.autocast(enabled=False)`) because squared magnitudes underflow
   in fp16. It is not precomputed in the DataLoader.

### Why wav2vec2-base was chosen

Self-supervised speech representations are the standard strong baseline for
ASVspoof-style tasks, and `wav2vec2-base` (95 M parameters, 768 hidden) is the
largest that fine-tunes inside 6 GB of VRAM with the staged-freezing scheme.
`model.py` accepts any HuggingFace speech encoder (`wavlm-base-plus`,
`wav2vec2-xls-r-300m`) via `encoder_name`, so the choice is a parameter, not a
hardcoding.

One subtlety the code handles explicitly: whether a checkpoint expects
zero-mean unit-variance input is a **property of that checkpoint**, not a
universal rule. `wav2vec2-base` was pretrained on raw waveforms
(`do_normalize=False`) while most XLS-R checkpoints expect normalised input.
`model.py` reads `do_normalize` off the feature extractor rather than assuming.

### Confidence it produces

A softmax probability over two classes. **This raw number is not calibrated** —
see [Section H](#h-metrics). The serving path applies a Platt calibration in
exactly one place (`detector.py::predict`) and returns both:

```json
"anti_spoof": {
  "synthetic_probability": 0.9999999998,   // calibrated — use this
  "raw_probability": 0.9697,               // what the checkpoint emitted
  "model_status": "neural"
}
```

## C.2 Speaker encoder — pretrained, not trained here

| | |
| --- | --- |
| **Name** | `speechbrain/spkrec-ecapa-voxceleb` |
| **Type** | ECAPA-TDNN speaker-embedding network |
| **Task** | Map a waveform to a fixed 192-dimensional speaker embedding |
| **Input** | Waveform, 16 kHz mono, variable length |
| **Output** | 192-d float vector |
| **Loaded in** | `backend/app/ml/speaker.py::SpeakerEncoder.__init__` |
| **Status** | **Pretrained, third-party, frozen.** Nothing about it is trained or fine-tuned in this project. |
| **Training data** | VoxCeleb (by SpeechBrain, not by us) |

Chosen because it is the standard open speaker-verification model, it ships
pretrained, and it needs no labelled data from us. It is imported **lazily** so
the anti-spoof path never acquires a hard dependency on it — if SpeechBrain is
missing, the speaker branch degrades and the rest of the system still runs.

Windows-specific detail: it is fetched with `LocalStrategy.COPY`, not the
default symlink strategy, because creating a symlink on Windows needs
administrator rights or Developer Mode and otherwise dies with
`WinError 1314`.

**It produces no probability.** It produces a vector. The decision comes from
cosine similarity against an enrolled centroid, compared to calibrated
thresholds — see [Section E](#e-the-detection-logic-step-by-step).

## C.3 Probability calibrator — fitted statistics, not a network

| | |
| --- | --- |
| **Type** | Two-parameter Platt scaling (logistic regression on the logit) |
| **Formula** | `calibrated = sigmoid(scale · logit(p) + bias)` |
| **Fitted values** | `scale = 3.754526`, `bias = 9.499884` (at an equal prior) |
| **Defined in** | `backend/app/ml/calibration.py` |
| **Fitted by** | `backend/scripts/calibrate_detector.py` |
| **Artifact** | `results/calibration/antispoof_calibration.json` (committed) |

**Why a bias term is essential and temperature scaling is not enough.**
Temperature scaling divides the logit by a scalar T. It is monotone and it
cannot change the sign of a logit — so a decision boundary that sits below
p = 0.5 stays below 0.5 for *every* T. Measured on the evaluation set, this
model's equal-error point is at a raw probability of **0.044**, deep inside the
bonafide cluster. No temperature could have fixed that. Only an affine map with
a bias can move a boundary. This is pinned by
`tests/test_affine_calibration.py::test_temperature_cannot_move_the_boundary`.

## C.4 Context analyser — rule-based, no ML

`backend/app/ml/context.py`. Thirteen optional signals, each with a weight out
of 100. Score = points triggered ÷ points that **could have been** assessed.

Weights (`ContextWeights`, version `v0-UNCALIBRATED`): unknown caller 14,
recent password reset 14, unusual amount 14, unexpected country 12, unknown
device 12, failed authentications 12, unknown beneficiary 12, new beneficiary
10, untrusted contact 8, odd hour 8.

**These are analyst intuition, not fitted values, and every response says so.**

## C.5 Fusion and policy — rule-based, no ML

`backend/app/ml/fusion.py` and `policy.py`. Weighted mean over available
branches; band thresholds map score → decision → actions. Weights:
synthetic speech 35, identity mismatch 30, audio forensics 10, prosody 10,
context 10, NLP 5. **Also placeholders**, also declared as such. The last three
branches do not exist, so their weight is never used.

## C.6 Ensemble logic — how the branches combine

There is no voting or stacking. `RiskFusion.fuse()`:

1. Each branch contributes `risk × weight`, but **only if it reported**.
2. `risk_score = (Σ weighted_risk / Σ weights_of_reporting_branches) × 100`.
3. Contributions are rescaled by `100 / weight_total` so the displayed points
   sum to the score shown above them.
4. **Override:** `if speaker_decision == "NO_MATCH": risk_score = max(risk_score,
   60)`.
5. If no branch reported at all → `INSUFFICIENT_EVIDENCE`, level `UNKNOWN`.

Step 2 is the important one. Missing branches are **renormalised over**, not
treated as zero. With only anti-spoof wired up, a spoof probability of 1.0 must
still be able to reach a high score — otherwise the system would silently cap
its own maximum at 35/100 and never escalate anything. The trade-off is that the
score's meaning shifts as branches come online, which is why
`available_signals` and `missing_signals` are part of the output rather than a
footnote.

---

# D. How voice becomes a decision (there is no text)

This section is deliberately explicit because it is the question most likely to
be asked, and the most likely to be answered wrongly by a teammate.

## D.1 There is no speech-to-text. At all.

- No transcription library, model, or API is imported anywhere.
- The word "transcript" does not appear in the backend.
- `requirements.txt` contains no ASR package.
- Nothing in the system knows *what words were spoken*.

The handoff document lists **NLP transcript analysis** as a planned branch
(the `nlp` weight of 5 exists in `FusionWeights`). **It is not implemented.**
`fusion.fuse()` accepts an `nlp_score` parameter, but no caller ever passes one,
so `nlp` is always in `missing_signals`.

## D.2 What is actually analysed

| Signal type | Used? | Where |
| --- | --- | --- |
| **Waveform (raw samples)** | **Yes** — the primary input | `model.py`, fed straight into wav2vec2 |
| **Spectral information** | **Yes** — log-Mel spectrogram, 80 bands, 0–8 kHz | `model.py::SpectralBranch` |
| **Learned SSL representations** | **Yes** — wav2vec2 hidden states | `model.py` |
| **Speaker embeddings** | **Yes** — 192-d ECAPA vectors | `speaker.py` |
| **Frame-energy statistics** | **Yes** — but only as an input gate | `speech_check.py` |
| **Metadata / context** | **Yes** — rule-based | `context.py` |
| **Hand-crafted acoustic features** (MFCC, pitch, ZCR, spectral centroid…) | **Computed and displayed, but NOT used in any decision** | `features.py` → dashboard only |
| **Text** | **No** | — |
| **Semantic information** | **No** | — |

## D.3 The exact path from microphone to number

Follow one 4-second clip:

1. **Bytes arrive.** `decode_upload()` reads them with `soundfile`, averages to
   mono, resamples to 16 kHz with `torchaudio.functional.resample`. Result:
   `float32` array in [−1, 1].
2. **Speech gate.** `assess_speech()` frames the audio (512-sample frames,
   160-sample hop), computes each frame's RMS, and takes the ratio of the 95th
   to the 10th percentile. Speech is *intermittent* — people pause between words
   — so this ratio is huge for speech and ~1 for any steady signal. Below **3.0**
   the clip is refused. (Measured: silence 0.0, white noise 1.09, pure tone 1.01,
   50 Hz hum 1.07, noise under a fade 2.71; real speech > 170,000,000 clean and
   6.2–6.9 at 10 dB SNR.)
3. **Length normalisation.** `detector.py::_prepare` peak-normalises, then
   centre-crops or **tiles** the clip to exactly 64,000 samples. Tiling rather
   than zero-padding is deliberate: bonafide and spoof clips have different
   length distributions in ASVspoof, so zero-padding would leak the label.
4. **Two branches run.**
   - wav2vec2 produces `(1, T≈199, 768)` hidden states; attentive pooling
     collapses them to `(1, 1536)`.
   - The mel branch produces an 80 × 401 log-Mel image; the CNN collapses it to
     `(1, 128)`.
5. **Fusion head.** Concatenated to `(1, 1664)`, through LayerNorm → Dropout →
   Linear(256) → ReLU → Dropout → Linear(2). Softmax over 2 logits.
6. **Calibration.** `σ(3.7545 · logit(p) + 9.4999)`.
7. **Threshold.** `p ≥ 0.5` ⇒ synthetic.

## D.4 So what is the model actually *detecting*?

Honestly: **statistical fingerprints of the synthesis process**, learned from
data rather than specified by hand. It was trained on 25,380 clips labelled
bonafide or spoof and learned whatever separates them. Nobody wrote a rule that
says "look for phase discontinuity" or "check the 7 kHz band".

The strongest evidence that it learned something *general* rather than
memorising ASVspoof: it flags **Windows SAPI text-to-speech** — a completely
different, much older synthesis family that appears nowhere in its training
data — at **0.969–0.980 raw**. That is a genuine out-of-distribution result,
reproducible on any Windows machine via `scripts/make_test_audio.py`.

The honest limits: it cannot explain *which* artefact triggered it, and it
cannot be expected to generalise to every future synthesiser. The measured
ablation in `results/README.md` shows exactly where generalisation breaks —
holding out the two voice-conversion attacks from training raised the
voice-conversion miss rate from **9.5% to 48.8%** while text-to-speech stayed
near zero. **Generalisation is bounded by synthesis family, not by attack
identity.** That is the project's strongest empirical finding and it is worth
stating out loud.

---

# E. The detection logic, step by step

## E.1 The three-row table the whole design rests on

| Scenario | Anti-spoof | Speaker identity | Caught by |
| --- | --- | --- | --- |
| Genuine caller | LOW | MATCH | — (correctly allowed) |
| Cloned voice | **HIGH** | may MATCH | **Anti-spoof** |
| Human impersonator | LOW | **NO_MATCH** | **Speaker verification only** |

Row 3 is why the speaker branch exists. An impersonator's speech is genuinely
human, so anti-spoof correctly reports LOW.

## E.2 Order of checks in `analyze()`

```
1. Decode and validate audio          → 400 on failure, nothing else runs
2. Speech-presence gate               → fail ⇒ anti-spoof branch withheld
3. Anti-spoof (if gate passed)        → calibrated p(synthetic)
4. Speaker verification (if identity claimed AND profile exists)
5. Context scoring (if context supplied)
6. Fusion over whatever reported
7. NO_MATCH override → floor at 60
8. Policy → actions
```

## E.3 Anti-spoof decision

`calibrated_p ≥ 0.5` ⇒ synthetic. Derived, not chosen — read off the DET curve
in `scripts/calibrate_detector.py`:

| Threshold | False alarm | Miss | Detects |
| --- | --- | --- | --- |
| 0.0940 | 5.03% | 2.68% | 97.32% |
| 0.1370 | 2.01% | 3.17% | 96.83% |
| 0.2775 | 1.01% | 4.16% | 95.84% |
| **0.5000** | **0.79%** | **5.05%** | **94.95%** |
| 0.9658 | 0.52% | 7.59% | 92.41% |

`DetectorConfig` uses `suspicious_threshold = 0.137` (≈2% FA) and
`high_risk_threshold = 0.50`.

## E.4 Speaker decision — three-way, deliberately

`app/ml/speaker.py::SpeakerThresholds.decide`:

```
similarity ≥ 0.4422  →  MATCH
similarity <  0.3271  →  NO_MATCH
otherwise             →  UNCERTAIN
```

Calibrated on 20 dev speakers (3,000 trials), verified to transfer to 67 unseen
eval speakers (6,700 trials) with essentially no change: **0.375% → 0.373% EER**.

A two-way split would force a guess on exactly the scores the system knows
least about. The UNCERTAIN band lets the policy engine ask for a second factor
instead of inventing confidence.

`similarity` = cosine similarity between the probe embedding and the enrolled
**centroid** — the L2-normalised mean of L2-normalised enrolment embeddings.
Normalising *before* averaging matters: raw ECAPA embeddings vary in magnitude
with utterance length and loudness, so a plain mean lets the longest enrolment
sample dominate.

## E.5 Enrolment quality gates

`SpeakerRegistry.enroll` refuses:
- fewer than **3** samples (5 recommended)
- a set whose mean pairwise cosine similarity is below **0.45**, unless
  explicitly overridden

Rationale in the code: a blurred prototype is not a one-off error — it is an
account that quietly matches several people for as long as it exists.

## E.6 Context scoring

Score = triggered points ÷ **available** points, not ÷ all conceivable points.
A call where only two signals were collected and both look bad reads as 100%
context risk. Dividing by the full weight of ten signals would dilute it to
~26% and hide the thing the operator needs to see.

Signals never collected go into `missing`, never scored as safe.

## E.7 Measured end-to-end demonstration

Reproduced against the current code (numbers from a live run, not estimates):

| Branches available | Risk | Decision |
| --- | --- | --- |
| anti-spoof only | **4.0** | ALLOW — monitor only |
| + identity NO_MATCH | **60.0** | SECONDARY_VERIFICATION |
| + identity + CEO-fraud context | **60.0** | SECONDARY_VERIFICATION |

The third row does not score higher than the second because the NO_MATCH
override already floors it at 60. The context still changes the outcome — in the
**actions**, adding `TRANSACTION_HOLD` and `MANAGER_APPROVAL`.

**The first row is the argument for the entire project**: with anti-spoof alone,
a human impersonator is allowed through.

---

# F. Datasets

## F.1 ASVspoof 2019 LA — the only dataset actually used

| | |
| --- | --- |
| **Source** | Public ASVspoof 2019 Logical Access challenge corpus |
| **Purpose** | Training and evaluating the anti-spoof model |
| **Audio format** | FLAC, 16 kHz mono |
| **Classes** | `bonafide` (real human) / `spoof` (TTS or voice conversion) |
| **Used via** | Kaggle (`asvpoof-2019-dataset-la`), not stored in this repository |

Splits, from `results/exp002-full/report.json` and the manifests:

| Split | Utterances | Attacks | Used for |
| --- | --- | --- | --- |
| Train | 25,380 | A01–A06 | Fine-tuning |
| Dev | 24,844 | A01–A06 | Validation, checkpoint selection, early stopping |
| **Eval** | **71,237** (7,355 bonafide + 63,882 spoof) | **A07–A19** | **Final reported metrics** |

The evaluation attacks **do not overlap** the training attacks. That is what
makes 2.95% EER an honest generalisation number and the 0.196% validation EER
misleading — validation used A01–A06, which the model trained on.

**A05 and A06 are the only voice-conversion attacks in train/dev**; A17–A19 are
the voice-conversion attacks in eval. This fact drove the project's key ablation
(see [Section H](#h-metrics)).

> **Not present in this repository.** `datasets/raw/LA.zip` is a **50 MB
> truncated fragment** of a 7.6 GB download that failed at 47 KB/s. The three
> files in `datasets/manifests/` are **0 bytes**. Training and evaluation were
> done on Kaggle, where the corpus was available. What *is* committed is the
> per-utterance score file (`kaggle_run/fixed/scores.csv`, 71,237 rows) and the
> evaluation reports under `results/`.

## F.2 VoxCeleb — used indirectly

The ECAPA-TDNN encoder was pretrained on VoxCeleb **by SpeechBrain**. We never
touch VoxCeleb; we consume the resulting weights. No training, no fine-tuning.

## F.3 ASVspoof 2019 LA ASV protocols — used for speaker calibration

`scripts/calibrate_speaker.py` builds target/non-target trials from the ASV
(not countermeasure) protocol files. Results in
`results/speaker-calibration/`: 20 dev speakers / 3,000 trials, and 67 eval
speakers / 6,700 trials.

## F.4 Locally generated test corpus — not training data

`scripts/make_test_audio.py` generates `datasets/test-audio/`:
9 Windows SAPI TTS clips (3 voices × 3 scripts) and 15 edge cases. **Purely for
testing the running system.** Nothing is trained on it. The `bonafide/` folder
is deliberately empty with a README explaining why — a synthetic stand-in
labelled bonafide would be worse than nothing.

## F.5 ASVspoof 2021 DF — referenced, never used

The handoff planned a cross-dataset evaluation on the 34.5 GB DF set. Notebooks
contain code paths for it. **No DF evaluation was ever run and no DF result
exists.**

---

# G. Training pipeline

**Yes, one model is trained here:** the anti-spoof classifier. The speaker
encoder is not.

## G.1 Where training happens

- Entry point: `backend/scripts/train_model.py`
- Trainer: `backend/app/ml/training.py::AntiSpoofTrainer`
- **Executed on Kaggle** (2× Tesla T4), not on the laptop. Notebooks in
  `notebooks/`.

## G.2 The pipeline

```
ASVspoof protocol files
   → scripts/build_manifest.py       (auto-detects protocol columns)
   → datasets/manifests/*.csv        (path, label, attack, speaker)
   → scripts/cache_dataset.py        (optional: decode once into a
                                      float16 memmap; parallel via Pool)
   → VoiceSpoofDataset               (app/ml/dataset.py)
       read_audio() via soundfile    — NOT torchaudio.load, which needs
                                       TorchCodec and was broken
       fit_length()                  — tiles rather than zero-pads
       peak_normalize()
       AudioAugmenter                — training split only
   → DataLoader (WeightedRandomSampler for class balance)
   → AntiSpoofModel
   → AdamW + cosine schedule with warmup + AMP
   → validate each epoch → EER
   → save best by EER → models/voxshield_antispoof.pt
   → scripts/evaluate_model.py on the eval split
   → scripts/calibrate_detector.py to fit the serving calibration
```

## G.3 Hyperparameters

From `TrainingConfig` in `backend/app/ml/config.py`:

| Setting | Value |
| --- | --- |
| Loss | `CrossEntropyLoss(label_smoothing=0.05)` |
| Optimizer | `AdamW`, weight decay 0.01 — **excluded from biases and norm layers** |
| Encoder LR | 1e-5 |
| Head LR | 1e-4 |
| Schedule | Linear warmup (10% of steps) then cosine decay, via `LambdaLR` |
| Epochs | 4 (config default); **5 were run** |
| Batch size | 8 |
| Gradient accumulation | 2 → effective batch 16 |
| Gradient clipping | `clip_grad_norm_` at 1.0 |
| Precision | AMP, `auto` |
| Seed | 1234 |
| Early stopping | patience 2, on validation EER |

## G.4 Staged fine-tuning

- **Epoch 1 — "frozen":** the whole SSL encoder is frozen; only the pooling,
  spectral branch and classifier train. Fast, stable, and it stops a randomly
  initialised classifier from wrecking the encoder with large early gradients.
- **Epochs 2+ — "finetune":** the top 4 transformer layers unfreeze
  (`unfreeze_top_layers = 4`). The convolutional feature extractor stays frozen
  always — those layers learn generic waveform filters, are unstable to
  fine-tune, and freezing them saves memory and time.

The optimizer is **rebuilt**, not extended, at the stage change, because AdamW
keeps per-parameter moment buffers.

## G.5 Augmentation — channel realism

`app/ml/augmentation.py`, training split only, each applied with its own
probability:

| Augmentation | Probability | What it simulates |
| --- | --- | --- |
| `telephone` | 0.35 | 300–3400 Hz band-pass (measured attenuation −35.5 dB outside the band) |
| `add_noise` | 0.30 | Additive white noise at random SNR |
| `random_gain` | 0.30 | Level variation |
| `codec` | 0.20 | μ-law companding |
| `reverb` | 0.15 | Room reflection |
| `clip` | — | Hard limiting, as a hot microphone would |
| `packet_dropout` | — | Zeroed spans, as VoIP packet loss does |

## G.6 Class balancing

ASVspoof LA is ~90% spoof. `balanced_sample_weights()` +
`WeightedRandomSampler` equalise the classes during training. **This is one
reason the raw output needed calibrating** — it shifts the model's implied prior
away from the data's.

## G.7 Checkpointing and resume

`save_resume_state` / `load_resume_state` persist weights, AdamW moment
buffers, scheduler state and RNG state, written atomically (`.tmp` + `replace`).
Built because Kaggle sessions time out at 12 hours.

## G.8 The actual run — `exp002-full`

From `results/exp002-full/summary.json`:

| Epoch | Stage | Train loss | Train acc | Val EER | Val ROC-AUC | Seconds |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | frozen | 0.4476 | 80.73% | 6.808% | 0.9830 | 701 |
| 2 | finetune | 0.2253 | 94.82% | 0.570% | 0.9995 | 739 |
| 3 | finetune | 0.1529 | 98.42% | 0.242% | 0.9999 | 739 |
| **4** | finetune | 0.1394 | 99.00% | **0.196%** | 0.9998 | 738 |
| 5 | finetune | 0.1355 | 99.20% | 0.196% | 0.9999 | 738 |

Best epoch 4. **≈12.3 minutes per epoch, ≈62 minutes total** on a Kaggle T4.

**Never quote the 0.196% as the headline.** It is validation on A01–A06 — the
attacks the model trained on.

---

# H. Metrics

## H.1 The critical distinction judges will probe

| | Model evaluation metric | Runtime detection confidence |
| --- | --- | --- |
| **What** | EER, minDCF, ROC-AUC, per-attack miss | Calibrated probability, fused 0–100 risk score |
| **Computed on** | 71,237 labelled eval utterances, offline | One live call, no ground truth |
| **Where** | `app/ml/metrics.py`, `scripts/evaluate_model.py` | `app/ml/detector.py`, `fusion.py` |
| **Means** | "how good is this model" | "how suspicious is this call" |
| **Threshold-free?** | EER and ROC-AUC yes | No — a decision needs a threshold |

They are not the same number and must not be conflated. The 0–100 risk score
is **not** a probability of fraud; it is a weighted combination of branch risks
with placeholder weights.

## H.2 Anti-spoof model evaluation — measured, on 71,237 eval utterances

| Metric | Value |
| --- | --- |
| **EER** | **2.951%** |
| ROC-AUC | 0.9863 |
| Normalised minDCF (prior 0.05) | 0.1474 |
| Accuracy at EER threshold | 97.05% |
| Miss at 1% false alarm | 3.76% |

Say **"normalised minDCF"**, never "t-DCF" — t-DCF requires an ASV subsystem
this project does not have.

## H.3 Calibration quality — measured on a held-out half

`scripts/calibrate_detector.py` splits the eval set 50/50 **stratified by
(label, attack)** so both halves contain all 13 attacks, fits on one half,
reports on the other:

| | Raw | Calibrated |
| --- | --- | --- |
| Expected Calibration Error | 0.1263 | **0.0147** |
| Brier score | 0.0843 | **0.0244** |
| Negative log-likelihood | 0.2898 | **0.0830** |
| Miss at p ≥ 0.5 | 11.86% | **2.81%** |
| EER | 2.993% | 2.993% (unchanged — calibration is monotone) |

## H.4 Per-attack breakdown at p ≥ 0.5 (held-out half)

| Attack | Miss rate | | Attack | Miss rate |
| --- | --- | --- | --- | --- |
| A07 | 0.57% | | A13 | 0.00% |
| A08 | 0.00% | | A14 | 0.24% |
| A09 | 0.00% | | A15 | 0.53% |
| A10 | **22.02%** | | A16 | 0.49% |
| A11 | 0.24% | | A17 | **13.23%** |
| A12 | 0.00% | | A18 | **24.22%** |
| | | | A19 | 4.11% |

Bonafide false alarm: **0.79%**. A10, A17 and A18 are the weak points and
should be named honestly rather than hidden behind the average.

## H.5 The ablation — the project's strongest finding

Two full training runs, compared in `results/README.md`:

| Run | Training attacks | Eval EER | Voice-conversion miss | A19 miss |
| --- | --- | --- | --- | --- |
| 1 | A01–A04 (VC held out) | 11.27% | 48.79% | 73.93% |
| **2** | **A01–A06 (VC included)** | **2.95%** | **9.52%** | **2.79%** |

Text-to-speech performance barely moved. **Generalisation is bounded by
synthesis family, not by attack identity.**

## H.6 Speaker verification

| | Dev (20 speakers) | Eval (67 unseen speakers) |
| --- | --- | --- |
| EER | 0.375% | **0.373%** |
| Trials | 600 target / 2,400 non-target | 1,340 / 5,360 |

Operating points at the shipped `balanced` thresholds: FAR 0.375%, FRR 0.500%.

## H.7 Metrics implemented in `app/ml/metrics.py`

`equal_error_rate` (with interpolation), `minimum_dcf`, `det_curve`,
`threshold_at_false_alarm`, `error_rates_at`, `calculate_metrics` (accuracy,
precision, recall, F1, confusion matrix, ROC-AUC), `per_attack_breakdown`,
`plot_curves` (DET and ROC PNGs).

Two bugs worth knowing about, both fixed and both regression-tested:
- `sklearn.roc_curve` prepends an **infinite** threshold; interpolating between
  `inf` and a finite value yields `NaN`. Replaced with `nextafter(max_score)`.
- `roc_curve` returns thresholds **descending**, so `acceptable[0]` is the
  *highest* threshold — nothing flagged, 100% miss. It should be
  `acceptable[-1]`. This bug was hiding the best operating point in the whole
  project (3.76% miss at 1% false alarm).

## H.8 What is NOT measured

- **Real telephone audio.** Telephone augmentation was on during training, but
  no actual phone call has ever been scored.
- **Any language other than English.**
- **Replay attacks.** ASVspoof LA is Logical Access only; the PA (Physical
  Access / replay) set was never used.
- **Adversarial examples** crafted against this model.
- **Fusion weight accuracy** — there is no labelled fraud data to fit them on.

---

# I. Real-time processing

| Question | Answer |
| --- | --- |
| Microphone input? | **Yes** — browser `getUserMedia`, both live and record-to-file |
| Uploaded audio? | **Yes** — `POST /api/v1/analyze` |
| Streaming? | **Yes** — WebSocket, binary frames |
| Chunks? | **4096 samples = 256 ms** of 16 kHz PCM16 |
| Windows? | **3.0 s** analysis window |
| Hop? | **1.0 s** — a score about once per second |
| Buffering? | One contiguous `np.ndarray` in `StreamingInferenceEngine` |
| WebSocket? | Yes — `/api/v1/calls/{call_id}/stream` |
| Polling? | Frontend polls `/api/v1/status` every 15 s for the health badge |
| Async? | FastAPI async endpoints; model inference in a **threadpool** |
| Batch inference? | No — one window at a time |

## What happens per chunk (256 ms)

1. Decode PCM16 → float32 (odd byte count tolerated).
2. VAD: RMS vs an adaptive noise floor with hangover.
3. Every 3rd chunk: recompute expensive prosody features (pitch tracking via
   `librosa.pyin` is roughly real-time on its own, so it runs on a duty cycle
   and the last value is carried forward).
4. If speech: append to the inference buffer, and to the speaker tracker.
5. If the buffer has ≥ 3 s: score one window, advance by 1 s. A single large
   chunk emits every window it covers.
6. If either branch produced something new: re-fuse, re-run policy.
7. Send one JSON message.

## Why the live window is shorter than the training window

Training used 4 s windows; live uses 3 s for latency. The detector **tiles** the
3 s window up to 4 s before scoring rather than feeding the encoder a length it
never saw. The two lengths are coupled through `DetectorConfig`, not hardcoded
in two places.

## Latency bottlenecks (measured)

| Stage | Cost |
| --- | --- |
| First model load | **~6 s** (362 MB checkpoint + wav2vec2 init) |
| Anti-spoof forward pass, warm | **~30 ms** (RTX 3050) |
| Speaker embedding | Runs on its own longer duty cycle |
| `librosa.pyin` pitch tracking | The most expensive per-chunk CPU work — hence the duty cycle |
| Inherent windowing delay | **3 s** before the first score |

**Scalability limits (honest):** the detector and encoder are process-wide
singletons (`get_shared`), which is correct for one process, but per-call
pipelines live in a plain in-memory dict (`AudioStreamManager.pipelines`) with
**no eviction, no cap and no cross-process sharing**. Multiple uvicorn workers
would each hold their own copy of the models and their own registry. There is no
queue, no batching and no GPU sharing strategy. This is single-node prototype
scale.

---

# J. Frontend

**Next.js 15.5 (App Router) + React 19 + TypeScript.** No CSS framework —
inline styles and a small shared component module. `npx tsc --noEmit` and
`npm run build` both pass.

## Structure

| Path | Purpose |
| --- | --- |
| `app/page.tsx` | Renders `<Dashboard/>` |
| `app/layout.tsx` | Root layout, metadata |
| `components/Dashboard.tsx` | Tab shell, backend health badge, footer with measured numbers |
| `components/AnalyzeTab.tsx` | Upload or record, claim identity, choose context, show result |
| `components/LiveCall.tsx` | WebSocket streaming call |
| `components/SpeakersTab.tsx` | Enrol / verify / list / delete |
| `components/StatusTab.tsx` | System diagnostics |
| `components/ResultView.tsx` | Renders one `AnalysisResult` |
| `components/ui.tsx` | Panel, Button, Badge, Row, TextInput, Notice, LevelMeter, Tabs |
| `lib/api.ts` | All typed API calls, colour helpers, formatting |
| `lib/recorder.ts` | Microphone capture → WAV encoding |

## The four tabs — what the user sees

**1. Analyse a recording.** Choose a file *or* record from the microphone with a
live level meter. Claim an identity (enrolled names appear as one-click chips).
Pick one of four context presets — *No context*, *Routine call*, *CEO fraud
pattern*, *Account takeover* — or write custom JSON. Press Analyse. The result
shows: the big fused risk number with level and decision badges; a "WHY"
breakdown listing each branch's point contribution; a "Not assessed" line; three
panels (anti-spoof with both calibrated and raw scores, speaker identity,
context); and the policy actions with blocking ones marked `!`.

**2. Live call.** Set identity and context *before* starting (both are fixed for
the life of the call). Start, and the risk builds over time; per-window and
smoothed probabilities, windows scored, speech ratio, and live policy actions.

**3. Speakers.** Records enrolment samples one at a time, showing a **different
prompt sentence each time** (varying the words makes the prototype describe a
person rather than one sentence). Each sample is playable and removable before
enrolling. Then verify: record one clip and see the similarity against the
thresholds. Lists everyone enrolled with consistency scores; delete anyone.

**4. System.** Checkpoint size, whether it is loaded, device, calibration
parameters and its held-out numbers, encoder availability, thresholds and their
source, fusion weights, and a "Declared limitations" panel.

## Why recording is done by hand

`lib/recorder.ts` does **not** use `MediaRecorder`. Chrome produces
`audio/webm;codecs=opus` and Safari produces `audio/mp4`; the backend decodes
with libsndfile, which reads neither. The upload would fail with "could not
decode … as audio", which looks like a backend bug and is not. So the recorder
captures raw float samples via `ScriptProcessorNode`, linearly resamples to
16 kHz, and writes a WAV header itself.

**`autoGainControl` is disabled** in both capture paths. AGC compresses the
quiet parts of speech toward the loud parts — and the backend's speech gate
reads exactly that dynamic range to distinguish speech from steady noise.

## Presentation rules the UI enforces

- A branch that did not run shows as **"NOT ASSESSED"**, never as zero risk.
- A decision is never shown without its actions.
- Uncalibrated inputs are labelled on the same screen as the score they made.
- A saturated probability renders as **">99.9%"**, not "100.0%" — the latter
  would claim a certainty no measurement supports.

## Not implemented in the frontend

No authentication or login. No history, no charts over time, no export, no
audit view, no multi-call view, no mobile-specific layout (it is responsive via
`auto-fit` grids but not designed for phones). **No frontend tests.**

---

# K. Backend

**FastAPI + Uvicorn**, Python 3.11. `backend/app/main.py` builds the app with a
lifespan that tries `init_db()` and **continues if it fails** — the audio path
does not touch Postgres, and refusing to start without Docker would take the
demo down over a dependency it does not use.

CORS is configured from `settings.cors_origin_list` (default
`http://localhost:3000`).

## Endpoints

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/` | Name, status, version |
| `GET` | `/api/v1/health` | Liveness — cheap, instant |
| `GET` | `/api/v1/status` | Full diagnostics: checkpoint, calibration + held-out numbers, device, encoder, thresholds and their source, fusion weights, declared warnings, rejected profiles |
| `POST` | `/api/v1/analyze` | The main endpoint — audio + optional identity + optional context |
| `POST` | `/api/v1/speakers/enroll` | identity + 3–5 audio files → profile |
| `POST` | `/api/v1/speakers/verify` | identity + one file → decision |
| `GET` | `/api/v1/speakers` | List enrolled, with active thresholds |
| `DELETE` | `/api/v1/speakers/{identity}` | Remove a profile |
| `POST` | `/api/v1/calls/demo` | Returns a fresh UUID `call_id` |
| `WS` | `/api/v1/calls/{call_id}/stream` | Live audio, `?claimed_identity=` and `?context=` |

Interactive docs at `/docs` (FastAPI's built-in Swagger UI).

`/api/v1/status` deliberately **does not construct anything** — loading the
detector takes ~6 s and 380 MB, so a health endpoint that did that would become
a way to exhaust the box. It reports what is already loaded and what is on disk.

## Model loading

Both models are **process-wide singletons** created lazily on first use, guarded
by a `threading.Lock` double-checked in `VoiceSpoofDetector.get_shared`.
Constructing one per WebSocket connection would download and instantiate the
encoder every time a call starts.

## Error handling

| Condition | Response |
| --- | --- |
| Missing / empty file | 400 with the filename |
| Undecodable audio | 400 naming the supported formats |
| < 1 second | 400 stating the actual duration |
| > 25 MB | 413 stating the limit |
| Invalid identity | 400 stating the rule |
| Invalid context | 400 naming the field and why |
| SpeechBrain missing | 503 on speaker endpoints; other branches still work |
| Bad WebSocket context | Close code 1008 with the reason (before accepting) |
| Postgres down | Logged warning, API starts anyway |

## External services

**None at runtime.** Both models load from local disk (the ECAPA weights are
downloaded from HuggingFace on first use and cached in `models/ecapa/`). No API
keys, no cloud calls, no telemetry.

---

# L. Database / storage

## L.1 Postgres — defined, created, but **never written to**

- **Technology:** PostgreSQL via SQLAlchemy 2.0 async + `asyncpg`.
- **Table:** `calls` (`app/models/call.py`) with columns `id` (UUID PK),
  `caller_name`, `status`, `risk_score`, `created_at`.
- **Service:** `app/services/call_service.py::create_call`.

**Traced and verified: `create_call` is never called by any endpoint.**
`POST /api/v1/calls/demo` simply returns `str(uuid.uuid4())` without touching
the database. `init_db()` runs `create_all` at startup, so the table is created
if Postgres is reachable — and then nothing ever inserts a row.

**Nothing about a call, a score or a decision is persisted.** This is a real gap
and judges may ask about it. Say so plainly.

## L.2 What *is* persisted — a JSON file

`models/speakers.json`, managed by `SpeakerRegistry.save/load`:

```json
{
  "thresholds": { "match": 0.4422, "no_match": 0.3271, "version": "..." },
  "profiles": [
    { "identity": "...", "centroid": [192 floats],
      "sample_count": 5, "consistency": 0.93,
      "encoder": "speechbrain/spkrec-ecapa-voxceleb",
      "created_at": "...", "threshold_version": "..." }
  ]
}
```

**Only the embedding centroid is stored. Audio is never written to disk.**
Asserted in `tests/test_speaker.py` (`test_store_contains_no_audio`,
`test_audio_is_not_persisted`).

The file is gitignored — correct, because embeddings are biometric data.

## L.3 Other on-disk artifacts

| Path | Contents | Committed? |
| --- | --- | --- |
| `models/voxshield_antispoof.pt` | Trained checkpoint, 362.6 MB | No (gitignored) |
| `models/ecapa/` | Downloaded ECAPA weights | No |
| `results/calibration/antispoof_calibration.json` | Platt parameters + provenance | **Yes** |
| `results/speaker-calibration/*.json` | Speaker thresholds | **Yes** |
| `results/exp002-full/` | Training history, reports, DET/ROC plots | **Yes** |
| `kaggle_run/fixed/scores.csv` | 71,237 per-utterance eval scores | Yes (kaggle_run is gitignored, but this copy exists locally) |
| `datasets/test-audio/` | Generated test corpus | No (regenerable) |

## L.4 No blockchain

The theme is "Blockchain & Cybersecurity" and the handoff §57 specifies a
hash-chained tamper-evident audit log. **None of it exists.** There is no
hashing, no chain, no ledger, no smart contract. This was explicitly descoped.

---

# M. Security

## M.1 Implemented

| Control | Where | What it does |
| --- | --- | --- |
| **Upload size limit** | `speakers.py::MAX_UPLOAD_BYTES` | 25 MB, returns 413 |
| **Minimum duration** | `speakers.py::MINIMUM_SECONDS` | 1 s, returns 400 |
| **Format validation** | `decode_upload` | Decodes with soundfile; unreadable → 400 |
| **Identity allowlist** | `SpeakerRegistry.validate_identity` | `^[A-Za-z0-9][A-Za-z0-9 ._@-]*$`, max 128 chars |
| **Context type + range validation** | `CallContext.from_dict` | Types, ranges, unknown-key rejection |
| **NaN / inf sanitisation** | `detector._prepare`, `assess_speech` | `np.nan_to_num` |
| **Odd-byte tolerance** | `_decode_pcm16` | A chunk split mid-sample cannot kill a call |
| **CORS allowlist** | `main.py` + `config.py` | Default `http://localhost:3000` only |
| **Environment config** | `pydantic-settings`, `.env` | `.env` is gitignored |
| **No audio persistence** | `SpeakerRegistry` | Only embeddings stored |
| **WebSocket pre-accept validation** | `calls.py` | Bad context closes with 1008 + reason |

### The identity allowlist is a real fix, not a hypothetical

The live registry had accumulated `../../../../tmp/pwned` and two
5,000-character identities — all accepted, stored and served back. Nothing
writes a file per identity today, so the traversal string was inert, but
`DELETE /api/v1/speakers/{identity}` puts it in a URL path and any future
per-identity storage would make it live. Now: allowlisted (not denylisted —
the set of characters a legitimate account reference needs is small and
knowable; the set of ways to smuggle a separator past a denylist is not), and
pre-existing invalid entries are skipped on load with a warning while the file
is left intact.

## M.2 NOT implemented — be honest about all of this

| Missing | Consequence |
| --- | --- |
| **Authentication** | Every endpoint is fully open |
| **Authorization** | Anyone can enrol, verify, or delete anyone's voice profile |
| **API keys / tokens** | None |
| **Rate limiting** | An attacker could brute-force voices against a profile, or exhaust the GPU |
| **HTTPS / TLS** | Runs on plain HTTP locally |
| **Encryption at rest** | `speakers.json` holds **unencrypted biometric embeddings** |
| **Audit logging** | Nothing records who did what |
| **Replay-attack defence** | No challenge phrase, no liveness check, no nonce |
| **Input sanitisation for log injection** | Identities are logged after validation, which limits it, but it is not a designed control |

`SpeakerRegistry` is deliberately a plain local store so that adding encryption
at rest is a single obvious place to change. **Embeddings remain biometric data
under DPDP/GDPR-style regimes and this store is not production-safe.**

**Prompt injection:** not applicable. There is no LLM anywhere in this system.

---

# N. Prevention and response

The problem statement says "Detection **and Prevention**". Here is exactly what
exists.

## N.1 The action set (`app/ml/policy.py`)

Ordered by friction imposed on the caller:

| Code | Description | Blocking |
| --- | --- | --- |
| `MONITOR` | Continue the call and record the risk event | No |
| `NOTIFY_AGENT` | Flag the elevated risk to the handling agent | No |
| `INCREASE_MONITORING` | Raise the analysis rate for the rest of the call | No |
| `DEVICE_CONFIRMATION` | Ask the caller to confirm on a registered device | No |
| `SECURITY_QUESTIONS` | Knowledge-based verification | No |
| `MFA` | Require multi-factor authentication | No |
| `CALLBACK` | End the call, call back on the number **already on file** | **Yes** |
| `TRANSACTION_HOLD` | Hold pending verification — **"do not reject it"** | **Yes** |
| `MANAGER_APPROVAL` | Require a second authorised person | **Yes** |
| `ESCALATE_SECURITY` | Raise an incident with the security team | No |

## N.2 Mapping

```
INSUFFICIENT_EVIDENCE → COLLECT_MORE_AUDIO, NOTIFY_AGENT
ALLOW                 → MONITOR
WARN                  → MONITOR, NOTIFY_AGENT, INCREASE_MONITORING
SECONDARY_VERIFICATION→ NOTIFY_AGENT, INCREASE_MONITORING, DEVICE_CONFIRMATION, MFA
HIGH_RISK_WORKFLOW    → NOTIFY_AGENT, MFA, CALLBACK, ESCALATE_SECURITY

+ identity NO_MATCH   → adds CALLBACK, ESCALATE_SECURITY
+ amount ≥ 10,000 and risk > ALLOW → adds TRANSACTION_HOLD, MANAGER_APPROVAL
+ amount ≥ 10,000 and risk = ALLOW → adds DEVICE_CONFIRMATION
+ uncalibrated inputs → adds a rationale line saying so
```

`requires_human = any(action.blocking)`.

## N.3 The one rule that is not a placeholder

> **Never block a high-value financial action on one AI voice score.**

Every action is a verification step, a reversible hold, or a human escalation.
None is "reject the transaction because the model said so". A false positive
costs a real customer a verification step; it must never mean a silently refused
legitimate payment. Asserted structurally (on action codes, not prose) in
`tests/test_context_policy.py::test_no_action_ever_rejects_a_transaction`.

## N.4 **What is simulated — state this plainly**

**VoxShield returns actions as JSON. It does not perform them.**

There is no telephony integration, no core-banking integration, no ticketing
system, no email or SMS gateway, no ability to actually terminate a call, place
a hold, or trigger an MFA challenge. The "prevention" layer is a **decision
API** whose output another system would act on.

Nothing in the code claims otherwise, and no teammate should either. The honest
framing: *"we produce the decision and the specific action; wiring that action
into a bank's systems is integration work we have not done."*

---

# O. Project phases

Reconstructed from the git history (34 commits, 2026-09-06 → 2026-09-10), the
handoff document's phase headings, and the actual file contents. Where the
evidence is ambiguous, it says so.

The handoff defines Phases 0–10. The git history shows the *first three phases
arrived together in one initial commit* — so their internal ordering is not
recoverable from git, only from the handoff's own numbering.

### Phase 0 — Problem definition and system planning
- **Goal:** Define the problem beyond "deepfake detection".
- **Evidence:** `VoxShield_Complete_Project_Handoff.md` §18. No code.
- **Key decision:** *VoxShield ≠ deepfake detector.* Fuse anti-spoof + identity
  + prosody + context + NLP + policy + audit.

### Phase 1 — Backend / project foundation
- **Implemented:** FastAPI app, config via pydantic-settings, SQLAlchemy async
  engine, `Call` model, health endpoint, Docker Compose for Postgres.
- **Files:** `app/main.py`, `config.py`, `database.py`, `models/call.py`,
  `api/health.py`.
- **Evidence:** all present in initial commit `54f51c1`.

### Phase 2 — Audio preprocessing / feature foundation
- **Implemented:** VAD, acoustic feature extractor, PCM processing, WebSocket
  streaming, per-call feature pipeline.
- **Files:** `audio/vad.py`, `audio/features.py`, `audio/processor.py`,
  `audio/metrics.py`, `audio/feature_pipeline.py`, `websocket/audio_stream.py`.
- **Note:** `processor.py` and `audio/metrics.py` were written here and are
  **now dead code** — `AudioProcessor` is defined but never instantiated
  anywhere. Verified by grep.

### Phase 3 / 3.6 — Dataset and model pipeline
- **Goal:** Get ASVspoof loaded and a model trainable.
- **Commits:** `f7c9b6b` "Implement ASVspoof DF dataset pipeline",
  `cf57448` "Fix FLAC loading and enforce 4 second audio".
- **Critical honesty note (handoff §10, §23):** Phase 3.6 was reported complete
  and **was not**. `datasets/raw/` did not exist and all three manifests were
  **0 bytes**. This is documented in the project's own handoff.

### Phase 3.7 — First real anti-spoof model
- **Commit:** `ed9cc87` "Make the anti-spoof pipeline trainable end to end".
- **What changed:** The pipeline could not previously run at all — no
  `train_model.py`, no `evaluate_model.py`, `SpectralBranch` did not exist
  despite the doc saying it did, dataset loading called a broken
  `torchaudio.load`, and PyTorch was a CPU-only build.
- **Implemented:** real `SpectralBranch` inside `forward`, attentive stats
  pooling, `AntiSpoofTrainer`, staged freezing, AMP, resume, progress
  reporting, `dataset.py` with memmapped caching, augmentation.

### Phase 3.7b — Cloud training migration
- **Commits:** `0ed4d13`, `58a9f35`, `2160a62`, `4937dd9`, `6f3d8ad`,
  `e6bffbb`, `b494a82`, `055aca5`.
- **Why:** the local download ran at 47 KB/s (43-hour ETA). Pivoted to Colab,
  then Kaggle when Colab's free tier was exhausted.
- **Notable bugs fixed:** notebooks collapsed to one line (source lists lacked
  trailing `\n`, and the validation had checked `"\n".join(...)` which silently
  reinserted them); `rmtree` deleting the kernel's own working directory.

### Phase 3.7c — Corpus discovery and protocol handling
- **Commits:** `c57137a` (five filesystem walks → one; 271 s → 0.21 s),
  `e68a63b` (two protocol-detection bugs: ASV protocols were being chosen over
  countermeasure protocols, making the test set 100% bonafide and discarding
  63,882 rows; and column detection sampling the first 2,000 rows, which are
  all bonafide).
- **Files:** `app/ml/discovery.py`.

### Phase 3.8 — Evaluation
- **Commits:** `3f9f2f6` (compare_runs), `c183767` (**the false-alarm operating
  point reported 100% miss** — `acceptable[0]` instead of `acceptable[-1]`),
  `522bd62` (record results).
- **Outcome:** run 1 → 11.27% EER; run 2 with A05/A06 restored → **2.95% EER**.
  The ablation became the project's strongest finding.

### Phase 4 — Speaker verification
- **Commits:** `1729315`, `32a5dfc`, `a870603`, `78ae47a`, `9c5355f`,
  `278cf7a`, `dc50e0c`.
- **Implemented:** `speaker.py` (registry, thresholds, encoder, live tracker),
  `speaker_calibration.py`, `fusion.py`, the speaker API, and fusion wired into
  the live path.
- **Note:** `278cf7a` untracked 79 MB of accidentally committed ECAPA weights.

### Phase 5 — Multi-branch audio forensics
- **Status: NOT IMPLEMENTED.** The handoff §50 specifies prosody and phase
  forensics. `FusionWeights` reserves 10 points each for `prosody` and
  `audio_forensics`, and `fuse()` accepts the parameters, but **no caller ever
  supplies them**. They are permanently in `missing_signals`.

### Phase 6 — Contextual fraud intelligence + policy engine
- **Commit:** `79adb4a`.
- **Implemented:** `context.py` (13 signals), `policy.py` (10 graded actions).

### Phase 7 — Blockchain audit layer
- **Status: NOT IMPLEMENTED, explicitly descoped by the project owner.**

### Phase 8 — Dashboard
- **Commit:** `811d1f2`.
- **Implemented:** four-tab testing console, in-browser WAV recording, typed
  API client. Replaced a scaffold that could only start a live call.

### Phase 9 — Adversarial testing
- **Status: PARTIAL.** `scripts/verify_system.py` covers degenerate and hostile
  *inputs* (silence, noise, traversal identities, malformed context, oversized
  and non-audio uploads). **No adversarial-example generation against the model
  itself.**

### Phase 10 — Production optimisation
- **Status: NOT IMPLEMENTED.** No ONNX export, no quantisation, no batching, no
  load testing, no containerised deployment of the API.

### Post-phase hardening (final session)
- **Commits:** `37dbb5b`, `291eee6`, `05ba5f9`, `811d1f2`, `23d1da0`.
- Platt calibration; the speech-presence gate; `INSUFFICIENT_EVIDENCE`; identity
  validation; the `/status` endpoint; the test corpus generator; the end-to-end
  verifier; `start.ps1`; `TESTING.md`.

> The handoff mentions "approximately 13 phases". The document itself defines
> Phases 0–10 plus the 3.6 / 3.7 / 3.8 sub-phases, which totals 13 numbered
> stages. A literal 13-item list beyond that is **not verifiable from the
> current codebase.**

---

# P. File-by-file architecture

## `backend/app/ml/` — the intelligence

| File | Purpose | Key contents |
| --- | --- | --- |
| `model.py` | The anti-spoof network | `AntiSpoofModel`, `SpectralBranch`, `AttentiveStatsPooling`, `AntiSpoofInference` |
| `detector.py` | Serving wrapper | `VoiceSpoofDetector` — rebuilds architecture from checkpoint metadata, applies calibration, process-wide singleton |
| `inference.py` | Streaming | `StreamingInferenceEngine` — buffer, window, hop, speech gate, smoothing |
| `risk.py` | Temporal aggregation | `TemporalRiskEngine` (exponential smoothing), `RiskThresholds` (hysteresis) |
| `calibration.py` | Probability calibration | `ProbabilityCalibrator` — Platt + temperature, prior shifting, ECE/Brier |
| `speaker.py` | Speaker verification | `SpeakerEncoder`, `SpeakerRegistry`, `SpeakerProfile`, `SpeakerThresholds`, `LiveSpeakerTracker`, identity validation |
| `speaker_calibration.py` | Threshold fitting | `build_trials`, `score_trials`, `calibrate` (three operating modes on an EER-relative budget) |
| `fusion.py` | Branch combination | `RiskFusion`, `FusionWeights`, `PolicyThresholds`, `Contribution`, `FusedRisk` |
| `context.py` | Contextual signals | `CallContext` (+ `from_dict` validation), `ContextWeights`, `ContextAnalyzer` |
| `policy.py` | Actions | `PolicyEngine`, `Action`, `PolicyDecision` |
| `metrics.py` | Evaluation | EER, minDCF, DET, ROC, per-attack breakdown, plots |
| `dataset.py` | Data loading | `load_manifest`, `read_audio`, `fit_length`, `WaveformStore` (memmap), `VoiceSpoofDataset`, `balanced_sample_weights` |
| `training.py` | Training loop | `AntiSpoofTrainer` — staged freezing, AMP, resume, speed probe |
| `augmentation.py` | Channel realism | `AudioAugmenter` — telephone, noise, gain, codec, reverb, clip, packet loss |
| `discovery.py` | Corpus location | Finds ASVspoof anywhere by **utterance-ID prefix**, not folder names |
| `progress.py` | Training UX | `TrainingProgress`, `RateTracker`, `PhaseTimer`, ETA estimation |
| `config.py` | Configuration | `DetectorConfig`, `TrainingConfig`, `resolve_path` |

## `backend/app/audio/` — signal processing

| File | Purpose | Used? |
| --- | --- | --- |
| `feature_pipeline.py` | Per-call orchestrator | **Yes** — the live path |
| `vad.py` | Energy VAD with adaptive noise floor | **Yes** |
| `speech_check.py` | Speech-presence gate | **Yes** — both paths |
| `features.py` | MFCC / pitch / spectral stats | **Display only** |
| `processor.py` | `AudioProcessor` | **No — dead code** |
| `metrics.py` | RMS / peak / ZCR helpers | Only by `processor.py`, itself dead |

## `backend/app/api/` and the rest

| File | Purpose |
| --- | --- |
| `api/analyze.py` | `POST /analyze` — the main endpoint |
| `api/speakers.py` | Enrol / verify / list / delete, `decode_upload`, threshold loading |
| `api/calls.py` | `POST /calls/demo`, the WebSocket route, `parse_context` |
| `api/health.py` | `/health` and `/status` |
| `websocket/audio_stream.py` | `AudioStreamManager` — per-call pipelines |
| `main.py` | App assembly, CORS, lifespan |
| `config.py` | `Settings` |
| `database.py` | Async engine, `Base`, `init_db` |
| `models/call.py` | `Call` ORM model — **table created, never written** |
| `services/call_service.py` | `create_call` — **never called** |

## `backend/scripts/`

| Script | Purpose |
| --- | --- |
| `train_model.py` | Train, with a speed probe that reports cost before it costs it |
| `evaluate_model.py` | EER / minDCF / ROC-AUC / per-attack / plots |
| `calibrate_detector.py` | Fit Platt calibration, derive operating points |
| `calibrate_speaker.py` | Fit speaker thresholds from ASV protocols |
| `build_manifest.py` | Protocol files → manifests, auto-detecting columns |
| `dataset_stats.py` | Verify manifests before committing to a run |
| `cache_dataset.py` | Decode once into a memmap |
| `compare_runs.py` | Side-by-side evaluation reports (the ablation) |
| `merge_scores.py` | Pool per-utterance scores across runs |
| `make_test_audio.py` | Generate the local test corpus |
| `verify_system.py` | 59 end-to-end checks against a running API |
| `package_code.py` | Zip source for Colab upload |

## Repository root

| Path | Purpose |
| --- | --- |
| `start.ps1` | One-command startup with preflight checks |
| `TESTING.md` | Case-by-case testing guide |
| `TRAINING.md` | Training runbook |
| `README.md` | Overview and status |
| `VoxShield_Complete_Project_Handoff.md` | The original design document |
| `results/` | Committed evaluation artifacts |
| `notebooks/` | 4 Kaggle/Colab training notebooks |
| `docker-compose.yml` | Postgres for local development |

---

# Q. Technology stack

| Layer | Technology | Version | Used for |
| --- | --- | --- | --- |
| **Frontend framework** | Next.js (App Router) | 15.5.0 | UI |
| | React | 19.1.0 | Components |
| | TypeScript | ^5 | Type safety |
| **Backend framework** | FastAPI | 0.141.1 | HTTP + WebSocket |
| | Uvicorn | 0.52.4 | ASGI server |
| | Pydantic / pydantic-settings | 2.13 / 2.15 | Validation, config |
| **Languages** | Python | 3.11.9 | Backend, ML |
| | TypeScript / JavaScript | — | Frontend |
| | PowerShell | — | `start.ps1` |
| **ML framework** | PyTorch | 2.14.0+cu130 | Model, training, inference |
| | torchaudio | 2.11.0+cu130 | Mel spectrogram, resampling |
| | Transformers | 4.57.6 | wav2vec2 loading |
| | SpeechBrain | 1.1.1 | ECAPA-TDNN |
| **AI models** | `facebook/wav2vec2-base` | — | SSL encoder (fine-tuned) |
| | `speechbrain/spkrec-ecapa-voxceleb` | — | Speaker embeddings (frozen) |
| **Speech-to-text** | **NONE** | — | **Not part of this system** |
| **Audio libraries** | soundfile (libsndfile) | 0.14.0 | Decoding WAV/FLAC/OGG |
| | librosa | 0.11.0 | Display features, pitch (pyin) |
| | NumPy | 2.4.6 | Arrays |
| | SciPy | 1.17.1 | Signal processing |
| **Metrics / data** | scikit-learn | 1.9.0 | ROC, DET |
| | pandas | 3.0.5 | Manifests, score files |
| | matplotlib | 3.11.1 | DET/ROC plots |
| **Database** | PostgreSQL + SQLAlchemy 2.0 + asyncpg | 2.0.52 / 0.31 | **Declared, never written to** |
| **Testing** | pytest, anyio, httpx | 9.1.1 | 312 tests |
| | requests, websockets | 2.34 / 17.1 | `verify_system.py` |
| **Training platform** | Kaggle (2× Tesla T4) | — | Where the model was trained |
| **Local hardware** | RTX 3050 6 GB Laptop GPU | — | Inference |
| **Cloud / external APIs** | **NONE at runtime** | — | Everything runs locally |
| **Deployment** | **None** — local only | — | No container for the API, no hosting |

---

# R. What is actually working right now

## Fully implemented and working — verified by running it

- Anti-spoof detection on uploaded audio, **2.95% EER measured**
- Platt calibration, applied in exactly one place
- Speech-presence gate rejecting silence, noise, tones, hum, DC, muted audio
- Speaker enrolment, verification, listing, deletion
- Three-way MATCH / UNCERTAIN / NO_MATCH with calibrated thresholds
- Context scoring across 13 signals with per-signal reasons
- Risk fusion with renormalisation over available branches
- `INSUFFICIENT_EVIDENCE` when nothing could be assessed
- Policy engine producing graded actions
- Live WebSocket streaming with per-second scoring
- Rolling speaker verification over a call
- `/status` diagnostics
- Four-tab frontend, in-browser WAV recording
- Identity and context input validation
- `start.ps1` with preflight checks
- Test corpus generation
- **312 unit tests pass; 59 end-to-end checks pass** (both run and verified)

## Implemented but needs verification

- **Behaviour on real human speech through a laptop microphone.** The false-alarm
  rate is measured on ASVspoof bonafide clips (clean, read speech). Nobody has
  measured it on a real mic in a real room. `TESTING.md` §1.2 covers what to do
  if your own voice reads as synthetic.
- **Real telephone audio.** Telephone augmentation was applied in training but
  no genuine phone call has been scored.
- **Concurrent load.** Never tested with more than one call at a time.
- **Non-Windows.** `make_test_audio.py` skips TTS generation off Windows;
  `start.ps1` is PowerShell-only. The backend itself should be portable but has
  not been run on Linux or macOS.

## Partially implemented

- **Database** — schema and service exist; **nothing ever writes a row**.
- **Adversarial testing** — hostile *inputs* covered; no adversarial examples
  against the model.
- **Speaker calibration modes** — `high_security` / `balanced` / `low_friction`
  exist; the shipped files have `low_friction` **identical to** `balanced`
  because they were generated before that bug was fixed. The system uses
  `balanced`, which is correct, and `/status` warns about the staleness.
- **`app/audio/processor.py` and `app/audio/metrics.py`** — written in Phase 2,
  now unreferenced dead code.

## Mocked / simulated

- **All policy actions.** They are returned as JSON. Nothing calls a phone
  system, holds a transaction, sends an MFA challenge, or opens a ticket.
- **`POST /api/v1/calls/demo`** returns a UUID and creates no state anywhere.
- **The context presets** in the UI are fixed example payloads, not data from a
  real banking system.

## Not implemented

- **Speech-to-text / NLP transcript analysis** (weight reserved, never supplied)
- **Prosody branch** (weight reserved, never supplied)
- **Phase / audio forensics branch** (weight reserved, never supplied)
- **Blockchain audit layer** — explicitly descoped
- **Authentication and authorization**
- **Rate limiting**
- **Encryption at rest for biometric embeddings**
- **Replay-attack detection** (ASVspoof PA never used)
- **Multi-language support**
- **ONNX / quantisation / edge optimisation**
- **Deployment** — no hosting, no container for the API
- **Frontend tests**
- **Cross-dataset evaluation** on ASVspoof 2021 DF

## Future scope

1. Fit the fusion weights on labelled fraud data — the single biggest gap
   between "prototype" and "product".
2. Prosody and phase-forensics branches (Phase 5).
3. NLP transcript analysis — which **would** require adding speech-to-text.
4. Blockchain audit layer (Phase 7).
5. Authentication, rate limiting, encryption at rest.
6. Cross-dataset evaluation on ASVspoof 2021 DF to measure generalisation.
7. Persist calls and decisions so the risk score can be audited after the fact.
8. Replace the energy VAD with a neural VAD (Silero/WebRTC) — the interface is
   already designed for this.

---

# S. Testing

## S.1 Unit and integration tests — `backend/tests/`, run with pytest

**Verified: `pytest -q` → 312 passed, 1 deselected.** Run at commit `23d1da0`.

| File | Tests | What it checks |
| --- | --- | --- |
| `test_affine_calibration.py` | 24 | Temperature cannot move a boundary; affine can; prior shifts; overflow safety; the shipped artifact's parameters and metadata |
| `test_context_policy.py` | 28 | Unknown vs known signals; per-signal scoring; type and range validation; **no action ever rejects a transaction**; identity mismatch forces out-of-band verification |
| `test_dataset.py` | 28 | Manifest loading, length fitting, memmap pickling, stratified subsampling |
| `test_speaker.py` | 28 | Centroid maths, enrolment gates, three-way decisions, **no audio persisted** |
| `test_speaker_calibration.py` | 22 | Trial construction (enrolment held out of probes), EER-relative budgets, mode ordering |
| `test_fusion.py` | 18 | Renormalisation, contributions summing to the score, NO_MATCH override, `INSUFFICIENT_EVIDENCE` |
| `test_speaker_hardening.py` | 17 | Path-traversal identities, oversized identities, stale-store loading, calibration fallback, the speech gate from both sides |
| `test_metrics.py` | 15 | EER interpolation, minDCF, the infinite-threshold NaN bug, the descending-threshold bug |
| `test_discovery.py` | 15 | Corpus location by utterance-ID prefix, CM-over-ASV protocol ranking |
| `test_inference.py` | 13 | Window/hop arithmetic, smoothing convergence, **silence and noise are not scored**, refused windows do not move the risk |
| `test_calibration.py` | 13 | Temperature scaling backward compatibility, ECE reduction, search-bound detection |
| `test_api_speakers.py` | 12 (async) | Enrol / verify / analyze endpoints via httpx |
| `test_live_pipeline.py` | 11 | Rolling tracker, encoder failure degrades rather than drops, odd byte counts |
| `test_augmentation.py` | 9 | Telephone band attenuation, noise SNR, packet dropout |
| `test_resume.py` | 9 | Weights + optimizer moments + scheduler + RNG round-trip |
| `test_risk.py` | 9 | Hysteresis, smoothing, reason generation |
| `test_vad.py` | 9 | Noise-floor tracking, the mid-speech lockout case |
| `test_audio_features.py` | 2 | Feature extraction shape and NaN safety |
| `test_health.py` | 2 (async) | Health endpoint |

## S.2 End-to-end verification — `scripts/verify_system.py`

**Verified: 59 passed, 0 failed, 0 skipped** against a live server on `:8000`.

Six groups:

1. **System (6)** — reachable, checkpoint present, calibrated, held-out numbers
   recorded, encoder available, fusion correctly declares itself uncalibrated.
2. **Spoof detection (9)** — all nine SAPI TTS files detected at ≥ 0.5.
3. **Edge cases (15)** — silence, white noise, pure tone, dial tone, mains hum,
   DC offset and muted audio all → `no_speech` + `INSUFFICIENT_EVIDENCE`;
   too-short, non-audio and empty → HTTP 400; stereo, 8 kHz, 48 kHz, clipped and
   quiet → still detected.
4. **Speaker (7)** — enrol one TTS voice, MATCH against itself (0.987),
   NO_MATCH against another (0.148), a 0.839 gap, unenrolled identity handled,
   impersonator escalates to risk 93.
5. **Input validation (8)** — four hostile identities and four malformed
   contexts all rejected with readable messages.
6. **Live stream (7)** — 111 browser-sized chunks, 24 windows scored, fused
   risk produced, reasons present, policy actions emitted, malformed context
   rejected at connect.

## S.3 What kinds of tests exist

| Type | Present? | Where |
| --- | --- | --- |
| Unit | Yes | `tests/` |
| Integration | Yes | `test_api_speakers.py`, `test_live_pipeline.py` |
| API | Yes | httpx in tests, requests in `verify_system.py` |
| Model behaviour | Yes | Calibration, metrics, gate behaviour |
| Audio / edge case | Yes | 15 degenerate inputs |
| Security | **Partial** — input validation only | `test_speaker_hardening.py` |
| **Frontend** | **None** | — |
| **Load / performance** | **None** | — |

## S.4 Sample test cases worth quoting

- `test_temperature_cannot_move_the_boundary` — proves *why* Platt scaling was
  necessary, not just that it works.
- `test_no_action_ever_rejects_a_transaction` — asserts on **action codes**, not
  prose, because `TRANSACTION_HOLD`'s own description contains the word "reject"
  precisely to say it is not one.
- `test_a_refused_window_does_not_move_the_risk` — silence during a call must
  not drag an established risk toward ALLOW.
- `test_the_threshold_sits_between_the_two_populations` — pins the speech gate
  from both sides so a later tweak that collapses either margin fails loudly.

---

# T. Questions judges are likely to ask

**Why is voice cloning dangerous?**
A convincing clone of a specific person can now be produced from a short sample,
while a large amount of financial authorisation still happens over the phone
where the only identity check is recognition. It scales: one cloned voice can be
used against many targets.

**How exactly do you detect a cloned voice?**
A fine-tuned `wav2vec2-base` transformer plus a log-Mel CNN, fused through
attentive statistics pooling into a two-class classifier. It consumes the raw
waveform and its spectrum — no text, no transcription. The output is calibrated
with Platt scaling, and `p ≥ 0.5` means synthetic. Trained on ASVspoof 2019 LA,
measured at 2.95% EER on 71,237 utterances covering 13 attacks it never trained
on.

**Why did you choose wav2vec2?**
Self-supervised speech representations are the standard strong baseline for this
task, and `wav2vec2-base` is the largest that fine-tunes inside 6 GB with staged
freezing. The encoder is a config parameter — WavLM and XLS-R also load.

**What dataset did you use?**
ASVspoof 2019 Logical Access: 25,380 train, 24,844 dev, 71,237 eval. The eval
attacks (A07–A19) do not overlap the training attacks (A01–A06). The corpus is
not in the repository — training ran on Kaggle — but the 71,237 per-utterance
scores and the evaluation reports are committed.

**How do you know the model is accurate?**
Evaluated on a held-out split with disjoint attack types: 2.95% EER, ROC-AUC
0.9863, normalised minDCF 0.1474. Calibration was fitted on one half of the eval
set and reported on the other. And it detects Windows SAPI TTS — a synthesis
family absent from the training data — at 0.97 raw, which is evidence of real
generalisation.

**What metrics do you use?**
EER, normalised minDCF, ROC-AUC, DET curves, per-attack miss rates, and
ECE/Brier for calibration. At runtime, a calibrated probability and a fused
0–100 risk score — **these are different things** and we keep them separate.

**How do you handle different accents?**
Honestly: **we do not, specifically.** ASVspoof 2019 LA is English (VCTK-derived)
read speech. Accent robustness is **not measured** and is a genuine limitation.
The pretrained encoder was trained on broader data, which may help, but we have
no number for it.

**How do you handle background noise?**
Training applied noise, telephone band-pass, μ-law codec, reverb, gain, clipping
and packet loss augmentation. Separately, the speech-presence gate holds down to
**10 dB SNR** (measured) before it refuses a clip. Beyond that it declines to
score rather than guessing.

**What happens with poor-quality audio?**
Three graded outcomes: under 1 second or undecodable → HTTP 400 with a specific
message; steady/silent/noise-only → scored as `no_speech`, branch withheld,
decision `INSUFFICIENT_EVIDENCE`; degraded but usable → scored normally, with
`speech_check` measurements returned so you can see the quality.

**How do you distinguish a real speaker from a cloned version?**
Two independent checks. Anti-spoof asks "was this generated?" from waveform and
spectrum. Speaker verification asks "is this the enrolled person?" via cosine
similarity between a 192-d ECAPA embedding and the enrolled centroid. A good
clone might pass the second and fail the first; a human impersonator fails the
second and passes the first. Both are needed.

**Can an attacker bypass the system?**
Yes, in known ways, and we can name them. (1) A synthesis method whose artefacts
differ from anything in ASVspoof — our own ablation shows exactly this failure,
with voice-conversion miss rising from 9.5% to 48.8% when that family was held
out. (2) A **replay attack** — we never trained on ASVspoof PA and have no
liveness check. (3) A very good clone that also matches the speaker embedding.
(4) There is **no authentication**, so the API itself is open. The mitigation is
architectural rather than model-based: no single branch decides, and no action
is irreversible.

**What are false positives and false negatives?**
A false positive flags a genuine caller as synthetic — measured at **0.79%**, so
about 1 in 127 real callers gets an extra verification step. A false negative
lets an attack through — **5.05%**, about 1 in 20. We deliberately chose an
operating point that tolerates more misses than false alarms, because the cost
of the two is asymmetric: a false positive costs a customer 30 seconds, and a
system that fires constantly gets switched off.

**How fast is detection?**
~30 ms per 3-second window on an RTX 3050, after a one-time ~6 s model load. In
the live stream a new score arrives roughly once per second.

**Is it truly real-time?**
Yes, with the definition stated: continuous streaming with ~1 s update latency
and a ~3 s warm-up before the first score. It is **not** sub-100 ms
utterance-level detection, and we should not claim that.

**How does speech-to-text fit into the system?**
**It does not. There is no speech-to-text anywhere in this project.** Detection
is entirely acoustic — waveform, spectrum, and speaker embeddings. NLP
transcript analysis is planned (a weight of 5 is reserved in `FusionWeights`)
but not implemented, and implementing it would mean adding an ASR component
that does not currently exist.

**What happens if the STT system makes a mistake?**
Not applicable — there is no STT. If one were added later, its errors would
affect only the NLP branch, and the fusion layer already handles a branch being
wrong or absent by renormalising over the others.

**What happens if the attacker uses a new voice cloning model?**
Detection degrades, and how much depends on whether it is a *new instance* of a
familiar family or a genuinely *new family*. We measured this: within-family
generalisation is strong (TTS miss near 0% across six unseen TTS attacks),
cross-family generalisation is weak (VC miss 48.8% when VC was held out).
Mitigation is retraining on new attack families as they appear, plus the fact
that the identity and context branches do not depend on synthesis artefacts at
all.

**What about replay attacks?**
**Not handled.** ASVspoof 2019 has a Physical Access set for exactly this and we
never used it. There is no liveness check and no challenge-response. This is a
known, stated gap.

**Is this speaker verification or deepfake detection?**
Both, plus context, fused. That is the design thesis. Measured: on a human
impersonator, anti-spoof alone gives risk 4.0 → ALLOW; adding identity gives
60.0 → SECONDARY_VERIFICATION. Either component alone misses a whole attack
class.

**How do you protect user audio data?**
Audio is never written to disk — only the 192-d embedding centroid, asserted by
two tests. But be honest about the rest: `models/speakers.json` holds
**unencrypted biometric embeddings**, there is **no authentication**, and there
is **no encryption at rest**. The store is deliberately a plain local file so
encryption is a single obvious place to add.

**How scalable is the system?**
Single-node prototype. Models are process-wide singletons, which is correct, but
per-call state lives in an in-memory dict with no eviction or cap, there is no
batching, no queue, no GPU-sharing strategy, and multiple workers would each
load their own copy. Scaling would mean externalising the registry, batching
inference, and a proper session store. Not done.

**What happens when the model is uncertain?**
It is designed into the output rather than hidden. Speaker verification has an
explicit **UNCERTAIN** band between 0.327 and 0.442. The fusion reports
`missing_signals` for every branch that could not run. When *no* branch could
run, the decision is `INSUFFICIENT_EVIDENCE` — never `ALLOW` — because "we could
not check" and "we checked and it is fine" must not produce the same answer.

**What is the biggest limitation?**
The fusion weights are uncalibrated placeholders. The two individual models are
measured and defensible; the number that combines them (35/30/10/10/10/5) is
engineering intuition with no labelled fraud data behind it. Every response says
so. Second biggest: no real telephone audio has ever been scored.

**Why is your risk score not a probability of fraud?**
Because it is a weighted average of branch risks using placeholder weights.
Calling it a fraud probability would imply a calibration we have not done. The
*anti-spoof* probability **is** calibrated (ECE 0.0147 held out); the fused
score is not, and is labelled advisory everywhere it appears.

**What is your future scope?**
In priority order: fit the fusion weights on labelled fraud data; add the
prosody and forensics branches; add authentication, rate limiting and encryption
at rest; persist decisions for audit; cross-dataset evaluation on ASVspoof 2021
DF; then the blockchain audit layer and NLP branch.

---

## One-paragraph summary for a pitch

VoxShield takes a voice call and answers three independent questions — is the
speech machine-generated, is the speaker who they claim to be, and is the
situation around the call suspicious — then fuses them into a single explainable
risk score with a specific, reversible action attached. The anti-spoof model is
a wav2vec2 transformer fused with a log-Mel CNN, fine-tuned on ASVspoof 2019 and
measured at **2.95% EER on 71,237 utterances covering 13 attack types it never
trained on**, with calibrated probabilities giving **94.95% detection at 0.79%
false alarm**. Speaker verification uses a pretrained ECAPA-TDNN at **0.373% EER
on 67 unseen speakers**. The system refuses to score audio it cannot assess,
reports every branch that did not run, and never returns an action that rejects
a customer's transaction — because a false positive should cost someone thirty
seconds, not their payment.
