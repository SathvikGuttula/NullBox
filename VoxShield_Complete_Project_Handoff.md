# VoxShield — Complete Project Handoff

> **Repository audit — 2026-09-08.** This document was checked line by line
> against the actual repository. Several claims were false. The corrections are
> inline and marked; the biggest ones:
>
> | claim in the doc | reality in the repo |
> |---|---|
> | Ryzen 9 / RTX 5050 / 8 GB VRAM | Ryzen 7 7840HS / **RTX 3050 6 GB** laptop |
> | Phase 3.6 dataset prep "complete" | `datasets/raw/` did not exist; all three manifests were **0 bytes** |
> | `train_model.py`, `evaluate_model.py` | **did not exist** — now written |
> | `SpectralBranch` exists but is unwired | **did not exist at all** — now written and wired |
> | `app/ml/risk.py` (§43, §44) | **did not exist** — now written |
> | Start with ASVspoof 2021 DF | DF is **evaluation-only**; start with ASVspoof 2019 LA |
> | (unstated) | PyTorch installed was **CPU-only**; `torchaudio.load` was broken |
>
> The operational guide is **`TRAINING.md`** in the repository root. It has the
> verified dataset download, the measured hardware, and a runbook.

## 1. Project Identity

**Project:** VoxShield

**SIH 2026 Problem Statement ID:** 26104

**Title:** AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks

**Organization:** AICTE

**Department:** Cyber Security Cell

**Category:** Software

**Theme:** Blockchain & Cybersecurity

### Core Problem

Modern voice-cloning systems can generate convincing speech from very small samples of a person's voice.

VoxShield is therefore not simply:

> "Is this audio AI generated?"

It is:

> "Is the person speaking genuine, does the voice actually belong to the claimed person, are there acoustic/behavioral signs of manipulation, and is the surrounding call/transaction suspicious enough to require intervention?"

This distinction should remain central to the project.

---

# 2. Final Intended VoxShield Architecture

```text
                    LIVE CALL / VOIP / AUDIO
                              │
                              ▼
                    Audio Capture Layer
                              │
                              ▼
                 Voice Activity Detection
                              │
                              ▼
                  2–4 second audio windows
                              │
                              ▼
                    Audio Preprocessing
                              │
          ┌───────────────────┼────────────────────┐
          │                   │                    │
          ▼                   ▼                    ▼
   Anti-Spoof Model     Speaker Verification   Audio Forensics
          │                   │          ┌─────────┼─────────┐
          │                   │          │         │         │
          │                   │       Spectral   Phase    Prosody
          │                   │
          ▼                   ▼
 Deepfake Probability   Identity Similarity
          │                   │
          └─────────────┬─────┘
                        │
                        ▼
                  Model Fusion Layer
                        │
                        ▼
                Audio Threat Score
                        │
              ┌─────────┴──────────┐
              │                    │
              ▼                    ▼
       Call Metadata          NLP / Context
       Fraud Signals          Fraud Analysis
              │                    │
              └──────────┬─────────┘
                         │
                         ▼
                  Dynamic Risk Engine
                         │
                         ▼
                     SCORE 0–100
                         │
             ┌───────────┼───────────┐
             ▼           ▼           ▼
            LOW      SUSPICIOUS      HIGH
             │           │           │
             ▼           ▼           ▼
           Allow       Warning      MFA /
                        User        Callback /
                                   Escalation
                         │
                         ▼
                 Audit Event Generated
                         │
                         ▼
              Tamper-Evident Blockchain
                      Audit Log
```

---

# 3. Anti-Spoof / Deepfake Detector

Answers:

> "Does this audio look synthetic, cloned, replayed or manipulated?"

It can examine:

- Speech embeddings
- Spectral characteristics
- Phase inconsistencies
- Vocoder artifacts
- Abnormal high-frequency structure
- Temporal artifacts
- Codec artifacts
- Generated speech fingerprints

Example output:

```text
spoof_probability = 0.91
```

---

# 4. Speaker Verification

Answers:

> "Does this voice actually belong to the claimed person?"

Example:

```text
Caller claims:
CEO = Alice

Stored Alice voice embedding:
[...embedding...]

Current caller embedding:
[...embedding...]

                ↓

cosine similarity
                ↓

identity match probability
```

This is essential.

A cloned voice may resemble the target while still producing subtle inconsistencies.

A real human attacker can also produce:

```text
Deepfake detection → genuine human
Speaker verification → WRONG PERSON
```

So anti-spoof detection alone is insufficient.

---

# 5. Behavioural / Prosody Analysis

This analyzes how someone speaks rather than only how their voice sounds.

Planned features:

```text
pitch
pitch variance
speaking rate
pause frequency
pause duration
energy
energy variance
rhythm
microprosody
syllable timing
intonation
voice stability
```

Example:

```text
Historical CEO:

Average speaking rate = 147 words/min
Pitch variance = normal
Pause pattern = consistent

Current call:

Speaking rate = 191 words/min
Pitch extremely stable
Abnormal pause placement
```

This creates another fraud signal.

---

# 6. Spectral / Audio-Forensic Branch

This branch exists because many synthetic voices sound convincing to humans but leave artifacts in the frequency domain.

Planned representations include:

```text
Log-Mel spectrogram
LFCC
CQCC
STFT
phase spectrum
spectral centroid
spectral rolloff
spectral flux
harmonic information
```

These can be processed using CNN/ResNet-style models.

---

# 7. SSL Speech Branch

Modern self-supervised speech models are intended to provide strong learned representations.

Models considered:

```text
Wav2Vec2
WavLM
HuBERT
XLS-R
```

The baseline implementation used:

```text
facebook/wav2vec2-base
```

Later improvements can replace or ensemble it with stronger models.

---

# 8. Other Anti-Spoof Architectures Considered

```text
CNN / ResNet spectrogram classifier
RawNet / RawNet2
AASIST
Wav2Vec2
WavLM
HuBERT
XLS-R
Artifact classifier
Spectral CNN
```

These are not intended to all run simultaneously in the initial version.

A later ensemble could resemble:

```text
WavLM / XLS-R
      +
AASIST
      +
Spectral CNN
      +
Prosody detector
      +
Speaker verification
```

Then fuse their scores.

---

# 9. Speaker Verification Model

Proposed primary architecture:

```text
ECAPA-TDNN
```

Workflow:

```text
voice
 ↓
VAD
 ↓
speech segment
 ↓
ECAPA-TDNN
 ↓
speaker embedding
 ↓
L2 normalization
 ↓
cosine similarity
 ↓
identity confidence
```

Enrollment should use multiple samples.

Recommended:

```text
3–5 clean recordings per enrolled speaker
```

Then:

```text
embedding 1
embedding 2
embedding 3
embedding 4
embedding 5
      │
      ▼
normalized centroid
      │
      ▼
stored speaker prototype
```

Prefer storing embeddings instead of unnecessary long-term raw audio.

Speaker embeddings are still biometric data and must be protected accordingly.

---

# 10. Contextual Fraud Engine

VoxShield should not decide solely from audio.

Examples of context:

```text
Unknown number
New device
International origin
Unusual time
High-value transaction
First-time beneficiary
Urgent payment request
Password reset request
Unusual location
Repeated failed verification
```

Example:

```text
Voice spoof score:        68
Identity mismatch:        75
Unknown phone number:     +8
High transaction value:   +10
Unusual time:             +4
```

These feed the final fraud engine.

---

# 11. NLP / Transcript Fraud Analysis

Later the audio stream should also go through speech-to-text.

```text
Audio
  ↓
ASR
  ↓
Transcript
  ↓
Fraud language classifier
```

Potential suspicious language indicators:

```text
urgency
authority impersonation
credential requests
OTP requests
financial instructions
social engineering
secrecy pressure
```

The ASR/NLP model was deliberately not finalized in the early phases.

---

# 12. Dynamic VoxShield Risk Score

The system eventually produces:

```text
0 ─────────────────────────── 100

LOW        SUSPICIOUS         HIGH
```

Example fusion concept:

```text
Anti-spoof detection       35%
Speaker verification      30%
Audio forensics           10%
Prosody                    10%
Contextual fraud           10%
NLP fraud                   5%
```

These percentages are illustrative only.

They must eventually be calibrated using validation data.

Do not hard-code them as scientifically established values.

---

# 13. What Happens After Detection

### LOW

```text
continue normally
record minimal event metadata
```

### SUSPICIOUS

Possible actions:

```text
show warning
request secondary verification
notify receiving employee
increase monitoring
```

### HIGH

Possible actions:

```text
require MFA
initiate callback verification
require transaction approval
escalate to security
prevent automated high-risk action
```

Important design principle:

**Do not automatically block major financial actions based solely on one AI voice score.**

High-impact intervention should combine multiple independent signals.

---

# 14. Blockchain Component

## Do NOT store audio on blockchain.

Bad design:

```text
call audio
     ↓
blockchain
```

This causes:

- Privacy problems
- Storage problems
- Performance problems
- Biometric data exposure

Instead blockchain is used for **tamper-evident security auditing**.

Example event:

```json
{
  "event_id": "VX-10482",
  "timestamp": "...",
  "session_hash": "...",
  "model_version": "...",
  "audio_evidence_hash": "...",
  "spoof_score": 0.91,
  "identity_score": 0.34,
  "risk_score": 87,
  "decision": "SECONDARY_VERIFICATION",
  "policy_version": "v1.3"
}
```

Hash this record:

```text
Security event
     ↓
SHA-256 hash
     ↓
Blockchain ledger
```

Blockchain proves:

- What decision occurred
- When it occurred
- Which model made it
- Which policy version was active
- Whether evidence was later modified

---

# 15. Privacy Model

VoxShield should follow:

```text
Edge inference whenever possible
Minimal audio retention
Encrypted speaker embeddings
Hash-based evidence records
Configurable retention
Role-based access control
No raw voice recordings on blockchain
```

---

# 16. Intended Real-Time Flow

A call should not wait until it finishes.

Use overlapping windows:

```text
0s────3s────6s────9s────12s────15s
|-----|
     |-----|
          |-----|
               |-----|
```

For every window:

```text
capture
 ↓
VAD
 ↓
normalize
 ↓
anti-spoof inference
 ↓
speaker verification
 ↓
forensics
 ↓
risk update
```

Then smooth the result over time.

Example:

```text
Window 1 → 0.21
Window 2 → 0.28
Window 3 → 0.81
Window 4 → 0.88
Window 5 → 0.92
```

---

# 17. Phase Status

Verified against the repository on 2026-09-08:

```text
0-2       COMPLETE
3.0-3.5   COMPLETE  (audio preprocessing, VAD, feature extraction)
3.6       NOT DONE  - dataset was never downloaded; datasets/raw/ absent,
                      all three manifests 0 bytes. The manifest builder that
                      existed hardcoded split="test" on every row and used the
                      2021 DF column layout against 2019 LA protocols.
3.7       CODE NOW EXISTS, NOT YET RUN - train_model.py was missing entirely.
                      Written 2026-09-08; blocked on the dataset download.
3.8       CODE NOW EXISTS, NOT YET RUN - evaluate_model.py was missing
                      entirely. Written 2026-09-08 with EER, normalised
                      minDCF, DET and per-attack breakdown.
```

Phase 3.6 was reported complete. It was not, and this is the single most
expensive error in the document: everything downstream was blocked on data
that had never been fetched.

---

# 18. PHASE 0 — Problem Definition / System Planning

## Status: COMPLETE

Established:

- SIH problem interpretation
- Threat model
- AI voice cloning focus
- Impersonation focus
- Privacy requirements
- Live inference requirement
- Blockchain's correct role
- Multi-model architecture
- Modular development strategy

---

# 19. PHASE 1 — Backend / Project Foundation

## Status: COMPLETE

Project root:

```text
C:\Users\Karthikeya Varma\voxshield
```

(Earlier revisions referenced `C:\Users\sathv\voxshield`, a different machine.
Read paths in this document as relative to the repository root rather than
copying them literally.)

Virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

The backend is Python-based and API-driven so ML components can be plugged in independently.

Intended API areas:

```text
health checks
audio analysis
speaker enrollment
speaker verification
risk analysis
event creation
audit access
```

Later deployment should support REST and potentially gRPC.

---

# 20. PHASE 2 — Audio Preprocessing / Feature Foundation

## Status: COMPLETE

Target pipeline:

```text
Audio
 ↓
mono
 ↓
16 kHz
 ↓
normalized
 ↓
fixed-duration segments/windows
 ↓
feature/model input
```

Initial training assumptions:

```text
sample rate = 16,000 Hz
clip length ≈ 4 seconds
```

Later real-time inference:

```text
2–3 second rolling windows
```

with overlap.

---

# 21. Test Issue Encountered

At one point:

```powershell
$env:PYTHONPATH="$PWD"; pytest -v
```

failed with:

```text
ModuleNotFoundError: No module named 'app'
```

Affected tests included:

```text
backend/tests/test_audio_features.py
backend/tests/test_health.py
backend/tests/test_inference.py
```

Cause:

```text
voxshield/
    backend/
        app/
```

while tests import:

```python
from app...
```

Correct approach - run from the backend directory, where `pytest.ini` sets
`pythonpath = .`:

```powershell
cd backend
pytest -v
```

or from the project root:

```powershell
$env:PYTHONPATH="$PWD\backend"
pytest -v
```

Two further causes were found on 2026-09-08 and fixed:

* `tests/confest.py` was misspelled, so pytest never loaded it and the
  `anyio_backend` fixture did not exist - the async API tests errored during
  collection rather than running. Renamed to `conftest.py`.
* `transformers` was never installed and was absent from `requirements.txt`, so
  the whole `app.*` import chain died at collection time. Added.

Keep this section so the imports are not incorrectly diagnosed as broken.

---

# 22. PHASE 3 — Dataset / Model Pipeline

Status verified 2026-09-08 - see section 17 for the corrected board.

```text
0-2      COMPLETE
3.0-3.5  COMPLETE
3.6      NOT DONE   (no dataset on disk, manifests empty)
3.7      CODE WRITTEN 2026-09-08, NOT YET RUN
3.8      CODE WRITTEN 2026-09-08, NOT YET RUN
```

---

# 23. PHASE 3.6 — Dataset Preparation

## Status: NOT DONE (verified 2026-09-08)

`datasets/raw/` does not exist and `datasets/manifests/{train,validation,test}.csv`
were all zero bytes. This is the current blocker for the whole project.

**CORRECTION - first benchmark.**

Earlier revisions recommended starting with **ASVspoof 2021 DF**. That cannot
work: DF is an *evaluation-only* partition. It ships no training split at all,
and it is over 100 GB. You cannot fit a model on it.

The correct starting point is **ASVspoof 2019 LA**:

```text
train        25,380 clips    attacks A01-A06
dev          24,844 clips    attacks A01-A06
eval         71,237 clips    attacks A07-A19   <- unseen in training
7.12 GB download, no login required
```

The eval attacks never appear in training, which is what makes the eval EER a
generalisation number rather than a memorisation number - exactly the "unseen
attacks" property section 61 asks for.

ASVspoof 2021 DF and ASVspoof 5 remain the right *later* additions, as
cross-dataset evaluation sets scored with a model trained on 2019 LA.

Verified download (Edinburgh DataShare, DSpace bitstream API):

```text
https://datashare.ed.ac.uk/server/api/core/bitstreams/a9f87c35-f055-4015-80e2-2fdff0d46269/content
```

Raw dataset location:

```text
datasets/raw/ASVspoof2019/LA/
```

---

# 24. Dataset Folder Layout

```text
voxshield/
│
├── datasets/
│   ├── raw/
│   │   └── asvspoof2021/
│   │
│   ├── processed/
│   │   ├── train/
│   │   │   ├── bonafide/
│   │   │   └── spoof/
│   │   ├── validation/
│   │   │   ├── bonafide/
│   │   │   └── spoof/
│   │   └── test/
│   │       ├── bonafide/
│   │       └── spoof/
│   │
│   └── manifests/
│       ├── train.csv
│       ├── validation.csv
│       └── test.csv
│
└── models/
```

---

# 25. Dataset Git Rules

Do not push large datasets/models.

`.gitignore`:

```gitignore
datasets/raw/
datasets/processed/

models/*.pt
models/*.bin
models/*.safetensors
```

---

# 26. Manifest Design

Actual structure as implemented:

```text
path,label,split,attack,speaker
.../ASVspoof2019_LA_train/flac/LA_T_1138215.flac,0,train,,LA_0079
.../ASVspoof2019_LA_train/flac/LA_T_1271820.flac,1,train,A01,LA_0079
```

`attack` enables the per-attack breakdown required by section 40. `speaker`
enables the speaker-leakage check in `scripts/dataset_stats.py`.

There is no `datasets/processed/` stage. Copying 120,000 FLAC files into a
second directory tree costs 7 GB and buys nothing - the manifest addresses the
originals in place.

Convention:

```text
0 = bonafide
1 = spoof
```

Important:

A generic manifest-builder implementation inferred labels from path names, but the proper final implementation should parse official ASVspoof protocol metadata whenever available.

Do not rely on filenames/folders when authoritative labels exist.

---

# 27. Training Dependencies

```powershell
pip install torch torchaudio transformers librosa soundfile scikit-learn pandas tqdm matplotlib
```

GPU check:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Hardware used for development (measured on the machine, 2026-09-08):

```text
AMD Ryzen 7 7840HS   8 cores / 16 threads
NVIDIA GeForce RTX 3050 6GB Laptop GPU   (Ampere, sm_86)
6 GB VRAM
15.3 GB system RAM
Driver 592.82 / CUDA 13.1
```

**CORRECTION.** Earlier revisions claimed a Ryzen 9 with an RTX 5050 and 8 GB
of VRAM. That is not this machine. Every batch size and memory figure must be
sized for **6 GB**, not 8.

6 GB is enough for wav2vec2-base fine-tuning at batch 8 with the convolutional
front end frozen and only the top transformer layers trainable. It is not
enough for full fine-tuning of xls-r-300m without gradient checkpointing. See
`TRAINING.md` for the measured configuration.

On Windows the PyPI torch wheels are CPU-only, so `pip install torch` silently
produces a build that cannot see the GPU - which is what was installed here.
Use the CUDA index:

```powershell
pip install --index-url https://download.pytorch.org/whl/cu130 `
    torch==2.14.0+cu130 torchaudio==2.11.0+cu130
```

---

# 28. Proposed ML Directory

```text
backend/
│
├── app/
│   └── ml/
│       ├── __init__.py
│       ├── config.py
│       ├── dataset.py
│       ├── model.py
│       ├── training.py
│       ├── metrics.py
│       ├── detector.py
│       ├── augmentation.py
│       └── risk.py
│
├── scripts/
│   ├── dataset_stats.py
│   ├── build_manifest.py
│   ├── train_model.py
│   └── evaluate_model.py
│
└── tests/
```

Models:

```text
models/
    voxshield_antispoof.pt
```

---

# 29. PHASE 3.7 — First Real Anti-Spoof Model

## Status: Code written 2026-09-08; not yet run (blocked on the dataset)

`backend/scripts/train_model.py` did not exist before this date. The trainer
now implements staged fine-tuning, AMP, gradient accumulation, class-balanced
sampling, cosine schedule with warmup, EER-based checkpoint selection and early
stopping - and prints per-step throughput with an ETA.

Baseline architecture:

```text
Audio waveform
      │
      ▼
facebook/wav2vec2-base
      │
      ▼
SSL embeddings
      │
      ▼
Temporal pooling
      │
      ├──────────────────────┐
      │                      │
      │                Spectral branch
      │                      │
      └──────────┬───────────┘
                 ▼
              Fusion
                 │
                 ▼
               MLP
                 │
                 ▼
          [bonafide, spoof]
```

---

# 30. Baseline Model Components

The main model contained:

```text
AntiSpoofOutput
SpectralBranch
AntiSpoofModel
AntiSpoofInference
```

SSL encoder:

```text
facebook/wav2vec2-base
```

Spectral branch:

```text
Conv2D
 ↓
BatchNorm
 ↓
ReLU
 ↓
MaxPool
 ↓
Conv2D
 ↓
BatchNorm
 ↓
ReLU
 ↓
Adaptive Average Pool
 ↓
64-dimensional vector
```

Channels:

```text
1
↓
32
↓
64
```

---

# 31. Fusion Classifier

```text
SSL features
      +
64D spectral features
      ↓
concatenate
      ↓
MLP
      ↓
256
      ↓
2 logits
```

Classes:

```text
0 → bonafide
1 → spoof
```

---

# 32. Critical Existing Limitation

**RESOLVED (2026-09-08).**

The situation was worse than described: `SpectralBranch` did not exist in
`app/ml/model.py` at all. There was no spectral code left unwired - there was
no spectral code.

It exists now and is computed inside `AntiSpoofModel.forward` on the same
device as the waveform, so the spectral tensor genuinely flows through the
branch during both training and inference. Building the log-Mel on the GPU also
keeps the DataLoader free of feature extraction, which matters on a laptop CPU.

Verify both paths:

```powershell
python backend\scripts\train_model.py --smoke-test                # spectral on
python backend\scripts\train_model.py --smoke-test --no-spectral  # ablation
```

---

# 33. Proper Spectral Implementation Still Required

Pipeline:

```text
waveform
 ↓
STFT
 ↓
Mel filterbank
 ↓
log
 ↓
Log-Mel spectrogram
 ↓
CNN
```

Potential PyTorch implementation:

```python
torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=512,
    hop_length=160,
    n_mels=80
)
```

Then:

```python
AmplitudeToDB()
```

The resulting tensor must actually be passed into `SpectralBranch` during training and inference.

---

# 34. Training Strategy

Baseline optimizer:

```text
AdamW
```

Initial learning rate:

```text
1e-5
```

Loss:

```text
CrossEntropyLoss
```

Device:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

Initial smoke-test batch:

```text
batch_size = 1
```

Recommended strategy:

### Stage 1

```text
Freeze Wav2Vec2
        ↓
Train classifier
```

### Stage 2

```text
Unfreeze final transformer layers
        ↓
Fine-tune with lower learning rate
```

Example starting values:

```text
Classifier LR:
1e-4

Encoder LR:
1e-5 or lower
```

---

# 35. Training Smoke Test

`backend/scripts/train_model.py` did not exist when this was written. It does
now, and the smoke test is a single flag:

```powershell
python backend\scripts\train_model.py --smoke-test
```

64 train clips, 32 validation, one epoch, no augmentation - it proves manifest
-> dataset -> model -> loss -> backward -> checkpoint -> EER in under a minute.

Before committing to a real run, size it:

```powershell
python backend\scripts\train_model.py --cache-dir datasets\cache --dry-run
```

That times a dozen real steps, discards the warm-up, and prints projected
per-epoch and total wall-clock plus peak VRAM - then stops.

Expected checkpoint:

```text
models/voxshield_antispoof.pt
```

The checkpoint stores the architecture, the full config and the git commit
alongside the weights, so it can always be traced back to the run that made it.

Do not choose epochs simply to maximize training accuracy. The trainer selects
the checkpoint by **validation EER** and stops early when it stops improving.

---

# 36. Data Augmentation Still Needed

For real phone-call detection, clean benchmark data alone is not enough.

Add:

```text
background noise
room impulse response
compression
packet-loss simulation
telephone bandpass
GSM codec
Opus codec
AAC
MP3
reverberation
volume changes
clipping
resampling
microphone coloration
```

Important target:

```text
clean audio
      ↓
telephone / VoIP simulation
      ↓
model
```

---

# 37. PHASE 3.8 — Evaluation

## Status: Code written 2026-09-08; not yet run (blocked on the dataset)

`backend/scripts/evaluate_model.py` did not exist before this date.

Previously provided evaluator computes:

```text
Accuracy
ROC-AUC
Confusion Matrix
Classification Report
```

`backend/scripts/evaluate_model.py` did not exist when this was written. It
does now, and it reports EER, normalised minDCF, ROC-AUC, the per-attack
breakdown and three operating points, and writes ROC/DET plots:

```powershell
python backend\scripts\evaluate_model.py `
    --manifest datasets\manifests\test.csv `
    --model models\voxshield_antispoof.pt `
    --calibrate-on datasets\manifests\validation.csv `
    --save-scores
```

Fit calibration on validation, report on test - never both on the same split.

---

# 38. Required Evaluation Metrics

### EER

Equal Error Rate:

```text
FAR = FRR
```

### ROC-AUC

Measures discrimination across thresholds.

### False Positive Rate

Important because falsely flagging genuine callers damages usability.

### False Negative Rate

Measures spoof attacks incorrectly accepted.

### Precision / Recall / F1

Useful for class performance.

### minDCF

Implemented in `app/ml/metrics.minimum_dcf`.

IMPORTANT: ASVspoof papers report *tandem* DCF (t-DCF), which folds in an ASV
subsystem's scores on the same trials. VoxShield has no enrolled ASV branch
yet, so t-DCF is not computable. What is implemented is NIST-style **normalised
minDCF**, reported with its `prior_spoof` stated. Call it normalised minDCF,
not t-DCF - reviewers will check.

### DET Curve

Useful for security/usability tradeoffs.

---

# 39. Never Report Only Accuracy

Instead of:

```text
Accuracy = 96%
```

report:

```text
Accuracy
Precision
Recall
F1
ROC-AUC
EER
FPR
FNR
minDCF
```

Also evaluate by attack type.

---

# 40. Required Per-Attack Evaluation

Eventually:

```text
Attack                 Detection

TTS                    ...
Voice conversion       ...
Replay                 ...
Neural vocoder         ...
Codec degraded         ...
Noisy spoof            ...
Unknown generator      ...
```

---

# 41. Cross-Dataset Testing

Important scientific test:

```text
Train:
ASVspoof 2021

Test:
different / unseen dataset
```

This checks whether VoxShield learned actual synthetic-speech artifacts rather than dataset-specific patterns.

Without cross-dataset evaluation, do not claim:

> "VoxShield detects any voice clone."

---

# 42. Real-Time Inference

Pipeline:

```text
Microphone / call
       ↓
audio buffer
       ↓
VAD
       ↓
~3 second window
       ↓
16 kHz preprocessing
       ↓
model
       ↓
spoof probability
       ↓
temporal risk engine
```

---

# 43. Temporal Risk Engine

Previously provided:

```python
from collections import deque


class TemporalRiskEngine:
    def __init__(self, window_size: int = 5, smoothing_alpha: float = 0.35):
        self.history = deque(maxlen=window_size)
        self.alpha = smoothing_alpha
        self.smoothed = 0.0

    def update(self, spoof_probability: float):

        self.history.append(spoof_probability)

        self.smoothed = (
            self.alpha * spoof_probability
            + (1 - self.alpha) * self.smoothed
        )

        risk_score = self.smoothed * 100

        if risk_score >= 85:
            level = "HIGH"

        elif risk_score >= 60:
            level = "SUSPICIOUS"

        else:
            level = "LOW"

        return {
            "risk_score": round(risk_score, 2),
            "risk_level": level,
            "raw_probability": round(spoof_probability, 4),
            "smoothed_probability": round(self.smoothed, 4),
            "samples_seen": len(self.history),
        }
```

Important:

```text
60
85
```

are engineering starting thresholds, not scientifically calibrated thresholds.

They should eventually be derived from validation data.

---

# 44. Risk Engine Unit Tests

```python
from app.ml.risk import TemporalRiskEngine


def test_risk_engine():
    engine = TemporalRiskEngine()

    result = engine.update(0.95)

    assert 0 <= result["risk_score"] <= 100


def test_low_risk():
    engine = TemporalRiskEngine()

    result = engine.update(0.10)

    assert result["risk_level"] == "LOW"
```

`app/ml/risk.py` did not exist when this was written. It does now, and it adds
hysteresis so a score hovering at a threshold boundary does not flip the
displayed level every window. Run:

```powershell
cd backend
pytest -v
```

---

# 45. Current Model-Training Checklist

- [ ] Verify `datasets/raw/asvspoof2021`
- [ ] Verify train/validation/test separation
- [ ] Verify official protocol labels
- [ ] Verify `train.csv`
- [ ] Verify `validation.csv`
- [ ] Verify `test.csv`
- [ ] Verify 16-kHz audio
- [ ] Add automatic resampling rather than rejecting non-16-kHz samples
- [ ] Verify Wav2Vec2 processor/input normalization
- [ ] Freeze SSL encoder for first baseline
- [ ] Train classifier
- [ ] Save checkpoint
- [ ] Evaluate validation set
- [ ] Add EER
- [ ] Add ROC curve
- [ ] Add DET curve
- [ ] Add minDCF
- [ ] Calibrate threshold
- [ ] Add actual Log-Mel spectral branch
- [ ] Train fusion model
- [ ] Add call/codec augmentation
- [ ] Test unseen attacks
- [ ] Test cross-dataset
- [ ] Measure inference latency
- [ ] Measure GPU/CPU memory
- [ ] Integrate rolling real-time inference

---

# 46. Phase 4 — Speaker Verification

## Status: NOT IMPLEMENTED

This is the next major capability after anti-spoof training is stable.

Architecture:

```text
Claimed identity
       │
       ▼
Stored enrollment embedding
       │
       │
Live audio
    ↓
ECAPA-TDNN
    ↓
live embedding
       │
       ▼
cosine similarity
       │
       ▼
identity score
```

---

# 47. Enrollment System

Conceptual API:

```text
POST /speaker/enroll
```

Input:

```text
identity
3–5 voice samples
```

Processing:

```text
VAD
quality check
embedding extraction
normalization
centroid
encryption
storage
```

Output:

```text
speaker profile created
```

---

# 48. Verification API

Concept:

```text
POST /speaker/verify
```

Input:

```text
claimed identity
live voice sample
```

Output:

```json
{
  "similarity": 0.71,
  "match_probability": 0.84,
  "verification": "MATCH"
}
```

Actual thresholds must be calibrated.

Do not assume a universal rule such as:

```text
cosine > 0.7 = same person
```

---

# 49. Phase 4 Attack Scenarios

Test:

```text
genuine Alice → claims Alice
genuine Bob → claims Alice
AI Alice clone → claims Alice
recording of Alice → claims Alice
AI Bob clone → claims Alice
noisy Alice → claims Alice
telephone Alice → claims Alice
```

---

# 50. Phase 5 — Multi-Branch Audio Forensics

## Status: NOT IMPLEMENTED

Proposed architecture:

```text
                    Audio
                      │
       ┌──────────────┼──────────────┐
       ▼              ▼              ▼
     SSL           Log-Mel         LFCC/CQCC
   Encoder            CNN            CNN
       │              │              │
       └──────────────┼──────────────┘
                      │
                    Fusion
                      │
            ┌─────────┴─────────┐
            ▼                   ▼
         Prosody            Phase/artifact
          branch                branch
            │                   │
            └─────────┬─────────┘
                      ▼
               Attention Fusion
                      │
                      ▼
               Spoof probability
```

---

# 51. Prosody Features

```text
F0 mean
F0 variance
F0 range
energy mean
energy variance
speaking rate
pause count
average pause
voiced/unvoiced ratio
jitter
shimmer
spectral centroid
spectral rolloff
zero-crossing rate
```

---

# 52. Phase-Aware Detection

Potential pipeline:

```text
complex STFT
   ↓
magnitude + phase
   ↓
phase consistency features
```

This directly addresses the problem statement's phase-analysis requirement.

---

# 53. AASIST / RawNet Experiments

After establishing the Wav2Vec2 baseline:

```text
Baseline A:
Wav2Vec2

Baseline B:
Spectral CNN

Baseline C:
AASIST

Baseline D:
Fusion
```

Compare scientifically rather than replacing the baseline blindly.

---

# 54. Phase 6 — Contextual Fraud Intelligence

## Status: NOT IMPLEMENTED

Potential inputs:

```text
caller_known
caller_country
trusted_contact
device_known
transaction_amount
beneficiary_age
beneficiary_known
login_risk
time_of_day
recent_password_reset
failed_authentication_count
```

Architecture:

```text
Voice threat
     +
Identity mismatch
     +
Prosody
     +
Call metadata
     +
Transaction context
     +
NLP
       ↓
Contextual Risk Engine
       ↓
0–100
```

---

# 55. Phase 6 — Policy Engine

Example initial policy:

```text
Risk 0–39
ALLOW

Risk 40–59
WARN

Risk 60–79
SECONDARY VERIFICATION

Risk 80–100
HIGH-RISK WORKFLOW
```

Thresholds must eventually be calibrated.

---

# 56. Secondary Verification

Possible actions:

```text
registered-device confirmation
MFA
callback to trusted number
security question workflow
manager approval
transaction hold
manual verification
```

The project should emphasize:

> VoxShield detects and prevents impersonation attacks rather than merely displaying an AI probability.

---

# 57. Phase 7 — Blockchain Audit Layer

## Status: NOT IMPLEMENTED

Suggested:

```text
Detection event
    ↓
canonical JSON
    ↓
SHA-256
    ↓
signed audit transaction
    ↓
permissioned blockchain
```

Possible technologies:

```text
Hyperledger Fabric
private Ethereum network
Polygon/EVM prototype
```

For SIH, a permissioned audit architecture is conceptually strong.

---

# 58. Database + Blockchain

Use both:

```text
PostgreSQL
   +
Blockchain
```

Database:

```text
full security event
```

Blockchain:

```text
event hash
timestamp
model version
policy version
decision hash
```

---

# 59. Phase 8 — Dashboard

## Status: NOT IMPLEMENTED

Suggested live screen:

```text
LIVE CALL
────────────────────────

Caller:
+91 XXXXX XXXXX

Voice Authenticity
████████░░ 82%

Identity Match
███░░░░░░░ 31%

Context Risk
███████░░░ 71%

Overall Risk
█████████░ 87%

STATUS:
HIGH RISK

Reason:
• AI voice characteristics detected
• Voice inconsistent with enrolled speaker
• Unrecognized caller
• High-value financial request
```

Useful visualizations:

```text
live waveform
spectrogram
risk gauge
timeline
speaker similarity
deepfake probability
reason codes
security events
blockchain proof
model version
```

---

# 60. Explainability

Do not show only:

```text
87% fraud
```

Show reasons:

```text
AI-generated speech artifacts        +28
Speaker identity mismatch            +24
Abnormal prosody                     +11
Unknown caller                       +8
Financial urgency detected           +12
```

---

# 61. Phase 9 — Adversarial Testing

## Status: NOT IMPLEMENTED

Test multiple generation sources.

Separate:

```text
Known attacks
Unknown attacks
```

Do not train and demonstrate using exactly the same spoof generators.

Unknown attacks are especially important.

---

# 62. Real-World Test Matrix

| Condition | Genuine | Fake |
|---|---:|---:|
| Clean microphone | ✓ | ✓ |
| Background noise | ✓ | ✓ |
| Phone speaker | ✓ | ✓ |
| VoIP codec | ✓ | ✓ |
| Low bitrate | ✓ | ✓ |
| Replay | ✓ | ✓ |
| Hindi | ✓ | ✓ |
| English | ✓ | ✓ |
| Telugu | ✓ | ✓ |
| Accent variation | ✓ | ✓ |
| Multiple speakers | ✓ | ✓ |

---

# 63. Multilingual Roadmap

Eventually evaluate:

```text
English
Hindi
Telugu
Tamil
Kannada
Malayalam
Marathi
Bengali
etc.
```

Do not claim support for every Indian language until validated.

Safer statement before full evaluation:

> The architecture is language-independent at the acoustic layer and is being evaluated for multilingual Indian speech.

---

# 64. Phase 10 — Production Optimization

## Status: NOT IMPLEMENTED

Potential path:

```text
PyTorch
 ↓
ONNX
 ↓
ONNX Runtime
 ↓
quantization
 ↓
INT8 / FP16
```

Or:

```text
TensorRT
```

Measure:

```text
model size
RAM
VRAM
CPU latency
GPU latency
real-time factor
```

---

# 65. Latency Target

Pipeline:

```text
audio window
     ↓
feature extraction
     ↓
model inference
     ↓
fusion
     ↓
risk
```

Prediction latency should eventually be substantially shorter than the audio window size so the system can keep up continuously.

---

# 66. Model Evolution Roadmap

```text
V0
Wav2Vec2 binary classifier

        ↓

V1
Wav2Vec2 + real Log-Mel CNN

        ↓

V2
SSL + spectral + prosody

        ↓

V3
Anti-spoof + ECAPA speaker verification

        ↓

V4
AASIST/RawNet ensemble

        ↓

V5
Audio + identity + NLP + context

        ↓

V6
Real-time security engine
```

---

# 67. Experiment Tracking

Every experiment should record:

```text
experiment ID
date
dataset
train split
validation split
model
pretrained checkpoint
learning rate
batch size
epochs
augmentations
seed
EER
ROC-AUC
F1
FPR
FNR
checkpoint
Git commit
```

Recommended:

```text
experiments/
    exp001/
        config.json
        metrics.json
        notes.md

    exp002/
        config.json
        metrics.json
        notes.md
```

---

# 68. Model Versioning

Do not simply keep:

```text
model.pt
```

Use:

```text
voxshield_v0_wav2vec.pt
voxshield_v1_spectral_ssl.pt
voxshield_v2_multibranch.pt
```

Track:

```text
model_id
dataset_version
training_commit
threshold_version
```

This also supports blockchain auditing.

---

# 69. Quality Gate Before Calling ML "Working"

```text
[ ] model trains
[ ] loss decreases
[ ] validation metrics generated
[ ] checkpoint loads
[ ] inference works on unseen WAV
[ ] genuine sample tested
[ ] synthetic sample tested
[ ] CPU inference tested
[ ] GPU inference tested
[ ] threshold calibrated
[ ] confusion matrix generated
[ ] EER measured
[ ] no train/test leakage
```

---

# 70. Security Requirements

Before production:

```text
API authentication
rate limiting
encrypted storage
embedding encryption
secret management
TLS
audit integrity
RBAC
input validation
audio-file validation
malicious file protection
DoS protection
model poisoning considerations
adversarial audio testing
```

---

# 71. Data Leakage Prevention

Ensure:

```text
speaker/train samples
```

do not leak into validation/test.

Ideally evaluate:

```text
speaker-disjoint splits
attack-disjoint tests
generator-disjoint tests
```

Otherwise metrics can be misleading.

---

# 72. Real-Time VAD

Use:

```text
microphone
 ↓
VAD
 ↓
speech only
 ↓
model
```

Potential choices:

```text
WebRTC VAD
Silero VAD
```

Select based on latency and compatibility.

---

# 73. Multiple Speakers / Diarization

Long-term:

```text
Call
 ↓
Speaker diarization
 ↓
Speaker A
Speaker B
 ↓
analyze independently
```

Not necessary for the earliest prototype.

---

# 74. Critical Model-Training Warning

Before claiming:

> "We built a hybrid Wav2Vec2 + spectral forensic model"

verify that the spectral tensor actually passes through:

```text
SpectralBranch
```

during both training and inference.

---

# 75. Second Major Warning

Do not claim:

```text
99% detection accuracy
```

from one validation split.

Performance can degrade under:

```text
new generators
different microphones
new codecs
other languages
background noise
telephone transmission
re-recording
```

The strongest SIH presentation should demonstrate these challenges.

---

# 76. Third Major Warning

Do not use a universal hardcoded speaker-verification threshold.

Use:

```text
similarity scores
 ↓
ROC / DET
 ↓
threshold selection
```

Potential operating modes:

```text
high-security
balanced
low-friction
```

---

# 77. Fourth Major Warning

Do not make blockchain the centerpiece of the AI architecture.

Detection happens through:

```text
signal processing
ML
speaker verification
context
```

Blockchain exists for:

```text
auditability
integrity
non-repudiation
evidence provenance
```

---

# 78. Proposed Complete Backend Architecture

```text
backend/
│
├── app/
│   ├── main.py
│   │
│   ├── api/
│   │   ├── health.py
│   │   ├── analyze.py
│   │   ├── speakers.py
│   │   ├── sessions.py
│   │   └── audit.py
│   │
│   ├── audio/
│   │   ├── loader.py
│   │   ├── preprocessing.py
│   │   ├── vad.py
│   │   ├── streaming.py
│   │   └── features.py
│   │
│   ├── ml/
│   │   ├── dataset.py
│   │   ├── model.py
│   │   ├── detector.py
│   │   ├── speaker_verification.py
│   │   ├── spectral.py
│   │   ├── prosody.py
│   │   ├── fusion.py
│   │   ├── risk.py
│   │   └── augmentation.py
│   │
│   ├── context/
│   │   ├── metadata.py
│   │   ├── nlp.py
│   │   └── fraud.py
│   │
│   ├── policy/
│   │   ├── engine.py
│   │   └── actions.py
│   │
│   ├── blockchain/
│   │   ├── audit.py
│   │   ├── hashing.py
│   │   └── client.py
│   │
│   └── database/
│       ├── models.py
│       └── session.py
│
├── scripts/
│   ├── build_manifest.py
│   ├── dataset_stats.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── calibrate_threshold.py
│   └── benchmark.py
│
└── tests/
```

---

# 79. Frontend Architecture

```text
frontend/
│
├── dashboard
├── live-call
├── incidents
├── speakers
├── analytics
├── audit
└── settings
```

Main demo should answer:

```text
WHO IS CALLING?
        ↓
IS VOICE SYNTHETIC?
        ↓
DOES VOICE MATCH?
        ↓
IS CONTEXT SUSPICIOUS?
        ↓
WHAT DOES VOXSHIELD DO?
```

---

# 80. Recommended Final Demo

## Demo A — Genuine

```text
Real person
Known identity
Normal conversation

Deepfake       LOW
Identity       MATCH
Context        LOW

Risk           12
Decision       ALLOW
```

## Demo B — AI Cloned CEO

```text
Cloned CEO voice
Requests urgent transfer

Deepfake       HIGH
Identity       suspicious
NLP            HIGH
Context        HIGH

Risk           91
Decision       MFA / callback
```

## Demo C — Real Attacker

```text
Real human
Claims to be CEO

Deepfake       LOW
Identity       MISMATCH
Context        suspicious

Risk           82
Decision       verification
```

Demo C is especially valuable because it proves VoxShield is more sophisticated than a generic AI-audio detector.

---

# 81. Recommended Final Innovation Statement

> **VoxShield is a real-time, multi-layer voice identity firewall. Instead of relying on a single deepfake classifier, it combines synthetic-speech forensics, speaker verification, behavioral voice consistency, contextual fraud signals and policy-driven secondary verification. Security decisions are recorded using tamper-evident cryptographic/blockchain auditing without putting users' raw voice data on-chain.**

---

# 82. Complete Status Board

| Component | Status |
|---|---|
| Problem definition | ✅ Complete |
| Threat model | ✅ Complete |
| Architecture design | ✅ Complete |
| Backend foundation | ✅ Complete |
| Basic audio pipeline | ✅ Complete |
| Audio preprocessing | ✅ Complete |
| Feature extraction foundation | ✅ Complete |
| ASVspoof dataset approach | ✅ Designed |
| Dataset downloaded | ❌ `datasets/raw/` does not exist |
| Dataset preparation | ❌ was reported complete; it was not |
| Training manifests | ❌ all three CSVs were 0 bytes |
| Manifest builder | ✅ rewritten, auto-detects protocol columns |
| Dataset verification + leakage check | ✅ `scripts/dataset_stats.py` |
| Waveform cache (training speed) | ✅ `scripts/cache_dataset.py` |
| Wav2Vec2 baseline architecture | ✅ implemented |
| Anti-spoof training script | ✅ written 2026-09-08, not yet run |
| Saved trained production checkpoint | ❌ none exists |
| EER | ✅ implemented + unit-tested |
| normalised minDCF | ✅ implemented (not t-DCF — see §38) |
| ROC-AUC | ✅ implemented |
| DET curve | ✅ implemented (normal-deviate scale) |
| Per-attack breakdown | ✅ implemented |
| Spectral CNN | ✅ written and wired into forward() |
| Log-Mel feature path | ✅ computed on GPU inside the model |
| Data augmentation | ✅ implemented |
| Codec simulation | ✅ mu-law, telephone band, 8 kHz round trip |
| Real-time inference | ✅ streaming engine fixed and unit-tested |
| Temporal risk smoothing | ✅ `app/ml/risk.py` with hysteresis |
| Probability calibration | ✅ temperature scaling (was a silent no-op) |
| Threshold calibration | ❌ needs a validation DET curve from a real run |
| Speaker verification | ❌ Next major phase |
| Speaker enrollment | ❌ |
| ECAPA-TDNN | ❌ |
| Cross-session consistency | ❌ |
| Prosody analysis | ❌ |
| Phase artifact analysis | ❌ |
| AASIST experiment | ❌ |
| RawNet experiment | ❌ |
| Multi-model fusion | ❌ |
| ASR | ❌ |
| NLP fraud detection | ❌ |
| Contextual risk engine | ❌ |
| Policy engine | ❌ |
| MFA/callback workflows | ❌ |
| Blockchain audit | ❌ |
| Dashboard | ❌ |
| Multilingual testing | ❌ |
| Cross-dataset testing | ❌ |
| Adversarial testing | ❌ |
| Edge optimization | ❌ |
| ONNX/TensorRT | ❌ |
| Final deployment | ❌ |

---

# 83. What the New Developer Should Do First

Do **not** start blockchain or frontend.

As of 2026-09-08 steps 1 and 2 are done and the code for 3 and 4 is written.
The only thing standing between the project and a trained model is the 7.12 GB
dataset download. **Start it now** - see `TRAINING.md` step 2.

Immediate order:

```text
STEP 1  [DONE]  Repository + tests audited and fixed
        ↓
STEP 2  [BLOCKED ON YOU]  Download ASVspoof 2019 LA, build manifests
        ↓
STEP 3  [CODE READY]
Get Wav2Vec2 baseline training end-to-end
        ↓
STEP 4  [CODE READY]
Evaluate EER/AUC/F1
        ↓
STEP 5
Add real Log-Mel spectral branch
        ↓
STEP 6
Add real-world augmentation
        ↓
STEP 7
Build ECAPA speaker verification
        ↓
STEP 8
Combine spoof + speaker scores
        ↓
STEP 9
Add prosody / forensic branches
        ↓
STEP 10
Build real-time pipeline
        ↓
STEP 11
Add context + policy
        ↓
STEP 12
Blockchain audit
        ↓
STEP 13
Dashboard
        ↓
STEP 14
Adversarial / multilingual evaluation
        ↓
STEP 15
Deployment optimization
```

---

# 84. What NOT to Work on Yet

Do not initially prioritize:

```text
fancy dashboard animations
mobile app
massive blockchain network
authentication UI
cloud deployment
beautiful graphs
LLM chatbot features
```

until this works:

```text
audio
 ↓
spoof detector
 ↓
speaker verifier
 ↓
fusion
 ↓
correct risk output
```

---

# 85. Next Milestone — VoxShield ML Core v1

The next major milestone should accept:

```text
test.wav
claimed_identity
```

and output something like:

```json
{
  "spoof_probability": 0.87,
  "speaker_similarity": 0.42,
  "speaker_match": false,
  "audio_risk": 81.2,
  "risk_level": "HIGH"
}
```

Once this works reliably, the project has moved from a collection of components into an actual voice-security system.

---

# 86. Final Handoff Summary

## Definitely completed

```text
Phase 0
Phase 1
Phase 2
```

These cover foundational architecture, backend/audio infrastructure and preprocessing.

## Previously reported complete

```text
Phase 3.6
Dataset preparation
```

Verify the actual dataset/manifests exist locally because datasets are intentionally not committed to Git.

## Designed/provided but execution should be verified

```text
Phase 3.7
Wav2Vec2 anti-spoof training

Phase 3.8
Evaluation + temporal risk / inference
```

## Immediate unfinished core

```text
Finish proper anti-spoof training
Proper spectral branch
EER/minDCF evaluation
Augmentation
Threshold calibration
Speaker verification
Speaker enrollment
Spoof + identity fusion
```

## Major later work

```text
Prosody
Phase/spectral forensics
AASIST / RawNet
Context fraud
ASR/NLP
Policy engine
MFA/callback workflows
Blockchain auditing
Dashboard
Multilingual testing
Cross-dataset testing
Deployment optimization
Adversarial evaluation
```

---

# 87. Core Architectural Principle

```text
VoxShield ≠ Deepfake Detector

VoxShield =
Deepfake Detection
        +
Speaker Identity Verification
        +
Audio Forensics
        +
Behavioral Consistency
        +
Fraud Context
        +
Risk & Policy Engine
        +
Tamper-Evident Audit
```

This is the intended end-state and the key idea the next developer should preserve.
