# VoxShield

AI-powered real-time voice integrity and impersonation prevention.

**SIH 2026 — Problem Statement 26104**, AICTE / Cyber Security Cell.
Theme: Blockchain & Cybersecurity.

## What this is (and is not)

VoxShield is not a deepfake detector with a dashboard. The question it answers
is not *"is this audio AI-generated?"* but:

> Is the person speaking genuine, does the voice belong to the claimed person,
> are there acoustic or behavioural signs of manipulation, and is the
> surrounding call suspicious enough to require intervention?

That distinction is the project. A real human impersonating a CEO passes any
anti-spoof model — it takes speaker verification to catch. A cloned voice may
match the target closely enough to pass verification — it takes anti-spoof to
catch. Neither alone is sufficient, which is why the architecture fuses:

```
synthetic-speech forensics
  + speaker identity verification
  + behavioural / prosodic consistency
  + contextual fraud signals
  + policy-driven secondary verification
  + tamper-evident audit
```

## Status

| layer | state |
|---|---|
| Backend API + websocket audio streaming | working |
| Audio preprocessing, VAD, feature extraction | working |
| Anti-spoof model (wav2vec2 + log-Mel CNN fusion) | **trained — 2.95% EER** on ASVspoof 2019 LA eval |
| Training + evaluation pipeline (EER / minDCF / DET) | working, [results/](results/) |
| Score calibration (Platt, prior-corrected) | **working** — ECE 0.126 -> 0.015 held out |
| Temporal risk engine | working, thresholds **derived from the DET curve** |
| Speaker verification (ECAPA-TDNN) | **working — 0.373% EER** on 67 unseen speakers |
| Risk fusion + policy engine | **working** — explainable 0-100 score, weights uncalibrated |
| Contextual fraud signals | **working**, weights uncalibrated |
| `/speakers/*`, `/analyze`, `/status`, live websocket | **working** |
| Dashboard | **working** — analyse, live call, enrolment, system status |
| Prosody / phase forensics | not started |
| NLP transcript analysis | not started |
| Blockchain audit layer | not started |

**Current result: 2.95 % EER, ROC-AUC 0.986** on 71,237 utterances across 13
attack types unseen in training — and 3.76 % miss at a 1 % false-alarm budget.
Full numbers, the per-attack breakdown and the training-data ablation are in
**[results/](results/README.md)**.

Speaker verification now closes the gap that anti-spoof cannot: a real human
impersonating someone passes any deepfake detector, and only an identity check
stops them. `POST /api/v1/analyze` takes audio plus a claimed identity and
returns a fused risk score with reason codes.

## Run it

```powershell
.\start.ps1
```

Checks Python, packages, GPU, checkpoint, calibration and ports before starting
anything, then opens the backend on `:8000` and the UI on `:3000`. Use
`.\start.ps1 -Check` to run the checks without starting anything.

Verify the whole system end to end - 59 checks against a running API with real
audio and the real checkpoint:

```powershell
cd backend
..\.venv\Scripts\python.exe scripts\make_test_audio.py   # once
..\.venv\Scripts\python.exe scriptserify_system.py
```

**[TESTING.md](TESTING.md)** is the case-by-case guide: every test, what to do,
and the value the correct answer has.

## On the numbers

The detector's raw output is not a probability. On the evaluation set the
equal-error point falls at a raw score of 0.044, so judging synthetic-vs-real
at 0.5 misses 12% of attacks. `scripts/calibrate_detector.py` fits a Platt
calibration on half the evaluation set and reports on the other half; the
serving path applies it in exactly one place, and the decision thresholds are
read off the resulting DET curve rather than chosen.

At the shipped operating point: **94.95% detection at 0.79% false alarm**,
measured on 35,619 held-out utterances across 13 unseen attack types.

The detector is a *speech* model, so it is not asked about anything else. Four
seconds of silence scores 0.999 synthetic and white noise 0.997 if you let it;
`app/audio/speech_check.py` refuses those windows and reports the branch as
unavailable rather than guessing. A call where nothing could be assessed
returns `INSUFFICIENT_EVIDENCE`, never `ALLOW`.

Fusion weights, policy bands and context weights remain **uncalibrated
placeholders** and say so in every response that uses them.

Remaining: prosody and phase forensics, NLP transcript analysis, and the
blockchain audit layer.

## Requirements

- Python 3.11+
- Node.js 20+
- Docker Desktop (for PostgreSQL)
- An NVIDIA GPU for training (CPU works but is ~100x slower)

## Setup

### Backend

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1

# CUDA build first - the PyPI wheels are CPU-only on Windows
pip install --index-url https://download.pytorch.org/whl/cu130 `
    torch==2.14.0+cu130 torchaudio==2.11.0+cu130

pip install -r backend\requirements.txt

Copy-Item backend\.env.example backend\.env
docker compose up -d

cd backend
uvicorn app.main:app --reload
```

API at <http://localhost:8000>, docs at <http://localhost:8000/docs>.

### Frontend

```powershell
cd frontend
npm install
Copy-Item .env.local.example .env.local
npm run dev
```

### Tests

Run from `backend/` — `pytest.ini` sets `pythonpath = .`, so `from app...`
resolves. Running from the repo root gives `ModuleNotFoundError: No module
named 'app'`, which is a working-directory problem, not a broken import.

```powershell
cd backend
pytest -v            # unit suite
pytest -v -m slow    # tests that download the pretrained encoder
```

## Training a model

Fastest path is Kaggle — **`notebooks/VoxShield_Kaggle_Training.ipynb`**.
ASVspoof 2019 LA is already a public Kaggle dataset, so there is no
download; attach it and train on a P100. A Colab notebook and the local
recipe are in **[TRAINING.md](TRAINING.md)**, along with what to report.

Short version:

```powershell
python backend\scripts\build_manifest.py --asvspoof2019-la datasets\raw\ASVspoof2019\LA
python backend\scripts\dataset_stats.py --all --check-audio
python backend\scripts\cache_dataset.py --all
python backend\scripts\train_model.py --smoke-test
python backend\scripts\train_model.py --cache-dir datasets\cache --dry-run
python backend\scripts\train_model.py --cache-dir datasets\cache --experiment-id exp001
python backend\scripts\evaluate_model.py --manifest datasets\manifests\test.csv --model models\voxshield_antispoof.pt
```

## Layout

```
backend/
  app/
    api/          REST routes
    audio/        VAD, feature extraction, per-call pipeline
    ml/           dataset, model, training, metrics, risk, calibration
    websocket/    live audio streaming
  scripts/        build_manifest, dataset_stats, cache_dataset,
                  train_model, evaluate_model
  tests/
datasets/
  raw/            downloaded corpora        (gitignored)
  cache/          memmapped waveform cache  (gitignored)
  manifests/      train/validation/test CSVs
experiments/      per-run config, history, per-step log  (gitignored)
models/           checkpoints                            (gitignored)
frontend/         Next.js dashboard
```

## Conventions

- Labels: **0 = bonafide, 1 = spoof**. Scores are P(spoof).
- Audio: mono, 16 kHz, 4-second training windows, 3-second live windows.
- Never report accuracy alone — the corpora are ~90% spoof. Report EER,
  normalised minDCF, ROC-AUC and the per-attack breakdown.
- Raw voice never goes on-chain. The audit layer stores hashes of decision
  records, not audio.
