# VoxShield — anti-spoof training runbook

Everything needed to go from an empty `datasets/` directory to an evaluated
checkpoint, on this machine, in about a day of mostly-unattended time.

**Your hardware** (measured, not assumed — the handoff doc is wrong about this):

| | |
|---|---|
| CPU | AMD Ryzen 7 7840HS, 8 cores / 16 threads |
| GPU | NVIDIA GeForce RTX 3050 **6 GB** Laptop GPU (Ampere, sm_86) |
| RAM | 15.3 GB |
| Disk | C: 195 GB free |
| Driver | 592.82, CUDA 13.1 |

The handoff doc claims a "Ryzen 9 / RTX 5050 / 8 GB VRAM". It is a 3050 with
6 GB. Every batch size and memory figure below is sized for 6 GB.

---

## The short version

```powershell
# 1. download LA.zip  (7.12 GB)  — YOU, unattended, start it first
# 2. extract it                  — YOU, ~10 min
python backend\scripts\build_manifest.py --asvspoof2019-la datasets\raw\ASVspoof2019\LA
python backend\scripts\dataset_stats.py --all --check-audio
python backend\scripts\cache_dataset.py --all
python backend\scripts\train_model.py --smoke-test
python backend\scripts\train_model.py --cache-dir datasets\cache --dry-run
python backend\scripts\train_model.py --cache-dir datasets\cache --experiment-id exp001
python backend\scripts\evaluate_model.py --manifest datasets\manifests\test.csv --model models\voxshield_antispoof.pt --calibrate-on datasets\manifests\validation.csv
```

Every step prints its own progress, throughput and ETA. Nothing runs blind.

**The environment is already fixed.** CUDA PyTorch, transformers and matplotlib
were installed and verified during the handoff:

```
torch      2.14.0+cu130
cuda       True | NVIDIA GeForce RTX 3050 6GB Laptop GPU
capability (8, 6)   bf16 True   6.0 GB VRAM
```

The full 90-test suite passes and both `train_model.py` and
`evaluate_model.py` were run end to end on GPU against a synthetic corpus. The
only thing missing is the audio.

**Measured on your GPU** (batch 8 x 2 accumulation, frozen encoder, bf16):

```
222 ms/step   72 samples/s   peak VRAM 1.96 GB of 6.0 GB
```

Extrapolated to the real 25,380-clip train split, that is roughly **6 minutes
per frozen-encoder epoch** and **10-13 minutes per fine-tuning epoch**, so the
default four-epoch schedule lands near **45-60 minutes** including validation.
Confirm with `--dry-run` once the data is down — the probe measures your actual
setup rather than trusting this paragraph.

Peak VRAM of 1.96 GB out of 6 GB means there is headroom: if the probe shows
the fine-tuning stage still under about 4.5 GB, `--batch-size 16
--gradient-accumulation 1` will be meaningfully faster.

---

## Step 1 — PyTorch (already done; here for the record)

The venv originally had **`torch 2.14.0+cpu`**. On Windows the PyPI torch
wheels are CPU-only, so `pip install torch` silently gives a build that cannot
see the GPU — training on CPU here is roughly a hundred times slower.

If you ever rebuild the venv, this is the command. `cu130` matches your CUDA
13.1 driver, and it is the **only** index that carries both halves of the pair:
cu126 tops out at torch 2.9.1, cu128 at torch 2.11.0.

```powershell
.venv\Scripts\python.exe -m pip install --upgrade `
    --index-url https://download.pytorch.org/whl/cu130 `
    torch==2.14.0+cu130 torchaudio==2.11.0+cu130

.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```

Verify:

```powershell
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expected:  2.14.0+cu130 True NVIDIA GeForce RTX 3050 6GB Laptop GPU
```

> **Do not run two `pip install` commands at once.** They compete for bandwidth
> and one dies with `ReadTimeoutError` while the shell still reports success.
> That happened twice during this handoff. Run them one after another, and use
> `--timeout 180 --retries 20` on a slow connection.

`transformers` is pinned `>=4.44,<5` in `requirements.txt` on purpose:
`model.py` reads the encoder's private `_get_feature_vector_attention_mask`
behind a `hasattr` guard, so a v5 rename would degrade the masked-pooling path
*silently* to no mask rather than raising.

### A note on `torchaudio.load`

torchaudio 2.9+ routes `torchaudio.load` through **TorchCodec**, which is a
separate install and is not present here — the call raises
`ImportError: TorchCodec is required for load_with_torchcodec`. The last commit
on this repo ("Fix FLAC loading") used `torchaudio.load`, so dataset loading
could never have worked. All audio I/O now goes through `soundfile`, which
reads FLAC natively, is faster for this workload, and does not break when
torchaudio reshuffles its backends. No action needed — just don't reintroduce
`torchaudio.load`.

---

## Train on Colab instead (recommended)

Two measurements settle this:

- **Your connection to Edinburgh DataShare runs at ~47 KB/s.** LA.zip would
  take **43 hours** locally. Over Colab's link it is typically minutes. This is
  measured, not estimated — the local download was started, timed, and stopped.
- **A free-tier T4 has 16 GB of VRAM against this laptop's 6 GB**, and is
  roughly 2–3x faster for this workload. It is Turing, so no bf16 — the trainer
  detects that and falls back to fp16 + GradScaler on its own.

Use **`notebooks/VoxShield_Colab_Training.ipynb`**. It handles the parts that
go wrong on Colab:

- **Never reads FLAC off Drive.** ASVspoof LA is ~121k small files and Drive's
  FUSE layer charges a round trip each; the GPU would starve. Archives are
  copied to local disk and extracted there.
- **`--resume` everywhere.** Free sessions cap near 12 h and drop when idle.
  Weights, AdamW moments, LR schedule and epoch counter are checkpointed every
  epoch and mirrored to Drive, so a dead session costs one epoch.
- **DF evaluated part-by-part.** The full DF set is ~40 GB extracted and will
  not co-exist with the training cache. Each part is extracted, scored,
  deleted; `merge_scores.py` pools the per-utterance scores at the end —
  because EER is a property of the whole score distribution, and averaging four
  per-part EERs gives a different, wrong number.

The rest of this document still applies — it is the same pipeline, and the
local path below remains valid for inference, the demo, and development.

## Step 2 — download the dataset (this is the long pole; start it now)

### Which dataset, and why not the one the handoff recommends

The handoff says to start with **ASVspoof 2021 DF**. You cannot: DF is an
*evaluation-only* release. Zenodo describes it as "the evaluation set for the
DF task", and the official ASVspoof 2021 site is explicit about training data:

> "Participants will be encouraged to prepare their own training and
> development data by using **ASVspoof 2019 training and development subsets**
> and data augmentation techniques."

No training data was released for 2021 at all. Every published DF result —
wav2vec2, AASIST, all of them — is a model trained on ASVspoof 2019 LA and
*scored* on 2021 DF. LA is not an alternative to DF; it is the prerequisite.

DF ships as four ~8.6 GB parts, 34.5 GB total. They are slices of one archive,
not four options — you need all four for the complete 611,829-utterance set,
though any single part is a usable ~25% subset if you say so when reporting.
Labels come separately: `DF-keys-full.tar.gz` (26 MB), which holds
`trial_metadata.txt`.

The standard, correct starting point is **ASVspoof 2019 LA**:

| split | clips | attacks | purpose |
|---|---:|---|---|
| train | 25,380 | A01–A06 | fitting |
| dev | 24,844 | A01–A06 | validation, threshold + calibration |
| eval | 71,237 | **A07–A19** | held-out test |

The eval attacks never appear in training. That is what makes the eval EER a
generalisation number rather than a memorisation number — and it is exactly the
"unseen attacks" property the handoff asks for in §61.

### Download

**7.12 GB, no login required.** Verified direct link (I fetched its headers:
`Content-Type: application/zip`, `Content-Length: 7640952520`,
`Accept-Ranges: bytes`):

```powershell
mkdir datasets\raw -Force
curl.exe -L -C - -o datasets\raw\LA.zip `
  "https://datashare.ed.ac.uk/server/api/core/bitstreams/a9f87c35-f055-4015-80e2-2fdff0d46269/content"
```

`-C -` resumes a partial download, so a dropped connection means re-running the
identical command, not starting over. Your connection measured ~2.4 MB/s during
this session, so budget roughly **50 minutes**.

Check the size before extracting — a truncated zip fails deep into extraction,
after you have already waited:

```powershell
(Get-Item datasets\raw\LA.zip).Length    # must be exactly 7640952520
```

Skip `PA.zip` (16.45 GB): it is the physical-access/replay partition, which is
a different threat model from voice cloning.

Landing page, if you would rather click:
<https://datashare.ed.ac.uk/handle/10283/3336>

### Two Windows settings worth changing first

```powershell
# Administrator PowerShell. Defender scans each of 121k FLAC opens otherwise.
Add-MpPreference -ExclusionPath "C:\Users\Karthikeya Varma\voxshield\datasets"
```

```powershell
# Reduces CUDA allocator fragmentation, which matters on a 6 GB card.
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
```

### Extract

```powershell
mkdir datasets\raw\ASVspoof2019
Expand-Archive -Path LA.zip -DestinationPath datasets\raw\ASVspoof2019 -Force
```

`Expand-Archive` is slow on 120k small files (10-20 min). If you have 7-Zip it
is several times faster:

```powershell
& "C:\Program Files\7-Zip\7z.exe" x LA.zip -odatasets\raw\ASVspoof2019
```

You should end up with:

```
datasets/raw/ASVspoof2019/LA/
├── ASVspoof2019_LA_train/flac/          25,380 .flac
├── ASVspoof2019_LA_dev/flac/            24,844 .flac
├── ASVspoof2019_LA_eval/flac/           71,237 .flac
└── ASVspoof2019_LA_cm_protocols/
    ├── ASVspoof2019.LA.cm.train.trn.txt
    ├── ASVspoof2019.LA.cm.dev.trl.txt
    └── ASVspoof2019.LA.cm.eval.trl.txt
```

Extracted size is about 7 GB, so budget ~15 GB total with the zip. You have 195 GB.

**Delete `LA.zip` afterwards** — it is already in `.gitignore`'s spirit but not
its letter, and 7 GB is 7 GB.

---

## Step 3 — build manifests

```powershell
python backend\scripts\build_manifest.py --asvspoof2019-la datasets\raw\ASVspoof2019\LA
```

Writes `datasets/manifests/{train,validation,test}.csv` with columns
`path,label,split,attack,speaker`, using the convention **0 = bonafide,
1 = spoof** throughout.

The script auto-detects which protocol column holds the id, the label and the
attack code by matching values against the files on disk, rather than
hardcoding indices. The old version hardcoded "label is column 5, attack is
column 8" — which is the 2021 DF layout, not the 2019 LA one — and stamped
`split="test"` on every row, so train and validation could never have been
separated.

Expect roughly:

```
[train]       25,380 rows   bonafide 2,580 (10.2%)   spoof 22,800 (89.8%)
[validation]  24,844 rows   bonafide 2,548 (10.3%)   spoof 22,296 (89.7%)
[test]        71,237 rows   bonafide 7,355 (10.3%)   spoof 63,882 (89.7%)
```

**That ~90% spoof imbalance is why accuracy is a useless metric here.** A model
that always answers "spoof" scores 90% accuracy and flags every real customer.
Training uses class-balanced sampling by default; evaluation reports EER.

### The attack-held-out split (do use this)

The same command also writes `train_heldout.csv` and `val_unseen.csv`, holding
A05 and A06 out of training and validating only on them.

This matters more than it sounds. Train and dev share the same six attacks
(A01–A06), so dev EER measures how well the model recognises attacks it trained
on. It saturates near zero within two or three epochs while performance on the
*eval* attacks (A07–A19) is still improving — so selecting the checkpoint on
dev EER picks a memorising model and reports a number that predicts nothing.

Validating on held-out attacks costs a few thousand training clips and buys a
model-selection signal that behaves like the real test:

```powershell
python backend\scripts\train_model.py `
    --cache-dir datasets\cache `
    --train-manifest datasets\manifests\train_heldout.csv `
    --validation-manifest datasets\manifests\val_unseen.csv `
    --experiment-id exp002
```

Change which attacks are held out with `--holdout-attacks A04,A05,A06`, or
disable with `--holdout-attacks ""`.

Report both runs. "EER on seen attacks vs EER on unseen attacks" is a far
stronger slide than one number, and it is the honest answer to "does this
generalise?"

---

## Step 4 — verify before you spend an hour on a bad run

```powershell
python backend\scripts\dataset_stats.py --all --check-audio
```

This opens the audio and checks for missing files, unreadable files, non-16 kHz
sample rates, clip durations, **file leakage between splits**, and **speaker
leakage between splits**. It also reports which eval attacks were unseen in
training.

Speaker leakage is the one that silently ruins a project: metrics look
excellent right until the model meets a voice it has not memorised. ASVspoof
2019 LA is speaker-disjoint by construction, so this should come back clean —
which is precisely why it is worth confirming rather than assuming.

---

## Step 5 — cache the waveforms (optional, pays for itself in 2 epochs)

```powershell
python backend\scripts\cache_dataset.py --all
```

Decodes each FLAC once into a memmapped float16 array. FLAC decoding — not the
GPU — is the bottleneck for the frozen-encoder stage, and it costs the same
several minutes *every epoch* otherwise.

`--all` caches train, validation, and the held-out pair if it exists. It
deliberately skips the test split: 71k clips read exactly once is not worth
11 GB of disk, and streaming FLAC is fine for a single pass. Cache it
explicitly if you plan repeated evaluation:

```powershell
python backend\scripts\cache_dataset.py --manifest datasets\manifests\test.csv --output datasets\cache\test
```

Disk cost at `--max-seconds 6`: roughly 4 GB train, 4 GB validation.

The cache directory is named after the manifest, so `--cache-dir datasets\cache`
with `--train-manifest ...\train_heldout.csv` picks up
`datasets\cache\train_heldout` automatically rather than silently training on
the wrong cached split.

---

## Step 6 — smoke test (under a minute)

```powershell
python backend\scripts\train_model.py --smoke-test
```

64 train clips, 32 validation, one epoch, no augmentation. This proves the
whole path works — manifest → dataset → model → loss → backward → checkpoint →
EER — before you commit real time. If this fails, nothing after it will work.

---

## Step 7 — size the real run *before* starting it

```powershell
python backend\scripts\train_model.py --cache-dir datasets\cache --dry-run
```

This runs a dozen real training steps, discards the warm-up ones, takes the
median, and prints:

```
  ---- projected runtime ----------------------------------------
  measured          118 ms/step   (67.8 samples/s)
  per epoch       3m07s   (1586 steps)
  whole run      12m28s   (4 epochs)
  peak vram 3.41/4.02G
  --------------------------------------------------------------
```

then stops. **This is the answer to "how long is an epoch going to take" — you
get it in about thirty seconds instead of finding out an hour in.**

Two caveats the output reminds you of:

- The probe runs during the *frozen encoder* stage. Epochs after the unfreeze
  backpropagate through four transformer layers and run roughly 1.7–2.2× slower.
- `peak vram` is the number to watch. If it approaches 6 GB, drop
  `--batch-size` to 4 and raise `--gradient-accumulation` to 4 — the effective
  batch stays the same and only wall-clock changes.

---

## Step 8 — train

```powershell
python backend\scripts\train_model.py --cache-dir datasets\cache --experiment-id exp001
```

### What you will see

```
Epoch 2/4   1586 steps x 16 samples
  [#####...............]  25%   397/1586  loss 0.1832  acc 93.71%  6.8it/s
      el 58s  eta 2m54s  run 9m12s  lr 8.42e-05  vram 3.41/4.02G
  epoch 2 done in 3m52s  |  train loss 0.1544  |  train acc 94.88%
      |  108.3 samples/s  |  vram 3.41/4.02G
  validation:  val_loss 0.1201   val_eer% 3.8410   val_auc 0.9912   val_acc% 95.60
  new best EER 3.841% -> saved models/voxshield_antispoof.pt
  2 epoch(s) left  ->  ~7m44s remaining
```

The bracketed line updates live in place. Every 25 steps a permanent copy is
printed so scrollback stays readable, and everything is also appended to
`experiments/exp001/progress.jsonl` for later plotting.

### The schedule, and why

**Stage 1 — epoch 1, encoder frozen.** Only the attentive pooling, the spectral
CNN and the classifier train. The head starts from random weights; letting its
first large gradients flow into a pretrained encoder is the standard way to
destroy features you did not pay to train. It is also much faster, since no
gradient crosses twelve transformer layers.

**Stage 2 — epochs 2+, top 4 layers unfrozen.** The encoder's top layers train
at `1e-5` while the head stays at `1e-4`. The convolutional front end stays
frozen permanently — generic waveform filters, unstable to fine-tune, and
freezing it is what keeps peak VRAM inside 6 GB.

The optimiser is **rebuilt** at the stage change rather than extended, because
AdamW keeps per-parameter moment buffers and newly unfrozen parameters have
none.

### Useful flags

| flag | effect |
|---|---|
| `--batch-size 4 --gradient-accumulation 4` | same effective batch, less VRAM |
| `--limit-train 5000` | quick experiment on a subset |
| `--epochs 6 --freeze-epochs 1` | longer schedule |
| `--no-augment` | clean-audio ablation (see below) |
| `--no-spectral` | SSL-only ablation |
| `--encoder microsoft/wavlm-base-plus` | stronger encoder, same memory class |
| `--gradient-checkpointing` | ~30% slower, large VRAM saving |
| `--precision fp16` | force fp16 (default `auto` picks bf16 on Ampere) |

Every run writes `experiments/<id>/{config.json,history.json,progress.jsonl,summary.json}`,
and the checkpoint stores the architecture and git commit alongside the weights —
so a `.pt` file can always be traced back to the recipe that made it.

---

## Step 9 — evaluate properly

```powershell
python backend\scripts\evaluate_model.py `
    --manifest datasets\manifests\test.csv `
    --model models\voxshield_antispoof.pt `
    --calibrate-on datasets\manifests\validation.csv `
    --save-scores
```

Reports EER, minDCF, ROC-AUC, a per-attack breakdown, and three operating
points, and writes ROC + DET plots.

Fit calibration on **validation** and report on **test** — never both on the
same split.

### What to actually put in the presentation

Never report accuracy alone. Report:

```
EER          the headline, threshold-free
ROC-AUC      discrimination across all thresholds
minDCF       cost-weighted, with the prior stated
per-attack   which attacks it catches and which it misses
at 1% FAR    "if we challenge 1% of genuine callers, we catch X% of attacks"
```

That last one is the number a bank actually cares about, and almost nobody
presents it.

> **On minDCF:** ASVspoof papers report *tandem* DCF (t-DCF), which folds in an
> ASV subsystem's scores on the same trials. VoxShield has no enrolled ASV
> branch yet, so t-DCF is not computable and this is NIST-style normalised DCF
> instead. Say "normalised minDCF", not "t-DCF" — reviewers will check.

### Sanity expectations

A wav2vec2-base + spectral fusion model trained this way should land in the
low single digits of EER on the LA eval set. If you see **under 0.5%**, be
suspicious before you are pleased — check `dataset_stats.py` for leakage and
confirm you are evaluating on `test.csv` and not on the split you trained on.

---

## Step 10 — the augmentation question (this decides your demo)

ASVspoof is studio-clean. A model trained only on it learns "clean +
slightly odd spectrum = spoof", and then collapses the first time a judge holds
a phone up to a laptop microphone — because now *every* sample is band-limited
and compressed, and the classifier cannot tell "degraded by a phone line" from
"degraded by a vocoder".

Augmentation is on by default and applies to bonafide and spoof alike, so
channel degradation becomes uninformative and only the synthesis artifact is
left to separate the classes. It runs on the CPU in the DataLoader workers,
costs well under a millisecond per clip, and needs no ffmpeg or sox:

| effect | p | what it simulates |
|---|---|---|
| telephone | 0.35 | 300–3400 Hz passband + 8 kHz round trip |
| mu-law codec | 0.20 | G.711 / PSTN quantisation |
| additive noise | 0.30 | 10–30 dB SNR |
| reverb | 0.15 | synthetic exponential RIR |
| gain | 0.30 | level variation |
| packet dropout | 0.10 | VoIP loss |
| clipping | 0.05 | hot microphone |

**Run the ablation.** Train once with `--no-augment` and once without the flag,
then evaluate both on the clean test set *and* on live microphone recordings.
The clean-trained model will usually win on clean test data and lose badly on
real audio. That contrast is a genuinely good slide.

---

## Where things stand

Verified against the repo, not the handoff doc:

| | before | now |
|---|---|---|
| CUDA PyTorch | CPU-only build | **2.14.0+cu130, verified `True`** |
| `transformers` | not installed | **4.57.6, pinned `<5`** |
| `train_model.py` / `evaluate_model.py` | did not exist | **written, run end to end on GPU** |
| `app/ml/risk.py` | did not exist | **written, 9 tests** |
| `SpectralBranch` | did not exist | **written and wired into `forward`** |
| Manifests | all three CSVs 0 bytes | builder rewritten and verified |
| Test suite | 2 collection errors, 0 runs | **90 tests, all passing** |
| Dataset | `datasets/raw/` absent | **still absent — your move** |
| Trained checkpoint | none | none (blocked on the dataset) |

The handoff lists Phase 3.6 (dataset preparation) as "previously reported
complete". It was not: there is no data and the manifests were empty. Phases
3.7 and 3.8 had no runnable entry point at all.

### Cross-dataset evaluation (later, optional)

The test that separates "learned synthesis artifacts" from "learned this
corpus" is scoring a model trained on ASVspoof against a corpus it has never
seen. **In-the-Wild** is the usual choice — 31,779 clips of real-world audio,
~8 GB:

```powershell
curl.exe -L -C - -o datasets\raw\release_in_the_wild.zip `
  "https://huggingface.co/datasets/mueller91/In-The-Wild/resolve/main/release_in_the_wild.zip"
```

Expect the EER to be several times worse than on LA eval. That is the normal,
honest result, and presenting it alongside the in-domain number is far more
credible than presenting only the good one.

## Known gaps, honestly

- **No speaker verification yet.** VoxShield's whole thesis is that anti-spoof
  alone is insufficient — Demo C in the handoff (a *real human* impersonating
  the CEO) needs ECAPA-TDNN, which is Phase 4 and not started.
- **Thresholds are uncalibrated.** 60 and 85 in `app/ml/risk.py` are
  engineering placeholders. Derive them from the validation DET curve with
  `metrics.threshold_at_false_alarm` before quoting any decision behaviour.
- **Single dataset.** Cross-dataset evaluation (train on ASVspoof, test on
  something else entirely) is the test that distinguishes "learned synthesis
  artifacts" from "learned this corpus". Until that is run, do not claim
  VoxShield detects arbitrary voice clones.
