# Results

Evaluation artifacts, kept in the repo so a number on a slide can always be
traced back to the run that produced it.

Per-utterance `scores.csv` files are gitignored — they are ~9 MB each and can
be regenerated from the checkpoint. Everything needed to *verify* a claim is
here.

---

## exp002-full — the current model

**wav2vec2-base + log-Mel CNN fusion, trained on ASVspoof 2019 LA train**
Evaluated on the full LA eval set: **71,237 utterances, 13 attack types
(A07–A19), none of which appear in training.**

| metric | value |
|---|---|
| **EER** | **2.951 %** |
| ROC-AUC | 0.9863 |
| normalised minDCF (prior_spoof = 0.05) | 0.1474 |

### Operating points

| threshold | false alarm | miss | reading |
|---|---|---|---|
| 0.5 (default) | 0.16 % | 12.02 % | conservative; few genuine callers challenged |
| 0.0443 (EER) | 2.95 % | 2.95 % | the balance point |
| **0.0537 (1 % FA)** | **0.99 %** | **3.76 %** | **challenge 1 in 100 genuine callers, catch 96.2 % of attacks** |

The last row is the one an operator cares about, and it is the one worth
quoting alongside EER.

### Per attack, at the EER threshold

| attack | family | miss rate | mean score |
|---|---|---:|---:|
| A07 | TTS | 0.04 % | 0.879 |
| A08 | TTS | 0.00 % | 0.974 |
| A09 | TTS | 0.00 % | 0.976 |
| **A10** | TTS | **9.32 %** | 0.470 |
| A11 | TTS | 0.08 % | 0.958 |
| A12 | TTS | 0.00 % | 0.971 |
| A13 | TTS/VC | 0.00 % | 0.978 |
| A14 | TTS/VC | 0.00 % | 0.923 |
| A15 | TTS/VC | 0.20 % | 0.912 |
| A16 | TTS | 0.14 % | 0.959 |
| **A17** | VC | **7.69 %** | 0.694 |
| **A18** | VC | **18.09 %** | 0.561 |
| A19 | VC | 2.79 % | 0.847 |
| bonafide | — | 2.95 % (FA) | 0.033 |

Ten of thirteen attacks are caught at under 0.2 % miss. The error is
concentrated in **A18** (voice conversion), **A10** (neural-vocoder TTS) and
**A17** (voice conversion).

---

## The ablation — why the training data mattered

An earlier run held **A05 and A06** out of training. Those are the *only two
voice-conversion attacks* in ASVspoof 2019 LA train and dev; **A17–A19** are
the voice-conversion attacks in eval. Withholding them removed the entire VC
family from training.

| | A05/A06 held out | full train set |
|---|---:|---:|
| EER | 11.27 % | **2.95 %** |
| ROC-AUC | 0.9496 | **0.9863** |
| minDCF | 0.3188 | **0.1474** |
| mean miss, TTS | 0.02 % | 1.37 % |
| **mean miss, VC** | **48.79 %** | **9.52 %** |
| A19 | 73.93 % | **2.79 %** |

**Generalisation is bounded by synthesis family, not by attack identity.** A
detector that has seen text-to-speech generalises to *unseen* TTS systems
almost perfectly, and does not transfer to voice conversion at all.

A10 moved the other way (0.08 % → 9.32 %). The held-out model flagged far more
audio as spoof overall — 11.26 % false alarm against 2.95 % — which flattered
every miss rate. The current model is much more conservative, and A10 sits
closest to its boundary.

---

## How to reproduce

```bash
python backend/scripts/build_manifest.py --discover <asvspoof-root>
python backend/scripts/cache_dataset.py --all --workers 16
python backend/scripts/train_model.py \
    --cache-dir datasets/cache \
    --train-manifest datasets/manifests/train.csv \
    --validation-manifest datasets/manifests/validation.csv \
    --batch-size 32 --precision fp16 \
    --epochs 5 --freeze-epochs 1 --unfreeze-top-layers 4
python backend/scripts/evaluate_model.py \
    --manifest datasets/manifests/test.csv \
    --model models/voxshield_antispoof.pt --save-scores
```

Trained on a Kaggle T4, 5 epochs, ~62 minutes. `history.json` has the
per-epoch curve; `summary.json` records the selected checkpoint;
`config.json` in the experiment directory has the exact recipe.

---

## Reporting rules

- **Never quote the validation EER (0.196 %).** Validation is A01–A06 —
  attacks the model trained on. The 15× gap to the 2.95 % test EER is
  memorisation, and quoting it would be dishonest.
- **Say "normalised minDCF", not "t-DCF".** Tandem DCF folds in an ASV
  subsystem's scores on the same trials; VoxShield has no enrolled ASV branch
  yet, so t-DCF is not computable here.
- **Never quote accuracy alone.** The set is 89.7 % spoof, so "always say
  spoof" scores 89.7 % and flags every genuine caller.
- **Always state the split.** "2.95 % EER" means nothing without "on ASVspoof
  2019 LA eval, 13 attacks unseen in training".
- These numbers are **clean-corpus** results with channel augmentation active
  during training. They are not a claim about live telephone audio, which has
  not been measured.
