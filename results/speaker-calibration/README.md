# Speaker verification — calibration

Pretrained **ECAPA-TDNN** (`speechbrain/spkrec-ecapa-voxceleb`, 192-d).
Nothing is trained here; this measures where the decision thresholds belong.

Trials are built from the bonafide utterances in the project's own manifests,
grouped by the `speaker` column:

- **target** — two bonafide utterances from the same speaker
- **nontarget** — two bonafide utterances from different speakers

Enrollment samples are held out of the probe pool, so no prototype ever
contains the utterance being scored against it.

---

## Result

| | speakers | trials | **EER** |
|---|---:|---:|---:|
| dev — calibrated on | 20 | 3,000 | 0.375 % |
| **eval — never seen during calibration** | **67** | **6,700** | **0.373 %** |

**The thresholds transfer.** Calibrated on 20 dev speakers and applied to 67
eval speakers who share none of them, the error rate is essentially unchanged
(0.375 % → 0.373 %). That is the check worth running: calibrating and reporting
on the same speakers would be the speaker-verification equivalent of quoting a
validation EER.

**0.373 % EER on unseen speakers is the number to report.**

## Operating points, eval speakers

| mode | match threshold | false accept | false reject |
|---|---:|---:|---:|
| high_security | 0.5177 | 0.07 % | 0.75 % |
| balanced | 0.4621 | 0.35 % | 0.37 % |
| no_match floor | 0.4326 | — | 0.07 % |

- **false accept** — an impostor admitted. The security cost.
- **false reject** — a genuine caller turned away. The usability cost.

Scores between `no_match` and `match` return **UNCERTAIN**, which the policy
engine should answer with a second factor rather than a guess.

## Known gap in these files

`low_friction` is **degenerate** in both runs — identical to `balanced`. That
was a bug in how the mode was derived: it used a false-*reject* budget, and
allowing more genuine rejections permits a *stricter* threshold, so the mode
came out at least as strict as `balanced` and collapsed onto it.

Fixed after these runs. All three modes now sit on the false-accept axis with
budgets defined relative to the measured EER, so they stay distinct whatever
the model quality. **Re-run the calibration notebook to populate a real
`low_friction`**; `balanced` and `high_security` in these files are correct and
usable as they stand.

## Caveats

- These describe **ASVspoof's clean studio audio**. They will not transfer to
  telephone conditions — re-calibrate before quoting anything about a phone
  deployment.
- ECAPA-TDNN is trained on VoxCeleb, which is predominantly English celebrity
  speech. Performance on other languages and demographics is unmeasured here.
- The 20-speaker dev set is thin for calibration. It held up on 67 eval
  speakers, but a larger and more varied enrollment population would make the
  thresholds more trustworthy.

## Reproduce

```bash
python backend/scripts/calibrate_speaker.py \
    --manifest datasets/manifests/validation.csv \
    --output speaker_thresholds_dev.json

python backend/scripts/calibrate_speaker.py \
    --manifest datasets/manifests/test.csv \
    --output speaker_thresholds_eval.json \
    --apply speaker_thresholds_dev.json
```

Or `notebooks/VoxShield_Kaggle_Speaker.ipynb` — about ten minutes on a T4.
