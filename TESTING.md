# Testing VoxShield

Every case, what to do, and what the correct answer looks like.

If a result does not match, that is a bug — the expected values here were all
measured on this machine, not estimated.

---

## Start it

```powershell
cd C:\Users\Karthikeya Varma\voxshield
.\start.ps1
```

This checks Python, packages, GPU, the checkpoint, the calibration, the test
corpus and the ports before starting anything, and names the fix for whatever
is missing. Two windows open: backend on `:8000`, frontend on `:3000`.

To check without starting: `.\start.ps1 -Check`
Backend only: `.\start.ps1 -BackendOnly`
Different port: `.\start.ps1 -BackendPort 8001`

Then open **http://localhost:3000**.

The first analysis takes about **6 seconds** — it loads a 380 MB checkpoint.
Every one after that is ~30 ms on the GPU.

---

## The 60-second check

Run this before anything else. It exercises all four subsystems against the
real checkpoint with real audio, and prints PASS or FAIL with the number it
actually got.

```powershell
cd backend
..\.venv\Scripts\python.exe scripts\make_test_audio.py   # once, generates the corpus
..\.venv\Scripts\python.exe scripts\verify_system.py
```

Expected last line:

```
  59 passed, 0 failed, 0 skipped
```

Anything other than 0 failures is a real problem. The exit code is the failure
count, so it works in CI.

Also run the unit suite:

```powershell
cd backend
..\.venv\Scripts\python.exe -m pytest -q
```

Expected: **312 passed**.

---

## What "correct" means here

Two numbers, both measured on 35,619 held-out ASVspoof 2019 LA evaluation
utterances covering 13 attack types the model never trained on:

| | |
| --- | --- |
| Detection at the shipped threshold | **94.95%** |
| False alarm at the same threshold | **0.79%** |
| Equal error rate | **2.95%** |

So roughly **1 in 20 attacks slips through**, and **1 in 127 genuine callers
gets challenged**. A single miss in your testing is not a bug. A *pattern* of
misses is.

Speaker verification: **0.373% EER** on 67 speakers not used for calibration.

---

# Part 1 — Anti-spoof

## 1.1 Synthetic speech is detected

**Do:** Analyse a recording tab → Choose a file → pick anything from
`datasets\test-audio\spoof\` → Analyse.

**Expect:**

| Field | Value |
| --- | --- |
| Synthetic probability | `>99.9%` |
| Raw model score | 0.968 – 0.980 |
| Verdict | SYNTHETIC |
| Fused risk | 100 / 100, HIGH |
| Decision | HIGH RISK WORKFLOW |

There are 9 files, 3 voices × 3 scripts. All nine should be detected.

**Why this is a real test.** These are Windows SAPI voices. ASVspoof 2019 LA
contains neural TTS and voice conversion — a completely different synthesis
family. The model has never seen anything like SAPI and still flags it at
0.97 raw. That is genuine out-of-distribution generalisation, not a memorised
artefact.

## 1.2 Your own voice is not flagged

**Do:** Analyse a recording → Record from microphone → say two or three full
sentences → Stop → Analyse. Leave identity and context empty.

**Expect:** Synthetic probability **below 50%**, verdict "genuine speech",
risk low.

**If your real voice reads as SYNTHETIC:** first check the **System** tab shows
"Calibrated: yes". If it does, this is the honest limitation to know about —
the model was trained on ASVspoof, which is clean read speech, and a laptop
microphone in a room is a different channel. Try again closer to the mic, in a
quieter room. A persistent false alarm on your voice is worth reporting; it is
the domain-shift case the evaluation set cannot measure.

## 1.3 Silence, noise and tones are refused — not guessed

**Do:** Analyse each file in `datasets\test-audio\edge\`.

| File | Expected |
| --- | --- |
| `silence.wav` | NOT ASSESSED · INSUFFICIENT EVIDENCE |
| `white_noise.wav` | NOT ASSESSED · INSUFFICIENT EVIDENCE |
| `pure_tone.wav` | NOT ASSESSED · INSUFFICIENT EVIDENCE |
| `dial_tone.wav` | NOT ASSESSED · INSUFFICIENT EVIDENCE |
| `mains_hum.wav` | NOT ASSESSED · INSUFFICIENT EVIDENCE |
| `dc_offset.wav` | NOT ASSESSED · INSUFFICIENT EVIDENCE |
| `speech_muted.wav` | NOT ASSESSED (below −60 dBFS) |

**This is the single most important test on the page.** Uncaught, the detector
scores four seconds of digital silence at **0.999 synthetic** and white noise
at **0.997** — it is a speech model, and asked about a class it never trained
on it answers confidently and wrongly. Hold music during a real call would
have fired the alarm every time.

Note the decision is **INSUFFICIENT EVIDENCE**, not ALLOW. "We could not check
this" and "we checked this and it is fine" must never produce the same answer.

## 1.4 Format handling

| File | Expected |
| --- | --- |
| `speech_stereo.wav` | Detected — channels averaged to mono |
| `speech_8k.wav` | Detected — telephone bandwidth, resampled up |
| `speech_48k.wav` | Detected — resampled down |
| `speech_clipped.wav` | Detected — clipping does not rescue a fake |
| `speech_quiet.wav` | Detected — peak-normalised before scoring |

`speech_8k.wav` is the one that matters for the pitch: real phone calls are
8 kHz.

## 1.5 Rejected uploads

| File | Expected |
| --- | --- |
| `too_short.wav` | Red error: "is 0.40s; at least 1.0s ... is needed" |
| `not_audio.wav` | Red error: "could not decode ... as audio" |
| `empty.wav` | Red error: "is empty" |
| An MP3 | Red error naming MP3 as unsupported |

Each message says what was wrong and what to do. None of them is a bare 500.

---

# Part 2 — Speaker verification

This is the branch that catches a **real human** impersonating someone. The
anti-spoof model cannot: an impersonator's speech is genuinely human, so it
correctly reports "not synthetic" and waves them through.

## 2.1 Enrol your voice

**Do:** Speakers tab → type an identity (e.g. `karthikeya`) → Record sample 1 →
read the prompt → Stop. Repeat until you have **5**. Then Enrol.

**Expect:** green message, `consistency` **above 0.6** (0.75–0.95 is typical
for one person in one room), and the note "Audio was discarded; only the
embedding centroid is stored."

**Read each prompt as written** — they change each time on purpose. Varying the
words makes the prototype describe a *person* rather than one sentence.

## 2.2 Verify as yourself

**Do:** Record and verify against your own identity.

**Expect:** **MATCH**, similarity **above 0.44**. Typically 0.6–0.9.

## 2.3 Verify as someone else — the impersonation case

**Do:** Have a friend or family member record against *your* identity.

**Expect:** **NO_MATCH**, similarity **below 0.33**. Typically 0.0–0.25.

**If nobody is available:** the verification script does this automatically
using two different Windows TTS voices, which ECAPA treats as different
speakers. Measured: 0.987 same-voice against 0.148 cross-voice, a gap of 0.839.

## 2.4 The three-way answer

Between 0.33 and 0.44 the answer is **UNCERTAIN**, deliberately. A two-way
split forces a guess on exactly the scores the system knows least about. If you
see UNCERTAIN, that is the design working — the policy engine asks for a second
factor instead of inventing confidence.

## 2.5 An identity nobody enrolled

**Do:** Verify against `does-not-exist`.

**Expect:** NO_MATCH with the reason "No enrolled voice profile for ...". Not an
error — a caller claiming an identity that does not exist is a fact about the
call, not a system failure.

## 2.6 Hostile identities

**Do:** Try enrolling or verifying with each of these:

| Identity | Expected |
| --- | --- |
| `../../../../tmp/pwned` | 400 — "may contain only letters, digits..." |
| `a/b` | 400 |
| 500 `X` characters | 400 — "identity is 500 characters; the limit is 128" |
| empty | 400 |

The live registry had accumulated the first and third of these before
validation existed. Both were accepted, stored, and served back.

---

# Part 3 — Call context

The branch that catches fraud when the voice is genuine and the person is real
but the *situation* is wrong.

## 3.1 Routine call

**Do:** Analyse → your own recording → context preset **Routine call**.

**Expect:** context risk **0%**, no reasons listed, 5 signals used.

## 3.2 CEO fraud pattern

**Do:** Same audio, context preset **CEO fraud pattern**.

**Expect:** context risk **100%**, 7 reasons including "Caller number is not
recognised", "Call placed at 23:00", "Requested amount is 125x the account's
typical transaction", "Beneficiary was added today".

Policy should now include **TRANSACTION HOLD** and **MANAGER APPROVAL**, and
"requires a human" should appear.

## 3.3 Unknown is not innocent

**Do:** Custom JSON → `{"caller_known": false}` → Analyse.

**Expect:** context risk **100%** — one signal was collected and it was bad.

This is the important behaviour: the score is a fraction of the risk that
*could* be assessed, not of all conceivable risk. Dividing by all ten possible
signals would show 14% and hide the thing the operator needs to see. Check the
"Not collected" count reads 12.

## 3.4 Rejected context

| Custom JSON | Expected error |
| --- | --- |
| `{"caller_known": false` | "Expecting ',' delimiter" |
| `[1, 2, 3]` | "context must be a JSON object" |
| `{"not_a_field": 1}` | "unknown context fields: ['not_a_field']" |
| `{"transaction_amount": "lots"}` | "must be int or float, got str" |
| `{"hour_of_day": 99}` | "must be between 0 and 23" |
| `{"failed_authentication_count": true}` | "got a boolean" |

The last one matters: `bool` is a subclass of `int` in Python, so an unguarded
check would silently accept `true` as a count of 1.

---

# Part 4 — Fusion: the three cases

These are the demonstrations. Each one shows something no single branch can do.

## Case A — genuine caller

| | |
| --- | --- |
| Audio | your own voice |
| Identity | your enrolled identity |
| Context | Routine call |

**Expect:** low risk, **ALLOW**, action MONITOR only.

## Case B — cloned voice

| | |
| --- | --- |
| Audio | any file from `datasets\test-audio\spoof\` |
| Identity | leave empty |j
| Context | No context |

**Expect:** risk 100, **HIGH RISK WORKFLOW**. Anti-spoof alone catches this.

## Case C — human impersonator (the one that matters)

| | |
| --- | --- |
| Audio | a **different person's** real voice |
| Identity | *your* enrolled identity |
| Context | CEO fraud pattern |

**Expect:** risk **60**, SECONDARY VERIFICATION, with CALLBACK and
ESCALATE SECURITY in the actions, and "requires a human".

**Now do the same thing with the identity field empty.** The anti-spoof branch
sees genuine human speech, reports low, and the call reads as **ALLOW**.

That contrast is the whole argument for the project. Reproduced against the
current code:

| Branches available | Risk | Decision |
| --- | --- | --- |
| anti-spoof only | **4.0** | ALLOW — monitor, nothing else |
| + identity (NO_MATCH) | **60.0** | SECONDARY VERIFICATION |
| + identity + CEO context | **60.0** | SECONDARY VERIFICATION |

Note the third row does not score higher than the second. A confirmed identity
mismatch floors the score at 60 on its own — someone is claiming to be a person
they are not, and that is not a matter of degree. The context still changes the
outcome, in the actions rather than the number: adding it brings
**TRANSACTION HOLD** and **MANAGER APPROVAL** into the response.

## 4.1 No action ever rejects a transaction

Look at the policy panel in any high-risk result. Every action is a
verification step, a reversible hold, or a human escalation. `TRANSACTION HOLD`
says explicitly "do not reject it".

This is the one rule in the system that is not a placeholder: a false positive
must cost a real customer a verification step, never a silently refused
legitimate payment.

---

# Part 5 — Live call

## 5.1 Your voice

**Do:** Live call tab → leave identity and context empty → Start live call →
talk for 15 seconds.

**Expect:**
- "connected" within a second
- First score after ~3 seconds of speech ("Buffering 0.0s of 3.0s" before that)
- Risk builds gradually rather than jumping — it is smoothed across windows
- Windows scored climbing roughly once a second

## 5.2 Play a synthetic file at your microphone

**Do:** Start a live call, then play any file from `datasets\test-audio\spoof\`
out loud near the mic.

**Expect:** risk climbing into HIGH within a few seconds.

Note this is a harder test than uploading the file — the audio has been through
a speaker, a room and a microphone. If it reads lower than the file test, that
is the channel mismatch, and it is real.

## 5.3 Stop talking

**Do:** During a call with an established high risk, go silent for 10 seconds.

**Expect:** the risk **holds** rather than falling to zero, and the anti-spoof
panel shows "NOT SCORED".

A pause is not evidence of innocence. Smoothing silence in as a zero would drag
an established risk toward ALLOW during exactly the moment an attacker would
want it to.

## 5.4 Live identity

**Do:** Start a call with your enrolled identity claimed. Talk for 20 seconds.

**Expect:** the identity panel appears after a few seconds and the similarity
figure **improves** as speech accumulates — the tracker keeps a running
centroid rather than averaging per-window scores.

## 5.5 Bad context on the websocket

**Do:** Not reachable from the UI. Use the API:
`ws://localhost:8000/api/v1/calls/test/stream?context=NOTJSON`

**Expect:** connection refused with a readable reason, not a silent 1011.

---

# Part 6 — System tab

**Do:** Open the System tab.

**Expect all of:**

| Field | Correct value |
| --- | --- |
| Badge | BACKEND READY |
| Checkpoint | 362.6 MB |
| Calibrated | yes |
| Platt scale / bias | 3.7545 / 9.4999 |
| Held-out utterances | 35619 |
| Detection at p ≥ 0.5 | 94.95% |
| False alarm at p ≥ 0.5 | 0.79% |
| Encoder installed | yes |
| Thresholds calibrated | yes (v1-20260909-balanced) |
| Fusion weights calibrated | **placeholders** |

The last row is supposed to say that. Under "Declared limitations" you should
see the uncalibrated-weights warning. **That warning is the system being
honest, not an error.**

You may also see "3 stored profiles were skipped" — those are the hostile
identities from earlier testing, now refused on load.

---

# What is real and what is not

Say this plainly if anyone asks.

**Measured:**
- Anti-spoof 2.95% EER, 94.95% detection at 0.79% false alarm, on 71,237
  utterances across 13 attack types held out of training
- Speaker verification 0.373% EER on 67 unseen speakers
- Calibration ECE 0.126 → 0.015 on a held-out half

**Placeholders, labelled as such everywhere they appear:**
- Fusion weights (35% spoof / 30% identity / 10% context …) — engineering
  guesses, not fitted
- Policy bands (40 / 60 / 80) — same
- Context weights — analyst intuition

**Not measured at all:**
- Real telephone audio. Augmentation was on during training, but no live phone
  call has ever been scored.
- Any language other than English.
- Adversarial attacks designed against this model specifically.

**Never quote the 0.196% validation EER.** Validation used A01–A06, which the
model trained on. **2.95%** is the number.

**Say "normalised minDCF", not "t-DCF".** t-DCF requires an ASV subsystem this
does not have.

---

# When something breaks

| Symptom | Cause and fix |
| --- | --- |
| "BACKEND UNREACHABLE" | Backend not running, or on another port. `.\start.ps1 -Check`. |
| Everything reads 50% synthetic | Calibration missing. System tab → Calibrated. Fix: `cd backend; python scripts\calibrate_detector.py` |
| Every real voice reads SYNTHETIC | Check calibration first. If fine, this is domain shift — see 1.2. |
| Microphone button does nothing | Browser permission. Click the padlock in the address bar. Chrome and Edge only allow this on `localhost` or HTTPS. |
| "could not decode ... as audio" | MP3 or M4A. Convert to WAV. |
| Enrolment refused as "inconsistent" | The samples do not sound like one person — background speech, or a different mic between samples. Re-record. |
| Port 8000 in use | `.\start.ps1 -BackendPort 8001`, or find it: `Get-NetTCPConnection -LocalPort 8000 -State Listen` |
| First call takes 6 seconds | Expected — 380 MB checkpoint loading. Subsequent calls ~30 ms. |
| `verify_system.py` says API unreachable | Start the backend first, or pass `--url http://127.0.0.1:<port>` |

---

# Regenerating things

```powershell
cd backend

# Test corpus (spoof samples + edge cases)
..\.venv\Scripts\python.exe scripts\make_test_audio.py

# Detector calibration, from the stored evaluation scores
..\.venv\Scripts\python.exe scripts\calibrate_detector.py

# Speaker thresholds - needs the ASVspoof ASV protocols
..\.venv\Scripts\python.exe scripts\calibrate_speaker.py --help
```

`scripts\calibrate_detector.py` prints the full operating-point table it
derives, including the per-attack miss rates. Worth reading once — it is where
every number in this document comes from.
