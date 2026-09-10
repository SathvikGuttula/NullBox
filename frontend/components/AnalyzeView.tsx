"use client";

/**
 * Analyse one recording.
 *
 * The flow is three numbered steps down a single column, because that is the
 * order the work happens in and skipping straight to the button with no audio
 * is the only real error state. Step 2 and 3 are optional and say so — the
 * product degrades to anti-spoof alone and reports that it did.
 *
 * Context presets exist because typing JSON at a demo is how a demo goes
 * wrong, and because the interesting cases — a genuine human impersonating
 * someone — are invisible without context.
 */

import {
  FileAudio,
  Mic,
  Play,
  Square,
  Trash2,
  Upload,
  Zap,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";

import {
  AnalysisResult,
  SystemStatus,
  analyze,
  listSpeakers,
} from "../lib/api";
import { MicRecorder, microphoneError } from "../lib/recorder";
import ResultView, { ResultSkeleton } from "./ResultView";
import {
  Button,
  Chip,
  Field,
  LevelMeter,
  Notice,
  Panel,
  Pill,
} from "./ui";

interface Preset {
  id: string;
  label: string;
  blurb: string;
  context: Record<string, unknown>;
}

const PRESETS: Preset[] = [
  {
    id: "none",
    label: "No context",
    blurb:
      "Audio only. The fusion sees one branch and says so — useful for showing what anti-spoof alone can and cannot tell you.",
    context: {},
  },
  {
    id: "routine",
    label: "Routine call",
    blurb:
      "Known caller, known device, known beneficiary, mid-afternoon. Every context signal is clear.",
    context: {
      caller_known: true,
      trusted_contact: true,
      device_known: true,
      beneficiary_known: true,
      recent_password_reset: false,
      failed_authentication_count: 0,
      hour_of_day: 14,
    },
  },
  {
    id: "ceo",
    label: "CEO fraud",
    blurb:
      "Unknown caller and device at 23:00, moving 125× the account's typical amount to a beneficiary added today. The classic shape.",
    context: {
      caller_known: false,
      trusted_contact: false,
      device_known: false,
      hour_of_day: 23,
      transaction_amount: 250000,
      typical_transaction_amount: 2000,
      beneficiary_known: false,
      beneficiary_age_days: 0,
    },
  },
  {
    id: "takeover",
    label: "Account takeover",
    blurb:
      "Password reset an hour ago, three failed logins, calling from an unexpected country on a new device.",
    context: {
      caller_known: true,
      device_known: false,
      recent_password_reset: true,
      failed_authentication_count: 3,
      caller_country: "IE",
      expected_country: "IN",
      hour_of_day: 3,
    },
  },
];

export default function AnalyzeView({ status }: { status: SystemStatus | null }) {
  const [file, setFile] = useState<File | null>(null);
  const [recorded, setRecorded] = useState<{ blob: Blob; seconds: number; url: string } | null>(null);
  const [dragging, setDragging] = useState(false);

  const [identity, setIdentity] = useState("");
  const [preset, setPreset] = useState("none");
  const [custom, setCustom] = useState("");
  const [useCustom, setUseCustom] = useState(false);

  const [enrolled, setEnrolled] = useState<string[]>([]);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [recording, setRecording] = useState(false);
  const [level, setLevel] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const recorder = useRef(new MicRecorder());

  useEffect(() => {
    listSpeakers()
      .then((data) => setEnrolled(data.speakers.map((entry) => entry.identity)))
      .catch(() => setEnrolled([]));
  }, [status?.speaker.enrolled]);

  useEffect(() => {
    if (!recording) return;
    const started = Date.now();
    const timer = setInterval(() => setElapsed((Date.now() - started) / 1000), 100);
    return () => clearInterval(timer);
  }, [recording]);

  // Object URLs are a real leak in a session where someone records repeatedly.
  useEffect(() => {
    return () => {
      if (recorded?.url) URL.revokeObjectURL(recorded.url);
    };
  }, [recorded?.url]);

  async function startRecording() {
    setError(null);
    try {
      recorder.current.onLevel = setLevel;
      await recorder.current.start();
      setRecording(true);
      setElapsed(0);
      setFile(null);
    } catch (exception) {
      setError(microphoneError(exception));
    }
  }

  function stopRecording() {
    const captured = recorder.current.stop();
    setRecording(false);
    setLevel(0);

    if (!captured) {
      setError("Nothing was captured. Check that the right microphone is selected.");
      return;
    }
    if (captured.seconds < 1) {
      setError(`Only ${captured.seconds.toFixed(1)}s captured. At least one second is needed.`);
      return;
    }

    if (recorded?.url) URL.revokeObjectURL(recorded.url);
    setRecorded({
      blob: captured.blob,
      seconds: captured.seconds,
      url: URL.createObjectURL(captured.blob),
    });
  }

  function chooseFile(next: File | null) {
    if (!next) return;
    if (recorded?.url) URL.revokeObjectURL(recorded.url);
    setRecorded(null);
    setFile(next);
    setError(null);
  }

  async function run() {
    const blob = file ?? recorded?.blob;
    if (!blob) {
      setError("Choose a file or record something first.");
      return;
    }

    let context: object = {};
    if (useCustom) {
      if (custom.trim()) {
        try {
          context = JSON.parse(custom);
        } catch (exception) {
          setError(`Context is not valid JSON: ${(exception as Error).message}`);
          return;
        }
      }
    } else {
      context = PRESETS.find((entry) => entry.id === preset)?.context ?? {};
    }

    setBusy(true);
    setError(null);
    setResult(null);

    try {
      setResult(
        await analyze(blob, file?.name ?? "recording.wav", {
          identity: identity.trim() || undefined,
          context,
        })
      );
    } catch (exception) {
      setError((exception as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const chosen = PRESETS.find((entry) => entry.id === preset);
  const ready = Boolean(file ?? recorded);
  const detectorLoaded = status?.anti_spoof.loaded ?? false;
  const thresholds = status
    ? {
        match: Number(status.speaker.thresholds.match ?? 0.55),
        no_match: Number(status.speaker.thresholds.no_match ?? 0.35),
      }
    : undefined;

  return (
    <div className="split">
      {/* ------------------------------------------------------ the controls */}
      <div className="stack stack-4 rise">
        <Panel title="1 · The audio" padded>
          {recording ? (
            <div className="stack stack-4">
              <div
                style={{
                  padding: "var(--s4)",
                  borderRadius: "var(--r-md)",
                  background: "var(--sig-critical-bg)",
                  border: "1px solid var(--sig-critical-line)",
                }}
              >
                <div className="row" style={{ marginBottom: "var(--s3)" }}>
                  <span className="live-dot" />
                  <span
                    className="mono"
                    style={{ fontSize: "var(--t-lg)", fontWeight: 600, letterSpacing: "-.02em" }}
                  >
                    {elapsed.toFixed(1)}s
                  </span>
                  <span className="label spacer">Recording</span>
                </div>
                <LevelMeter level={level} />
              </div>

              <Button variant="danger" onClick={stopRecording} icon={<Square size={14} />} block>
                Stop
              </Button>

              <p className="field-hint">
                Say two or three full sentences. The detector needs speech with pauses in
                it, not one held vowel.
              </p>
            </div>
          ) : (
            <div className="stack stack-4">
              <label
                className={`dropzone${dragging ? " over" : ""}`}
                onDragOver={(event) => {
                  event.preventDefault();
                  setDragging(true);
                }}
                onDragLeave={() => setDragging(false)}
                onDrop={(event) => {
                  event.preventDefault();
                  setDragging(false);
                  chooseFile(event.dataTransfer.files?.[0] ?? null);
                }}
              >
                <input
                  type="file"
                  accept=".wav,.flac,.ogg,audio/*"
                  onChange={(event) => chooseFile(event.target.files?.[0] ?? null)}
                  aria-label="Choose an audio file"
                />
                <Upload size={20} className="dim" aria-hidden />
                <div>
                  <div style={{ fontSize: "var(--t-md)", fontWeight: 550 }}>
                    Drop a file, or click to choose
                  </div>
                  <div className="field-hint" style={{ marginTop: 3 }}>
                    WAV, FLAC or OGG. MP3 and M4A are not supported.
                  </div>
                </div>
              </label>

              <div className="row" style={{ gap: "var(--s3)" }}>
                <span className="label">or</span>
                <div style={{ flex: 1, height: 1, background: "var(--line-soft)" }} />
              </div>

              <Button onClick={startRecording} icon={<Mic size={14} />} block>
                Record from microphone
              </Button>
            </div>
          )}

          {file && !recording && (
            <div className="row" style={{ marginTop: "var(--s4)", gap: "var(--s2)" }}>
              <FileAudio size={14} style={{ color: "var(--brand)", flex: "none" }} />
              <span className="mono" style={{ fontSize: "var(--t-sm)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {file.name}
              </span>
              <span className="label spacer">{(file.size / 1024).toFixed(0)} KB</span>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setFile(null)}
                ariaLabel="Remove file"
              >
                <Trash2 size={13} />
              </Button>
            </div>
          )}

          {recorded && !recording && (
            <div className="stack stack-3" style={{ marginTop: "var(--s4)" }}>
              <div className="row" style={{ gap: "var(--s2)" }}>
                <Pill signal="clear">Recorded</Pill>
                <span className="mono dim" style={{ fontSize: "var(--t-sm)" }}>
                  {recorded.seconds.toFixed(1)}s · 16 kHz mono
                </span>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setRecorded(null)}
                  ariaLabel="Discard recording"
                >
                  <Trash2 size={13} />
                </Button>
              </div>
              <audio controls src={recorded.url} style={{ width: "100%", height: 32 }} />
            </div>
          )}
        </Panel>

        <Panel title="2 · Claimed identity" aside={<span className="label">Optional</span>}>
          <div className="stack stack-3">
            <Field
              hint="Who the caller says they are. Without it, a low risk score says nothing about who is speaking."
            >
              <input
                className="input"
                value={identity}
                onChange={(event) => setIdentity(event.target.value)}
                placeholder="anti-spoof only"
                aria-label="Claimed identity"
              />
            </Field>

            {enrolled.length > 0 ? (
              <div className="chipset">
                {enrolled.map((name) => (
                  <Chip
                    key={name}
                    active={identity === name}
                    onClick={() => setIdentity(identity === name ? "" : name)}
                  >
                    {name}
                  </Chip>
                ))}
              </div>
            ) : (
              <Notice tone="info">
                Nobody is enrolled yet. Enrol a voice on the Speakers tab to exercise the
                identity branch — it is the one that catches a human impersonator.
              </Notice>
            )}
          </div>
        </Panel>

        <Panel title="3 · Call context" aside={<span className="label">Optional</span>}>
          <div className="stack stack-3">
            <div className="chipset">
              {PRESETS.map((entry) => (
                <Chip
                  key={entry.id}
                  active={!useCustom && preset === entry.id}
                  onClick={() => {
                    setPreset(entry.id);
                    setUseCustom(false);
                  }}
                >
                  {entry.label}
                </Chip>
              ))}
              <Chip active={useCustom} onClick={() => setUseCustom(true)}>
                Custom JSON
              </Chip>
            </div>

            {useCustom ? (
              <Field hint="Unknown fields and wrong types are rejected with an explanation rather than silently ignored — try it.">
                <textarea
                  className="textarea"
                  rows={6}
                  value={custom}
                  onChange={(event) => setCustom(event.target.value)}
                  placeholder={'{ "caller_known": false, "hour_of_day": 23 }'}
                  aria-label="Custom context JSON"
                />
              </Field>
            ) : (
              <p className="prose" style={{ fontSize: "var(--t-sm)" }}>
                {chosen?.blurb}
              </p>
            )}
          </div>
        </Panel>

        <div className="stack stack-3">
          <Button
            variant="primary"
            size="lg"
            onClick={run}
            disabled={busy || !ready}
            icon={busy ? undefined : <Zap size={15} />}
            block
          >
            {busy ? "Analysing…" : "Analyse"}
          </Button>

          {!ready && !busy && (
            <p className="field-hint" style={{ textAlign: "center" }}>
              Choose a file or record something first.
            </p>
          )}
          {busy && !detectorLoaded && (
            <p className="field-hint" style={{ textAlign: "center" }}>
              First run loads a 362 MB checkpoint — about six seconds.
            </p>
          )}
        </div>

        {error && <Notice tone="error">{error}</Notice>}
      </div>

      {/* -------------------------------------------------------- the result */}
      <div style={{ minWidth: 0 }}>
        {busy ? (
          <ResultSkeleton
            note={detectorLoaded ? "Scoring…" : "Loading the checkpoint…"}
          />
        ) : result ? (
          <ResultView result={result} thresholds={thresholds} />
        ) : (
          <div className="panel rise rise-2" style={{ minHeight: 380, display: "grid", placeItems: "center" }}>
            <div className="empty">
              <div className="empty-icon">
                <Play size={18} />
              </div>
              <h4>No analysis yet</h4>
              <p>
                Add a recording on the left and press Analyse. The result will show every
                branch that could run — and name the ones that could not.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
