"use client";

/**
 * Analyse one recording: upload a file or record from the microphone, claim an
 * identity, describe the call, and see every branch of the result.
 *
 * The context presets are the point of this screen. Anti-spoof alone cannot
 * distinguish a genuine customer from a genuine human impersonating one - both
 * are real speech - so the interesting demos need context, and typing JSON by
 * hand at a demo is how a demo goes wrong.
 */

import { useEffect, useRef, useState } from "react";

import { AnalysisResult, analyze, listSpeakers } from "../lib/api";
import { MicRecorder, microphoneError } from "../lib/recorder";
import ResultView from "./ResultView";
import { Badge, Button, COLOURS, LevelMeter, Notice, Panel, TextInput } from "./ui";

interface Preset {
  id: string;
  label: string;
  description: string;
  context: Record<string, unknown>;
}

const PRESETS: Preset[] = [
  {
    id: "none",
    label: "No context",
    description:
      "Audio only. The fusion sees one branch and says so - useful for showing what anti-spoof alone can and cannot tell you.",
    context: {},
  },
  {
    id: "routine",
    label: "Routine call",
    description:
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
    id: "ceo-fraud",
    label: "CEO fraud pattern",
    description:
      "Unknown caller and device at 23:00, asking to move 125x the account's typical amount to a beneficiary added today. The classic shape.",
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
    description:
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

export default function AnalyzeTab() {
  const [file, setFile] = useState<File | null>(null);
  const [recorded, setRecorded] = useState<{ blob: Blob; seconds: number } | null>(
    null
  );
  const [identity, setIdentity] = useState("");
  const [preset, setPreset] = useState("none");
  const [customContext, setCustomContext] = useState("");
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
      .then((data) => setEnrolled(data.speakers.map((s) => s.identity)))
      .catch(() => setEnrolled([]));
  }, [result]);

  useEffect(() => {
    if (!recording) return;
    const started = Date.now();
    const timer = setInterval(() => setElapsed((Date.now() - started) / 1000), 100);
    return () => clearInterval(timer);
  }, [recording]);

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
      setError(
        `Only ${captured.seconds.toFixed(1)}s captured. At least 1 second is needed.`
      );
      return;
    }
    setRecorded({ blob: captured.blob, seconds: captured.seconds });
  }

  async function run() {
    const blob = file ?? recorded?.blob;
    if (!blob) {
      setError("Choose a file or record something first.");
      return;
    }

    let context: object = {};

    if (useCustom) {
      if (customContext.trim()) {
        try {
          context = JSON.parse(customContext);
        } catch (exception) {
          setError(`Context is not valid JSON: ${(exception as Error).message}`);
          return;
        }
      }
    } else {
      context = PRESETS.find((p) => p.id === preset)?.context ?? {};
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

  const chosen = PRESETS.find((p) => p.id === preset);
  const ready = Boolean(file ?? recorded);

  return (
    <div style={{ display: "grid", gap: 18 }}>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
          gap: 16,
        }}
      >
        <Panel title="1. The audio">
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
            <label>
              <input
                type="file"
                accept=".wav,.flac,.ogg,audio/*"
                style={{ display: "none" }}
                onChange={(event) => {
                  setFile(event.target.files?.[0] ?? null);
                  setRecorded(null);
                  setError(null);
                }}
              />
              <span
                style={{
                  display: "inline-block",
                  padding: "10px 16px",
                  borderRadius: 9,
                  border: `1px solid ${COLOURS.border}`,
                  background: "#111827",
                  fontSize: 13,
                  fontWeight: 600,
                  cursor: "pointer",
                }}
              >
                Choose a file
              </span>
            </label>

            {recording ? (
              <Button variant="danger" onClick={stopRecording}>
                Stop ({elapsed.toFixed(1)}s)
              </Button>
            ) : (
              <Button onClick={startRecording}>Record from microphone</Button>
            )}
          </div>

          {recording && (
            <div style={{ marginTop: 14 }}>
              <LevelMeter level={level} />
              <div style={{ fontSize: 11.5, color: COLOURS.muted, marginTop: 7 }}>
                Speak normally for five to ten seconds. Say a couple of full
                sentences — the detector needs speech with pauses in it, not a
                single held vowel.
              </div>
            </div>
          )}

          {file && (
            <div style={{ marginTop: 12, fontSize: 12.5 }}>
              <Badge colour={COLOURS.accent}>FILE</Badge>{" "}
              <code>{file.name}</code>{" "}
              <span style={{ color: COLOURS.muted }}>
                ({(file.size / 1024).toFixed(0)} KB)
              </span>
            </div>
          )}

          {recorded && !recording && (
            <div style={{ marginTop: 12, fontSize: 12.5 }}>
              <Badge colour={COLOURS.good}>RECORDED</Badge>{" "}
              {recorded.seconds.toFixed(1)}s of 16 kHz mono WAV
              <audio
                controls
                src={URL.createObjectURL(recorded.blob)}
                style={{ width: "100%", marginTop: 10, height: 34 }}
              />
            </div>
          )}

          <div style={{ fontSize: 11, color: COLOURS.muted, marginTop: 12 }}>
            WAV, FLAC and OGG are accepted. MP3 and M4A are not — the backend
            decodes with libsndfile, which does not read them.
          </div>
        </Panel>

        <Panel title="2. Claimed identity">
          <TextInput
            value={identity}
            onChange={setIdentity}
            placeholder="leave empty for anti-spoof only"
            hint="Who the caller says they are. Without it, a low risk score says nothing about who is speaking."
          />

          {enrolled.length > 0 ? (
            <div style={{ display: "flex", gap: 7, flexWrap: "wrap", marginTop: 4 }}>
              {enrolled.map((name) => (
                <button
                  key={name}
                  onClick={() => setIdentity(name)}
                  style={{
                    padding: "5px 11px",
                    borderRadius: 999,
                    border: `1px solid ${
                      identity === name ? COLOURS.accent : COLOURS.border
                    }`,
                    background: identity === name ? `${COLOURS.accent}20` : "transparent",
                    color: identity === name ? COLOURS.accent : COLOURS.dim,
                    fontSize: 11.5,
                    cursor: "pointer",
                    fontFamily: "inherit",
                  }}
                >
                  {name}
                </button>
              ))}
            </div>
          ) : (
            <Notice tone="info">
              Nobody is enrolled yet. Enrol a voice on the Speakers tab to
              exercise the identity branch — it is the one that catches a human
              impersonator.
            </Notice>
          )}
        </Panel>
      </div>

      <Panel
        title="3. Call context"
        subtitle="Everything about the call that is not the audio. Anything omitted counts as not collected, never as safe."
      >
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 12 }}>
          {PRESETS.map((option) => (
            <button
              key={option.id}
              onClick={() => {
                setPreset(option.id);
                setUseCustom(false);
              }}
              style={{
                padding: "8px 14px",
                borderRadius: 9,
                border: `1px solid ${
                  !useCustom && preset === option.id ? COLOURS.accent : COLOURS.border
                }`,
                background:
                  !useCustom && preset === option.id
                    ? `${COLOURS.accent}18`
                    : "transparent",
                color:
                  !useCustom && preset === option.id ? COLOURS.accent : COLOURS.dim,
                fontSize: 12.5,
                cursor: "pointer",
                fontFamily: "inherit",
                fontWeight: 600,
              }}
            >
              {option.label}
            </button>
          ))}
          <button
            onClick={() => setUseCustom(true)}
            style={{
              padding: "8px 14px",
              borderRadius: 9,
              border: `1px solid ${useCustom ? COLOURS.accent : COLOURS.border}`,
              background: useCustom ? `${COLOURS.accent}18` : "transparent",
              color: useCustom ? COLOURS.accent : COLOURS.dim,
              fontSize: 12.5,
              cursor: "pointer",
              fontFamily: "inherit",
              fontWeight: 600,
            }}
          >
            Custom JSON
          </button>
        </div>

        {useCustom ? (
          <>
            <textarea
              value={customContext}
              onChange={(event) => setCustomContext(event.target.value)}
              placeholder='{"caller_known": false, "hour_of_day": 23}'
              rows={6}
              style={{
                width: "100%",
                padding: 12,
                borderRadius: 9,
                border: `1px solid ${COLOURS.border}`,
                background: COLOURS.inset,
                color: COLOURS.text,
                fontSize: 12.5,
                fontFamily: "ui-monospace, Consolas, monospace",
              }}
            />
            <div style={{ fontSize: 11, color: COLOURS.muted, marginTop: 6 }}>
              Unknown fields and wrong types are rejected with an explanation
              rather than silently ignored — try it.
            </div>
          </>
        ) : (
          <>
            <div style={{ fontSize: 12.5, color: COLOURS.dim, lineHeight: 1.6 }}>
              {chosen?.description}
            </div>
            {Object.keys(chosen?.context ?? {}).length > 0 && (
              <pre
                style={{
                  marginTop: 12,
                  padding: 12,
                  borderRadius: 9,
                  background: COLOURS.inset,
                  border: `1px solid ${COLOURS.border}`,
                  fontSize: 11.5,
                  color: COLOURS.dim,
                  overflowX: "auto",
                }}
              >
                {JSON.stringify(chosen?.context, null, 2)}
              </pre>
            )}
          </>
        )}
      </Panel>

      <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
        <Button variant="primary" onClick={run} disabled={busy || !ready}>
          {busy ? "Analysing..." : "Analyse"}
        </Button>
        {!ready && (
          <span style={{ fontSize: 12.5, color: COLOURS.muted }}>
            Choose a file or record something first.
          </span>
        )}
        {busy && (
          <span style={{ fontSize: 12.5, color: COLOURS.muted }}>
            The first call loads a 380 MB checkpoint and takes a few seconds.
          </span>
        )}
      </div>

      {error && <Notice tone="error">{error}</Notice>}

      {result && <ResultView result={result} />}
    </div>
  );
}
