"use client";

/**
 * Enrol a voice, verify against it, and see who is enrolled.
 *
 * Enrolment takes several recordings on purpose. One sample makes a prototype
 * that reflects one recording condition rather than one person, and the
 * registry refuses a set whose samples do not agree with each other - a
 * blurred prototype is not a one-off error, it is an account that quietly
 * matches several people for as long as it exists.
 *
 * Only the embedding centroid is stored. The audio never reaches disk.
 */

import { useEffect, useRef, useState } from "react";

import {
  SpeakerSummary,
  Verification,
  deleteSpeaker,
  enrollSpeaker,
  listSpeakers,
  verifySpeaker,
} from "../lib/api";
import { MicRecorder, microphoneError } from "../lib/recorder";
import { Badge, Button, COLOURS, LevelMeter, Notice, Panel, Row, TextInput } from "./ui";

const REQUIRED = 3;
const RECOMMENDED = 5;

const PROMPTS = [
  "My name is on this account and I am authorising this call.",
  "The quick brown fox jumps over the lazy dog near the river bank.",
  "Please transfer the balance to the account we discussed on Tuesday.",
  "I would like to confirm the details of my most recent transaction.",
  "Security is important to me, so I am happy to verify my identity.",
];

export default function SpeakersTab() {
  const [speakers, setSpeakers] = useState<SpeakerSummary[]>([]);
  const [thresholds, setThresholds] = useState<Record<string, unknown>>({});

  const [identity, setIdentity] = useState("");
  const [samples, setSamples] = useState<{ blob: Blob; seconds: number }[]>([]);

  const [recording, setRecording] = useState(false);
  const [level, setLevel] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const recorder = useRef(new MicRecorder());

  const [verifyIdentity, setVerifyIdentity] = useState("");
  const [verifyResult, setVerifyResult] = useState<Verification | null>(null);

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  async function refresh() {
    try {
      const data = await listSpeakers();
      setSpeakers(data.speakers);
      setThresholds(data.thresholds);
    } catch (exception) {
      setError((exception as Error).message);
    }
  }

  useEffect(() => {
    refresh();
  }, []);

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
    } catch (exception) {
      setError(microphoneError(exception));
    }
  }

  function stopRecording(onDone: (blob: Blob, seconds: number) => void) {
    const captured = recorder.current.stop();
    setRecording(false);
    setLevel(0);

    if (!captured) {
      setError("Nothing was captured. Check the microphone selection.");
      return;
    }
    if (captured.seconds < 1) {
      setError(`Only ${captured.seconds.toFixed(1)}s captured; at least 1s is needed.`);
      return;
    }
    if (captured.peak < 0.01) {
      setError(
        "That recording is almost silent. Check the right microphone is selected " +
          "and speak closer to it."
      );
      return;
    }
    onDone(captured.blob, captured.seconds);
  }

  async function enrol() {
    if (samples.length < REQUIRED) {
      setError(`Record at least ${REQUIRED} samples (${RECOMMENDED} is better).`);
      return;
    }

    setBusy(true);
    setError(null);
    setSuccess(null);

    try {
      const profile = await enrollSpeaker(
        identity.trim(),
        samples.map((s) => s.blob)
      );
      setSuccess(
        `Enrolled '${profile.identity}' from ${profile.samples} samples ` +
          `(consistency ${profile.consistency.toFixed(3)}). ${profile.note}`
      );
      setSamples([]);
      setVerifyIdentity(profile.identity);
      await refresh();
    } catch (exception) {
      setError((exception as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function runVerify(blob: Blob) {
    if (!verifyIdentity.trim()) {
      setError("Type or pick the identity to verify against.");
      return;
    }

    setBusy(true);
    setError(null);
    setVerifyResult(null);

    try {
      setVerifyResult(await verifySpeaker(verifyIdentity.trim(), blob));
    } catch (exception) {
      setError((exception as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function remove(name: string) {
    try {
      await deleteSpeaker(name);
      setSuccess(`Deleted '${name}'.`);
      await refresh();
    } catch (exception) {
      setError((exception as Error).message);
    }
  }

  const decisionColour =
    verifyResult?.decision === "MATCH"
      ? COLOURS.good
      : verifyResult?.decision === "UNCERTAIN"
      ? COLOURS.warn
      : COLOURS.bad;

  return (
    <div style={{ display: "grid", gap: 18 }}>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(340px, 1fr))",
          gap: 16,
        }}
      >
        <Panel
          title="Enrol a voice"
          subtitle={`${REQUIRED} samples minimum, ${RECOMMENDED} recommended. Say something different each time - varying the words is what makes the prototype describe a person rather than one sentence.`}
        >
          <TextInput
            label="Identity"
            value={identity}
            onChange={setIdentity}
            placeholder="e.g. karthikeya"
            hint="Letters, digits, spaces and . _ @ - only. Up to 128 characters."
          />

          <div
            style={{
              padding: 12,
              borderRadius: 9,
              background: COLOURS.inset,
              border: `1px solid ${COLOURS.border}`,
              marginBottom: 12,
            }}
          >
            <div style={{ fontSize: 11, color: COLOURS.muted, letterSpacing: 1 }}>
              READ THIS ONE
            </div>
            <div style={{ fontSize: 14, marginTop: 6, lineHeight: 1.5 }}>
              {PROMPTS[samples.length % PROMPTS.length]}
            </div>
          </div>

          {recording ? (
            <>
              <Button
                variant="danger"
                onClick={() =>
                  stopRecording((blob, seconds) =>
                    setSamples((current) => [...current, { blob, seconds }])
                  )
                }
              >
                Stop ({elapsed.toFixed(1)}s)
              </Button>
              <div style={{ marginTop: 12 }}>
                <LevelMeter level={level} />
              </div>
            </>
          ) : (
            <Button onClick={startRecording} disabled={busy}>
              Record sample {samples.length + 1}
            </Button>
          )}

          {samples.length > 0 && (
            <div style={{ marginTop: 14 }}>
              {samples.map((sample, index) => (
                <div
                  key={index}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 10,
                    padding: "6px 0",
                    borderBottom: `1px solid ${COLOURS.border}44`,
                  }}
                >
                  <span style={{ fontSize: 12, color: COLOURS.dim, minWidth: 74 }}>
                    Sample {index + 1}
                  </span>
                  <span style={{ fontSize: 12, color: COLOURS.muted }}>
                    {sample.seconds.toFixed(1)}s
                  </span>
                  <audio
                    controls
                    src={URL.createObjectURL(sample.blob)}
                    style={{ height: 30, flex: 1 }}
                  />
                  <Button
                    variant="ghost"
                    onClick={() =>
                      setSamples((current) => current.filter((_, i) => i !== index))
                    }
                    style={{ padding: "4px 9px", fontSize: 11 }}
                  >
                    remove
                  </Button>
                </div>
              ))}

              <div style={{ marginTop: 14, display: "flex", gap: 10, alignItems: "center" }}>
                <Button
                  variant="primary"
                  onClick={enrol}
                  disabled={busy || samples.length < REQUIRED || !identity.trim()}
                >
                  Enrol {samples.length} sample{samples.length === 1 ? "" : "s"}
                </Button>
                <span style={{ fontSize: 12, color: COLOURS.muted }}>
                  {samples.length < REQUIRED
                    ? `${REQUIRED - samples.length} more needed`
                    : samples.length < RECOMMENDED
                    ? `${RECOMMENDED - samples.length} more recommended`
                    : "ready"}
                </span>
              </div>
            </div>
          )}
        </Panel>

        <Panel
          title="Verify against an enrolled voice"
          subtitle="Record and score one clip. Try it as yourself, then have someone else read the same line."
        >
          <TextInput
            label="Identity to verify against"
            value={verifyIdentity}
            onChange={setVerifyIdentity}
            placeholder="e.g. karthikeya"
          />

          {speakers.length > 0 && (
            <div style={{ display: "flex", gap: 7, flexWrap: "wrap", marginBottom: 14 }}>
              {speakers.map((speaker) => (
                <button
                  key={speaker.identity}
                  onClick={() => setVerifyIdentity(speaker.identity)}
                  style={{
                    padding: "5px 11px",
                    borderRadius: 999,
                    border: `1px solid ${
                      verifyIdentity === speaker.identity ? COLOURS.accent : COLOURS.border
                    }`,
                    background:
                      verifyIdentity === speaker.identity
                        ? `${COLOURS.accent}20`
                        : "transparent",
                    color:
                      verifyIdentity === speaker.identity ? COLOURS.accent : COLOURS.dim,
                    fontSize: 11.5,
                    cursor: "pointer",
                    fontFamily: "inherit",
                  }}
                >
                  {speaker.identity}
                </button>
              ))}
            </div>
          )}

          {recording ? (
            <Button variant="danger" onClick={() => stopRecording((blob) => runVerify(blob))}>
              Stop and verify ({elapsed.toFixed(1)}s)
            </Button>
          ) : (
            <Button onClick={startRecording} disabled={busy || speakers.length === 0}>
              Record and verify
            </Button>
          )}

          {speakers.length === 0 && (
            <Notice tone="info">Enrol someone first — there is nothing to verify against.</Notice>
          )}

          {verifyResult && (
            <div style={{ marginTop: 16 }}>
              <div
                style={{
                  padding: 14,
                  borderRadius: 10,
                  border: `1px solid ${decisionColour}55`,
                  background: `${decisionColour}12`,
                }}
              >
                <Badge colour={decisionColour}>{verifyResult.decision}</Badge>
                <div
                  style={{
                    fontSize: 30,
                    fontWeight: 800,
                    marginTop: 8,
                    fontVariantNumeric: "tabular-nums",
                  }}
                >
                  {verifyResult.similarity.toFixed(3)}
                  <span style={{ fontSize: 13, color: COLOURS.muted }}> similarity</span>
                </div>
              </div>

              <div style={{ marginTop: 12 }}>
                <Row
                  label="Match threshold"
                  value={String(thresholds.match ?? "—")}
                  hint="at or above this is MATCH"
                />
                <Row
                  label="No-match threshold"
                  value={String(thresholds.no_match ?? "—")}
                  hint="below this is NO_MATCH"
                />
                <Row
                  label="Calibration"
                  value={String(thresholds.version ?? "—")}
                />
              </div>

              {verifyResult.reasons?.length > 0 && (
                <ul
                  style={{
                    margin: "12px 0 0",
                    paddingLeft: 18,
                    fontSize: 12,
                    color: COLOURS.dim,
                    lineHeight: 1.65,
                  }}
                >
                  {verifyResult.reasons.map((reason, index) => (
                    <li key={index}>{reason}</li>
                  ))}
                </ul>
              )}

              <div style={{ fontSize: 11, color: COLOURS.muted, marginTop: 10 }}>
                Between the two thresholds the answer is UNCERTAIN, deliberately.
                A two-way split forces a guess on exactly the scores the system
                knows least about.
              </div>
            </div>
          )}
        </Panel>
      </div>

      {error && <Notice tone="error">{error}</Notice>}
      {success && <Notice tone="good">{success}</Notice>}

      <Panel
        title={`Enrolled voices (${speakers.length})`}
        subtitle="Only the embedding centroid is stored. No audio is kept."
      >
        {speakers.length === 0 ? (
          <div style={{ fontSize: 12.5, color: COLOURS.muted }}>
            Nobody enrolled yet.
          </div>
        ) : (
          speakers.map((speaker) => (
            <div
              key={speaker.identity}
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                gap: 12,
                padding: "9px 0",
                borderBottom: `1px solid ${COLOURS.border}44`,
              }}
            >
              <div>
                <div style={{ fontWeight: 700, fontSize: 13 }}>{speaker.identity}</div>
                <div style={{ fontSize: 11.5, color: COLOURS.muted, marginTop: 2 }}>
                  {speaker.samples} samples · consistency{" "}
                  {speaker.consistency.toFixed(3)}
                  {speaker.consistency < 0.6 && " · loosely consistent"}
                </div>
              </div>
              <Button
                variant="ghost"
                onClick={() => remove(speaker.identity)}
                style={{ padding: "5px 11px", fontSize: 11.5 }}
              >
                delete
              </Button>
            </div>
          ))
        )}
      </Panel>
    </div>
  );
}
