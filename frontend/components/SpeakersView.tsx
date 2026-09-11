"use client";

/**
 * Enrol a voice, verify against it, manage who is enrolled.
 *
 * Enrolment is the one genuinely multi-step flow in the product, so it gets a
 * visible progress state: how many samples you have, how many you need, and a
 * different prompt each time. Varying the words is not decoration — it makes
 * the prototype describe a person rather than one sentence, and the registry
 * refuses a set whose samples do not agree with each other.
 */

import {
  Check,
  Fingerprint,
  Mic,
  Square,
  Trash2,
  UserPlus,
} from "lucide-react";
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
import {
  Button,
  Chip,
  DataList,
  DataRow,
  Empty,
  Field,
  LevelMeter,
  Notice,
  Panel,
  Pill,
  Toast,
  signalForVerification,
  signalVar,
} from "./ui";
import { SimilarityScale } from "./viz";

const REQUIRED = 3;
const RECOMMENDED = 5;

const PROMPTS = [
  "My name is on this account and I am authorising this call.",
  "The quick brown fox jumps over the lazy dog near the river bank.",
  "Please transfer the balance to the account we discussed on Tuesday.",
  "I would like to confirm the details of my most recent transaction.",
  "Security is important to me, so I am happy to verify my identity.",
];

type Sample = { blob: Blob; seconds: number; url: string };

export default function SpeakersView({ onChange }: { onChange: () => void }) {
  const [speakers, setSpeakers] = useState<SpeakerSummary[]>([]);
  const [thresholds, setThresholds] = useState<Record<string, unknown>>({});
  const [loading, setLoading] = useState(true);

  const [identity, setIdentity] = useState("");
  const [samples, setSamples] = useState<Sample[]>([]);

  const [mode, setMode] = useState<"enrol" | "verify" | null>(null);
  const [level, setLevel] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const recorder = useRef(new MicRecorder());

  const [verifyAs, setVerifyAs] = useState("");
  const [verification, setVerification] = useState<Verification | null>(null);

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [toast, setToast] = useState<{ tone: "good" | "error"; message: string } | null>(null);

  async function refresh() {
    try {
      const data = await listSpeakers();
      setSpeakers(data.speakers);
      setThresholds(data.thresholds);
    } catch (exception) {
      setError((exception as Error).message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  useEffect(() => {
    if (!mode) return;
    const started = Date.now();
    const timer = setInterval(() => setElapsed((Date.now() - started) / 1000), 100);
    return () => clearInterval(timer);
  }, [mode]);

  /**
   * Object URLs are revoked on unmount only.
   *
   * Depending this effect on `samples` looked right and was a real bug: the
   * cleanup fired on every change and revoked every URL in the *previous*
   * array, including the ones still present in the new one — so recording a
   * second sample silently broke the first sample's player. A ref keeps the
   * current list available to a cleanup that runs exactly once.
   */
  const samplesRef = useRef<Sample[]>([]);
  useEffect(() => {
    samplesRef.current = samples;
  }, [samples]);
  useEffect(() => {
    return () => samplesRef.current.forEach((sample) => URL.revokeObjectURL(sample.url));
  }, []);

  async function record(next: "enrol" | "verify") {
    setError(null);
    try {
      recorder.current.onLevel = setLevel;
      await recorder.current.start();
      setMode(next);
      setElapsed(0);
    } catch (exception) {
      setError(microphoneError(exception));
    }
  }

  function finish(onDone: (blob: Blob, seconds: number) => void) {
    const captured = recorder.current.stop();
    setMode(null);
    setLevel(0);

    if (!captured) {
      setError("Nothing was captured. Check the microphone selection.");
      return;
    }
    if (captured.seconds < 1) {
      setError(`Only ${captured.seconds.toFixed(1)}s captured; at least one second is needed.`);
      return;
    }
    if (captured.peak < 0.01) {
      setError(
        "That recording is almost silent. Check the right microphone is selected and speak closer to it."
      );
      return;
    }
    onDone(captured.blob, captured.seconds);
  }

  async function enrol() {
    setBusy(true);
    setError(null);
    try {
      const profile = await enrollSpeaker(
        identity.trim(),
        samples.map((sample) => sample.blob)
      );
      setToast({
        tone: "good",
        message: `Enrolled "${profile.identity}" from ${profile.samples} samples — consistency ${profile.consistency.toFixed(3)}. Audio was discarded; only the embedding centroid is stored.`,
      });
      samples.forEach((sample) => URL.revokeObjectURL(sample.url));
      setSamples([]);
      setVerifyAs(profile.identity);
      setIdentity("");
      await refresh();
      onChange();
    } catch (exception) {
      setError((exception as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function runVerify(blob: Blob) {
    if (!verifyAs.trim()) {
      setError("Pick the identity to verify against.");
      return;
    }
    setBusy(true);
    setError(null);
    setVerification(null);
    try {
      setVerification(await verifySpeaker(verifyAs.trim(), blob));
    } catch (exception) {
      setError((exception as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function remove(name: string) {
    try {
      await deleteSpeaker(name);
      setToast({ tone: "good", message: `Deleted "${name}".` });
      if (verifyAs === name) {
        setVerifyAs("");
        setVerification(null);
      }
      await refresh();
      onChange();
    } catch (exception) {
      setToast({ tone: "error", message: (exception as Error).message });
    }
  }

  const enough = samples.length >= REQUIRED;
  const matchAt = Number(thresholds.match ?? 0.55);
  const noMatchAt = Number(thresholds.no_match ?? 0.35);

  return (
    <>
      <div className="grid grid-pair">
        {/* ------------------------------------------------------- enrolment */}
        <Panel
          title="Enrol a voice"
          aside={
            <span className="label">
              {samples.length}/{RECOMMENDED}
            </span>
          }
          className="rise"
          note="Only the embedding centroid is stored. Audio never reaches disk."
        >
          <div className="stack stack-4">
            <Field
              label="Identity"
              hint="Letters, digits, spaces and . _ @ - only, up to 128 characters."
            >
              <input
                className="input"
                value={identity}
                onChange={(event) => setIdentity(event.target.value)}
                placeholder="e.g. karthikeya"
                disabled={mode === "enrol"}
                aria-label="Identity to enrol"
              />
            </Field>

            {/* Sample progress. Reads at a glance without counting rows. */}
            <div className="row" style={{ gap: "var(--s1)" }}>
              {Array.from({ length: RECOMMENDED }, (_, index) => {
                const done = index < samples.length;
                const required = index < REQUIRED;
                return (
                  <div
                    key={index}
                    title={
                      done
                        ? `Sample ${index + 1} recorded`
                        : required
                        ? "Required"
                        : "Recommended"
                    }
                    style={{
                      flex: 1,
                      height: 3,
                      borderRadius: 2,
                      background: done
                        ? "var(--sig-clear)"
                        : required
                        ? "var(--line-strong)"
                        : "var(--line-soft)",
                      transition: "background var(--dur-2) var(--ease)",
                    }}
                  />
                );
              })}
            </div>

            <div
              style={{
                padding: "var(--s4)",
                borderRadius: "var(--r-md)",
                background: "var(--surface-inset)",
                border: "1px solid var(--line-soft)",
              }}
            >
              <div className="label" style={{ marginBottom: 6 }}>
                Read this one
              </div>
              <p style={{ fontSize: "var(--t-base)", lineHeight: 1.5 }}>
                {PROMPTS[samples.length % PROMPTS.length]}
              </p>
            </div>

            {mode === "enrol" ? (
              <div className="stack stack-3">
                <div className="row">
                  <span className="live-dot" />
                  <span className="mono" style={{ fontSize: "var(--t-md)", fontWeight: 600 }}>
                    {elapsed.toFixed(1)}s
                  </span>
                  <div className="spacer" style={{ width: 130 }}>
                    <LevelMeter level={level} segments={16} />
                  </div>
                </div>
                <Button
                  variant="danger"
                  block
                  icon={<Square size={14} />}
                  onClick={() =>
                    finish((blob, seconds) =>
                      setSamples((current) => [
                        ...current,
                        { blob, seconds, url: URL.createObjectURL(blob) },
                      ])
                    )
                  }
                >
                  Stop
                </Button>
              </div>
            ) : (
              <Button
                block
                icon={<Mic size={14} />}
                onClick={() => record("enrol")}
                disabled={busy || !identity.trim()}
              >
                Record sample {samples.length + 1}
              </Button>
            )}

            {samples.length > 0 && (
              <div className="stack stack-2">
                {samples.map((sample, index) => (
                  <div className="sample" key={sample.url}>
                    <span className="sample-n">{index + 1}</span>
                    <audio controls src={sample.url} />
                    <span className="mono dim" style={{ fontSize: "var(--t-xs)" }}>
                      {sample.seconds.toFixed(1)}s
                    </span>
                    <Button
                      variant="ghost"
                      size="sm"
                      ariaLabel={`Remove sample ${index + 1}`}
                      onClick={() => {
                        URL.revokeObjectURL(sample.url);
                        setSamples((current) => current.filter((_, i) => i !== index));
                      }}
                    >
                      <Trash2 size={12} />
                    </Button>
                  </div>
                ))}
              </div>
            )}

            <Button
              variant="primary"
              block
              icon={<UserPlus size={14} />}
              onClick={enrol}
              disabled={busy || !enough || !identity.trim() || mode === "enrol"}
            >
              {enough
                ? `Enrol ${samples.length} sample${samples.length === 1 ? "" : "s"}`
                : `${REQUIRED - samples.length} more sample${REQUIRED - samples.length === 1 ? "" : "s"} needed`}
            </Button>
          </div>
        </Panel>

        {/* --------------------------------------------------------- verify */}
        <Panel
          title="Verify against an enrolled voice"
          className="rise rise-1"
          note="Try it as yourself, then have someone else read the same line. The gap between the two is the whole point."
        >
          <div className="stack stack-4">
            {speakers.length === 0 ? (
              <Empty icon={<Fingerprint size={18} />} title="Nothing to verify against">
                Enrol someone first.
              </Empty>
            ) : (
              <>
                <Field label="Identity">
                  <div className="chipset">
                    {speakers.map((speaker) => (
                      <Chip
                        key={speaker.identity}
                        active={verifyAs === speaker.identity}
                        disabled={mode === "verify"}
                        onClick={() => setVerifyAs(speaker.identity)}
                      >
                        {speaker.identity}
                      </Chip>
                    ))}
                  </div>
                </Field>

                {mode === "verify" ? (
                  <div className="stack stack-3">
                    <div className="row">
                      <span className="live-dot" />
                      <span className="mono" style={{ fontSize: "var(--t-md)", fontWeight: 600 }}>
                        {elapsed.toFixed(1)}s
                      </span>
                      <div className="spacer" style={{ width: 130 }}>
                        <LevelMeter level={level} segments={16} />
                      </div>
                    </div>
                    <Button
                      variant="danger"
                      block
                      icon={<Square size={14} />}
                      onClick={() => finish((blob) => runVerify(blob))}
                    >
                      Stop and verify
                    </Button>
                  </div>
                ) : (
                  <Button
                    block
                    icon={<Mic size={14} />}
                    onClick={() => record("verify")}
                    disabled={busy || !verifyAs}
                  >
                    {busy ? "Verifying…" : "Record and verify"}
                  </Button>
                )}

                {verification && (
                  <div className="stack stack-4 fade">
                    <div
                      style={{
                        padding: "var(--s4)",
                        borderRadius: "var(--r-md)",
                        border: `1px solid var(--sig-${signalForVerification(verification.decision)}-line)`,
                        background: `var(--sig-${signalForVerification(verification.decision)}-bg)`,
                      }}
                    >
                      <Pill signal={signalForVerification(verification.decision)}>
                        {verification.decision.replace("_", " ")}
                      </Pill>
                      <div
                        className="mono"
                        style={{
                          fontSize: "var(--t-3xl)",
                          fontWeight: 600,
                          letterSpacing: "-.035em",
                          marginTop: "var(--s3)",
                          color: signalVar(signalForVerification(verification.decision)),
                        }}
                      >
                        {verification.similarity.toFixed(3)}
                        <span
                          style={{
                            fontSize: "var(--t-sm)",
                            color: "var(--text-3)",
                            fontWeight: 500,
                            letterSpacing: 0,
                          }}
                        >
                          {" "}
                          similarity
                        </span>
                      </div>
                    </div>

                    <SimilarityScale
                      similarity={verification.similarity}
                      matchAt={matchAt}
                      noMatchAt={noMatchAt}
                    />

                    <DataList>
                      <DataRow label="Match at" value={matchAt.toFixed(4)} />
                      <DataRow label="No match below" value={noMatchAt.toFixed(4)} />
                      <DataRow
                        label="Calibration"
                        value={String(thresholds.version ?? "—")}
                      />
                    </DataList>

                    {verification.reasons?.length > 0 && (
                      <ul
                        style={{
                          paddingLeft: 16,
                          fontSize: "var(--t-sm)",
                          color: "var(--text-2)",
                          lineHeight: 1.6,
                        }}
                      >
                        {verification.reasons.map((reason, index) => (
                          <li key={index} style={{ marginBottom: 4 }}>
                            {reason}
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                )}
              </>
            )}
          </div>
        </Panel>
      </div>

      {error && (
        <div style={{ marginTop: "var(--s4)" }}>
          <Notice tone="error">{error}</Notice>
        </div>
      )}

      {/* ---------------------------------------------------------- roster */}
      <div style={{ marginTop: "var(--s5)" }}>
        <Panel
          title="Enrolled voices"
          aside={<span className="label">{speakers.length} enrolled</span>}
          className="rise rise-2"
        >
          {loading ? (
            <div className="stack stack-2">
              {[0, 1].map((index) => (
                <div key={index} className="skeleton" style={{ height: 44 }} />
              ))}
            </div>
          ) : speakers.length === 0 ? (
            <Empty icon={<UserPlus size={18} />} title="Nobody enrolled yet">
              Record five samples on the left to create the first profile.
            </Empty>
          ) : (
            <div className="stack stack-2">
              {speakers.map((speaker) => {
                const weak = speaker.consistency < 0.6;
                return (
                  <div
                    key={speaker.identity}
                    className="row"
                    style={{
                      padding: "var(--s3) var(--s4)",
                      borderRadius: "var(--r-md)",
                      border: "1px solid var(--line-soft)",
                      background: "var(--surface-2)",
                      gap: "var(--s4)",
                    }}
                  >
                    <span
                      style={{
                        width: 28,
                        height: 28,
                        flex: "none",
                        borderRadius: "var(--r-sm)",
                        display: "grid",
                        placeItems: "center",
                        background: "var(--surface-3)",
                        color: "var(--text-3)",
                      }}
                    >
                      <Check size={13} />
                    </span>

                    <div style={{ minWidth: 0 }}>
                      <div style={{ fontWeight: 600, fontSize: "var(--t-md)" }}>
                        {speaker.identity}
                      </div>
                      <div className="mono dim" style={{ fontSize: "var(--t-xs)" }}>
                        {speaker.samples} samples · consistency {speaker.consistency.toFixed(3)}
                      </div>
                    </div>

                    <div className="spacer row" style={{ gap: "var(--s2)" }}>
                      {weak && <Pill signal="watch">Loosely consistent</Pill>}
                      <Button
                        variant="ghost"
                        size="sm"
                        ariaLabel={`Delete ${speaker.identity}`}
                        onClick={() => remove(speaker.identity)}
                      >
                        <Trash2 size={13} />
                      </Button>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </Panel>
      </div>

      {toast && (
        <Toast tone={toast.tone} message={toast.message} onDismiss={() => setToast(null)} />
      )}
    </>
  );
}
