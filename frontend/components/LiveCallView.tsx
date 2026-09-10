"use client";

/**
 * The live call.
 *
 * What this screen shows that the file view cannot is *movement*. Risk builds
 * over a call — a caller who drifts upward over ninety seconds looks nothing
 * like one who spikes on a single bad window — so the trace is the primary
 * readout here and the current number is secondary to it.
 *
 * Identity and context are set before the call opens and locked once it is
 * running, mirroring the backend: letting either change mid-stream would let a
 * caller re-aim the identity check at whichever profile happened to match.
 */

import {
  Activity,
  Fingerprint,
  PhoneCall,
  PhoneOff,
  Radio,
  Waves,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";

import {
  API_URL,
  ContextAssessment,
  FusedRisk,
  PolicyDecision,
  SystemStatus,
  Verification,
  WS_URL,
  createCall,
  formatProbability,
  listSpeakers,
  prettyDecision,
} from "../lib/api";
import { microphoneError } from "../lib/recorder";
import {
  AnimatedNumber,
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
  Signal,
  signalForDecision,
  signalForLevel,
  signalForVerification,
  signalVar,
} from "./ui";
import { RiskScale, SignalTrace } from "./viz";

interface Analysis {
  vad: { speech: boolean; confidence: number; rms: number };
  stream: {
    total_audio_ms: number;
    speech_audio_ms: number;
    speech_ratio: number;
    chunks: number;
    speaker_branch: boolean;
    claimed_identity: string | null;
  };
  deepfake?: {
    available: boolean;
    windows_scored: number;
    buffered_seconds: number;
    result?: {
      synthetic_probability: number | null;
      smoothed_probability: number;
      model_status: string;
      reasons: string[];
    } | null;
  };
  speaker?: Verification | null;
  context?: ContextAssessment | null;
  risk?: FusedRisk | null;
  policy?: PolicyDecision | null;
}

const CONTEXT_PRESETS: { id: string; label: string; context: object }[] = [
  { id: "none", label: "No context", context: {} },
  {
    id: "routine",
    label: "Routine call",
    context: {
      caller_known: true,
      trusted_contact: true,
      device_known: true,
      beneficiary_known: true,
      hour_of_day: 14,
    },
  },
  {
    id: "suspicious",
    label: "Suspicious call",
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
];

export default function LiveCallView({ status }: { status: SystemStatus | null }) {
  const [active, setActive] = useState(false);
  const [connected, setConnected] = useState(false);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [history, setHistory] = useState<number[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [level, setLevel] = useState(0);
  const [seconds, setSeconds] = useState(0);

  const [identity, setIdentity] = useState("");
  const [preset, setPreset] = useState("none");
  const [enrolled, setEnrolled] = useState<string[]>([]);

  const socketRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);

  useEffect(() => {
    listSpeakers()
      .then((data) => setEnrolled(data.speakers.map((entry) => entry.identity)))
      .catch(() => setEnrolled([]));
  }, [status?.speaker.enrolled]);

  useEffect(() => () => stop(), []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!active) return;
    const started = Date.now();
    const timer = setInterval(() => setSeconds((Date.now() - started) / 1000), 250);
    return () => clearInterval(timer);
  }, [active]);

  function stop() {
    processorRef.current?.disconnect();
    processorRef.current = null;

    audioRef.current?.close().catch(() => {});
    audioRef.current = null;

    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;

    if (socketRef.current?.readyState === WebSocket.OPEN) socketRef.current.close();
    socketRef.current = null;

    setConnected(false);
    setLevel(0);
  }

  async function start() {
    setError(null);
    setAnalysis(null);
    setHistory([]);
    setSeconds(0);

    try {
      const { call_id } = await createCall();

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true,
          // Off deliberately: AGC flattens the difference between the loud and
          // quiet parts of speech, and the backend measures exactly that
          // dynamic range to tell speech from steady noise.
          autoGainControl: false,
        },
      });
      streamRef.current = stream;

      const query = new URLSearchParams();
      if (identity.trim()) query.set("claimed_identity", identity.trim());
      const context = CONTEXT_PRESETS.find((entry) => entry.id === preset)?.context ?? {};
      if (Object.keys(context).length > 0) query.set("context", JSON.stringify(context));

      const suffix = query.toString() ? `?${query}` : "";
      const socket = new WebSocket(`${WS_URL}/api/v1/calls/${call_id}/stream${suffix}`);
      socket.binaryType = "arraybuffer";

      socket.onopen = () => {
        setConnected(true);
        capture(stream, socket);
      };

      socket.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          if (message.type !== "voice_analysis" || !message.analysis) return;

          const next: Analysis = message.analysis;
          setAnalysis(next);

          if (next.risk) {
            setHistory((current) => {
              const score = next.risk!.risk_score;
              if (current.length && current[current.length - 1] === score) return current;
              // Two hundred points is roughly three minutes at one per second,
              // which is longer than any call anyone demos.
              return [...current, score].slice(-200);
            });
          }
        } catch {
          // A malformed frame must not end the call.
        }
      };

      socket.onerror = () =>
        setError(`Could not reach the backend at ${API_URL}. Check the System tab.`);

      socket.onclose = (event) => {
        setConnected(false);
        // 1008 is the policy-violation close the backend uses to reject a bad
        // context before accepting, with the reason attached.
        if (event.code === 1008 && event.reason) {
          setError(event.reason);
          setActive(false);
        }
      };

      socketRef.current = socket;
    } catch (exception) {
      setError(microphoneError(exception));
      setActive(false);
      stop();
    }
  }

  function capture(stream: MediaStream, socket: WebSocket) {
    const context = new AudioContext({ sampleRate: 16000 });
    audioRef.current = context;

    const source = context.createMediaStreamSource(stream);
    const processor = context.createScriptProcessor(4096, 1, 1);
    processorRef.current = processor;

    processor.onaudioprocess = (event) => {
      if (socket.readyState !== WebSocket.OPEN) return;

      const input = event.inputBuffer.getChannelData(0);
      const pcm = new Int16Array(input.length);

      let peak = 0;
      for (let i = 0; i < input.length; i++) {
        const sample = Math.max(-1, Math.min(1, input[i]));
        pcm[i] = sample < 0 ? sample * 32768 : sample * 32767;
        if (Math.abs(sample) > peak) peak = Math.abs(sample);
      }

      setLevel(peak);
      socket.send(pcm.buffer);
    };

    source.connect(processor);

    // Through a silent gain node: ScriptProcessorNode only fires while
    // connected to a destination, and routing it to the speakers would echo
    // the microphone back into the room.
    const silent = context.createGain();
    silent.gain.value = 0;
    processor.connect(silent);
    silent.connect(context.destination);
  }

  function toggle() {
    if (active) {
      stop();
      setActive(false);
    } else {
      setActive(true);
      start();
    }
  }

  const risk = analysis?.risk ?? null;
  const deepfake = analysis?.deepfake?.result ?? null;
  const buffered = analysis?.deepfake?.buffered_seconds ?? 0;
  const insufficient = risk?.decision === "INSUFFICIENT_EVIDENCE";
  const signal: Signal = risk
    ? insufficient
      ? "unknown"
      : signalForLevel(risk.risk_level)
    : "neutral";

  return (
    <div className="stack stack-5">
      {/* ------------------------------------------------------------ setup */}
      <Panel
        title="Call setup"
        aside={
          <span className="label">
            {active ? "Locked for the call" : "Fixed once the call starts"}
          </span>
        }
        className="rise"
      >
        <div className="grid grid-2" style={{ alignItems: "start" }}>
          <div className="stack stack-3">
            <Field label="Claimed identity">
              <input
                className="input"
                value={identity}
                onChange={(event) => setIdentity(event.target.value)}
                placeholder="anti-spoof only"
                disabled={active}
                aria-label="Claimed identity"
              />
            </Field>
            {enrolled.length > 0 && (
              <div className="chipset">
                {enrolled.map((name) => (
                  <Chip
                    key={name}
                    active={identity === name}
                    disabled={active}
                    onClick={() => setIdentity(identity === name ? "" : name)}
                  >
                    {name}
                  </Chip>
                ))}
              </div>
            )}
          </div>

          <div className="stack stack-3">
            <span className="label">Call context</span>
            <div className="chipset">
              {CONTEXT_PRESETS.map((entry) => (
                <Chip
                  key={entry.id}
                  active={preset === entry.id}
                  disabled={active}
                  onClick={() => setPreset(entry.id)}
                >
                  {entry.label}
                </Chip>
              ))}
            </div>
          </div>
        </div>

        <div className="row row-wrap" style={{ marginTop: "var(--s5)", gap: "var(--s4)" }}>
          <Button
            variant={active ? "danger" : "primary"}
            size="lg"
            onClick={toggle}
            icon={active ? <PhoneOff size={15} /> : <PhoneCall size={15} />}
          >
            {active ? "End call" : "Start live call"}
          </Button>

          {active && (
            <>
              <div className="row" style={{ gap: "var(--s2)" }}>
                {connected ? (
                  <>
                    <span className="live-dot" />
                    <span className="label" style={{ color: "var(--sig-critical)" }}>
                      Live
                    </span>
                  </>
                ) : (
                  <span className="label">Connecting…</span>
                )}
              </div>
              <span className="mono dim" style={{ fontSize: "var(--t-md)" }}>
                {Math.floor(seconds / 60)}:{String(Math.floor(seconds % 60)).padStart(2, "0")}
              </span>
              <div style={{ flex: 1, minWidth: 140, maxWidth: 260 }}>
                <LevelMeter level={level} />
              </div>
            </>
          )}
        </div>

        {error && (
          <div style={{ marginTop: "var(--s4)" }}>
            <Notice tone="error">{error}</Notice>
          </div>
        )}
      </Panel>

      {!active && !analysis && (
        <Panel className="rise rise-1">
          <Empty icon={<Radio size={18} />} title="No call running">
            Start a call and keep talking. The first score arrives once three seconds of
            speech have accumulated, then updates about once a second.
          </Empty>
        </Panel>
      )}

      {analysis && (
        <>
          {/* ------------------------------------------------------- trace */}
          <Panel
            title="Risk over the call"
            aside={
              <span className="label">
                {analysis.deepfake?.windows_scored ?? 0} windows scored
              </span>
            }
            className="rise rise-1"
            note="What matters live is not the current number but whether it is climbing. Dashed rules mark the decision bands at 40, 60 and 80."
          >
            <div className="stack stack-4">
              {risk ? (
                <div
                  className="row"
                  style={{ alignItems: "flex-end", gap: "var(--s6)" }}
                  aria-live="polite"
                  aria-atomic="true"
                >
                  <div>
                    <div className="label" style={{ marginBottom: 4 }}>
                      Current
                    </div>
                    <div
                      className="mono"
                      style={{
                        fontSize: "var(--t-4xl)",
                        fontWeight: 600,
                        letterSpacing: "-.04em",
                        lineHeight: 1,
                        color: signalVar(signal),
                      }}
                    >
                      {insufficient ? "—" : <AnimatedNumber value={risk.risk_score} />}
                    </div>
                  </div>
                  <div className="stack stack-2" style={{ paddingBottom: 4 }}>
                    <Pill signal={signal}>{insufficient ? "Unknown" : risk.risk_level}</Pill>
                    <Pill signal={signalForDecision(risk.decision)}>
                      {prettyDecision(risk.decision)}
                    </Pill>
                  </div>
                </div>
              ) : (
                <div className="stack stack-2">
                  <div className="label">Buffering</div>
                  <div className="row" style={{ gap: "var(--s3)" }}>
                    <div
                      style={{
                        flex: 1,
                        height: 4,
                        borderRadius: 2,
                        background: "var(--surface-3)",
                        overflow: "hidden",
                      }}
                    >
                      <div
                        style={{
                          width: `${Math.min((buffered / 3) * 100, 100)}%`,
                          height: "100%",
                          background: "var(--brand)",
                          transition: "width var(--dur-2) var(--ease)",
                        }}
                      />
                    </div>
                    <span className="mono dim" style={{ fontSize: "var(--t-sm)" }}>
                      {buffered.toFixed(1)}s / 3.0s
                    </span>
                  </div>
                </div>
              )}

              <SignalTrace history={history} />

              {risk && <RiskScale score={risk.risk_score} signal={signal} unknown={insufficient} />}
            </div>
          </Panel>

          {/* ------------------------------------------------------ branches */}
          <div className="grid grid-3 rise rise-2">
            <Panel title="Anti-spoof">
              {deepfake && deepfake.model_status === "neural" ? (
                <DataList>
                  <DataRow
                    label="This window"
                    value={formatProbability(deepfake.synthetic_probability)}
                  />
                  <DataRow
                    label="Smoothed"
                    qualifier="across the call"
                    value={formatProbability(deepfake.smoothed_probability)}
                  />
                  <DataRow
                    label="Windows"
                    value={analysis.deepfake?.windows_scored ?? 0}
                  />
                </DataList>
              ) : deepfake?.model_status === "no_speech" ? (
                <div className="stack stack-3">
                  <Pill signal="unknown">Not scored</Pill>
                  <p className="prose" style={{ fontSize: "var(--t-sm)" }}>
                    {deepfake.reasons?.[0] ?? "No speech in this window."}
                  </p>
                  <p className="field-hint">
                    The standing risk is held rather than dragged toward zero — a pause
                    is not evidence of innocence.
                  </p>
                </div>
              ) : (
                <Empty icon={<Waves size={16} />} title="Buffering">
                  {buffered.toFixed(1)}s of the 3.0s window.
                </Empty>
              )}
            </Panel>

            <Panel title="Identity">
              {analysis.speaker ? (
                <div className="stack stack-3">
                  <Pill signal={signalForVerification(analysis.speaker.decision)}>
                    {analysis.speaker.decision.replace("_", " ")}
                  </Pill>
                  <DataList>
                    <DataRow label="Claimed" value={analysis.speaker.identity} />
                    <DataRow
                      label="Similarity"
                      value={analysis.speaker.similarity.toFixed(3)}
                    />
                    {analysis.speaker.speech_seconds_analysed !== undefined && (
                      <DataRow
                        label="Speech used"
                        qualifier="improves"
                        value={`${analysis.speaker.speech_seconds_analysed.toFixed(1)}s`}
                      />
                    )}
                  </DataList>
                </div>
              ) : analysis.stream.speaker_branch ? (
                <Empty icon={<Fingerprint size={16} />} title="Accumulating">
                  Building an identity estimate from the speech so far.
                </Empty>
              ) : (
                <Empty icon={<Fingerprint size={16} />} title="Branch off">
                  {analysis.stream.claimed_identity
                    ? `No enrolled profile for "${analysis.stream.claimed_identity}".`
                    : "No identity claimed, so this is anti-spoof only."}
                </Empty>
              )}
            </Panel>

            <Panel title="Stream">
              <DataList>
                <DataRow
                  label="Voice activity"
                  value={
                    <span style={{ color: analysis.vad.speech ? "var(--sig-clear)" : "var(--text-3)" }}>
                      {analysis.vad.speech ? "speech" : "silence"}
                    </span>
                  }
                />
                <DataRow
                  label="Speech ratio"
                  value={`${(analysis.stream.speech_ratio * 100).toFixed(0)}%`}
                />
                <DataRow
                  label="Audio received"
                  value={`${(analysis.stream.total_audio_ms / 1000).toFixed(1)}s`}
                />
                <DataRow label="Chunks" qualifier="256 ms" value={analysis.stream.chunks} />
              </DataList>
            </Panel>
          </div>

          {/* -------------------------------------------------------- reasons */}
          {risk && risk.reasons.length > 0 && (
            <Panel title="Reasons" className="rise rise-3">
              <ul className="stack stack-2" style={{ listStyle: "none" }}>
                {risk.reasons.map((reason, index) => (
                  <li
                    key={index}
                    style={{
                      display: "flex",
                      gap: "var(--s2)",
                      fontSize: "var(--t-md)",
                      lineHeight: 1.55,
                    }}
                  >
                    <Activity
                      size={13}
                      style={{ color: signalVar(signal), flex: "none", marginTop: 4 }}
                    />
                    {reason}
                  </li>
                ))}
              </ul>

              {risk.missing_signals.length > 0 && (
                <div style={{ marginTop: "var(--s4)" }}>
                  <Notice tone="unknown">
                    <strong>Not assessed:</strong>{" "}
                    {risk.missing_signals.join(", ").replace(/_/g, " ")}
                  </Notice>
                </div>
              )}
            </Panel>
          )}

          {/* --------------------------------------------------------- policy */}
          {analysis.policy && (
            <Panel
              title="Policy"
              aside={
                analysis.policy.requires_human ? (
                  <Pill signal="verify">Human required</Pill>
                ) : (
                  <Pill signal="neutral">Automatic</Pill>
                )
              }
              className="rise rise-4"
            >
              <div className="chipset">
                {analysis.policy.actions.map((action) => (
                  <span
                    key={action.code}
                    className={`pill ${action.blocking ? "pill-verify" : "pill-neutral"}`}
                    title={action.description}
                  >
                    {action.code.replace(/_/g, " ")}
                  </span>
                ))}
              </div>
            </Panel>
          )}
        </>
      )}
    </div>
  );
}
