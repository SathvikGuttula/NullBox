"use client";

/**
 * The live call: microphone in, fused risk and policy actions out, once a
 * second.
 *
 * Two things this shows that the file-analysis screen cannot. Risk builds over
 * a call rather than arriving at once - the score is smoothed across windows,
 * so a single odd window does not fire the whole system. And the identity
 * estimate improves as speech accumulates, because the tracker keeps a running
 * centroid rather than averaging per-window similarities.
 *
 * The stream sends 16-bit PCM at 16 kHz in 4096-sample chunks, which is 256 ms.
 * That is exactly what the verification script replays, so a bug reproduced
 * here reproduces there.
 */

import { useEffect, useRef, useState } from "react";

import {
  API_URL,
  WS_URL,
  ContextAssessment,
  FusedRisk,
  PolicyDecision,
  Verification,
  createCall,
  decisionColour,
  formatProbability,
  listSpeakers,
  prettyDecision,
  riskColour,
} from "../lib/api";
import { microphoneError } from "../lib/recorder";
import { Badge, Button, COLOURS, LevelMeter, Notice, Panel, Row, TextInput } from "./ui";

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

const CONTEXT_PRESETS: Record<string, object> = {
  none: {},
  routine: {
    caller_known: true,
    trusted_contact: true,
    device_known: true,
    beneficiary_known: true,
    hour_of_day: 14,
  },
  suspicious: {
    caller_known: false,
    trusted_contact: false,
    device_known: false,
    hour_of_day: 23,
    transaction_amount: 250000,
    typical_transaction_amount: 2000,
    beneficiary_known: false,
    beneficiary_age_days: 0,
  },
};

export default function LiveCall() {
  const [active, setActive] = useState(false);
  const [connected, setConnected] = useState(false);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [level, setLevel] = useState(0);

  const [identity, setIdentity] = useState("");
  const [preset, setPreset] = useState("none");
  const [enrolled, setEnrolled] = useState<string[]>([]);

  const socketRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const contextRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);

  useEffect(() => {
    listSpeakers()
      .then((data) => setEnrolled(data.speakers.map((s) => s.identity)))
      .catch(() => setEnrolled([]));
  }, []);

  useEffect(() => {
    return () => stop();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function stop() {
    processorRef.current?.disconnect();
    processorRef.current = null;

    contextRef.current?.close().catch(() => {});
    contextRef.current = null;

    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;

    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.close();
    }
    socketRef.current = null;

    setConnected(false);
    setLevel(0);
  }

  async function start() {
    setError(null);
    setAnalysis(null);

    try {
      const { call_id } = await createCall();

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true,
          // Off deliberately: automatic gain control flattens the difference
          // between the loud and quiet parts of speech, and that difference is
          // exactly what the backend measures to tell speech from steady noise.
          autoGainControl: false,
        },
      });
      streamRef.current = stream;

      const query = new URLSearchParams();
      if (identity.trim()) query.set("claimed_identity", identity.trim());

      const context = CONTEXT_PRESETS[preset];
      if (Object.keys(context).length > 0) {
        query.set("context", JSON.stringify(context));
      }

      const suffix = query.toString() ? `?${query}` : "";
      const socket = new WebSocket(
        `${WS_URL}/api/v1/calls/${call_id}/stream${suffix}`
      );
      socket.binaryType = "arraybuffer";

      socket.onopen = () => {
        setConnected(true);
        capture(stream, socket);
      };

      socket.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          if (message.type === "voice_analysis" && message.analysis) {
            setAnalysis(message.analysis);
          }
        } catch {
          // A malformed frame should not end the call.
        }
      };

      socket.onerror = () =>
        setError(
          `Could not reach the backend at ${API_URL}. Is it running? Check the System tab.`
        );

      socket.onclose = (event) => {
        setConnected(false);
        // 1008 is the policy-violation close the backend uses to reject a bad
        // context before accepting the socket, with the reason attached.
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
    const audio = new AudioContext({ sampleRate: 16000 });
    contextRef.current = audio;

    const source = audio.createMediaStreamSource(stream);
    const processor = audio.createScriptProcessor(4096, 1, 1);
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

    // Routed through a silent gain node: ScriptProcessorNode only fires while
    // connected to a destination, and connecting it to the speakers would echo
    // the microphone back into the room.
    const silent = audio.createGain();
    silent.gain.value = 0;
    processor.connect(silent);
    silent.connect(audio.destination);
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

  const risk = analysis?.risk;
  const deepfake = analysis?.deepfake?.result;

  return (
    <div style={{ display: "grid", gap: 16 }}>
      <Panel title="Call setup" subtitle="Both are fixed for the life of the call.">
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
            gap: 16,
          }}
        >
          <div>
            <TextInput
              label="Claimed identity"
              value={identity}
              onChange={setIdentity}
              placeholder="leave empty for anti-spoof only"
            />
            {enrolled.length > 0 && (
              <div style={{ display: "flex", gap: 7, flexWrap: "wrap" }}>
                {enrolled.map((name) => (
                  <button
                    key={name}
                    onClick={() => setIdentity(name)}
                    disabled={active}
                    style={{
                      padding: "5px 11px",
                      borderRadius: 999,
                      border: `1px solid ${
                        identity === name ? COLOURS.accent : COLOURS.border
                      }`,
                      background: identity === name ? `${COLOURS.accent}20` : "transparent",
                      color: identity === name ? COLOURS.accent : COLOURS.dim,
                      fontSize: 11.5,
                      cursor: active ? "not-allowed" : "pointer",
                      fontFamily: "inherit",
                    }}
                  >
                    {name}
                  </button>
                ))}
              </div>
            )}
          </div>

          <div>
            <div
              style={{
                fontSize: 11,
                letterSpacing: 1,
                color: COLOURS.muted,
                marginBottom: 6,
                textTransform: "uppercase",
              }}
            >
              Call context
            </div>
            <div style={{ display: "flex", gap: 7, flexWrap: "wrap" }}>
              {[
                ["none", "None"],
                ["routine", "Routine call"],
                ["suspicious", "Suspicious call"],
              ].map(([id, label]) => (
                <button
                  key={id}
                  onClick={() => setPreset(id)}
                  disabled={active}
                  style={{
                    padding: "8px 13px",
                    borderRadius: 9,
                    border: `1px solid ${preset === id ? COLOURS.accent : COLOURS.border}`,
                    background: preset === id ? `${COLOURS.accent}18` : "transparent",
                    color: preset === id ? COLOURS.accent : COLOURS.dim,
                    fontSize: 12,
                    cursor: active ? "not-allowed" : "pointer",
                    fontFamily: "inherit",
                    fontWeight: 600,
                  }}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div style={{ marginTop: 16, display: "flex", gap: 12, alignItems: "center" }}>
          <Button variant={active ? "danger" : "primary"} onClick={toggle}>
            {active ? "End call" : "Start live call"}
          </Button>
          <span style={{ display: "flex", alignItems: "center", gap: 7, fontSize: 12 }}>
            <span
              style={{
                width: 9,
                height: 9,
                borderRadius: "50%",
                background: connected ? COLOURS.good : COLOURS.border,
              }}
            />
            {connected ? "connected" : "offline"}
          </span>
        </div>

        {active && (
          <div style={{ marginTop: 14 }}>
            <LevelMeter level={level} />
            <div style={{ fontSize: 11.5, color: COLOURS.muted, marginTop: 7 }}>
              Keep talking. The first score arrives once three seconds of speech
              have accumulated, then updates about once a second.
            </div>
          </div>
        )}
      </Panel>

      {error && <Notice tone="error">{error}</Notice>}

      {analysis && (
        <>
          {risk ? (
            <Panel style={{ borderColor: `${decisionColour(risk.decision)}55` }}>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "flex-start",
                  gap: 16,
                  flexWrap: "wrap",
                }}
              >
                <div>
                  <div style={{ fontSize: 11, letterSpacing: 1.2, color: COLOURS.muted }}>
                    LIVE RISK
                  </div>
                  <div
                    style={{
                      fontSize: 44,
                      fontWeight: 800,
                      lineHeight: 1.1,
                      marginTop: 4,
                      color: riskColour(risk.risk_level),
                    }}
                  >
                    {risk.risk_score.toFixed(0)}
                    <span style={{ fontSize: 16, color: COLOURS.muted }}> / 100</span>
                  </div>
                  <div style={{ marginTop: 8, display: "flex", gap: 8 }}>
                    <Badge colour={riskColour(risk.risk_level)}>{risk.risk_level}</Badge>
                    <Badge colour={decisionColour(risk.decision)}>
                      {prettyDecision(risk.decision)}
                    </Badge>
                  </div>
                </div>

                <div style={{ textAlign: "right", fontSize: 12, color: COLOURS.muted }}>
                  <div>{analysis.deepfake?.windows_scored ?? 0} windows scored</div>
                  <div style={{ marginTop: 3 }}>
                    {(analysis.stream.speech_audio_ms / 1000).toFixed(1)}s of speech
                  </div>
                </div>
              </div>

              {risk.reasons.length > 0 && (
                <ul
                  style={{
                    margin: "14px 0 0",
                    paddingLeft: 18,
                    fontSize: 12.5,
                    color: COLOURS.text,
                    lineHeight: 1.7,
                  }}
                >
                  {risk.reasons.map((reason, index) => (
                    <li key={index}>{reason}</li>
                  ))}
                </ul>
              )}

              {risk.missing_signals.length > 0 && (
                <div style={{ marginTop: 12, fontSize: 11.5, color: COLOURS.muted }}>
                  <strong>Not assessed:</strong>{" "}
                  {risk.missing_signals.join(", ").replace(/_/g, " ")}
                </div>
              )}

              {!risk.calibrated && (
                <Notice tone="warn">
                  Fusion weights are uncalibrated placeholders — advisory only.
                </Notice>
              )}
            </Panel>
          ) : (
            <Panel title="Live risk">
              <div style={{ fontSize: 13, color: COLOURS.dim }}>
                Waiting for enough speech. The detector needs a three-second
                window before it scores anything.
              </div>
            </Panel>
          )}

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
              gap: 16,
            }}
          >
            <Panel title="Anti-spoof">
              {deepfake && deepfake.model_status === "neural" ? (
                <>
                  <Row
                    label="This window"
                    value={formatProbability(deepfake.synthetic_probability)}
                  />
                  <Row
                    label="Smoothed"
                    value={formatProbability(deepfake.smoothed_probability)}
                    hint="across the call"
                  />
                  <Row label="Windows scored" value={analysis.deepfake?.windows_scored ?? 0} />
                </>
              ) : deepfake?.model_status === "no_speech" ? (
                <>
                  <Badge colour={COLOURS.unknown}>NOT SCORED</Badge>
                  <div style={{ fontSize: 12.5, color: COLOURS.dim, marginTop: 10 }}>
                    {deepfake.reasons?.[0] ?? "No speech in this window."}
                  </div>
                  <div style={{ fontSize: 11, color: COLOURS.muted, marginTop: 8 }}>
                    The standing risk is held rather than dragged toward zero —
                    a pause is not evidence of innocence.
                  </div>
                </>
              ) : (
                <div style={{ fontSize: 12.5, color: COLOURS.dim }}>
                  Buffering ({(analysis.deepfake?.buffered_seconds ?? 0).toFixed(1)}s of
                  3.0s).
                </div>
              )}
            </Panel>

            <Panel title="Identity">
              {analysis.speaker ? (
                <>
                  <Row label="Claimed" value={analysis.speaker.identity} />
                  <Row label="Similarity" value={analysis.speaker.similarity.toFixed(3)} />
                  <Row
                    label="Decision"
                    value={
                      <Badge
                        colour={
                          analysis.speaker.decision === "MATCH"
                            ? COLOURS.good
                            : analysis.speaker.decision === "UNCERTAIN"
                            ? COLOURS.warn
                            : COLOURS.bad
                        }
                      >
                        {analysis.speaker.decision}
                      </Badge>
                    }
                  />
                  {analysis.speaker.speech_seconds_analysed !== undefined && (
                    <Row
                      label="Speech analysed"
                      value={`${analysis.speaker.speech_seconds_analysed.toFixed(1)}s`}
                      hint="estimate improves"
                    />
                  )}
                </>
              ) : analysis.stream.speaker_branch ? (
                <div style={{ fontSize: 12.5, color: COLOURS.dim }}>
                  Accumulating speech for an identity estimate.
                </div>
              ) : (
                <>
                  <Badge colour={COLOURS.unknown}>OFF</Badge>
                  <div style={{ fontSize: 12.5, color: COLOURS.dim, marginTop: 10 }}>
                    {analysis.stream.claimed_identity
                      ? `No enrolled profile for '${analysis.stream.claimed_identity}'.`
                      : "No identity claimed, so this is anti-spoof only."}
                  </div>
                </>
              )}
            </Panel>

            <Panel title="Stream">
              <Row
                label="Voice activity"
                value={analysis.vad.speech ? "speech" : "silence"}
              />
              <Row
                label="Speech ratio"
                value={`${(analysis.stream.speech_ratio * 100).toFixed(0)}%`}
              />
              <Row
                label="Audio received"
                value={`${(analysis.stream.total_audio_ms / 1000).toFixed(1)}s`}
              />
              <Row label="Chunks" value={analysis.stream.chunks} />
            </Panel>
          </div>

          {analysis.policy && (
            <Panel
              title="Policy"
              subtitle="What to do about it. No action refuses the customer's transaction."
            >
              {analysis.policy.actions.map((action) => (
                <div
                  key={action.code}
                  style={{
                    display: "flex",
                    gap: 12,
                    alignItems: "flex-start",
                    padding: "8px 0",
                    borderBottom: `1px solid ${COLOURS.border}44`,
                  }}
                >
                  <span
                    style={{
                      color: action.blocking ? COLOURS.warn : COLOURS.muted,
                      fontWeight: 800,
                      minWidth: 16,
                    }}
                  >
                    {action.blocking ? "!" : "•"}
                  </span>
                  <div>
                    <div style={{ fontWeight: 700, fontSize: 12.5 }}>
                      {action.code.replace(/_/g, " ")}
                    </div>
                    <div style={{ fontSize: 12, color: COLOURS.dim, marginTop: 2 }}>
                      {action.description}
                    </div>
                  </div>
                </div>
              ))}
            </Panel>
          )}
        </>
      )}
    </div>
  );
}
