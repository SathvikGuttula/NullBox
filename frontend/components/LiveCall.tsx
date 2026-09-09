"use client";

import { useEffect, useRef, useState } from "react";

import AudioVisualizer from "./AudioVisualizer";

interface LiveCallProps {
  active: boolean;
  onToggle: () => void;
  onRiskUpdate?: (score: number) => void;
}

interface VoiceFeatures {
  duration_ms: number;
  energy: number;
  spectral_centroid: number;
  spectral_bandwidth: number;
  spectral_rolloff: number;
  spectral_flatness: number;
  zero_crossing_rate: number;
  pitch_mean_hz: number;
  pitch_std_hz: number;
  voiced_ratio: number;
  mfcc_mean: number[];
  mfcc_std: number[];
}

interface Analysis {
  vad: {
    speech: boolean;
    confidence: number;
    rms: number;
    noise_floor: number;
  };

  features: VoiceFeatures;

  stream: {
    total_audio_ms: number;
    speech_audio_ms: number;
    speech_ratio: number;
  };

  deepfake?: {
    available: boolean;
    result?: {
      smoothed_probability: number;
      model_status: string;
    };
  };

  speaker?: {
    identity: string;
    similarity: number;
    decision: string;
    calibrated: boolean;
    speech_seconds_analysed?: number;
    reasons: string[];
  } | null;

  risk?: {
    risk_score: number;
    risk_level: string;
    decision: string;
    reasons: string[];
    available_signals: string[];
    missing_signals: string[];
    calibrated: boolean;
  } | null;
}

interface ServerMessage {
  type: string;
  call_id?: string;
  segment_id?: number;
  analysis?: Analysis;
}

export default function LiveCall({
  active,
  onToggle,
  onRiskUpdate,
}: LiveCallProps) {
  const [connected, setConnected] =
    useState(false);

  const [analysis, setAnalysis] =
    useState<Analysis | null>(null);

  const [riskScore, setRiskScore] =
    useState<number>(0);

  const [riskLevel, setRiskLevel] =
    useState<string>("LOW");

  const [decision, setDecision] =
    useState<string>("ALLOW");

  const [reasons, setReasons] =
    useState<string[]>([]);

  const [
    speakerSimilarity,
    setSpeakerSimilarity,
  ] = useState<number | null>(null);

  const [
    speakerDecision,
    setSpeakerDecision,
  ] = useState<string | null>(null);

  const [deepfakeScore, setDeepfakeScore] =
    useState<number | null>(null);

  const [modelStatus, setModelStatus] =
    useState("initializing");

  const [error, setError] =
    useState<string | null>(null);

  const socketRef =
    useRef<WebSocket | null>(null);

  const streamRef =
    useRef<MediaStream | null>(null);

  const processorRef =
    useRef<ScriptProcessorNode | null>(null);

  const audioContextRef =
    useRef<AudioContext | null>(null);

  const callIdRef =
    useRef<string | null>(null);

  useEffect(() => {
    if (!active) {
      stopCall();
      return;
    }

    startCall();

    return () => {
      stopCall();
    };
  }, [active]);

  async function startCall() {
    try {
      setError(null);

      const response =
        await fetch(
          "http://localhost:8000/api/v1/calls/demo",
          {
            method: "POST",
          }
        );

      if (!response.ok) {
        throw new Error(
          "Unable to create demo call"
        );
      }

      const data =
        await response.json();

      callIdRef.current =
        data.call_id;

      const stream =
        await navigator.mediaDevices
          .getUserMedia({
            audio: {
              channelCount: 1,
              sampleRate: 16000,
              echoCancellation: true,
              noiseSuppression: true,
              autoGainControl: false,
            },
          });

      streamRef.current =
        stream;

      const socket =
        new WebSocket(
          `ws://localhost:8000/api/v1/calls/${data.call_id}/stream`
        );

      socket.binaryType =
        "arraybuffer";

      socket.onopen = () => {
        setConnected(true);

        startAudioCapture(
          stream,
          socket
        );
      };

      socket.onmessage =
        (event) => {
          try {
            const message:
              ServerMessage =
              JSON.parse(
                event.data
              );

            if (
              message.type ===
                "voice_analysis" &&
              message.analysis
            ) {
              setAnalysis(
                message.analysis
              );
              const deepfake = message.analysis.deepfake;

              if (
                deepfake?.available &&
                deepfake.result
              ) {
                setDeepfakeScore(
                  deepfake.result
                    .smoothed_probability
                );

                setModelStatus(
                  deepfake.result
                    .model_status
                );
              }

              // The fused score, not a placeholder. It only exists once at
              // least one branch has reported, so leave the previous value
              // standing rather than flashing 0 between updates.
              const risk = message.analysis.risk;

              if (risk) {
                setRiskScore(risk.risk_score);
                setRiskLevel(risk.risk_level);
                setDecision(risk.decision);
                setReasons(risk.reasons ?? []);
                onRiskUpdate?.(risk.risk_score);
              }

              const speaker = message.analysis.speaker;

              if (speaker) {
                setSpeakerSimilarity(
                  speaker.similarity
                );
                setSpeakerDecision(
                  speaker.decision
                );
              }
            }
          } catch (err) {
            console.error(
              "Invalid server message",
              err
            );
          }
        };

      socket.onerror = () => {
        setError(
          "WebSocket connection error"
        );
      };

      socket.onclose = () => {
        setConnected(false);
      };

      socketRef.current =
        socket;

    } catch (err) {
      console.error(err);

      setError(
        err instanceof Error
          ? err.message
          : "Unable to start call"
      );

      onToggle();
    }
  }

  function startAudioCapture(
    stream: MediaStream,
    socket: WebSocket
  ) {
    const audioContext =
      new AudioContext({
        sampleRate: 16000,
      });

    audioContextRef.current =
      audioContext;

    const source =
      audioContext
        .createMediaStreamSource(
          stream
        );

    const processor =
      audioContext
        .createScriptProcessor(
          4096,
          1,
          1
        );

    processorRef.current =
      processor;

    processor.onaudioprocess =
      (event) => {
        if (
          socket.readyState !==
          WebSocket.OPEN
        ) {
          return;
        }

        const input =
          event.inputBuffer
            .getChannelData(0);

        const pcm =
          new Int16Array(
            input.length
          );

        for (
          let i = 0;
          i < input.length;
          i++
        ) {
          const sample =
            Math.max(
              -1,
              Math.min(
                1,
                input[i]
              )
            );

          pcm[i] =
            sample < 0
              ? sample * 32768
              : sample * 32767;
        }

        socket.send(
          pcm.buffer
        );
      };

    source.connect(
      processor
    );

    processor.connect(
      audioContext.destination
    );
  }

  function stopCall() {
    processorRef.current
      ?.disconnect();

    processorRef.current =
      null;

    audioContextRef.current
      ?.close()
      .catch(() => {});

    audioContextRef.current =
      null;

    streamRef.current
      ?.getTracks()
      .forEach(
        (track) => track.stop()
      );

    streamRef.current =
      null;

    if (
      socketRef.current &&
      socketRef.current.readyState ===
        WebSocket.OPEN
    ) {
      socketRef.current.close();
    }

    socketRef.current =
      null;

    setConnected(false);

    setAnalysis(null);
  }

  const features =
    analysis?.features;

  return (
    <div
      style={{
        background: "#0d1117",
        border:
          "1px solid #1f2937",
        borderRadius: 16,
        padding: 24,
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent:
            "space-between",
          alignItems: "center",
          marginBottom: 20,
        }}
      >
        <div>
          <div
            style={{
              fontSize: 12,
              color: "#6b7280",
              letterSpacing: 1,
            }}
          >
            VOICE STREAM
          </div>

          <div
            style={{
              fontSize: 22,
              fontWeight: 700,
              marginTop: 6,
            }}
          >
            {active
              ? "LIVE ANALYSIS"
              : "NO ACTIVE CALL"}
          </div>
        </div>

        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            fontSize: 12,
          }}
        >
          <span
            style={{
              width: 9,
              height: 9,
              borderRadius: "50%",
              background:
                connected
                  ? "#22c55e"
                  : "#374151",
            }}
          />

          {connected
            ? "CONNECTED"
            : "OFFLINE"}
        </div>
      </div>

      <AudioVisualizer
        level={
          analysis?.vad.rms ?? 0
        }
        active={active}
      />

      {analysis && (
        <>
          <div
            style={{
              marginTop: 16,
              padding: 14,
              background:
                "#080b10",
              borderRadius: 10,
            }}
          >
            <div
              style={{
                fontSize: 11,
                color: "#6b7280",
                letterSpacing: 1,
              }}
            >
              VOICE ACTIVITY
            </div>

            <div
              style={{
                marginTop: 8,
                fontSize: 18,
                fontWeight: 700,
              }}
            >
              {analysis.vad.speech
                ? "SPEECH DETECTED"
                : "SILENCE"}
            </div>

            <div
              style={{
                color: "#6b7280",
                fontSize: 12,
                marginTop: 5,
              }}
            >
              VAD confidence:{" "}
              {(
                analysis.vad
                  .confidence * 100
              ).toFixed(1)}
              %
            </div>
          </div>

          <div
            style={{
              display: "grid",
              gridTemplateColumns:
                "repeat(2, 1fr)",
              gap: 10,
              marginTop: 16,
            }}
          >
            <Metric
              label="PITCH"
              value={
                features
                  ? `${features.pitch_mean_hz.toFixed(1)} Hz`
                  : "-"
              }
            />

            <Metric
              label="PITCH VARIATION"
              value={
                features
                  ? `${features.pitch_std_hz.toFixed(1)} Hz`
                  : "-"
              }
            />

            <Metric
              label="SPECTRAL CENTROID"
              value={
                features
                  ? `${features.spectral_centroid.toFixed(0)} Hz`
                  : "-"
              }
            />

            <Metric
              label="SPECTRAL BANDWIDTH"
              value={
                features
                  ? `${features.spectral_bandwidth.toFixed(0)} Hz`
                  : "-"
              }
            />

            <Metric
              label="SPECTRAL FLATNESS"
              value={
                features
                  ? features
                      .spectral_flatness
                      .toFixed(4)
                  : "-"
              }
            />

            <Metric
              label="VOICED RATIO"
              value={
                features
                  ? `${(
                      features.voiced_ratio *
                      100
                    ).toFixed(1)}%`
                  : "-"
              }
            />
          </div>

          <div
            style={{
              marginTop: 16,
              padding: 14,
              background:
                "#080b10",
              borderRadius: 10,
            }}
          >
            <div
              style={{
                fontSize: 11,
                color: "#6b7280",
                letterSpacing: 1,
              }}
            >
              STREAM ANALYTICS
            </div>

            <div
              style={{
                display: "flex",
                justifyContent:
                  "space-between",
                marginTop: 10,
                fontSize: 13,
              }}
            >
              <span>
                Speech ratio
              </span>

              <strong>
                {(
                  analysis.stream
                    .speech_ratio *
                  100
                ).toFixed(1)}
                %
              </strong>
            </div>
          </div>

          {analysis.risk && (
            <div
              style={{
                marginTop: 16,
                padding: 14,
                background: "#080b10",
                borderRadius: 10,
                border: `1px solid ${
                  riskLevel === "HIGH"
                    ? "#7f1d1d"
                    : riskLevel ===
                      "SUSPICIOUS"
                    ? "#78350f"
                    : "#14532d"
                }`,
              }}
            >
              <div
                style={{
                  display: "flex",
                  justifyContent:
                    "space-between",
                  alignItems:
                    "baseline",
                }}
              >
                <span
                  style={{
                    fontSize: 11,
                    color: "#6b7280",
                    letterSpacing: 1,
                  }}
                >
                  VOXSHIELD RISK
                </span>

                <span
                  style={{
                    fontSize: 11,
                    color:
                      riskLevel ===
                      "HIGH"
                        ? "#fca5a5"
                        : riskLevel ===
                          "SUSPICIOUS"
                        ? "#fcd34d"
                        : "#86efac",
                    letterSpacing: 1,
                  }}
                >
                  {riskLevel}
                </span>
              </div>

              <div
                style={{
                  fontSize: 34,
                  fontWeight: 700,
                  marginTop: 6,
                }}
              >
                {riskScore.toFixed(0)}
                <span
                  style={{
                    fontSize: 15,
                    color: "#6b7280",
                  }}
                >
                  {" "}
                  / 100
                </span>
              </div>

              <div
                style={{
                  fontSize: 13,
                  color: "#9ca3af",
                  marginTop: 2,
                }}
              >
                {decision.replace(
                  /_/g,
                  " "
                )}
              </div>

              {speakerSimilarity !==
                null && (
                <div
                  style={{
                    display: "flex",
                    justifyContent:
                      "space-between",
                    marginTop: 12,
                    fontSize: 13,
                  }}
                >
                  <span>
                    Identity match
                  </span>
                  <strong>
                    {speakerSimilarity.toFixed(
                      2
                    )}{" "}
                    <span
                      style={{
                        color:
                          "#6b7280",
                      }}
                    >
                      {speakerDecision}
                    </span>
                  </strong>
                </div>
              )}

              {reasons.length > 0 && (
                <ul
                  style={{
                    margin: "12px 0 0",
                    paddingLeft: 18,
                    fontSize: 12,
                    color: "#d1d5db",
                    lineHeight: 1.6,
                  }}
                >
                  {reasons.map(
                    (reason, i) => (
                      <li key={i}>
                        {reason}
                      </li>
                    )
                  )}
                </ul>
              )}

              {analysis.risk
                .missing_signals
                .length > 0 && (
                <div
                  style={{
                    marginTop: 10,
                    fontSize: 11,
                    color: "#6b7280",
                  }}
                >
                  Not yet contributing:{" "}
                  {analysis.risk.missing_signals.join(
                    ", "
                  )}
                </div>
              )}

              {!analysis.risk
                .calibrated && (
                <div
                  style={{
                    marginTop: 8,
                    fontSize: 11,
                    color: "#fcd34d",
                  }}
                >
                  Fusion weights are
                  uncalibrated
                  placeholders.
                </div>
              )}
            </div>
          )}
        </>
      )}

      {error && (
        <div
          style={{
            marginTop: 14,
            padding: 12,
            borderRadius: 10,
            background: "#1c1111",
            color: "#fca5a5",
            fontSize: 13,
          }}
        >
          {error}
        </div>
      )}

      <button
        onClick={onToggle}
        style={{
          width: "100%",
          padding:
            "12px 16px",
          borderRadius: 10,
          border:
            "1px solid #374151",
          background:
            active
              ? "#111827"
              : "#172554",
          color: "#fff",
          cursor: "pointer",
          marginTop: 20,
        }}
      >
        {active
          ? "End Demo Call"
          : "Start Voice Analysis"}
      </button>
    </div>
  );
}

function Metric({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div
      style={{
        background:
          "#080b10",
        borderRadius: 10,
        padding: 12,
      }}
    >
      <div
        style={{
          color:
            "#6b7280",
          fontSize: 10,
          letterSpacing: 1,
        }}
      >
        {label}
      </div>

      <div
        style={{
          fontSize: 14,
          fontWeight: 700,
          marginTop: 5,
        }}
      >
        {value}
      </div>
    </div>
  );
}