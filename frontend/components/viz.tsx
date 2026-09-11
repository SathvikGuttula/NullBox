"use client";

/**
 * Instrument readouts.
 *
 * These are the three components that could not be lifted from any other
 * product, because each one is drawn against this system's own calibration.
 * A generic gauge would show a number the user already has. These show the
 * number *in relation to the thresholds that decide what happens next*, which
 * is the question an analyst is actually asking.
 */

import { useEffect, useRef } from "react";

import { Signal, signalVar } from "./ui";

/* ------------------------------------------------------------ risk scale */

/**
 * Where this score falls among the decision bands.
 *
 * The bands are the policy thresholds from the backend (40 / 60 / 80), drawn
 * to scale and tinted with the signal each one triggers. The marker is a hard
 * edge rather than a soft dot: the difference between 59 and 61 is a different
 * action, so the readout should not look approximate.
 */

const BANDS = [
  { from: 0,  to: 40,  signal: "clear"    as Signal, label: "Allow" },
  { from: 40, to: 60,  signal: "watch"    as Signal, label: "Warn" },
  { from: 60, to: 80,  signal: "verify"   as Signal, label: "Verify" },
  { from: 80, to: 100, signal: "critical" as Signal, label: "Escalate" },
];

export function RiskScale({
  score,
  signal,
  unknown = false,
}: {
  score: number;
  signal: Signal;
  unknown?: boolean;
}) {
  const clamped = Math.max(0, Math.min(100, score));

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      <div
        role="img"
        aria-label={
          unknown
            ? "No score — nothing could be assessed"
            : `Risk ${clamped.toFixed(0)} of 100`
        }
        style={{
          position: "relative",
          display: "flex",
          gap: 2,
          height: 10,
          borderRadius: 2,
          overflow: "hidden",
        }}
      >
        {BANDS.map((band) => (
          <div
            key={band.label}
            style={{
              flex: band.to - band.from,
              background: unknown
                ? "var(--surface-3)"
                : `color-mix(in srgb, ${signalVar(band.signal)} 22%, var(--surface-3))`,
            }}
          />
        ))}

        {!unknown && (
          <div
            style={{
              position: "absolute",
              left: `${clamped}%`,
              top: -2,
              bottom: -2,
              width: 3,
              marginLeft: -1.5,
              borderRadius: 1,
              background: signalVar(signal),
              boxShadow: "0 0 0 2px var(--surface)",
              transition: "left var(--dur-4) var(--ease-out)",
            }}
          />
        )}
      </div>

      <div
        className="mono"
        style={{
          display: "flex",
          fontSize: "var(--t-2xs)",
          color: "var(--text-4)",
          letterSpacing: ".04em",
        }}
      >
        {BANDS.map((band) => (
          <span key={band.label} style={{ flex: band.to - band.from }}>
            {band.from}
          </span>
        ))}
        <span style={{ marginLeft: -8 }}>100</span>
      </div>
    </div>
  );
}

/* -------------------------------------------------------- threshold ruler */

/**
 * Where this sample's calibrated probability sits among the measured
 * operating points.
 *
 * Plotted in **log-odds**, not linearly. The calibration is deliberately sharp
 * — a confident spoof comes back at 0.99999 and a clean bonafide at 0.03 — so
 * on a linear axis every real sample piles up against one edge and the chart
 * says nothing. Log-odds is the scale the calibration actually works in, and it
 * spreads those extremes into something readable.
 *
 * The ticks are the real false-alarm budgets from
 * `scripts/calibrate_detector.py`, measured on 35,619 held-out utterances.
 */

const OPERATING_POINTS = [
  { p: 0.0940, label: "5% FA",   detail: "97.3% detected" },
  { p: 0.1370, label: "2% FA",   detail: "96.8% detected" },
  { p: 0.2775, label: "1% FA",   detail: "95.8% detected" },
  { p: 0.5000, label: "shipped", detail: "94.95% detected at 0.79% false alarm", primary: true },
  { p: 0.9658, label: "0.5% FA", detail: "92.4% detected" },
];

const SPAN = 14; // log-odds shown either side of the boundary

function toPercent(probability: number): number {
  const clipped = Math.min(Math.max(probability, 1e-12), 1 - 1e-12);
  const logOdds = Math.log(clipped / (1 - clipped));
  return ((Math.max(-SPAN, Math.min(SPAN, logOdds)) + SPAN) / (SPAN * 2)) * 100;
}

export function ThresholdRuler({ probability }: { probability: number }) {
  const position = toPercent(probability);
  const synthetic = probability >= 0.5;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      <div
        style={{
          position: "relative",
          height: 34,
          borderRadius: "var(--r-sm)",
          background:
            "linear-gradient(90deg, var(--sig-clear-bg), var(--surface-3) 48%, var(--surface-3) 52%, var(--sig-critical-bg))",
          border: "1px solid var(--line-soft)",
          overflow: "hidden",
        }}
      >
        {OPERATING_POINTS.map((point) => {
          const left = toPercent(point.p);
          return (
            <div
              key={point.label}
              title={`p ≥ ${point.p} — ${point.detail}`}
              style={{
                position: "absolute",
                left: `${left}%`,
                top: 0,
                bottom: 0,
                width: point.primary ? 2 : 1,
                marginLeft: point.primary ? -1 : 0,
                background: point.primary ? "var(--text-3)" : "var(--line-strong)",
              }}
            />
          );
        })}

        <div
          style={{
            position: "absolute",
            left: `${position}%`,
            top: 3,
            bottom: 3,
            width: 4,
            marginLeft: -2,
            borderRadius: 2,
            background: synthetic ? "var(--sig-critical)" : "var(--sig-clear)",
            boxShadow: "0 0 0 2px var(--surface), 0 0 12px -2px currentColor",
            transition: "left var(--dur-4) var(--ease-out)",
          }}
        />
      </div>

      <div
        className="mono"
        style={{
          position: "relative",
          height: 13,
          fontSize: "var(--t-2xs)",
          color: "var(--text-4)",
        }}
      >
        <span style={{ position: "absolute", left: 0 }}>genuine</span>
        <span
          style={{
            position: "absolute",
            left: `${toPercent(0.5)}%`,
            transform: "translateX(-50%)",
            color: "var(--text-3)",
            fontWeight: 600,
          }}
        >
          0.50
        </span>
        <span style={{ position: "absolute", right: 0 }}>synthetic</span>
      </div>
    </div>
  );
}

/* --------------------------------------------------------- signal trace */

/**
 * Risk over the life of a call.
 *
 * The single most useful thing to show during a live call is not the current
 * number — it is whether the number is *climbing*. A caller who starts clean
 * and drifts up over ninety seconds looks completely different from one who
 * spikes on a single bad window, and only a trace makes that visible.
 *
 * Drawn on canvas rather than as SVG because it redraws about once a second
 * for the length of a call, and canvas keeps that off the DOM entirely.
 */
export function SignalTrace({
  history,
  height = 84,
}: {
  history: number[];
  height?: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const context = canvas.getContext("2d");
    if (!context) return;

    const ratio = window.devicePixelRatio || 1;
    const width = canvas.clientWidth;

    canvas.width = Math.max(1, Math.floor(width * ratio));
    canvas.height = Math.floor(height * ratio);
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);

    const styles = getComputedStyle(canvas);
    const read = (token: string) => styles.getPropertyValue(token).trim();

    const y = (value: number) => height - (value / 100) * height;

    // Band boundaries first, so the trace is read against them.
    context.lineWidth = 1;
    context.setLineDash([2, 4]);
    for (const [value, token] of [
      [40, "--sig-watch"],
      [60, "--sig-verify"],
      [80, "--sig-critical"],
    ] as const) {
      context.strokeStyle = read(token);
      context.globalAlpha = 0.34;
      context.beginPath();
      context.moveTo(0, y(value) + 0.5);
      context.lineTo(width, y(value) + 0.5);
      context.stroke();
    }
    context.setLineDash([]);
    context.globalAlpha = 1;

    if (history.length < 2) return;

    // Fit the whole call into the width; oldest sample at the left edge.
    const step = width / Math.max(history.length - 1, 1);
    const point = (index: number) => [index * step, y(history[index])] as const;

    const last = history[history.length - 1];
    const stroke =
      last >= 80 ? read("--sig-critical")
      : last >= 60 ? read("--sig-verify")
      : last >= 40 ? read("--sig-watch")
      : read("--sig-clear");

    // Area fill under the trace, so the shape reads even at a glance.
    const fill = context.createLinearGradient(0, 0, 0, height);
    fill.addColorStop(0, stroke);
    fill.addColorStop(1, "transparent");
    context.globalAlpha = 0.16;
    context.beginPath();
    context.moveTo(0, height);
    history.forEach((_, index) => {
      const [px, py] = point(index);
      context.lineTo(px, py);
    });
    context.lineTo(width, height);
    context.closePath();
    context.fillStyle = fill;
    context.fill();
    context.globalAlpha = 1;

    context.beginPath();
    history.forEach((_, index) => {
      const [px, py] = point(index);
      if (index === 0) context.moveTo(px, py);
      else context.lineTo(px, py);
    });
    context.strokeStyle = stroke;
    context.lineWidth = 1.75;
    context.lineJoin = "round";
    context.lineCap = "round";
    context.stroke();

    // Emphasised endpoint — the value that is current.
    const [lx, ly] = point(history.length - 1);
    context.beginPath();
    context.arc(lx, ly, 3, 0, Math.PI * 2);
    context.fillStyle = stroke;
    context.fill();
    context.beginPath();
    context.arc(lx, ly, 5.5, 0, Math.PI * 2);
    context.strokeStyle = stroke;
    context.globalAlpha = 0.35;
    context.lineWidth = 1.25;
    context.stroke();
  }, [history, height]);

  return (
    <canvas
      ref={canvasRef}
      style={{ width: "100%", height, display: "block" }}
      role="img"
      aria-label={
        history.length
          ? `Risk over the call, currently ${history[history.length - 1].toFixed(0)} of 100`
          : "No risk history yet"
      }
    />
  );
}

/* --------------------------------------------------- similarity readout */

/**
 * A speaker similarity plotted against its two calibrated thresholds.
 *
 * The band between them is deliberately visible: it is the UNCERTAIN zone,
 * and the whole point of a three-way decision is that the system says so
 * rather than guessing on the scores it knows least about.
 */
export function SimilarityScale({
  similarity,
  matchAt,
  noMatchAt,
}: {
  similarity: number;
  matchAt: number;
  noMatchAt: number;
}) {
  const place = (value: number) => Math.max(0, Math.min(1, value)) * 100;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
      <div
        style={{
          position: "relative",
          height: 8,
          borderRadius: 2,
          overflow: "hidden",
          background: "var(--sig-critical-bg)",
        }}
      >
        <div
          style={{
            position: "absolute",
            left: `${place(noMatchAt)}%`,
            width: `${place(matchAt) - place(noMatchAt)}%`,
            top: 0,
            bottom: 0,
            background: "var(--sig-watch-bg)",
          }}
        />
        <div
          style={{
            position: "absolute",
            left: `${place(matchAt)}%`,
            right: 0,
            top: 0,
            bottom: 0,
            background: "var(--sig-clear-bg)",
          }}
        />
        <div
          style={{
            position: "absolute",
            left: `${place(similarity)}%`,
            top: -2,
            bottom: -2,
            width: 3,
            marginLeft: -1.5,
            borderRadius: 1,
            background: "var(--text)",
            boxShadow: "0 0 0 2px var(--surface)",
            transition: "left var(--dur-4) var(--ease-out)",
          }}
        />
      </div>

      <div
        className="mono"
        style={{ position: "relative", height: 12, fontSize: "var(--t-2xs)", color: "var(--text-4)" }}
      >
        <span style={{ position: "absolute", left: `${place(noMatchAt)}%`, transform: "translateX(-50%)" }}>
          {noMatchAt.toFixed(2)}
        </span>
        <span style={{ position: "absolute", left: `${place(matchAt)}%`, transform: "translateX(-50%)" }}>
          {matchAt.toFixed(2)}
        </span>
      </div>
    </div>
  );
}
