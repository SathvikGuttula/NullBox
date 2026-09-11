"use client";

/**
 * Design-system primitives.
 *
 * Everything here is a thin wrapper over the classes in globals.css. The point
 * is that a component never invents a colour, a radius or a spacing value —
 * it picks a role, and the stylesheet decides how that role looks. That is
 * what stops the interface drifting into seven slightly different cards.
 */

import {
  AlertTriangle,
  Check,
  Info,
  ShieldAlert,
  X,
} from "lucide-react";
import {
  CSSProperties,
  ReactNode,
  useEffect,
  useRef,
  useState,
} from "react";

/* ---------------------------------------------------------------- signals */

/**
 * The five states everything in this product resolves to.
 *
 * `unknown` is the one that matters. A branch that could not be assessed is
 * not low risk and not high risk — it is absent, and it gets its own colour so
 * that "we did not check" can never be read as "we checked and it is fine".
 */
export type Signal = "clear" | "watch" | "verify" | "critical" | "unknown" | "neutral";

export function signalForLevel(level: string): Signal {
  if (level === "HIGH") return "critical";
  if (level === "SUSPICIOUS") return "verify";
  if (level === "UNKNOWN") return "unknown";
  if (level === "LOW") return "clear";
  return "neutral";
}

export function signalForDecision(decision: string): Signal {
  switch (decision) {
    case "HIGH_RISK_WORKFLOW": return "critical";
    case "SECONDARY_VERIFICATION": return "verify";
    case "WARN": return "watch";
    case "INSUFFICIENT_EVIDENCE": return "unknown";
    case "ALLOW": return "clear";
    default: return "neutral";
  }
}

export function signalForVerification(decision: string): Signal {
  if (decision === "MATCH") return "clear";
  if (decision === "UNCERTAIN") return "watch";
  if (decision === "NO_MATCH") return "critical";
  return "neutral";
}

/** The CSS variable holding a signal's colour, for inline use on one element. */
export function signalVar(signal: Signal): string {
  return signal === "neutral" ? "var(--text-3)" : `var(--sig-${signal})`;
}

/* ------------------------------------------------------------------ layout */

export function Panel({
  title,
  aside,
  note,
  children,
  padded = true,
  className = "",
  style,
}: {
  title?: string;
  aside?: ReactNode;
  note?: ReactNode;
  children: ReactNode;
  padded?: boolean;
  className?: string;
  style?: CSSProperties;
}) {
  return (
    <section className={`panel ${className}`} style={style}>
      {title && (
        <header className="panel-head">
          <h3>{title}</h3>
          {aside && <div className="spacer">{aside}</div>}
        </header>
      )}
      <div className={padded ? "panel-body" : ""}>{children}</div>
      {note && <div className="panel-note">{note}</div>}
    </section>
  );
}

export function Field({
  label,
  hint,
  children,
}: {
  label?: string;
  hint?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="field">
      {label && <span className="label">{label}</span>}
      {children}
      {hint && <span className="field-hint">{hint}</span>}
    </div>
  );
}

/* ----------------------------------------------------------------- pills */

export function Pill({
  signal = "neutral",
  children,
  plain = false,
}: {
  signal?: Signal | "brand";
  children: ReactNode;
  plain?: boolean;
}) {
  return (
    <span className={`pill pill-${signal}${plain ? " pill-plain" : ""}`}>
      {children}
    </span>
  );
}

/* --------------------------------------------------------------- buttons */

type ButtonProps = {
  children: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  variant?: "default" | "primary" | "danger" | "ghost";
  size?: "sm" | "md" | "lg";
  icon?: ReactNode;
  block?: boolean;
  title?: string;
  ariaLabel?: string;
  type?: "button" | "submit";
};

export function Button({
  children,
  onClick,
  disabled,
  variant = "default",
  size = "md",
  icon,
  block,
  title,
  ariaLabel,
  type = "button",
}: ButtonProps) {
  const classes = [
    "btn",
    variant !== "default" ? `btn-${variant}` : "",
    size !== "md" ? `btn-${size}` : "",
    block ? "btn-block" : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <button
      type={type}
      className={classes}
      onClick={onClick}
      disabled={disabled}
      title={title}
      aria-label={ariaLabel}
    >
      {icon}
      {children}
    </button>
  );
}

export function Chip({
  children,
  active,
  onClick,
  disabled,
}: {
  children: ReactNode;
  active?: boolean;
  onClick?: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      className="chip"
      aria-pressed={active}
      onClick={onClick}
      disabled={disabled}
    >
      {children}
    </button>
  );
}

/* --------------------------------------------------------------- notices */

const NOTICE_ICON = {
  info: Info,
  warn: AlertTriangle,
  error: ShieldAlert,
  good: Check,
  unknown: Info,
} as const;

export function Notice({
  tone = "info",
  children,
}: {
  tone?: keyof typeof NOTICE_ICON;
  children: ReactNode;
}) {
  const Icon = NOTICE_ICON[tone];
  return (
    <div className={`notice notice-${tone}`} role={tone === "error" ? "alert" : undefined}>
      <Icon size={14} aria-hidden />
      <div>{children}</div>
    </div>
  );
}

/* ------------------------------------------------------------- data rows */

export function DataList({ children }: { children: ReactNode }) {
  return <dl className="dl">{children}</dl>;
}

export function DataRow({
  label,
  qualifier,
  value,
}: {
  label: string;
  qualifier?: string;
  value: ReactNode;
}) {
  return (
    <div className="dl-row">
      <dt>
        {label}
        {qualifier && <span className="qual">{qualifier}</span>}
      </dt>
      <dd>{value}</dd>
    </div>
  );
}

/* ----------------------------------------------------------- empty state */

export function Empty({
  icon,
  title,
  children,
}: {
  icon: ReactNode;
  title: string;
  children?: ReactNode;
}) {
  return (
    <div className="empty">
      <div className="empty-icon">{icon}</div>
      <h4>{title}</h4>
      {children && <p>{children}</p>}
    </div>
  );
}

/* ---------------------------------------------------------- level meter */

/**
 * Segmented input meter.
 *
 * Segments rather than a continuous bar: this is a level instrument, and a
 * discrete readout is easier to judge at a glance than a sliding edge. The top
 * segments turn amber and then red so clipping is visible before it is heard.
 */
export function LevelMeter({ level, segments = 24 }: { level: number; segments?: number }) {
  const lit = Math.round(Math.min(level * 1.35, 1) * segments);

  return (
    <div className="meter" role="meter" aria-valuemin={0} aria-valuemax={1}
         aria-valuenow={Number(level.toFixed(2))} aria-label="Microphone input level">
      {Array.from({ length: segments }, (_, i) => {
        const on = i < lit;
        const share = i / segments;
        const tone = share > 0.9 ? "peak" : share > 0.72 ? "hot" : "on";
        return (
          <span
            key={i}
            className={`meter-seg${on ? ` ${tone}` : ""}`}
            style={{ height: `${38 + share * 62}%` }}
          />
        );
      })}
    </div>
  );
}

/* --------------------------------------------------------- animated value */

/**
 * Counts to a new value instead of snapping to it.
 *
 * During a live call the risk score updates about once a second, and a number
 * that jumps is genuinely harder to read than one that travels — the eye
 * follows the movement and lands on the new value. Short enough (400 ms) that
 * it never feels like waiting.
 */
export function AnimatedNumber({
  value,
  decimals = 0,
  duration = 400,
}: {
  value: number;
  decimals?: number;
  duration?: number;
}) {
  const [shown, setShown] = useState(value);
  const from = useRef(value);
  const frame = useRef<number | null>(null);

  useEffect(() => {
    const reduced =
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    if (reduced || from.current === value) {
      from.current = value;
      setShown(value);
      return;
    }

    const start = performance.now();
    const origin = from.current;
    const delta = value - origin;

    const step = (now: number) => {
      const t = Math.min((now - start) / duration, 1);
      // Same out-easing as the CSS, so motion feels like one system.
      const eased = 1 - Math.pow(1 - t, 3);
      setShown(origin + delta * eased);
      if (t < 1) frame.current = requestAnimationFrame(step);
      else from.current = value;
    };

    frame.current = requestAnimationFrame(step);
    return () => {
      if (frame.current !== null) cancelAnimationFrame(frame.current);
      from.current = value;
    };
  }, [value, duration]);

  return <>{shown.toFixed(decimals)}</>;
}

/* ---------------------------------------------------------------- toasts */

export function Toast({
  tone,
  message,
  onDismiss,
}: {
  tone: "good" | "error";
  message: string;
  onDismiss: () => void;
}) {
  useEffect(() => {
    const timer = setTimeout(onDismiss, tone === "error" ? 9000 : 5000);
    return () => clearTimeout(timer);
  }, [onDismiss, tone]);

  return (
    <div
      className="rise"
      role="status"
      style={{
        position: "fixed",
        right: "var(--s5)",
        bottom: "var(--s5)",
        zIndex: 60,
        maxWidth: 400,
        display: "flex",
        alignItems: "flex-start",
        gap: "var(--s3)",
        padding: "var(--s3) var(--s4)",
        borderRadius: "var(--r-md)",
        border: `1px solid var(--sig-${tone === "good" ? "clear" : "critical"}-line)`,
        background: `var(--sig-${tone === "good" ? "clear" : "critical"}-bg)`,
        boxShadow: "var(--lift-3)",
        fontSize: "var(--t-sm)",
        lineHeight: 1.55,
      }}
    >
      {tone === "good" ? (
        <Check size={15} style={{ color: "var(--sig-clear)", flex: "none", marginTop: 2 }} />
      ) : (
        <ShieldAlert size={15} style={{ color: "var(--sig-critical)", flex: "none", marginTop: 2 }} />
      )}
      <span style={{ flex: 1 }}>{message}</span>
      <button
        className="btn btn-ghost btn-sm btn-icon"
        onClick={onDismiss}
        aria-label="Dismiss"
      >
        <X size={13} />
      </button>
    </div>
  );
}
