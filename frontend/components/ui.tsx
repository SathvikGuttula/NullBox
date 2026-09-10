"use client";

/**
 * Shared presentation pieces.
 *
 * The dashboard used inline styles scattered across every component, so the
 * same panel border was written seven different ways. These are the pieces
 * that were actually repeated, extracted once.
 */

import { CSSProperties, ReactNode } from "react";

export const COLOURS = {
  background: "#070a0f",
  panel: "#0d1117",
  inset: "#080b10",
  border: "#1f2937",
  text: "#e5e7eb",
  muted: "#6b7280",
  dim: "#9ca3af",
  good: "#4ade80",
  warn: "#fbbf24",
  bad: "#f87171",
  unknown: "#a78bfa",
  accent: "#60a5fa",
};

export function Panel({
  title,
  subtitle,
  children,
  style,
}: {
  title?: string;
  subtitle?: string;
  children: ReactNode;
  style?: CSSProperties;
}) {
  return (
    <div
      style={{
        background: COLOURS.panel,
        border: `1px solid ${COLOURS.border}`,
        borderRadius: 14,
        padding: 20,
        ...style,
      }}
    >
      {title && (
        <div
          style={{
            fontSize: 11,
            letterSpacing: 1.2,
            color: COLOURS.muted,
            marginBottom: subtitle ? 4 : 14,
            textTransform: "uppercase",
          }}
        >
          {title}
        </div>
      )}
      {subtitle && (
        <div style={{ fontSize: 12, color: COLOURS.dim, marginBottom: 14 }}>
          {subtitle}
        </div>
      )}
      {children}
    </div>
  );
}

export function Button({
  children,
  onClick,
  disabled,
  variant = "default",
  style,
}: {
  children: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  variant?: "default" | "primary" | "danger" | "ghost";
  style?: CSSProperties;
}) {
  const backgrounds: Record<string, string> = {
    default: "#111827",
    primary: "#1d4ed8",
    danger: "#7f1d1d",
    ghost: "transparent",
  };

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        padding: "10px 16px",
        borderRadius: 9,
        border: `1px solid ${variant === "primary" ? "#2563eb" : COLOURS.border}`,
        background: disabled ? "#0b0f16" : backgrounds[variant],
        color: disabled ? COLOURS.muted : COLOURS.text,
        cursor: disabled ? "not-allowed" : "pointer",
        fontSize: 13,
        fontWeight: 600,
        transition: "background 120ms",
        ...style,
      }}
    >
      {children}
    </button>
  );
}

export function Badge({
  children,
  colour = COLOURS.muted,
}: {
  children: ReactNode;
  colour?: string;
}) {
  return (
    <span
      style={{
        display: "inline-block",
        padding: "3px 9px",
        borderRadius: 999,
        border: `1px solid ${colour}44`,
        background: `${colour}18`,
        color: colour,
        fontSize: 11,
        fontWeight: 700,
        letterSpacing: 0.6,
      }}
    >
      {children}
    </span>
  );
}

export function Row({
  label,
  value,
  hint,
}: {
  label: string;
  value: ReactNode;
  hint?: string;
}) {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "baseline",
        gap: 16,
        padding: "7px 0",
        borderBottom: `1px solid ${COLOURS.border}55`,
      }}
    >
      <span style={{ color: COLOURS.dim, fontSize: 13 }}>
        {label}
        {hint && (
          <span style={{ color: COLOURS.muted, fontSize: 11, marginLeft: 6 }}>
            {hint}
          </span>
        )}
      </span>
      <span
        style={{
          fontWeight: 600,
          fontSize: 13,
          textAlign: "right",
          fontVariantNumeric: "tabular-nums",
        }}
      >
        {value}
      </span>
    </div>
  );
}

export function TextInput({
  value,
  onChange,
  placeholder,
  label,
  hint,
}: {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  label?: string;
  hint?: string;
}) {
  return (
    <label style={{ display: "block", marginBottom: 12 }}>
      {label && (
        <div
          style={{
            fontSize: 11,
            letterSpacing: 1,
            color: COLOURS.muted,
            marginBottom: 6,
            textTransform: "uppercase",
          }}
        >
          {label}
        </div>
      )}
      <input
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder={placeholder}
        style={{
          width: "100%",
          padding: "10px 12px",
          borderRadius: 9,
          border: `1px solid ${COLOURS.border}`,
          background: COLOURS.inset,
          color: COLOURS.text,
          fontSize: 13,
          fontFamily: "inherit",
        }}
      />
      {hint && (
        <div style={{ fontSize: 11, color: COLOURS.muted, marginTop: 5 }}>
          {hint}
        </div>
      )}
    </label>
  );
}

export function Notice({
  children,
  tone = "info",
}: {
  children: ReactNode;
  tone?: "info" | "warn" | "error" | "good";
}) {
  const colours = {
    info: COLOURS.accent,
    warn: COLOURS.warn,
    error: COLOURS.bad,
    good: COLOURS.good,
  };
  const colour = colours[tone];

  return (
    <div
      style={{
        padding: "10px 13px",
        borderRadius: 9,
        border: `1px solid ${colour}44`,
        background: `${colour}12`,
        color: colour,
        fontSize: 12.5,
        lineHeight: 1.55,
        marginTop: 10,
      }}
    >
      {children}
    </div>
  );
}

export function LevelMeter({ level }: { level: number }) {
  const percent = Math.min(level * 140, 100);

  return (
    <div
      style={{
        height: 8,
        background: COLOURS.inset,
        borderRadius: 999,
        overflow: "hidden",
        border: `1px solid ${COLOURS.border}`,
      }}
    >
      <div
        style={{
          width: `${percent}%`,
          height: "100%",
          background:
            percent > 92 ? COLOURS.bad : percent > 12 ? COLOURS.good : COLOURS.muted,
          transition: "width 60ms linear",
        }}
      />
    </div>
  );
}

export function Tabs({
  tabs,
  active,
  onChange,
}: {
  tabs: { id: string; label: string }[];
  active: string;
  onChange: (id: string) => void;
}) {
  return (
    <div
      style={{
        display: "flex",
        gap: 6,
        borderBottom: `1px solid ${COLOURS.border}`,
        marginBottom: 22,
        flexWrap: "wrap",
      }}
    >
      {tabs.map((tab) => (
        <button
          key={tab.id}
          onClick={() => onChange(tab.id)}
          style={{
            padding: "10px 16px",
            border: "none",
            borderBottom: `2px solid ${
              active === tab.id ? COLOURS.accent : "transparent"
            }`,
            background: "transparent",
            color: active === tab.id ? COLOURS.text : COLOURS.muted,
            cursor: "pointer",
            fontSize: 13.5,
            fontWeight: active === tab.id ? 700 : 500,
            fontFamily: "inherit",
          }}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}
