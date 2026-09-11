"use client";

import {
  Activity,
  AudioLines,
  Cpu,
  Monitor,
  Moon,
  ShieldCheck,
  Sun,
  UserRoundCheck,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { API_URL, SystemStatus, getStatus } from "../lib/api";
import AnalyzeView from "./AnalyzeView";
import LiveCallView from "./LiveCallView";
import SpeakersView from "./SpeakersView";
import StatusView from "./StatusView";
import { Pill } from "./ui";

/**
 * The app shell.
 *
 * A left rail rather than a tab bar. Four destinations is exactly the range
 * where a persistent rail pays for itself: every destination stays visible and
 * addressable, the current one is unambiguous, and there is somewhere durable
 * to put the health indicator — which an analyst needs to trust before they
 * trust anything on screen.
 *
 * On narrow screens the rail becomes a bottom bar instead of collapsing into a
 * hamburger, because hiding four destinations behind a tap costs more than the
 * 56px it saves.
 */

type ViewId = "analyze" | "live" | "speakers" | "system";

const VIEWS: {
  id: ViewId;
  label: string;
  short: string;
  icon: typeof AudioLines;
  title: string;
  subtitle: string;
}[] = [
  {
    id: "analyze",
    label: "Analyse",
    short: "Analyse",
    icon: AudioLines,
    title: "Analyse a recording",
    subtitle: "Score one clip on every branch that can run",
  },
  {
    id: "live",
    label: "Live call",
    short: "Live",
    icon: Activity,
    title: "Live call",
    subtitle: "Streaming analysis, updated once a second",
  },
  {
    id: "speakers",
    label: "Speakers",
    short: "Speakers",
    icon: UserRoundCheck,
    title: "Speaker profiles",
    subtitle: "Enrol a voice, then verify against it",
  },
  {
    id: "system",
    label: "System",
    short: "System",
    icon: Cpu,
    title: "System",
    subtitle: "What is loaded, calibrated and ready",
  },
];

export default function Dashboard() {
  const [view, setView] = useState<ViewId>("analyze");
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [reachable, setReachable] = useState<boolean | null>(null);

  const refresh = useCallback(async () => {
    try {
      const next = await getStatus();
      setStatus(next);
      setReachable(true);
    } catch {
      setReachable(false);
      setStatus(null);
    }
  }, []);

  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, 20000);
    return () => clearInterval(timer);
  }, [refresh]);

  /**
   * 1-4 jumps between views.
   *
   * Guarded on the event target: an analyst typing "3" into the transaction
   * amount should not be thrown onto the Speakers tab. Plain digits rather
   * than a modifier because nothing else in the product claims them, and a
   * bare key is one press instead of two.
   */
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.metaKey || event.ctrlKey || event.altKey) return;

      const target = event.target as HTMLElement | null;
      const tag = target?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || target?.isContentEditable) return;

      const index = Number(event.key) - 1;
      if (Number.isInteger(index) && index >= 0 && index < VIEWS.length) {
        event.preventDefault();
        setView(VIEWS[index].id);
      }
    };

    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const active = VIEWS.find((entry) => entry.id === view)!;
  const enrolled = status?.speaker.enrolled ?? 0;
  const headingRef = useRef<HTMLHeadingElement | null>(null);

  // Without this a keyboard user who presses "2" keeps focus on the rail
  // button they left, and a screen reader announces nothing at all.
  const firstRender = useRef(true);
  useEffect(() => {
    if (firstRender.current) {
      firstRender.current = false;
      return;
    }
    headingRef.current?.focus();
  }, [view]);

  return (
    <div className="shell">
      <nav className="rail" aria-label="Sections">
        <div className="brand-lockup">
          <span className="brand-mark" aria-hidden>
            <ShieldCheck size={17} strokeWidth={2.2} />
          </span>
          <span>
            <span className="brand-name">VoxShield</span>
            <span className="brand-tag">Voice integrity</span>
          </span>
        </div>

        <div className="nav">
          {VIEWS.map((entry, index) => {
            const Icon = entry.icon;
            const current = entry.id === view;
            return (
              <button
                key={entry.id}
                className="nav-item"
                aria-current={current ? "page" : undefined}
                onClick={() => setView(entry.id)}
              >
                <Icon size={16} strokeWidth={2} aria-hidden />
                <span>{entry.label}</span>
                {entry.id === "speakers" && enrolled > 0 ? (
                  <span className="nav-count">{enrolled}</span>
                ) : (
                  <kbd className="kbd" aria-hidden>{index + 1}</kbd>
                )}
              </button>
            );
          })}
        </div>

        <div className="rail-foot stack stack-3">
          <ThemeToggle />
          <BackendState reachable={reachable} status={status} />
        </div>
      </nav>

      <div className="main">
        <header className="topbar">
          <div>
            <h1 className="topbar-title" tabIndex={-1} ref={headingRef}>
              {active.title}
            </h1>
            <div className="topbar-sub">{active.subtitle}</div>
          </div>
          <div className="topbar-actions">
            <span className="mono" style={{ fontSize: "var(--t-2xs)", color: "var(--text-4)" }}>
              {API_URL.replace(/^https?:\/\//, "")}
            </span>
            <Pill signal={reachable === false ? "critical" : reachable ? "clear" : "neutral"}>
              {reachable === false ? "Offline" : reachable ? "Ready" : "Checking"}
            </Pill>
          </div>
        </header>

        <main className="view">
          {/* Keyed so a view change replays its entrance rather than swapping
              silently — it makes the navigation feel like it did something. */}
          <div className="view-inner" key={view}>
            {view === "analyze" && <AnalyzeView status={status} />}
            {view === "live" && <LiveCallView status={status} />}
            {view === "speakers" && <SpeakersView onChange={refresh} />}
            {view === "system" && <StatusView status={status} reachable={reachable} onRefresh={refresh} />}
          </div>
        </main>
      </div>
    </div>
  );
}

type Theme = "light" | "dark" | "system";

/**
 * Theme control.
 *
 * Three states, not two. "System" is the default and the honest one — most
 * people never touch this, and following the OS is right for them. The explicit
 * choices exist because an analyst demoing on a projector in a bright room has
 * a reason to override it, and that reason is not the OS's business.
 *
 * The choice is written to the document element so the CSS can win in both
 * directions, and to localStorage so it survives a reload.
 */
function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>("system");

  useEffect(() => {
    try {
      setTheme((localStorage.getItem("voxshield-theme") as Theme | null) ?? "system");
    } catch {
      // Reading storage throws outright in some privacy modes, not just writing.
    }
  }, []);

  function choose(next: Theme) {
    setTheme(next);
    try {
      localStorage.setItem("voxshield-theme", next);
    } catch {
      // Private browsing. The choice still applies for this session.
    }
    if (next === "system") document.documentElement.removeAttribute("data-theme");
    else document.documentElement.setAttribute("data-theme", next);
  }

  const options: { id: Theme; icon: typeof Sun; label: string }[] = [
    { id: "light", icon: Sun, label: "Light" },
    { id: "system", icon: Monitor, label: "Match system" },
    { id: "dark", icon: Moon, label: "Dark" },
  ];

  return (
    <div className="theme-toggle" role="group" aria-label="Colour theme">
      {options.map((option) => {
        const Icon = option.icon;
        return (
          <button
            key={option.id}
            onClick={() => choose(option.id)}
            aria-pressed={theme === option.id}
            title={option.label}
            aria-label={option.label}
          >
            <Icon size={13} />
          </button>
        );
      })}
    </div>
  );
}

/**
 * The rail's footer state.
 *
 * Deliberately more than a green dot. Two things silently break this product —
 * a missing checkpoint and a missing calibration — and both leave the API
 * answering plausibly. Surfacing them here means nobody demos an uncalibrated
 * detector by accident.
 */
function BackendState({
  reachable,
  status,
}: {
  reachable: boolean | null;
  status: SystemStatus | null;
}) {
  if (reachable === false) {
    return (
      <div
        style={{
          padding: "var(--s3)",
          borderRadius: "var(--r-md)",
          background: "var(--sig-critical-bg)",
          border: "1px solid var(--sig-critical-line)",
          fontSize: "var(--t-xs)",
          lineHeight: 1.5,
          color: "var(--text-2)",
        }}
      >
        <div style={{ color: "var(--sig-critical)", fontWeight: 600, marginBottom: 4 }}>
          Backend unreachable
        </div>
        Run <code className="mono">.\start.ps1</code> from the repository root.
      </div>
    );
  }

  if (!status) {
    return <div className="skeleton" style={{ height: 58, borderRadius: "var(--r-md)" }} />;
  }

  const calibrated = Boolean(
    (status.anti_spoof.calibration as Record<string, unknown>).present
  );
  const warnings: string[] = [];
  if (!status.anti_spoof.checkpoint_present) warnings.push("No checkpoint");
  if (!calibrated) warnings.push("Not calibrated");
  if (!status.speaker.speechbrain_installed) warnings.push("No speaker encoder");

  return (
    <div
      style={{
        padding: "var(--s3)",
        borderRadius: "var(--r-md)",
        background: "var(--surface-2)",
        border: "1px solid var(--line-soft)",
      }}
    >
      <div className="label" style={{ marginBottom: 6 }}>
        Detector
      </div>
      {warnings.length > 0 ? (
        <div style={{ fontSize: "var(--t-xs)", color: "var(--sig-watch)", lineHeight: 1.5 }}>
          {warnings.join(" · ")}
        </div>
      ) : (
        <div className="mono" style={{ fontSize: "var(--t-xs)", color: "var(--text-2)", lineHeight: 1.6 }}>
          <div>
            2.95<span style={{ color: "var(--text-4)" }}>% EER</span>
          </div>
          <div style={{ color: "var(--text-3)" }}>
            {status.anti_spoof.device ?? "not loaded"}
          </div>
        </div>
      )}
    </div>
  );
}
