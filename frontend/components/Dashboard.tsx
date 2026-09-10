"use client";

import { useEffect, useState } from "react";

import { API_URL, getStatus } from "../lib/api";
import AnalyzeTab from "./AnalyzeTab";
import LiveCall from "./LiveCall";
import SpeakersTab from "./SpeakersTab";
import StatusTab from "./StatusTab";
import { Badge, COLOURS, Tabs } from "./ui";

const TABS = [
  { id: "analyze", label: "Analyse a recording" },
  { id: "live", label: "Live call" },
  { id: "speakers", label: "Speakers" },
  { id: "status", label: "System" },
];

export default function Dashboard() {
  const [active, setActive] = useState("analyze");
  const [online, setOnline] = useState<boolean | null>(null);
  const [enrolled, setEnrolled] = useState(0);

  useEffect(() => {
    let cancelled = false;

    const poll = () =>
      getStatus()
        .then((status) => {
          if (cancelled) return;
          setOnline(status.ready);
          setEnrolled(status.speaker.enrolled);
        })
        .catch(() => {
          if (!cancelled) setOnline(false);
        });

    poll();
    const timer = setInterval(poll, 15000);

    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [active]);

  return (
    <main
      style={{
        minHeight: "100vh",
        background: COLOURS.background,
        color: COLOURS.text,
        padding: "28px clamp(16px, 4vw, 40px) 60px",
        fontFamily:
          "system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
      }}
    >
      <header
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
          gap: 20,
          marginBottom: 26,
          flexWrap: "wrap",
        }}
      >
        <div>
          <div style={{ fontSize: 24, fontWeight: 800, letterSpacing: 1.5 }}>
            VOXSHIELD
          </div>
          <div style={{ color: COLOURS.muted, marginTop: 4, fontSize: 13 }}>
            Voice integrity, speaker identity and call context, fused into one
            decision
          </div>
        </div>

        <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
          {enrolled > 0 && (
            <Badge colour={COLOURS.muted}>
              {enrolled} voice{enrolled === 1 ? "" : "s"} enrolled
            </Badge>
          )}
          <Badge
            colour={
              online === null ? COLOURS.muted : online ? COLOURS.good : COLOURS.bad
            }
          >
            {online === null
              ? "CHECKING"
              : online
              ? "BACKEND READY"
              : "BACKEND UNREACHABLE"}
          </Badge>
        </div>
      </header>

      {online === false && (
        <div
          style={{
            padding: "12px 15px",
            borderRadius: 10,
            border: `1px solid ${COLOURS.bad}44`,
            background: `${COLOURS.bad}12`,
            color: COLOURS.bad,
            fontSize: 12.5,
            marginBottom: 22,
            lineHeight: 1.6,
          }}
        >
          No backend at <code>{API_URL}</code>. Start it from the repository
          root with <code>.\start.ps1</code>, or{" "}
          <code>cd backend &amp;&amp; python -m uvicorn app.main:app --port 8000</code>.
          The System tab has the details.
        </div>
      )}

      <Tabs tabs={TABS} active={active} onChange={setActive} />

      {active === "analyze" && <AnalyzeTab />}
      {active === "live" && <LiveCall />}
      {active === "speakers" && <SpeakersTab />}
      {active === "status" && <StatusTab />}

      <footer
        style={{
          marginTop: 44,
          paddingTop: 18,
          borderTop: `1px solid ${COLOURS.border}`,
          fontSize: 11.5,
          color: COLOURS.muted,
          lineHeight: 1.7,
        }}
      >
        Anti-spoof: 2.95% EER on 71,237 ASVspoof 2019 LA evaluation utterances
        across 13 attack types held out of training. At the shipped operating
        point, 94.95% detection at 0.79% false alarm. Speaker verification:
        0.373% EER on 67 unseen speakers. Fusion weights and policy bands are
        uncalibrated placeholders and are labelled as such wherever they appear.
      </footer>
    </main>
  );
}
