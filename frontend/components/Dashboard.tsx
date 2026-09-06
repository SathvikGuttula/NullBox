"use client";

import { useState } from "react";

import LiveCall from "./LiveCall";
import RiskGauge from "./RiskGauge";

export default function Dashboard() {

  const [active, setActive] =
    useState(false);

  const [riskScore, setRiskScore] =
    useState(0);

  return (
    <main
      style={{
        minHeight: "100vh",
        padding: 32,
        background: "#070a0f",
      }}
    >

      <header
        style={{
          display: "flex",
          justifyContent:
            "space-between",
          alignItems: "center",
          marginBottom: 32,
        }}
      >

        <div>

          <div
            style={{
              fontSize: 26,
              fontWeight: 800,
              letterSpacing: 1,
            }}
          >
            VOXSHIELD
          </div>

          <div
            style={{
              color: "#6b7280",
              marginTop: 4,
            }}
          >
            Real-Time Voice Security Platform
          </div>

        </div>

        <div
          style={{
            padding:
              "8px 14px",
            border:
              "1px solid #1f2937",
            borderRadius: 999,
            fontSize: 12,
          }}
        >
          ● SYSTEM ONLINE
        </div>

      </header>

      <section
        style={{
          display: "grid",
          gridTemplateColumns:
            "minmax(280px, 0.8fr) minmax(400px, 1.2fr)",
          gap: 20,
        }}
      >

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
              color: "#6b7280",
              fontSize: 12,
              letterSpacing: 1,
              marginBottom: 20,
            }}
          >
            CURRENT VOICE RISK
          </div>

          <RiskGauge
            score={riskScore}
          />

          <div
            style={{
              textAlign: "center",
              marginTop: 20,
              color: "#6b7280",
              fontSize: 13,
            }}
          >
            Risk fusion engine
          </div>

        </div>

        <LiveCall
          active={active}
          onToggle={() =>
            setActive(!active)
          }
          onRiskUpdate={
            setRiskScore
          }
        />

      </section>

      <section
        style={{
          display: "grid",
          gridTemplateColumns:
            "repeat(4, 1fr)",
          gap: 16,
          marginTop: 20,
        }}
      >

        {[
          [
            "ACTIVE CALLS",
            active ? "1" : "0",
          ],
          [
            "AUDIO STREAM",
            active ? "LIVE" : "OFFLINE",
          ],
          [
            "SPEECH ENGINE",
            active
              ? "RUNNING"
              : "IDLE",
          ],
          [
            "AUDIT EVENTS",
            "0",
          ],
        ].map(
          ([label, value]) => (

            <div
              key={label}
              style={{
                background:
                  "#0d1117",
                border:
                  "1px solid #1f2937",
                borderRadius: 14,
                padding: 20,
              }}
            >

              <div
                style={{
                  color:
                    "#6b7280",
                  fontSize: 11,
                  letterSpacing: 1,
                }}
              >
                {label}
              </div>

              <div
                style={{
                  fontSize: 22,
                  fontWeight: 800,
                  marginTop: 10,
                }}
              >
                {value}
              </div>

            </div>

          )
        )}

      </section>

    </main>
  );
}