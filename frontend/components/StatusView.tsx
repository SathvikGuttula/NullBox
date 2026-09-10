"use client";

/**
 * What is actually loaded, and what is only claimed.
 *
 * This screen exists because every field on it has been silently wrong at some
 * point in this project — a detector reporting itself untrained because a path
 * resolved against the wrong directory, a registry falling back to placeholder
 * thresholds because its calibration file was gitignored — and in each case the
 * API kept answering plausibly. Look here before demonstrating anything.
 *
 * The declared limitations are presented as the system being honest about
 * itself, not as errors, because that is what they are.
 */

import {
  Cpu,
  Database,
  Gauge,
  RefreshCw,
  ServerCrash,
  ShieldCheck,
  Sliders,
  Waves,
} from "lucide-react";

import { API_URL, SystemStatus } from "../lib/api";
import {
  Button,
  DataList,
  DataRow,
  Empty,
  Notice,
  Panel,
  Pill,
} from "./ui";

export default function StatusView({
  status,
  reachable,
  onRefresh,
}: {
  status: SystemStatus | null;
  reachable: boolean | null;
  onRefresh: () => void;
}) {
  if (reachable === false) {
    return (
      <Panel className="rise">
        <Empty icon={<ServerCrash size={18} />} title="Backend unreachable">
          The interface is pointed at <code className="mono">{API_URL}</code>, and nothing
          answered there.
        </Empty>

        <div className="stack stack-4" style={{ maxWidth: 560, margin: "0 auto" }}>
          <pre
            className="mono"
            style={{
              margin: 0,
              padding: "var(--s4)",
              borderRadius: "var(--r-md)",
              background: "var(--surface-inset)",
              border: "1px solid var(--line-soft)",
              fontSize: "var(--t-sm)",
              lineHeight: 1.7,
              overflowX: "auto",
            }}
          >
            {`cd backend\npython -m uvicorn app.main:app --port 8000`}
          </pre>
          <p className="field-hint" style={{ textAlign: "center" }}>
            Or <code className="mono">.\start.ps1</code> from the repository root, which
            checks everything first and names whatever is missing.
          </p>
          <Button icon={<RefreshCw size={14} />} onClick={onRefresh} block>
            Try again
          </Button>
        </div>
      </Panel>
    );
  }

  if (!status) {
    return (
      <div className="grid grid-2">
        {[0, 1, 2, 3].map((index) => (
          <div key={index} className="panel" style={{ padding: "var(--s5)" }}>
            <div className="stack stack-3">
              <div className="skeleton" style={{ height: 11, width: 90 }} />
              <div className="skeleton" style={{ height: 13, width: "70%" }} />
              <div className="skeleton" style={{ height: 13, width: "52%" }} />
              <div className="skeleton" style={{ height: 13, width: "61%" }} />
            </div>
          </div>
        ))}
      </div>
    );
  }

  const anti = status.anti_spoof;
  const calibration = anti.calibration as Record<string, unknown>;
  const calibrated = Boolean(calibration.present) && !calibration.error;
  const detection = calibration["detection_rate_at_0.5"] as number | undefined;
  const falseAlarm = calibration["false_alarm_at_0.5"] as number | undefined;
  const speaker = status.speaker;
  const metadata = anti.checkpoint_metadata as Record<string, unknown> | undefined;

  return (
    <div className="stack stack-5">
      <div className="row row-wrap rise">
        <Pill signal={status.ready ? "clear" : "critical"}>
          {status.ready ? "Ready" : "Not ready"}
        </Pill>
        <span className="mono dim" style={{ fontSize: "var(--t-sm)" }}>
          {status.service} v{status.version} · {API_URL}
        </span>
        <span className="spacer">
          <Button variant="ghost" size="sm" icon={<RefreshCw size={13} />} onClick={onRefresh}>
            Refresh
          </Button>
        </span>
      </div>

      {status.warnings.length > 0 && (
        <Panel
          title="Declared limitations"
          aside={<span className="label">{status.warnings.length}</span>}
          className="rise rise-1"
          note="These are the system being honest about itself, not errors. A build that hid them would be worse."
        >
          <div className="stack stack-3">
            {status.warnings.map((warning, index) => (
              <Notice key={index} tone="warn">
                {warning}
              </Notice>
            ))}
          </div>
        </Panel>
      )}

      <div className="grid grid-2 rise rise-2">
        {/* ---------------------------------------------------- anti-spoof */}
        <Panel
          title="Anti-spoof detector"
          aside={
            calibrated ? (
              <Pill signal="clear">Calibrated</Pill>
            ) : (
              <Pill signal="critical">Uncalibrated</Pill>
            )
          }
          note={
            metadata
              ? `Checkpoint from experiment ${String(metadata.experiment_id ?? "?")}, epoch ${String(metadata.epoch ?? "?")}. Its validation EER is not the headline number — validation used attacks the model trained on. The evaluation figure is 2.95%.`
              : undefined
          }
        >
          <DataList>
            <DataRow
              label="Checkpoint"
              value={
                anti.checkpoint_present ? (
                  <span style={{ color: "var(--sig-clear)" }}>{anti.checkpoint_mb} MB</span>
                ) : (
                  <span style={{ color: "var(--sig-critical)" }}>missing</span>
                )
              }
            />
            <DataRow label="Loaded in memory" value={anti.loaded ? "yes" : "not yet"} />
            {anti.device && <DataRow label="Device" value={anti.device} />}
            {calibrated && (
              <>
                <DataRow
                  label="Platt scale / bias"
                  value={`${calibration.scale} / ${calibration.bias}`}
                />
                <DataRow
                  label="Held-out utterances"
                  value={Number(calibration.held_out_utterances ?? 0).toLocaleString()}
                />
              </>
            )}
          </DataList>

          {calibrated && detection !== undefined && falseAlarm !== undefined && (
            <div
              className="grid"
              style={{
                gridTemplateColumns: "1fr 1fr",
                gap: "var(--s3)",
                marginTop: "var(--s4)",
              }}
            >
              <Metric
                label="Detected"
                value={`${(detection * 100).toFixed(2)}%`}
                tone="clear"
                caption="at p ≥ 0.5"
              />
              <Metric
                label="False alarm"
                value={`${(falseAlarm * 100).toFixed(2)}%`}
                tone="watch"
                caption="at p ≥ 0.5"
              />
            </div>
          )}
        </Panel>

        {/* ------------------------------------------------------- speaker */}
        <Panel
          title="Speaker verification"
          aside={
            speaker.speechbrain_installed ? (
              <Pill signal="clear">Encoder ready</Pill>
            ) : (
              <Pill signal="critical">No encoder</Pill>
            )
          }
        >
          <DataList>
            <DataRow label="Enrolled voices" value={speaker.enrolled} />
            <DataRow
              label="Thresholds"
              value={
                (speaker.thresholds.calibrated as boolean) ? (
                  <span style={{ color: "var(--sig-clear)" }}>calibrated</span>
                ) : (
                  <span style={{ color: "var(--sig-watch)" }}>placeholders</span>
                )
              }
            />
            <DataRow
              label="Match / no match"
              value={`${speaker.thresholds.match} / ${speaker.thresholds.no_match}`}
            />
            <DataRow label="Version" value={String(speaker.thresholds.version ?? "—")} />
          </DataList>

          {speaker.identities.length > 0 && (
            <div className="chipset" style={{ marginTop: "var(--s4)" }}>
              {speaker.identities.map((name) => (
                <span key={name} className="pill pill-neutral pill-plain">
                  {name}
                </span>
              ))}
            </div>
          )}

          {speaker.rejected_profiles && speaker.rejected_profiles.length > 0 && (
            <div style={{ marginTop: "var(--s4)" }}>
              <Notice tone="warn">
                {speaker.rejected_profiles.length} stored profile
                {speaker.rejected_profiles.length === 1 ? " was" : "s were"} skipped
                because the identities are no longer valid. They predate identity
                validation and cannot be recreated.
              </Notice>
            </div>
          )}
        </Panel>

        {/* -------------------------------------------------------- fusion */}
        <Panel
          title="Fusion weights"
          aside={
            status.fusion.calibrated ? (
              <Pill signal="clear">Calibrated</Pill>
            ) : (
              <Pill signal="watch">Placeholders</Pill>
            )
          }
          note="Engineering placeholders, not measured quantities. What is not arbitrary is the structure: contributions are additive, each is reported with its own magnitude, and every decision carries reason codes."
        >
          <div className="stack stack-2">
            {Object.entries(status.fusion.weights).map(([name, weight]) => {
              const implemented = ["synthetic_speech", "identity_mismatch", "context"].includes(
                name
              );
              return (
                <div key={name} className="stack stack-2">
                  <div
                    className="row"
                    style={{ fontSize: "var(--t-sm)", gap: "var(--s2)" }}
                  >
                    <span style={{ color: implemented ? "var(--text)" : "var(--text-4)" }}>
                      {name.replace(/_/g, " ")}
                    </span>
                    {!implemented && <span className="label">not implemented</span>}
                    <span
                      className="mono spacer"
                      style={{ color: implemented ? "var(--text-2)" : "var(--text-4)" }}
                    >
                      {weight}
                    </span>
                  </div>
                  <div
                    style={{
                      height: 3,
                      borderRadius: 2,
                      background: "var(--surface-3)",
                      overflow: "hidden",
                    }}
                  >
                    <div
                      style={{
                        width: `${(weight / 35) * 100}%`,
                        height: "100%",
                        borderRadius: 2,
                        background: implemented ? "var(--brand)" : "var(--line-strong)",
                      }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </Panel>

        {/* --------------------------------------------------------- audio */}
        <Panel title="Audio pipeline" note="The live window is deliberately shorter than the training window — a latency trade. Shorter windows are tiled up to the training length before scoring, so the encoder never sees a length it did not train on.">
          <DataList>
            {Object.entries(status.audio).map(([name, value]) => (
              <DataRow key={name} label={name.replace(/_/g, " ")} value={value} />
            ))}
          </DataList>
        </Panel>
      </div>

      {/* ------------------------------------------------ measured results */}
      <Panel
        title="Measured performance"
        aside={<span className="label">Evaluation set, disjoint attacks</span>}
        className="rise rise-3"
        note="Never quote the 0.196% validation EER — validation used A01–A06, the attacks the model trained on. 2.95% is the number."
      >
        <div className="grid grid-3">
          <Metric label="Anti-spoof EER" value="2.95%" tone="clear" caption="71,237 utterances · 13 unseen attacks" icon={<ShieldCheck size={14} />} />
          <Metric label="Speaker EER" value="0.373%" tone="clear" caption="67 unseen speakers" icon={<Waves size={14} />} />
          <Metric label="ROC-AUC" value="0.9863" tone="neutral" caption="anti-spoof, evaluation split" icon={<Gauge size={14} />} />
          <Metric label="Normalised minDCF" value="0.1474" tone="neutral" caption="prior 0.05 — not t-DCF" icon={<Sliders size={14} />} />
          <Metric label="Calibration ECE" value="0.0147" tone="clear" caption="from 0.1263 raw, held out" icon={<Cpu size={14} />} />
          <Metric label="Persisted decisions" value="0" tone="watch" caption="nothing is written to the database" icon={<Database size={14} />} />
        </div>
      </Panel>
    </div>
  );
}

function Metric({
  label,
  value,
  caption,
  tone,
  icon,
}: {
  label: string;
  value: string;
  caption?: string;
  tone: "clear" | "watch" | "critical" | "neutral";
  icon?: React.ReactNode;
}) {
  const colour = tone === "neutral" ? "var(--text)" : `var(--sig-${tone})`;

  return (
    <div
      style={{
        padding: "var(--s4)",
        borderRadius: "var(--r-md)",
        background: "var(--surface-2)",
        border: "1px solid var(--line-soft)",
      }}
    >
      <div className="row" style={{ gap: "var(--s2)", marginBottom: 6 }}>
        {icon && <span style={{ color: "var(--text-4)", display: "flex" }}>{icon}</span>}
        <span className="label">{label}</span>
      </div>
      <div
        className="mono"
        style={{
          fontSize: "var(--t-xl)",
          fontWeight: 600,
          letterSpacing: "-.03em",
          color: colour,
          lineHeight: 1.1,
        }}
      >
        {value}
      </div>
      {caption && (
        <div style={{ fontSize: "var(--t-xs)", color: "var(--text-3)", marginTop: 4, lineHeight: 1.45 }}>
          {caption}
        </div>
      )}
    </div>
  );
}
