"use client";

/**
 * What is actually loaded, and what is only claimed.
 *
 * Look at this before demonstrating anything. Every field here has been
 * silently wrong at some point in this project - a detector reporting itself
 * untrained because a path resolved against the wrong directory, a registry
 * falling back to placeholder thresholds because its calibration file was
 * gitignored - and in each case the API kept answering plausibly.
 */

import { useEffect, useState } from "react";

import { API_URL, SystemStatus, getStatus } from "../lib/api";
import { Badge, Button, COLOURS, Notice, Panel, Row } from "./ui";

export default function StatusTab() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  async function refresh() {
    setLoading(true);
    try {
      setStatus(await getStatus());
      setError(null);
    } catch (exception) {
      setError((exception as Error).message);
      setStatus(null);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  if (error) {
    return (
      <Panel title="Backend unreachable">
        <Notice tone="error">{error}</Notice>
        <div style={{ fontSize: 12.5, color: COLOURS.dim, marginTop: 12, lineHeight: 1.7 }}>
          The UI is pointed at <code>{API_URL}</code>. Start the backend with:
          <pre
            style={{
              marginTop: 10,
              padding: 12,
              borderRadius: 9,
              background: COLOURS.inset,
              border: `1px solid ${COLOURS.border}`,
              fontSize: 12,
              overflowX: "auto",
            }}
          >
            {`cd backend\npython -m uvicorn app.main:app --port 8000`}
          </pre>
          Or from the repository root, <code>.\start.ps1</code>, which checks
          everything first and tells you what is missing.
        </div>
        <div style={{ marginTop: 12 }}>
          <Button onClick={refresh}>Try again</Button>
        </div>
      </Panel>
    );
  }

  if (!status) {
    return <Panel title="System">{loading ? "Loading..." : "No data."}</Panel>;
  }

  const anti = status.anti_spoof;
  const calibration = anti.calibration as Record<string, unknown>;
  const speaker = status.speaker;
  const detection = calibration["detection_rate_at_0.5"] as number | undefined;
  const falseAlarm = calibration["false_alarm_at_0.5"] as number | undefined;

  return (
    <div style={{ display: "grid", gap: 16 }}>
      <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
        <Badge colour={status.ready ? COLOURS.good : COLOURS.bad}>
          {status.ready ? "READY" : "NOT READY"}
        </Badge>
        <span style={{ fontSize: 12.5, color: COLOURS.muted }}>
          {status.service} v{status.version} at <code>{API_URL}</code>
        </span>
        <Button onClick={refresh} variant="ghost" style={{ padding: "5px 11px", fontSize: 11.5 }}>
          refresh
        </Button>
      </div>

      {status.warnings.length > 0 && (
        <Panel
          title="Declared limitations"
          subtitle="These are the system being honest about itself, not errors."
        >
          {status.warnings.map((warning, index) => (
            <Notice key={index} tone="warn">
              {warning}
            </Notice>
          ))}
        </Panel>
      )}

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(330px, 1fr))",
          gap: 16,
        }}
      >
        <Panel title="Anti-spoof detector">
          <Row
            label="Checkpoint"
            value={
              anti.checkpoint_present ? (
                <span style={{ color: COLOURS.good }}>{anti.checkpoint_mb} MB</span>
              ) : (
                <span style={{ color: COLOURS.bad }}>missing</span>
              )
            }
          />
          <Row label="Loaded into memory" value={anti.loaded ? "yes" : "not yet"} />
          {anti.device && <Row label="Device" value={anti.device} />}
          <Row
            label="Calibrated"
            value={
              calibration.present ? (
                <span style={{ color: COLOURS.good }}>yes</span>
              ) : (
                <span style={{ color: COLOURS.bad }}>NO</span>
              )
            }
          />
          {Boolean(calibration.present) && (
            <>
              <Row
                label="Platt scale / bias"
                value={`${calibration.scale} / ${calibration.bias}`}
              />
              <Row
                label="Held-out utterances"
                value={String(calibration.held_out_utterances ?? "—")}
              />
              {detection !== undefined && (
                <Row
                  label="Detection at p ≥ 0.5"
                  value={`${(detection * 100).toFixed(2)}%`}
                />
              )}
              {falseAlarm !== undefined && (
                <Row
                  label="False alarm at p ≥ 0.5"
                  value={`${(falseAlarm * 100).toFixed(2)}%`}
                />
              )}
            </>
          )}

          {anti.checkpoint_metadata && (
            <div style={{ fontSize: 11, color: COLOURS.muted, marginTop: 10 }}>
              Checkpoint from experiment{" "}
              <code>{String(anti.checkpoint_metadata.experiment_id ?? "?")}</code>, epoch{" "}
              {String(anti.checkpoint_metadata.epoch ?? "?")}. Its validation EER is
              not the headline number — validation used attacks the model
              trained on. The evaluation figure is 2.95%.
            </div>
          )}
        </Panel>

        <Panel title="Speaker verification">
          <Row
            label="Encoder installed"
            value={
              speaker.speechbrain_installed ? (
                <span style={{ color: COLOURS.good }}>yes</span>
              ) : (
                <span style={{ color: COLOURS.bad }}>no</span>
              )
            }
          />
          <Row label="Enrolled voices" value={speaker.enrolled} />
          <Row
            label="Thresholds calibrated"
            value={
              (speaker.thresholds.calibrated as boolean) ? (
                <span style={{ color: COLOURS.good }}>yes</span>
              ) : (
                <span style={{ color: COLOURS.warn }}>placeholders</span>
              )
            }
          />
          <Row label="Match / no-match" value={`${speaker.thresholds.match} / ${speaker.thresholds.no_match}`} />
          <Row label="Version" value={String(speaker.thresholds.version ?? "—")} />

          {speaker.identities.length > 0 && (
            <div style={{ marginTop: 10, fontSize: 12, color: COLOURS.dim }}>
              {speaker.identities.join(", ")}
            </div>
          )}

          {speaker.rejected_profiles && speaker.rejected_profiles.length > 0 && (
            <Notice tone="warn">
              {speaker.rejected_profiles.length} stored profile(s) were skipped
              because their identities are no longer valid:{" "}
              {speaker.rejected_profiles
                .map((entry) => entry.identity.slice(0, 30))
                .join(", ")}
              . They predate identity validation and cannot be recreated.
            </Notice>
          )}
        </Panel>

        <Panel title="Fusion and policy">
          <Row
            label="Weights calibrated"
            value={
              status.fusion.calibrated ? (
                <span style={{ color: COLOURS.good }}>yes</span>
              ) : (
                <span style={{ color: COLOURS.warn }}>placeholders</span>
              )
            }
          />
          <Row label="Version" value={status.fusion.weights_version} />
          {Object.entries(status.fusion.weights).map(([name, weight]) => (
            <Row key={name} label={name.replace(/_/g, " ")} value={weight} />
          ))}
          <div style={{ fontSize: 11, color: COLOURS.muted, marginTop: 10 }}>
            Engineering placeholders, not measured quantities. What is not
            arbitrary is the structure: contributions are additive, each is
            reported with its own magnitude, and every decision carries reason
            codes.
          </div>
        </Panel>

        <Panel title="Audio pipeline">
          {Object.entries(status.audio).map(([name, value]) => (
            <Row key={name} label={name.replace(/_/g, " ")} value={value} />
          ))}
          <div style={{ fontSize: 11, color: COLOURS.muted, marginTop: 10 }}>
            The live window is deliberately shorter than the training window —
            a latency trade. Shorter windows are tiled up to the training length
            before scoring, so the encoder never sees a length it did not train
            on.
          </div>
        </Panel>
      </div>
    </div>
  );
}
