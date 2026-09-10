"use client";

/**
 * One analysis result, rendered so that every number says what it means.
 *
 * Three rules this follows, all of them things the backend is careful about
 * and which a UI can quietly undo:
 *
 *   - A branch that did not run is shown as "not assessed", never as zero
 *     risk. The backend puts those in `missing_signals` precisely so this can
 *     be displayed rather than hidden.
 *   - A decision is never shown without its actions. A bare "87" moves the
 *     decision onto whoever is reading it, under time pressure, which is the
 *     condition social engineering exploits.
 *   - Uncalibrated inputs are labelled on the same screen as the score they
 *     produced, not in a footnote.
 */

import {
  AnalysisResult,
  decisionColour,
  formatProbability,
  prettyDecision,
  riskColour,
} from "../lib/api";
import { Badge, COLOURS, Notice, Panel, Row } from "./ui";

export default function ResultView({ result }: { result: AnalysisResult }) {
  const { anti_spoof, speaker, context, risk, policy, notes } = result;

  return (
    <div style={{ display: "grid", gap: 16 }}>
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
            <div
              style={{ fontSize: 11, letterSpacing: 1.2, color: COLOURS.muted }}
            >
              FUSED RISK
            </div>
            <div
              style={{
                fontSize: 46,
                fontWeight: 800,
                lineHeight: 1.1,
                marginTop: 4,
                color: riskColour(risk.risk_level),
              }}
            >
              {risk.risk_score.toFixed(0)}
              <span style={{ fontSize: 17, color: COLOURS.muted }}> / 100</span>
            </div>
            <div style={{ marginTop: 8, display: "flex", gap: 8 }}>
              <Badge colour={riskColour(risk.risk_level)}>
                {risk.risk_level}
              </Badge>
              <Badge colour={decisionColour(risk.decision)}>
                {prettyDecision(risk.decision)}
              </Badge>
            </div>
          </div>

          <div style={{ textAlign: "right", fontSize: 12, color: COLOURS.muted }}>
            <div>{result.duration_seconds}s of audio</div>
            <div style={{ marginTop: 3 }}>
              {risk.available_signals.length} of{" "}
              {risk.available_signals.length + risk.missing_signals.length}{" "}
              branches reported
            </div>
          </div>
        </div>

        {risk.decision === "INSUFFICIENT_EVIDENCE" && (
          <Notice tone="warn">
            <strong>Not a clean result — an absent one.</strong> No branch could
            be assessed, so this call has not been cleared; it has not been
            checked. A score of 0 here means &quot;nothing to score&quot;, not
            &quot;nothing wrong&quot;.
          </Notice>
        )}

        {risk.contributions.length > 0 && risk.decision !== "INSUFFICIENT_EVIDENCE" && (
          <div style={{ marginTop: 18 }}>
            <div
              style={{
                fontSize: 11,
                letterSpacing: 1.2,
                color: COLOURS.muted,
                marginBottom: 8,
              }}
            >
              WHY
            </div>
            {risk.contributions.map((contribution, index) => (
              <div
                key={index}
                style={{
                  display: "flex",
                  gap: 12,
                  alignItems: "baseline",
                  padding: "6px 0",
                  borderBottom: `1px solid ${COLOURS.border}44`,
                }}
              >
                <span
                  style={{
                    minWidth: 52,
                    fontWeight: 700,
                    fontVariantNumeric: "tabular-nums",
                    color: contribution.points > 0 ? COLOURS.warn : COLOURS.muted,
                  }}
                >
                  {contribution.points > 0
                    ? `+${contribution.points.toFixed(0)}`
                    : "—"}
                </span>
                <span style={{ fontSize: 13, color: COLOURS.text }}>
                  {contribution.detail}
                </span>
              </div>
            ))}
          </div>
        )}

        {risk.missing_signals.length > 0 && (
          <div style={{ marginTop: 12, fontSize: 11.5, color: COLOURS.muted }}>
            <strong>Not assessed:</strong>{" "}
            {risk.missing_signals.join(", ").replace(/_/g, " ")}. These
            contributed nothing in either direction — they are not evidence of
            safety.
          </div>
        )}

        {!risk.calibrated && (
          <Notice tone="warn">
            The fusion weights are uncalibrated placeholders. This score is
            advisory — do not automate an irreversible action on it.
          </Notice>
        )}
      </Panel>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))",
          gap: 16,
        }}
      >
        <Panel title="Anti-spoof">
          {anti_spoof.model_status === "neural" ? (
            <>
              <Row
                label="Synthetic probability"
                value={
                  <span
                    style={{
                      color:
                        (anti_spoof.synthetic_probability ?? 0) >= 0.5
                          ? COLOURS.bad
                          : COLOURS.good,
                    }}
                  >
                    {formatProbability(anti_spoof.synthetic_probability)}
                  </span>
                }
                hint="calibrated"
              />
              <Row
                label="Raw model score"
                value={anti_spoof.raw_probability?.toFixed(4) ?? "n/a"}
                hint="uncalibrated"
              />
              <Row
                label="Verdict"
                value={
                  (anti_spoof.synthetic_probability ?? 0) >= 0.5
                    ? "SYNTHETIC"
                    : "genuine speech"
                }
              />
              <div style={{ fontSize: 11, color: COLOURS.muted, marginTop: 10 }}>
                At this operating point the detector catches 94.95% of attacks
                with a 0.79% false-alarm rate, measured on 35,619 held-out
                utterances across 13 attack types it never trained on.
              </div>
            </>
          ) : anti_spoof.model_status === "no_speech" ? (
            <>
              <Badge colour={COLOURS.unknown}>NOT ASSESSED</Badge>
              <div
                style={{ fontSize: 12.5, color: COLOURS.dim, marginTop: 10, lineHeight: 1.6 }}
              >
                {anti_spoof.speech_check?.reason}
              </div>
              {anti_spoof.speech_check && (
                <div style={{ marginTop: 12 }}>
                  <Row
                    label="Level (RMS)"
                    value={anti_spoof.speech_check.rms.toFixed(5)}
                  />
                  <Row
                    label="Dynamic range"
                    value={anti_spoof.speech_check.dynamic_range.toFixed(1)}
                    hint="speech is >10"
                  />
                </div>
              )}
              <div style={{ fontSize: 11, color: COLOURS.muted, marginTop: 10 }}>
                The detector is a speech model. Asked about silence or noise it
                answers confidently and wrongly — 0.999 and 0.997 synthetic
                respectively, measured — so the branch is withheld instead.
              </div>
            </>
          ) : (
            <>
              <Badge colour={COLOURS.muted}>{anti_spoof.model_status}</Badge>
              <div style={{ fontSize: 12.5, color: COLOURS.dim, marginTop: 10 }}>
                The anti-spoof branch did not contribute.
              </div>
            </>
          )}
        </Panel>

        <Panel title="Speaker identity">
          {speaker ? (
            <>
              <Row
                label="Claimed identity"
                value={<code style={{ fontSize: 12 }}>{speaker.identity}</code>}
              />
              <Row
                label="Similarity"
                value={speaker.similarity.toFixed(3)}
                hint="cosine"
              />
              <Row
                label="Decision"
                value={
                  <Badge
                    colour={
                      speaker.decision === "MATCH"
                        ? COLOURS.good
                        : speaker.decision === "UNCERTAIN"
                        ? COLOURS.warn
                        : COLOURS.bad
                    }
                  >
                    {speaker.decision}
                  </Badge>
                }
              />
              {speaker.reasons?.length > 0 && (
                <ul
                  style={{
                    margin: "12px 0 0",
                    paddingLeft: 18,
                    fontSize: 12,
                    color: COLOURS.dim,
                    lineHeight: 1.65,
                  }}
                >
                  {speaker.reasons.map((reason, index) => (
                    <li key={index}>{reason}</li>
                  ))}
                </ul>
              )}
            </>
          ) : (
            <>
              <Badge colour={COLOURS.unknown}>NOT ASSESSED</Badge>
              <div
                style={{ fontSize: 12.5, color: COLOURS.dim, marginTop: 10, lineHeight: 1.6 }}
              >
                No identity was claimed, so nothing here says whether the
                speaker is who they claim to be. A low risk score above is an
                anti-spoof result only.
              </div>
            </>
          )}
        </Panel>

        <Panel title="Call context">
          {context ? (
            <>
              <Row
                label="Context risk"
                value={`${(context.risk * 100).toFixed(0)}%`}
                hint={`${context.points.toFixed(0)} of ${context.possible.toFixed(0)} pts`}
              />
              <Row label="Signals used" value={context.signals_used.length} />
              <Row label="Not collected" value={context.missing.length} />
              {context.reasons.length > 0 && (
                <ul
                  style={{
                    margin: "12px 0 0",
                    paddingLeft: 18,
                    fontSize: 12,
                    color: COLOURS.dim,
                    lineHeight: 1.65,
                  }}
                >
                  {context.reasons.map((reason, index) => (
                    <li key={index}>{reason}</li>
                  ))}
                </ul>
              )}
              <div style={{ fontSize: 11, color: COLOURS.muted, marginTop: 10 }}>
                Scored as a fraction of the risk that could be assessed, not of
                all conceivable risk — otherwise two bad signals out of two look
                harmless.
              </div>
            </>
          ) : (
            <>
              <Badge colour={COLOURS.unknown}>NOT COLLECTED</Badge>
              <div
                style={{ fontSize: 12.5, color: COLOURS.dim, marginTop: 10, lineHeight: 1.6 }}
              >
                No call metadata was supplied. This is the branch that catches a
                genuine human impersonator — real voice, wrong person, urgent
                unusual request — which neither of the other two can.
              </div>
            </>
          )}
        </Panel>
      </div>

      <Panel
        title="Policy — what to do about it"
        subtitle="Every action is a verification step, a reversible hold, or a human escalation. None of them refuses the customer's transaction."
      >
        {policy.actions.map((action) => (
          <div
            key={action.code}
            style={{
              display: "flex",
              gap: 12,
              alignItems: "flex-start",
              padding: "9px 0",
              borderBottom: `1px solid ${COLOURS.border}44`,
            }}
          >
            <span
              style={{
                color: action.blocking ? COLOURS.warn : COLOURS.muted,
                fontWeight: 800,
                minWidth: 18,
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

        {policy.requires_human && (
          <Notice tone="warn">
            This decision requires a human. Actions marked
            &nbsp;<strong>!</strong>&nbsp; cannot be completed automatically.
          </Notice>
        )}

        {policy.rationale.length > 0 && (
          <ul
            style={{
              margin: "14px 0 0",
              paddingLeft: 18,
              fontSize: 12,
              color: COLOURS.muted,
              lineHeight: 1.65,
            }}
          >
            {policy.rationale.map((line, index) => (
              <li key={index}>{line}</li>
            ))}
          </ul>
        )}
      </Panel>

      {notes.length > 0 && (
        <Panel title="Notes from the analyser">
          <ul
            style={{
              margin: 0,
              paddingLeft: 18,
              fontSize: 12.5,
              color: COLOURS.dim,
              lineHeight: 1.7,
            }}
          >
            {notes.map((note, index) => (
              <li key={index}>{note}</li>
            ))}
          </ul>
        </Panel>
      )}
    </div>
  );
}
