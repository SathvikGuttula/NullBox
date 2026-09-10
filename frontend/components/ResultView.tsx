"use client";

/**
 * One analysis result.
 *
 * The layout follows what an analyst reads in order: the verdict and what it
 * means, then why, then what to do, then the branch detail if they want it.
 * Three rules the backend is careful about, which a UI can quietly undo:
 *
 *   - A branch that did not run reads as "not assessed", never as zero risk.
 *   - A decision is never shown without its actions. A bare number moves the
 *     decision onto whoever is reading it, under time pressure, which is the
 *     condition social engineering exploits.
 *   - Uncalibrated inputs are labelled on the same screen as the score they
 *     produced, not in a footnote.
 */

import {
  CircleAlert,
  Fingerprint,
  Gauge,
  Hand,
  ListChecks,
  ScanLine,
  Waves,
} from "lucide-react";

import {
  AnalysisResult,
  formatProbability,
  prettyDecision,
} from "../lib/api";
import {
  AnimatedNumber,
  DataList,
  DataRow,
  Notice,
  Panel,
  Pill,
  Signal,
  signalForDecision,
  signalForLevel,
  signalForVerification,
  signalVar,
} from "./ui";
import { RiskScale, SimilarityScale, ThresholdRuler } from "./viz";

export default function ResultView({
  result,
  thresholds,
}: {
  result: AnalysisResult;
  thresholds?: { match: number; no_match: number };
}) {
  const { anti_spoof, speaker, context, risk, policy, notes } = result;

  const insufficient = risk.decision === "INSUFFICIENT_EVIDENCE";
  const signal: Signal = insufficient ? "unknown" : signalForLevel(risk.risk_level);
  const decisionSignal = signalForDecision(risk.decision);
  const colour = signalVar(insufficient ? "unknown" : decisionSignal);

  const reported = risk.available_signals.length;
  const total = reported + risk.missing_signals.length;
  const maxPoints = Math.max(...risk.contributions.map((c) => c.points), 1);

  return (
    <div className="stack stack-4">
      {/* ---------------------------------------------------------- verdict */}
      <section
        className="verdict rise"
        style={{ ["--verdict-colour" as string]: colour }}
      >
        <div className="verdict-body">
          <div className="stack stack-4" style={{ minWidth: 0 }}>
            <div>
              <div className="label" style={{ marginBottom: 6 }}>
                Fused risk
              </div>
              <div className="verdict-score">
                {insufficient ? (
                  <span style={{ fontSize: "var(--t-3xl)", letterSpacing: "-.02em" }}>
                    Not assessed
                  </span>
                ) : (
                  <>
                    <AnimatedNumber value={risk.risk_score} />
                    <span className="of"> / 100</span>
                  </>
                )}
              </div>
            </div>

            <div className="row row-wrap" style={{ gap: "var(--s2)" }}>
              <Pill signal={signal}>{insufficient ? "Unknown" : risk.risk_level}</Pill>
              <Pill signal={decisionSignal}>{prettyDecision(risk.decision)}</Pill>
              {policy.requires_human && <Pill signal="verify">Needs a human</Pill>}
            </div>

            <RiskScale score={risk.risk_score} signal={signal} unknown={insufficient} />
          </div>

          <div className="verdict-meta">
            <div className="label">Branches</div>
            <div
              className="mono"
              style={{ fontSize: "var(--t-xl)", fontWeight: 600, letterSpacing: "-.02em" }}
            >
              {reported}
              <span style={{ color: "var(--text-4)" }}>/{total}</span>
            </div>
            <div style={{ fontSize: "var(--t-xs)", color: "var(--text-3)" }}>
              {result.duration_seconds}s of audio
            </div>
          </div>
        </div>

        {insufficient && (
          <div style={{ padding: "0 var(--s6) var(--s5)" }}>
            <Notice tone="unknown">
              <strong>Not a clean result — an absent one.</strong> No branch could be
              assessed, so this call has not been cleared; it has not been checked. A
              score of zero here means &ldquo;nothing to score&rdquo;, not
              &ldquo;nothing wrong&rdquo;.
            </Notice>
          </div>
        )}
      </section>

      {/* ------------------------------------------------------------- why */}
      {!insufficient && risk.contributions.length > 0 && (
        <Panel
          title="Why"
          aside={<span className="label">Points, out of 100</span>}
          className="rise rise-1"
        >
          <div className="contrib">
            {risk.contributions.map((contribution, index) => (
              <div className="contrib-row" key={`${contribution.source}-${index}`}>
                <div className={`contrib-pts${contribution.points <= 0 ? " zero" : ""}`}>
                  {contribution.points > 0 ? `+${contribution.points.toFixed(0)}` : "—"}
                </div>
                <div style={{ minWidth: 0 }}>
                  <div className="contrib-text">{contribution.detail}</div>
                  {contribution.points > 0 && (
                    <div className="contrib-bar">
                      <div
                        className="contrib-fill"
                        style={{
                          width: `${(contribution.points / maxPoints) * 100}%`,
                          background: colour,
                          animation: "growX var(--dur-4) var(--ease-out) both",
                          animationDelay: `${140 + index * 60}ms`,
                        }}
                      />
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>

          {risk.missing_signals.length > 0 && (
            <div style={{ marginTop: "var(--s4)" }}>
              <Notice tone="unknown">
                <strong>Not assessed:</strong>{" "}
                {risk.missing_signals.join(", ").replace(/_/g, " ")}. These contributed
                nothing in either direction — they are not evidence of safety.
              </Notice>
            </div>
          )}
        </Panel>
      )}

      {/* ----------------------------------------------------------- policy */}
      <Panel
        title="What to do about it"
        aside={
          policy.requires_human ? (
            <Pill signal="verify">Human required</Pill>
          ) : (
            <Pill signal="neutral">Automatic</Pill>
          )
        }
        className="rise rise-2"
        note={
          <>
            Every action is a verification step, a reversible hold, or a human
            escalation. None of them refuses the customer&rsquo;s transaction — a false
            positive should cost someone thirty seconds, not their payment.
          </>
        }
      >
        <div>
          {policy.actions.map((action) => (
            <div className="action" key={action.code}>
              <span className={`action-icon ${action.blocking ? "blocking" : "passive"}`}>
                {action.blocking ? <Hand size={12} /> : <ListChecks size={12} />}
              </span>
              <div>
                <div className="action-code">{action.code.replace(/_/g, " ")}</div>
                <div className="action-desc">{action.description}</div>
              </div>
            </div>
          ))}
        </div>

        {policy.rationale.length > 0 && (
          <ul
            style={{
              margin: "var(--s4) 0 0",
              paddingLeft: 16,
              fontSize: "var(--t-sm)",
              color: "var(--text-3)",
              lineHeight: 1.6,
            }}
          >
            {policy.rationale.map((line, index) => (
              <li key={index} style={{ marginBottom: 4 }}>
                {line}
              </li>
            ))}
          </ul>
        )}
      </Panel>

      {/* ---------------------------------------------------------- branches */}
      <div className="grid grid-2 rise rise-3">
        <AntiSpoofPanel result={result} />
        <SpeakerPanel result={result} thresholds={thresholds} />
      </div>

      {context && <ContextPanel result={result} />}

      {(notes.length > 0 || !risk.calibrated) && (
        <div className="stack stack-3 rise rise-5">
          {!risk.calibrated && (
            <Notice tone="warn">
              Fusion weights are uncalibrated placeholders. This score is advisory —
              do not automate an irreversible action on it.
            </Notice>
          )}
          {notes.map((note, index) => (
            <Notice key={index} tone="info">
              {note}
            </Notice>
          ))}
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------ anti-spoof */

function AntiSpoofPanel({ result }: { result: AnalysisResult }) {
  const { anti_spoof } = result;

  if (anti_spoof.model_status === "neural" && anti_spoof.synthetic_probability !== null) {
    const probability = anti_spoof.synthetic_probability;
    const synthetic = probability >= 0.5;

    return (
      <Panel
        title="Anti-spoof"
        aside={
          <Pill signal={synthetic ? "critical" : "clear"}>
            {synthetic ? "Synthetic" : "Genuine speech"}
          </Pill>
        }
        note="Measured at this operating point: 94.95% of attacks detected at a 0.79% false-alarm rate, on 35,619 held-out utterances across 13 attack types the model never trained on."
      >
        <div className="stack stack-5">
          <div>
            <div className="label" style={{ marginBottom: 8 }}>
              Calibrated probability
            </div>
            <div
              className="mono"
              style={{
                fontSize: "var(--t-2xl)",
                fontWeight: 600,
                letterSpacing: "-.03em",
                color: synthetic ? "var(--sig-critical)" : "var(--sig-clear)",
              }}
            >
              {formatProbability(probability)}
            </div>
          </div>

          <ThresholdRuler probability={probability} />

          <DataList>
            <DataRow
              label="Raw model score"
              qualifier="uncalibrated"
              value={anti_spoof.raw_probability?.toFixed(4) ?? "—"}
            />
            {anti_spoof.speech_check && (
              <DataRow
                label="Dynamic range"
                qualifier="speech is >10"
                value={anti_spoof.speech_check.dynamic_range.toFixed(1)}
              />
            )}
          </DataList>
        </div>
      </Panel>
    );
  }

  if (anti_spoof.model_status === "no_speech") {
    return (
      <Panel
        title="Anti-spoof"
        aside={<Pill signal="unknown">Not assessed</Pill>}
        note="The detector is a speech model. Asked about silence or noise it answers confidently and wrongly — 0.999 and 0.997 synthetic respectively, measured — so the branch is withheld instead of guessed."
      >
        <div className="stack stack-4">
          <p className="prose" style={{ fontSize: "var(--t-md)" }}>
            {anti_spoof.speech_check?.reason}
          </p>
          {anti_spoof.speech_check && (
            <DataList>
              <DataRow label="Level" qualifier="RMS" value={anti_spoof.speech_check.rms.toFixed(5)} />
              <DataRow
                label="Dynamic range"
                qualifier="threshold 3.0"
                value={anti_spoof.speech_check.dynamic_range.toFixed(1)}
              />
              <DataRow label="Duration" value={`${anti_spoof.speech_check.seconds.toFixed(1)}s`} />
            </DataList>
          )}
        </div>
      </Panel>
    );
  }

  return (
    <Panel title="Anti-spoof" aside={<Pill signal="neutral">{anti_spoof.model_status}</Pill>}>
      <div className="empty">
        <div className="empty-icon">
          <Waves size={18} />
        </div>
        <h4>Branch did not contribute</h4>
        <p>No trained checkpoint is loaded, so nothing was scored.</p>
      </div>
    </Panel>
  );
}

/* --------------------------------------------------------------- speaker */

function SpeakerPanel({
  result,
  thresholds,
}: {
  result: AnalysisResult;
  thresholds?: { match: number; no_match: number };
}) {
  const { speaker } = result;

  if (!speaker) {
    return (
      <Panel title="Speaker identity" aside={<Pill signal="unknown">Not assessed</Pill>}>
        <div className="empty">
          <div className="empty-icon">
            <Fingerprint size={18} />
          </div>
          <h4>No identity claimed</h4>
          <p>
            Nothing here says whether the speaker is who they claim to be. A low risk
            score above is an anti-spoof result only.
          </p>
        </div>
      </Panel>
    );
  }

  const signal = signalForVerification(speaker.decision);

  return (
    <Panel
      title="Speaker identity"
      aside={<Pill signal={signal}>{speaker.decision.replace("_", " ")}</Pill>}
      note="Between the two thresholds the answer is UNCERTAIN, deliberately — a two-way split would force a guess on exactly the scores the system knows least about."
    >
      <div className="stack stack-5">
        <div>
          <div className="label" style={{ marginBottom: 8 }}>
            Cosine similarity against {speaker.identity}
          </div>
          <div
            className="mono"
            style={{
              fontSize: "var(--t-2xl)",
              fontWeight: 600,
              letterSpacing: "-.03em",
              color: signalVar(signal),
            }}
          >
            {speaker.similarity.toFixed(3)}
          </div>
        </div>

        {thresholds && (
          <SimilarityScale
            similarity={speaker.similarity}
            matchAt={thresholds.match}
            noMatchAt={thresholds.no_match}
          />
        )}

        {speaker.reasons?.length > 0 && (
          <ul
            style={{
              paddingLeft: 16,
              fontSize: "var(--t-sm)",
              color: "var(--text-2)",
              lineHeight: 1.6,
            }}
          >
            {speaker.reasons.map((reason, index) => (
              <li key={index} style={{ marginBottom: 4 }}>
                {reason}
              </li>
            ))}
          </ul>
        )}
      </div>
    </Panel>
  );
}

/* --------------------------------------------------------------- context */

function ContextPanel({ result }: { result: AnalysisResult }) {
  const context = result.context!;
  const percent = context.risk * 100;
  const signal: Signal =
    percent >= 70 ? "critical" : percent >= 35 ? "watch" : "clear";

  return (
    <Panel
      title="Call context"
      aside={<Pill signal={signal}>{percent.toFixed(0)}% of assessable risk</Pill>}
      className="rise rise-4"
      note="Scored as a fraction of the risk that could be assessed, not of all conceivable risk — otherwise two bad signals out of two look harmless."
    >
      <div className="grid grid-2">
        <div className="stack stack-4">
          <DataList>
            <DataRow
              label="Points"
              value={`${context.points.toFixed(0)} / ${context.possible.toFixed(0)}`}
            />
            <DataRow label="Signals collected" value={context.signals_used.length} />
            <DataRow label="Not collected" value={context.missing.length} />
          </DataList>
        </div>

        <div>
          {context.reasons.length > 0 ? (
            <ul className="stack stack-2" style={{ listStyle: "none" }}>
              {context.reasons.map((reason, index) => (
                <li
                  key={index}
                  style={{
                    display: "flex",
                    gap: "var(--s2)",
                    fontSize: "var(--t-sm)",
                    lineHeight: 1.5,
                    color: "var(--text-2)",
                  }}
                >
                  <CircleAlert
                    size={13}
                    style={{ color: signalVar(signal), flex: "none", marginTop: 3 }}
                  />
                  {reason}
                </li>
              ))}
            </ul>
          ) : (
            <div
              className="row"
              style={{ gap: "var(--s2)", fontSize: "var(--t-sm)", color: "var(--text-3)" }}
            >
              <Gauge size={14} /> Every collected signal is clear.
            </div>
          )}
        </div>
      </div>
    </Panel>
  );
}

/* --------------------------------------------------------- loading state */

/**
 * A skeleton shaped like the result it is waiting for.
 *
 * The first analysis of a session loads a 362 MB checkpoint and takes about six
 * seconds. A spinner for six seconds reads as a hang; a layout that is already
 * the right shape reads as work in progress.
 */
export function ResultSkeleton({ note }: { note?: string }) {
  return (
    <div className="stack stack-4 fade">
      <div className="panel" style={{ padding: "var(--s6)" }}>
        <div className="stack stack-4">
          <div className="skeleton" style={{ height: 11, width: 72 }} />
          <div className="skeleton" style={{ height: 54, width: 190 }} />
          <div className="skeleton" style={{ height: 10, width: "100%" }} />
        </div>
      </div>
      <div className="panel" style={{ padding: "var(--s5)" }}>
        <div className="stack stack-3">
          <div className="skeleton" style={{ height: 13, width: "62%" }} />
          <div className="skeleton" style={{ height: 13, width: "44%" }} />
          <div className="skeleton" style={{ height: 13, width: "53%" }} />
        </div>
      </div>
      {note && (
        <div
          className="row"
          style={{ gap: "var(--s2)", fontSize: "var(--t-sm)", color: "var(--text-3)" }}
        >
          <ScanLine size={14} className="dim" />
          {note}
        </div>
      )}
    </div>
  );
}

