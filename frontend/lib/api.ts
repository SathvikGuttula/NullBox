/**
 * One place that knows where the backend is, and one shape for every result.
 *
 * The URLs used to be hardcoded as http://localhost:8000 inside the call
 * component, so pointing the UI at a backend on another port meant editing
 * source. They come from the environment now, with the same defaults.
 */

export const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export const WS_URL =
  process.env.NEXT_PUBLIC_WS_URL ??
  API_URL.replace(/^http/, "ws");

export interface SpeechCheck {
  has_speech: boolean;
  reason: string;
  rms: number;
  dynamic_range: number;
  seconds: number;
}

export interface AntiSpoof {
  synthetic_probability: number | null;
  raw_probability: number | null;
  model_status: string;
  speech_check?: SpeechCheck;
}

export interface Verification {
  identity: string;
  similarity: number;
  decision: "MATCH" | "UNCERTAIN" | "NO_MATCH";
  threshold_version: string;
  calibrated: boolean;
  enrollment_samples?: number;
  enrollment_consistency?: number;
  speech_seconds_analysed?: number;
  reasons: string[];
}

export interface ContextAssessment {
  risk: number;
  points: number;
  possible: number;
  reasons: string[];
  signals_used: string[];
  missing: string[];
  calibrated: boolean;
}

export interface Contribution {
  source: string;
  points: number;
  detail: string;
}

export interface FusedRisk {
  risk_score: number;
  risk_level: string;
  decision: string;
  contributions: Contribution[];
  reasons: string[];
  available_signals: string[];
  missing_signals: string[];
  calibrated: boolean;
}

export interface PolicyAction {
  code: string;
  description: string;
  blocking: boolean;
}

export interface PolicyDecision {
  decision: string;
  actions: PolicyAction[];
  rationale: string[];
  requires_human: boolean;
  calibrated: boolean;
}

export interface AnalysisResult {
  duration_seconds: number;
  anti_spoof: AntiSpoof;
  speaker: Verification | null;
  context: ContextAssessment | null;
  risk: FusedRisk;
  policy: PolicyDecision;
  notes: string[];
}

export interface SpeakerSummary {
  identity: string;
  samples: number;
  consistency: number;
  created_at: string;
}

export interface SystemStatus {
  service: string;
  version: string;
  ready: boolean;
  anti_spoof: {
    checkpoint_present: boolean;
    checkpoint_mb: number | null;
    loaded: boolean;
    device?: string;
    trained?: boolean;
    thresholds: Record<string, unknown>;
    calibration: Record<string, unknown>;
    checkpoint_metadata?: Record<string, unknown>;
  };
  speaker: {
    speechbrain_installed: boolean;
    enrolled: number;
    identities: string[];
    thresholds: Record<string, unknown>;
    rejected_profiles?: { identity: string; reason: string }[];
  };
  fusion: {
    weights_version: string;
    calibrated: boolean;
    weights: Record<string, number>;
    bands: Record<string, unknown>;
  };
  audio: Record<string, number>;
  warnings: string[];
}

/**
 * FastAPI puts its error message in `detail`. Surfacing that verbatim matters
 * here: the backend's 400s explain exactly what was wrong with the input
 * ("identity may contain only...", "transaction_amount must be int or float"),
 * and replacing them with "Request failed" throws away the useful half.
 */
async function unwrap<T>(response: Response): Promise<T> {
  const text = await response.text();

  let body: unknown;
  try {
    body = text ? JSON.parse(text) : {};
  } catch {
    throw new Error(text.slice(0, 300) || `HTTP ${response.status}`);
  }

  if (!response.ok) {
    const detail = (body as { detail?: unknown })?.detail;
    throw new Error(
      typeof detail === "string"
        ? detail
        : detail
        ? JSON.stringify(detail)
        : `HTTP ${response.status}`
    );
  }

  return body as T;
}

export async function getStatus(): Promise<SystemStatus> {
  return unwrap(await fetch(`${API_URL}/api/v1/status`, { cache: "no-store" }));
}

export async function analyze(
  file: Blob,
  filename: string,
  options: { identity?: string; context?: object } = {}
): Promise<AnalysisResult> {
  const form = new FormData();
  form.append("sample", file, filename);

  if (options.identity) {
    form.append("claimed_identity", options.identity);
  }
  if (options.context && Object.keys(options.context).length > 0) {
    form.append("context", JSON.stringify(options.context));
  }

  return unwrap(
    await fetch(`${API_URL}/api/v1/analyze`, { method: "POST", body: form })
  );
}

export async function listSpeakers(): Promise<{
  count: number;
  thresholds: Record<string, unknown>;
  speakers: SpeakerSummary[];
}> {
  return unwrap(
    await fetch(`${API_URL}/api/v1/speakers`, { cache: "no-store" })
  );
}

export async function enrollSpeaker(
  identity: string,
  samples: Blob[]
): Promise<{
  identity: string;
  samples: number;
  consistency: number;
  note: string;
}> {
  const form = new FormData();
  form.append("identity", identity);
  samples.forEach((blob, index) =>
    form.append("samples", blob, `sample_${index + 1}.wav`)
  );

  return unwrap(
    await fetch(`${API_URL}/api/v1/speakers/enroll`, {
      method: "POST",
      body: form,
    })
  );
}

export async function verifySpeaker(
  identity: string,
  sample: Blob
): Promise<Verification> {
  const form = new FormData();
  form.append("identity", identity);
  form.append("sample", sample, "probe.wav");

  return unwrap(
    await fetch(`${API_URL}/api/v1/speakers/verify`, {
      method: "POST",
      body: form,
    })
  );
}

export async function deleteSpeaker(identity: string): Promise<void> {
  const response = await fetch(
    `${API_URL}/api/v1/speakers/${encodeURIComponent(identity)}`,
    { method: "DELETE" }
  );
  await unwrap(response);
}

export async function createCall(): Promise<{ call_id: string }> {
  return unwrap(
    await fetch(`${API_URL}/api/v1/calls/demo`, { method: "POST" })
  );
}

// -- presentation helpers --------------------------------------------------
//
// Colour no longer lives here. It is decided by `signalForLevel` and friends in
// components/ui.tsx, which map a backend state to one of the five design-system
// signals rather than to a literal — so a palette change happens in the
// stylesheet and nowhere else. The hex tables that used to sit here were the
// last place in the app that hardcoded a colour.


/**
 * A calibrated probability saturates: a confident spoof comes back as
 * 0.9999999998, and rendering that as "100.0%" claims a certainty no
 * measurement supports. Cap the displayed value instead.
 */
export function formatProbability(value: number | null): string {
  if (value === null || value === undefined) return "n/a";
  if (value >= 0.9995) return ">99.9%";
  if (value <= 0.0005) return "<0.1%";
  return `${(value * 100).toFixed(1)}%`;
}

export function prettyDecision(decision: string): string {
  return decision.replace(/_/g, " ");
}
