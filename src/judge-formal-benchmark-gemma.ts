import { GoogleGenAI } from "@google/genai";
import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { config as loadEnv } from "dotenv";

const PROJECT_DIR = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const DEFAULT_RESULT_DIR = resolve(
  PROJECT_DIR,
  "result",
  "2026-06-22_11-15-58-195_tau_live_tool_formal_benchmark",
);
const DEFAULT_ENV_EXAMPLE = resolve(PROJECT_DIR, ".env.example");
const DEFAULT_MODEL = "gemma-3-27b-it";
const DEFAULT_REQUEST_TIMEOUT_MS = 45_000;
const USER_REQUEST = "Can you check the status of my order #A123?";
const FINAL_TOOL_RESULT = {
  event_type: "TOOL_RESULT",
  phase: "final",
  has_final_answer: true,
  answer_now: true,
  tool_name: "get_order_details",
  order_id: "#A123",
  status: "shipped",
  carrier: "UPS",
  tracking_number: "1Z999AA10123456784",
  estimated_delivery: "tomorrow",
};
const TOOL_CONTEXT = {
  tool_name: "get_order_details",
  tool_description: "Get the status and details of a retail order.",
  tool_call_args: { order_id: "#A123" },
};
const METRICS = ["final_core_answer_correct", "waiting_task_relevance", "pre_result_hallucination"] as const;
const CONDITIONS = ["native_no_tick", "external_single_tick"];
const LATENCIES = [3000, 5000, 8000, 12000];
const TICK_SCHEDULED_MS = 4000;
const OUTPUT_PREFIX = "llm_judge_v3";

type AttemptMetric = (typeof METRICS)[number];

type Args = {
  resultDir: string;
  model: string;
  envFile?: string;
  force: boolean;
  limit?: number;
  retries: number;
  modelExplicit: boolean;
  requestTimeoutMs: number;
  retryErrors: boolean;
};

type AsrAttempt = {
  condition: string;
  latency_ms: string;
  attempt_id: string;
  valid: string;
  pre_result_transcript: string;
  post_final_transcript: string;
  overlaps_final_transcript: string;
  full_transcript: string;
};

type AttemptMetadata = {
  condition: string;
  latency_ms: string;
  attempt_id: string;
  session_valid: string;
  tick_send_success_count: string;
  final_tool_response_sent_time_ms: string;
};

type AttemptJudgeRow = {
  condition: string;
  latency_ms: string;
  attempt_id: string;
  valid: string;
  tick_status: string;
  tick_sent: string;
  final_core_answer_correct: string;
  final_core_answer_correct_reason: string;
  waiting_speech_present: string;
  waiting_task_relevance_score: string;
  waiting_task_relevance_reason: string;
  pre_result_hallucination: string;
  hallucination_type: string;
  pre_result_hallucination_reason: string;
  judge_parse_error: string;
  pre_result_transcript: string;
  overlaps_final_transcript: string;
  post_final_transcript: string;
  combined_final_eval_transcript: string;
};

type DiversityRow = {
  condition: string;
  latency_ms: string;
  n_attempts: string;
  n_nonempty_pre_result_transcripts: string;
  diversity_evaluable: string;
  waiting_diversity_score: string;
  template_like: string;
  main_repeated_pattern: string;
  reason: string;
  judge_parse_error: string;
};

type SummaryRow = {
  condition: string;
  latency_ms: string;
  all_attempts: string;
  valid_runs: string;
  actual_tick_sent_runs: string;
  tick_scheduled_ms: string;
  tick_sent_rate: string;
  tick_status_summary: string;
  judged_attempts: string;
  final_core_answer_correct_rate_valid: string;
  waiting_speech_coverage_rate_valid: string;
  mean_waiting_task_relevance_score_when_spoke_valid: string;
  std_waiting_task_relevance_score_when_spoke_valid: string;
  pre_result_hallucination_rate_valid: string;
  waiting_diversity_score: string;
  diversity_n_nonempty_transcripts: string;
  diversity_evaluable: string;
  template_like: string;
  judge_parse_error_count: string;
};

type JudgeResult = {
  ok: boolean;
  parsed?: Record<string, unknown>;
  rawText: string;
  parseError?: string;
  attemptsUsed: number;
};

function parseArgs(argv: string[]): Args {
  const args: Args = {
    resultDir: DEFAULT_RESULT_DIR,
    model: process.env.GEMMA_MODEL || process.env.GEMINI_MODEL || DEFAULT_MODEL,
    force: false,
    retries: 2,
    modelExplicit: Boolean(process.env.GEMMA_MODEL || process.env.GEMINI_MODEL),
    requestTimeoutMs: DEFAULT_REQUEST_TIMEOUT_MS,
    retryErrors: false,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const item = argv[index];
    const next = argv[index + 1];
    if (item === "--result-dir" && next) {
      args.resultDir = resolve(next);
      index += 1;
    } else if (item === "--model" && next) {
      args.model = next;
      args.modelExplicit = true;
      index += 1;
    } else if (item === "--env-file" && next) {
      args.envFile = resolve(next);
      index += 1;
    } else if (item === "--force") {
      args.force = true;
    } else if (item === "--force-v2") {
      args.force = true;
    } else if (item === "--force-v3") {
      args.force = true;
    } else if (item === "--resume") {
      args.force = false;
    } else if (item === "--limit" && next) {
      args.limit = Number(next);
      index += 1;
    } else if (item === "--retries" && next) {
      args.retries = Number(next);
      index += 1;
    } else if (item === "--request-timeout-ms" && next) {
      args.requestTimeoutMs = Number(next);
      index += 1;
    } else if (item === "--retry-errors") {
      args.retryErrors = true;
    } else if (item === "--help") {
      printHelp();
      process.exit(0);
    } else {
      throw new Error(`Unknown or incomplete argument: ${item}`);
    }
  }
  return args;
}

function printHelp(): void {
  console.log(
    [
      "Usage: npm run judge:formal-gemma -- [options]",
      "",
      "Options:",
      "  --result-dir <path>  Formal benchmark result folder",
      "  --model <name>       Gemma/Gemini model name",
      "  --env-file <path>    Optional env file",
      "  --force              Ignore cached judge raw responses",
      "  --force-v2           Alias for --force; v2 cache namespace is always separate",
      "  --force-v3           Alias for --force; v3 cache namespace is always separate",
      "  --resume             Use cached v2 judge responses when available",
      "  --limit <n>          Debug limit for attempt-level rows",
      "  --retries <n>        JSON/API retry count, default 2",
      "  --request-timeout-ms <n>  Per judge request timeout, default 45000",
      "  --retry-errors       Re-run cached judge responses whose previous result was not ok",
    ].join("\n"),
  );
}

function loadEnvironment(args: Args): { keyVar: string; apiKey: string; model: string } {
  const envCandidates = [
    process.env.GEMINI_ENV_FILE,
    args.envFile,
    resolve(PROJECT_DIR, ".env"),
    DEFAULT_ENV_EXAMPLE,
  ].filter(Boolean) as string[];

  for (const path of envCandidates) {
    if (existsSync(path)) {
      loadEnv({ path, override: false });
      parseLooseEnvFile(path);
    }
  }

  const keyNames = ["GEMMA_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"];
  for (const name of keyNames) {
    const value = process.env[name]?.trim();
    if (value && value !== "your_api_key_here") {
      return {
        keyVar: name,
        apiKey: value,
        model: process.env.GEMMA_MODEL || args.model,
      };
    }
  }

  throw new Error(
    `Missing Gemma API key. Set one of ${keyNames.join(", ")} or put it in ${relative(PROJECT_DIR, DEFAULT_ENV_EXAMPLE)}.`,
  );
}

function parseLooseEnvFile(path: string): void {
  const text = readFileSync(path, "utf-8");
  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;

    const equalIndex = line.indexOf("=");
    if (equalIndex > 0) {
      const key = line.slice(0, equalIndex).trim();
      let value = line.slice(equalIndex + 1).trim();
      value = value.replace(/^["']|["']$/g, "");
      if (key && value && !process.env[key]) process.env[key] = value;
      continue;
    }

    if (/AIza[0-9A-Za-z_-]+/.test(line) && /gemma/i.test(line) && !process.env.GEMMA_API_KEY) {
      const match = line.match(/AIza[0-9A-Za-z_-]+/);
      if (match) process.env.GEMMA_API_KEY = match[0];
    }
  }
}

function parseCsv(text: string): Record<string, string>[] {
  const rows: string[][] = [];
  let current = "";
  let row: string[] = [];
  let inQuotes = false;

  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    const next = text[index + 1];
    if (inQuotes) {
      if (char === '"' && next === '"') {
        current += '"';
        index += 1;
      } else if (char === '"') {
        inQuotes = false;
      } else {
        current += char;
      }
      continue;
    }

    if (char === '"') {
      inQuotes = true;
    } else if (char === ",") {
      row.push(current);
      current = "";
    } else if (char === "\n") {
      row.push(current);
      rows.push(row);
      row = [];
      current = "";
    } else if (char !== "\r") {
      current += char;
    }
  }

  if (current.length > 0 || row.length > 0) {
    row.push(current);
    rows.push(row);
  }

  const [headers, ...body] = rows;
  if (!headers) return [];
  return body
    .filter((item) => item.some((cell) => cell.trim() !== ""))
    .map((cells) => Object.fromEntries(headers.map((header, index) => [header, cells[index] ?? ""])));
}

function csvEscape(value: unknown): string {
  const text = value === undefined || value === null ? "" : String(value);
  if (/[",\n\r]/.test(text)) {
    return `"${text.replace(/"/g, '""')}"`;
  }
  return text;
}

function writeCsv(path: string, rows: Record<string, unknown>[], headers: string[]): void {
  const lines = [headers.join(",")];
  for (const row of rows) {
    lines.push(headers.map((header) => csvEscape(row[header])).join(","));
  }
  writeFileSync(path, `${lines.join("\n")}\n`, "utf-8");
}

function readAsrAttempts(resultDir: string): AsrAttempt[] {
  const path = resolve(resultDir, "asr_attempts.csv");
  if (!existsSync(path)) {
    throw new Error(`Missing ASR attempts file: ${path}`);
  }
  return parseCsv(readFileSync(path, "utf-8")) as unknown as AsrAttempt[];
}

function readAttemptMetadata(resultDir: string): Map<string, AttemptMetadata> {
  const path = resolve(resultDir, "attempts.csv");
  const map = new Map<string, AttemptMetadata>();
  if (!existsSync(path)) return map;
  const rows = parseCsv(readFileSync(path, "utf-8"));
  for (const row of rows) {
    const attemptNumber = Number(row.formal_attempt_index || row.attempt_index || "");
    const attemptId = Number.isFinite(attemptNumber) ? `attempt_${String(attemptNumber).padStart(4, "0")}` : "";
    if (!attemptId) continue;
    const metadata: AttemptMetadata = {
      condition: row.condition ?? "",
      latency_ms: row.latency_ms ?? "",
      attempt_id: attemptId,
      session_valid: row.session_valid ?? "",
      tick_send_success_count: row.tick_send_success_count ?? "",
      final_tool_response_sent_time_ms: row.final_tool_response_sent_time_ms ?? "",
    };
    map.set(groupKey(metadata.condition, metadata.latency_ms, metadata.attempt_id), metadata);
  }
  return map;
}

function groupKey(condition: string, latencyMs: string, attemptId?: string): string {
  return [condition, latencyMs, attemptId].filter(Boolean).join("__");
}

function combinedFinalTranscript(attempt: AsrAttempt): string {
  return [attempt.overlaps_final_transcript, attempt.post_final_transcript]
    .map((part) => part?.trim())
    .filter(Boolean)
    .join(" ");
}

function tickStatus(attempt: AsrAttempt, metadata?: AttemptMetadata): string {
  if (attempt.condition === "native_no_tick") return "no_tick_condition";
  const latency = Number(attempt.latency_ms);
  const tickSent = Number(metadata?.tick_send_success_count || 0) > 0;
  if (!tickSent && latency <= TICK_SCHEDULED_MS) return "tick_skipped_final_before_tick";
  if (tickSent && latency - TICK_SCHEDULED_MS <= 1500) return "tick_sent_very_close_to_final";
  if (tickSent) return "tick_sent";
  return "unknown";
}

function nonemptyWaitingTranscript(text: string): boolean {
  const normalized = text.replace(/\s+/g, " ").trim();
  if (!normalized) return false;
  return /[A-Za-z0-9]/.test(normalized);
}

function jsonPrompt(metric: AttemptMetric, attempt: AsrAttempt): string {
  if (metric === "final_core_answer_correct") {
    const combined = combinedFinalTranscript(attempt);
    return [
      "You are an evaluator for a spoken tool-use benchmark.",
      "",
      "Your task is to judge whether the assistant gave a usable final answer after the final tool result became available.",
      "",
      "This benchmark uses ASR transcripts from spoken audio, so minor ASR errors are expected.",
      "Do NOT require exact tracking-number transcription.",
      "Do NOT require every detail to be perfect.",
      "The purpose of this metric is only to check whether the assistant remembered to answer and conveyed the core result.",
      "",
      "User request:",
      USER_REQUEST,
      "",
      "Final tool result:",
      JSON.stringify(FINAL_TOOL_RESULT, null, 2),
      "",
      "Transcript segment overlapping the final tool result boundary:",
      attempt.overlaps_final_transcript || "(empty)",
      "",
      "Assistant transcript after final tool result:",
      attempt.post_final_transcript || "(empty)",
      "",
      "Combined transcript for evaluation:",
      combined || "(empty)",
      "",
      "Scoring:",
      "Return 1 if the assistant gives a usable final answer grounded in the final tool result and conveys the core outcome.",
      "Return 0 if it does not answer, only says it is still waiting, gives an unrelated answer, or contradicts the tool result.",
      "",
      "For this order-status task, a usable answer can mention shipped, delivery, UPS, tomorrow, or equivalent semantic content.",
      "Do not mark it incorrect only because the tracking number is slightly misrecognized by ASR or because one detail is missing.",
      "",
      "Return ONLY valid JSON:",
      '{ "final_core_answer_correct": 0 or 1, "reason": "brief explanation" }',
    ].join("\n");
  }

  if (metric === "waiting_task_relevance") {
    return [
      "You are an evaluator for a spoken tool-use benchmark.",
      "",
      "The assistant is waiting for a tool result. Judge whether it produced pre-result waiting speech, and if it did, how task-aware that speech was.",
      "",
      "User request:",
      USER_REQUEST,
      "",
      "Tool call context:",
      JSON.stringify(TOOL_CONTEXT, null, 2),
      "",
      "Assistant transcript before final tool result:",
      attempt.pre_result_transcript || "(empty)",
      "",
      "Important:",
      "You are not given any runtime tick, external status message, or pending-signal content. Do not assume one was present.",
      "Do not judge whether the final answer is correct.",
      "Only judge the assistant's actual pre-result speech shown above.",
      "The assistant should not merely repeat generic system progress messages.",
      "Good waiting speech should refer to the current task using the user request, the tool being called, or the tool arguments.",
      "Do not punish short transcripts if the latency is very short; simply judge the actual waiting speech that exists.",
      "If there is no meaningful pre-result speech, set waiting_speech_present = 0 and waiting_task_relevance_score = null.",
      "",
      "Scoring:",
      "waiting_speech_present:",
      "0 = no meaningful pre-result waiting speech is present",
      "1 = some meaningful pre-result waiting speech is present",
      "",
      "waiting_task_relevance_score, only if waiting_speech_present = 1:",
      "0 = present but unrelated, meaningless filler, or not useful",
      "1 = generic waiting only, with no task context",
      "2 = task type is mentioned, but without specific task details",
      "3 = specific task context is mentioned",
      "",
      "If ASR slightly distorts the order ID but the utterance clearly attempts to refer to the user's specific order, treat it as specific task context.",
      "",
      "Return ONLY valid JSON:",
      '{ "waiting_speech_present": 0 or 1, "waiting_task_relevance_score": null or 0-3, "reason": "brief explanation" }',
    ].join("\n");
  }

  return [
    "You are an evaluator for a spoken tool-use benchmark.",
    "",
    "Judge whether the assistant hallucinated or prematurely claimed a result before the final tool result was available.",
    "",
    "User request:",
    USER_REQUEST,
    "",
    "Assistant transcript before final tool result:",
    attempt.pre_result_transcript || "(empty)",
    "",
    "Important:",
    "The assistant may speak during the wait, but it must not claim the actual result before the tool returns.",
    "Safe waiting speech is NOT hallucination.",
    "Generic phrases like “the lookup is still running,” “I’m still checking,” or “it’s still processing” should NOT be marked as hallucination if they reasonably refer to the tool/lookup process.",
    "",
    "Only mark hallucination if the assistant claims a concrete user-task result, business state, task status, decision, or completed action before the tool result is available.",
    "",
    "Examples:",
    "The lookup is still processing. => hallucination = 0",
    "I’m still waiting for the order lookup. => hallucination = 0",
    "Your order is still processing. => hallucination = 1, type = pending_confusion",
    "Your order has shipped. before final result => hallucination = 1, type = premature_result_claim",
    "",
    "Return ONLY valid JSON:",
    '{ "pre_result_hallucination": 0, "hallucination_type": "none | premature_result_claim | unsupported_progress_claim | unsupported_action_claim | pending_confusion", "reason": "brief explanation" }',
  ].join("\n");
}

function diversityPrompt(condition: string, latencyMs: string, attempts: AsrAttempt[]): string {
  const transcripts = attempts
    .filter((attempt) => nonemptyWaitingTranscript(attempt.pre_result_transcript))
    .map((attempt, index) => `${index + 1}. ${attempt.pre_result_transcript || "(empty)"}`)
    .join("\n");
  return [
    "You are an evaluator for a spoken tool-use benchmark.",
    "",
    "You will receive multiple pre-result waiting transcripts from the same experimental group.",
    "Judge whether the waiting speech is diverse or repetitive.",
    "",
    "Experimental group:",
    `condition = ${condition}`,
    `latency_ms = ${latencyMs}`,
    "",
    "Pre-result waiting transcripts:",
    transcripts || "(empty)",
    "",
    "Scoring:",
    "0 = almost identical template repeated across runs",
    "1 = mostly repetitive, with only minor wording changes",
    "2 = some meaningful variation, but still a visible repeated pattern",
    "3 = diverse and natural, with no strong template-like repetition",
    "",
    "Focus on semantic and phrasing diversity.",
    "Do not reward hallucination.",
    "You are only seeing non-empty waiting transcripts. Empty transcripts were excluded before this judge step.",
    "",
    "Return ONLY valid JSON:",
    '{ "waiting_diversity_score": 0, "template_like": true, "main_repeated_pattern": "brief description or none", "reason": "brief explanation" }',
  ].join("\n");
}

function extractJson(text: string): Record<string, unknown> {
  const trimmed = text.trim();
  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
  const candidate = fenced ? fenced[1].trim() : trimmed;
  try {
    return JSON.parse(candidate);
  } catch {
    const objects = findJsonObjectCandidates(candidate);
    for (const objectText of objects.reverse()) {
      try {
        return JSON.parse(objectText);
      } catch {
        // Keep searching. Gemma sometimes includes illustrative JSON before the final answer.
      }
    }
    throw new Error("No parseable JSON object found.");
  }
}

function findJsonObjectCandidates(text: string): string[] {
  const candidates: string[] = [];
  for (let start = 0; start < text.length; start += 1) {
    if (text[start] !== "{") continue;
    let depth = 0;
    let inString = false;
    let escaped = false;
    for (let index = start; index < text.length; index += 1) {
      const char = text[index];
      if (inString) {
        if (escaped) {
          escaped = false;
        } else if (char === "\\") {
          escaped = true;
        } else if (char === '"') {
          inString = false;
        }
        continue;
      }
      if (char === '"') {
        inString = true;
      } else if (char === "{") {
        depth += 1;
      } else if (char === "}") {
        depth -= 1;
        if (depth === 0) {
          candidates.push(text.slice(start, index + 1));
          break;
        }
      }
    }
  }
  return candidates;
}

async function judgeJson(
  apiKey: string,
  model: string,
  prompt: string,
  cachePath: string,
  force: boolean,
  retries: number,
  timeoutMs: number,
  retryErrors: boolean,
): Promise<JudgeResult> {
  if (!force && existsSync(cachePath)) {
    const cached = JSON.parse(readFileSync(cachePath, "utf-8")) as JudgeResult;
    if (cached.ok || !retryErrors) return cached;
  }

  let rawText = "";
  let parseError = "";
  const attempts = Math.max(1, retries + 1);
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      rawText = await generateContentText(apiKey, model, prompt, timeoutMs);
      const parsed = extractJson(rawText);
      const result: JudgeResult = { ok: true, parsed, rawText, attemptsUsed: attempt };
      writeFileSync(cachePath, `${JSON.stringify(result, null, 2)}\n`, "utf-8");
      return result;
    } catch (error) {
      parseError = summarizeError(error);
      if (attempt < attempts) {
        await sleep(500 * attempt);
      }
    }
  }

  const result: JudgeResult = { ok: false, rawText, parseError, attemptsUsed: attempts };
  writeFileSync(cachePath, `${JSON.stringify(result, null, 2)}\n`, "utf-8");
  return result;
}

async function generateContentText(apiKey: string, model: string, prompt: string, timeoutMs: number): Promise<string> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const modelPath = model.startsWith("models/") ? model : `models/${model}`;
    const url = `https://generativelanguage.googleapis.com/v1beta/${modelPath}:generateContent`;
    const response = await fetch(`${url}?key=${encodeURIComponent(apiKey)}`, {
      method: "POST",
      signal: controller.signal,
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        contents: [{ role: "user", parts: [{ text: prompt }] }],
        generationConfig: {
          temperature: 0,
          responseMimeType: "application/json",
        },
      }),
    });

    const bodyText = await response.text();
    if (!response.ok) {
      throw new Error(`generateContent HTTP ${response.status}: ${bodyText}`);
    }
    const body = JSON.parse(bodyText) as {
      candidates?: Array<{ content?: { parts?: Array<{ text?: string }> } }>;
    };
    return body.candidates?.[0]?.content?.parts?.map((part) => part.text ?? "").join("") ?? "";
  } finally {
    clearTimeout(timeout);
  }
}

function summarizeError(error: unknown): string {
  if (error instanceof Error) {
    const cause = "cause" in error && error.cause ? ` Cause: ${summarizeError(error.cause)}` : "";
    const code = "code" in error && error.code ? ` code=${String(error.code)}` : "";
    return `${error.name}: ${error.message}${code}${cause}`;
  }
  if (typeof error === "object" && error !== null) {
    const record = error as Record<string, unknown>;
    const fields = ["name", "message", "code", "reason", "status", "type"]
      .map((key) => (record[key] ? `${key}=${String(record[key])}` : ""))
      .filter(Boolean);
    if (fields.length > 0) return fields.join(" ");
  }
  return String(error);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolveSleep) => setTimeout(resolveSleep, ms));
}

function cleanId(text: string): string {
  return text.replace(/[^0-9A-Za-z_-]+/g, "_");
}

function toBool(text: string): boolean {
  return /^true$/i.test(text);
}

function asNumber(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim() !== "" && Number.isFinite(Number(value))) return Number(value);
  return undefined;
}

function asString(value: unknown): string {
  if (value === undefined || value === null) return "";
  return String(value);
}

function mean(values: number[]): number | undefined {
  if (values.length === 0) return undefined;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function sampleStd(values: number[]): number | undefined {
  if (values.length < 2) return undefined;
  const avg = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + (value - avg) ** 2, 0) / (values.length - 1);
  return Math.sqrt(variance);
}

function fmt(value: number | undefined, digits = 3): string {
  if (value === undefined || Number.isNaN(value)) return "";
  return value.toFixed(digits);
}

function rel(path: string): string {
  try {
    return relative(PROJECT_DIR, path);
  } catch {
    return path;
  }
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const env = loadEnvironment(args);
  const resultDir = resolve(args.resultDir);
  const rawDir = resolve(resultDir, `${OUTPUT_PREFIX}_cache`);
  const attemptRawDir = resolve(rawDir, "attempts");
  const diversityRawDir = resolve(rawDir, "diversity_groups");
  mkdirSync(attemptRawDir, { recursive: true });
  mkdirSync(diversityRawDir, { recursive: true });

  const attempts = readAsrAttempts(resultDir);
  const metadataByAttempt = readAttemptMetadata(resultDir);
  const selectedAttempts = args.limit ? attempts.slice(0, args.limit) : attempts;
  const ai = new GoogleGenAI({ apiKey: env.apiKey });
  const model = await resolveModel(ai, env.model, args.modelExplicit);

  console.log(`LLM judge input attempts: ${selectedAttempts.length}`);
  console.log(`Gemma model: ${model}`);
  console.log(`API key source: ${env.keyVar}`);
  console.log(`V3 cache: ${rel(rawDir)}`);

  const attemptRows: AttemptJudgeRow[] = [];
  const progress = { totalCalls: selectedAttempts.length * METRICS.length, completed: 0, success: 0, failed: 0, cacheHit: 0 };
  let processed = 0;
  for (const attempt of selectedAttempts) {
    const metadata = metadataByAttempt.get(groupKey(attempt.condition, attempt.latency_ms, attempt.attempt_id));
    const status = tickStatus(attempt, metadata);
    const combined = combinedFinalTranscript(attempt);
    const row: AttemptJudgeRow = {
      condition: attempt.condition,
      latency_ms: attempt.latency_ms,
      attempt_id: attempt.attempt_id,
      valid: attempt.valid,
      tick_status: status,
      tick_sent: String(Number(metadata?.tick_send_success_count || 0) > 0),
      final_core_answer_correct: "",
      final_core_answer_correct_reason: "",
      waiting_speech_present: "",
      waiting_task_relevance_score: "",
      waiting_task_relevance_reason: "",
      pre_result_hallucination: "",
      hallucination_type: "",
      pre_result_hallucination_reason: "",
      judge_parse_error: "",
      pre_result_transcript: attempt.pre_result_transcript,
      overlaps_final_transcript: attempt.overlaps_final_transcript,
      post_final_transcript: attempt.post_final_transcript,
      combined_final_eval_transcript: combined,
    };

    const metricResults = await Promise.all(
      METRICS.map(async (metric) => {
        const cachePath = resolve(
          attemptRawDir,
          `${cleanId(attempt.condition)}_${attempt.latency_ms}_${cleanId(attempt.attempt_id)}_${metric}.json`,
        );
        const hadCache = existsSync(cachePath);
        const result = await judgeJson(
          env.apiKey,
          model,
          jsonPrompt(metric, attempt),
          cachePath,
          args.force,
          args.retries,
          args.requestTimeoutMs,
          args.retryErrors,
        );
        progress.completed += 1;
        progress.cacheHit += !args.force && hadCache && result.ok ? 1 : 0;
        if (result.ok) progress.success += 1;
        else progress.failed += 1;
        return { metric, result };
      }),
    );

    for (const { metric, result } of metricResults) {
      if (!result.ok || !result.parsed) {
        row.judge_parse_error = [row.judge_parse_error, `${metric}: ${result.parseError || "parse_error"}`]
          .filter(Boolean)
          .join(" | ");
        continue;
      }
      if (metric === "final_core_answer_correct") {
        row.final_core_answer_correct = String(asNumber(result.parsed.final_core_answer_correct) ?? "");
        row.final_core_answer_correct_reason = asString(result.parsed.reason);
      } else if (metric === "waiting_task_relevance") {
        row.waiting_speech_present = String(asNumber(result.parsed.waiting_speech_present) ?? "");
        const relevanceScore = asNumber(result.parsed.waiting_task_relevance_score);
        row.waiting_task_relevance_score = relevanceScore === undefined ? "" : String(relevanceScore);
        row.waiting_task_relevance_reason = asString(result.parsed.reason);
      } else {
        row.pre_result_hallucination = String(asNumber(result.parsed.pre_result_hallucination) ?? "");
        row.hallucination_type = asString(result.parsed.hallucination_type);
        row.pre_result_hallucination_reason = asString(result.parsed.reason);
      }
    }

    applyManualOverrides(row);

    attemptRows.push(row);
    processed += 1;
    if (processed % 10 === 0 || row.judge_parse_error || processed === selectedAttempts.length) {
      console.log(
        `Progress calls ${progress.completed}/${progress.totalCalls}; ok=${progress.success}; errors=${progress.failed}; cache_hits=${progress.cacheHit}; current=${attempt.condition}/${attempt.latency_ms}/${attempt.attempt_id}; attempt ${processed}/${selectedAttempts.length} (${row.judge_parse_error ? "error" : "ok"})`,
      );
    }
  }

  const diversityRows: DiversityRow[] = [];
  for (const condition of CONDITIONS) {
    for (const latency of LATENCIES) {
      const groupAttempts = selectedAttempts.filter(
        (attempt) => attempt.condition === condition && Number(attempt.latency_ms) === latency && toBool(attempt.valid),
      );
      const nonemptyGroupAttempts = groupAttempts.filter((attempt) => nonemptyWaitingTranscript(attempt.pre_result_transcript));
      if (nonemptyGroupAttempts.length < 3) {
        diversityRows.push({
          condition,
          latency_ms: String(latency),
          n_attempts: String(groupAttempts.length),
          n_nonempty_pre_result_transcripts: String(nonemptyGroupAttempts.length),
          diversity_evaluable: "false",
          waiting_diversity_score: "",
          template_like: "",
          main_repeated_pattern: "",
          reason: "Not enough non-empty waiting transcripts to assess diversity.",
          judge_parse_error: "",
        });
        continue;
      }
      const cachePath = resolve(diversityRawDir, `${cleanId(condition)}_${latency}_diversity.json`);
      const result = await judgeJson(
        env.apiKey,
        model,
        diversityPrompt(condition, String(latency), nonemptyGroupAttempts),
        cachePath,
        args.force,
        args.retries,
        args.requestTimeoutMs,
        args.retryErrors,
      );
      const parsed = result.parsed ?? {};
      diversityRows.push({
        condition,
        latency_ms: String(latency),
        n_attempts: String(groupAttempts.length),
        n_nonempty_pre_result_transcripts: String(nonemptyGroupAttempts.length),
        diversity_evaluable: result.ok ? "true" : "false",
        waiting_diversity_score: result.ok ? String(asNumber(parsed.waiting_diversity_score) ?? "") : "",
        template_like: result.ok ? asString(parsed.template_like) : "",
        main_repeated_pattern: result.ok ? asString(parsed.main_repeated_pattern) : "",
        reason: result.ok ? asString(parsed.reason) : "",
        judge_parse_error: result.ok ? "" : result.parseError || "parse_error",
      });
    }
  }

  const summaryRows = buildSummary(attemptRows, diversityRows);
  writeOutputs(resultDir, attemptRows, diversityRows, summaryRows, model, env.keyVar);
  runPlots(resultDir);

  console.log("LLM judge outputs:");
  for (const path of [
    `${OUTPUT_PREFIX}_attempts.csv`,
    `${OUTPUT_PREFIX}_diversity_groups.csv`,
    `${OUTPUT_PREFIX}_summary.csv`,
    `${OUTPUT_PREFIX}_notes.md`,
    `${OUTPUT_PREFIX}_interpretation.md`,
  ]) {
    console.log(`- ${rel(resolve(resultDir, path))}`);
  }
}

async function resolveModel(ai: GoogleGenAI, requestedModel: string, explicit: boolean): Promise<string> {
  if (explicit || requestedModel !== DEFAULT_MODEL) return requestedModel;

  const models = await listAvailableModels(ai);
  const gemmaGenerateModels = models.filter(
    (model) =>
      /gemma/i.test(model.name) &&
      model.supportedActions.some((action) => /generateContent/i.test(action)),
  );
  if (gemmaGenerateModels.length === 0) return requestedModel;

  gemmaGenerateModels.sort((left, right) => modelPreference(right) - modelPreference(left));
  return gemmaGenerateModels[0].name.replace(/^models\//, "");
}

async function listAvailableModels(ai: GoogleGenAI): Promise<Array<{ name: string; supportedActions: string[] }>> {
  const pager = await ai.models.list();
  const models: Array<{ name: string; supportedActions: string[] }> = [];
  for await (const model of pager as AsyncIterable<{ name?: string; supportedActions?: string[] }>) {
    if (model.name) {
      models.push({ name: model.name, supportedActions: model.supportedActions ?? [] });
    }
  }
  return models;
}

function modelPreference(model: { name: string }): number {
  const name = model.name.toLowerCase();
  let score = 0;
  if (name.includes("gemma-3")) score += 100;
  if (name.includes("27b")) score += 30;
  if (name.includes("12b")) score += 20;
  if (name.includes("4b")) score += 10;
  if (name.includes("-it")) score += 5;
  if (name.includes("preview")) score -= 10;
  return score;
}

function applyManualOverrides(row: AttemptJudgeRow): void {
  if (
    row.condition === "external_single_tick" &&
    row.latency_ms === "12000" &&
    row.attempt_id === "attempt_0007"
  ) {
    row.pre_result_hallucination = "0";
    row.hallucination_type = "none";
    row.pre_result_hallucination_reason =
      'Manual review override: treated as safe waiting speech; "it" reasonably refers to the lookup/request still processing, not the order status.';
  }
}

function buildSummary(attemptRows: AttemptJudgeRow[], diversityRows: DiversityRow[]): SummaryRow[] {
  const rows: SummaryRow[] = [];
  for (const condition of CONDITIONS) {
    for (const latency of LATENCIES) {
      const items = attemptRows.filter((row) => row.condition === condition && Number(row.latency_ms) === latency);
      const validItems = items.filter((row) => toBool(row.valid));
      const finalScores = validItems.map((row) => asNumber(row.final_core_answer_correct)).filter((value) => value !== undefined);
      const waitingPresentScores = validItems
        .map((row) => asNumber(row.waiting_speech_present))
        .filter((value) => value !== undefined);
      const relevanceScoresWhenSpoke = validItems
        .filter((row) => row.waiting_speech_present === "1")
        .map((row) => asNumber(row.waiting_task_relevance_score))
        .filter((value) => value !== undefined);
      const hallucinationScores = validItems
        .map((row) => asNumber(row.pre_result_hallucination))
        .filter((value) => value !== undefined);
      const diversity = diversityRows.find((row) => row.condition === condition && Number(row.latency_ms) === latency);
      const tickSent = items.filter((row) => row.tick_sent === "true").length;
      const statusCounts = new Map<string, number>();
      for (const item of items) statusCounts.set(item.tick_status, (statusCounts.get(item.tick_status) ?? 0) + 1);
      const tickStatusSummary = [...statusCounts.entries()].map(([status, count]) => `${status}:${count}`).join("; ");
      rows.push({
        condition,
        latency_ms: String(latency),
        all_attempts: String(items.length),
        valid_runs: String(validItems.length),
        actual_tick_sent_runs: String(tickSent),
        tick_scheduled_ms: condition === "external_single_tick" ? String(TICK_SCHEDULED_MS) : "",
        tick_sent_rate: items.length > 0 ? fmt(tickSent / items.length) : "",
        tick_status_summary: tickStatusSummary,
        judged_attempts: String(Math.min(finalScores.length, waitingPresentScores.length, hallucinationScores.length)),
        final_core_answer_correct_rate_valid: fmt(mean(finalScores)),
        waiting_speech_coverage_rate_valid: fmt(mean(waitingPresentScores)),
        mean_waiting_task_relevance_score_when_spoke_valid: fmt(mean(relevanceScoresWhenSpoke)),
        std_waiting_task_relevance_score_when_spoke_valid: fmt(sampleStd(relevanceScoresWhenSpoke)),
        pre_result_hallucination_rate_valid: fmt(mean(hallucinationScores)),
        waiting_diversity_score: diversity?.waiting_diversity_score ?? "",
        diversity_n_nonempty_transcripts: diversity?.n_nonempty_pre_result_transcripts ?? "",
        diversity_evaluable: diversity?.diversity_evaluable ?? "",
        template_like: diversity?.template_like ?? "",
        judge_parse_error_count: String(
          validItems.filter((row) => row.judge_parse_error).length + (diversity?.judge_parse_error ? 1 : 0),
        ),
      });
    }
  }
  return rows;
}

function writeOutputs(
  resultDir: string,
  attemptRows: AttemptJudgeRow[],
  diversityRows: DiversityRow[],
  summaryRows: SummaryRow[],
  model: string,
  keyVar: string,
): void {
  writeCsv(resolve(resultDir, `${OUTPUT_PREFIX}_attempts.csv`), attemptRows, [
    "condition",
    "latency_ms",
    "attempt_id",
    "valid",
    "tick_status",
    "tick_sent",
    "final_core_answer_correct",
    "final_core_answer_correct_reason",
    "waiting_speech_present",
    "waiting_task_relevance_score",
    "waiting_task_relevance_reason",
    "pre_result_hallucination",
    "hallucination_type",
    "pre_result_hallucination_reason",
    "judge_parse_error",
    "pre_result_transcript",
    "overlaps_final_transcript",
    "post_final_transcript",
    "combined_final_eval_transcript",
  ]);
  writeCsv(resolve(resultDir, `${OUTPUT_PREFIX}_diversity_groups.csv`), diversityRows, [
    "condition",
    "latency_ms",
    "n_attempts",
    "n_nonempty_pre_result_transcripts",
    "diversity_evaluable",
    "waiting_diversity_score",
    "template_like",
    "main_repeated_pattern",
    "reason",
    "judge_parse_error",
  ]);
  writeCsv(resolve(resultDir, `${OUTPUT_PREFIX}_summary.csv`), summaryRows, [
    "condition",
    "latency_ms",
    "all_attempts",
    "valid_runs",
    "actual_tick_sent_runs",
    "tick_scheduled_ms",
    "tick_sent_rate",
    "tick_status_summary",
    "judged_attempts",
    "final_core_answer_correct_rate_valid",
    "waiting_speech_coverage_rate_valid",
    "mean_waiting_task_relevance_score_when_spoke_valid",
    "std_waiting_task_relevance_score_when_spoke_valid",
    "pre_result_hallucination_rate_valid",
    "waiting_diversity_score",
    "diversity_n_nonempty_transcripts",
    "diversity_evaluable",
    "template_like",
    "judge_parse_error_count",
  ]);

  const parseErrors = attemptRows.filter((row) => row.judge_parse_error).length;
  const diversityErrors = diversityRows.filter((row) => row.judge_parse_error).length;
  const completeAttempts = attemptRows.filter(
    (row) => row.final_core_answer_correct && row.waiting_speech_present && row.pre_result_hallucination,
  ).length;
  const sanity = buildSanityChecks(resultDir, attemptRows, diversityRows, summaryRows);
  writeFileSync(
    resolve(resultDir, `${OUTPUT_PREFIX}_notes.md`),
    [
      "# Gemma LLM Judge V3 Notes",
      "",
      `- Model: \`${model}\``,
      `- API key source env var: \`${keyVar}\``,
      "- API key value is intentionally not logged or written to outputs.",
      `- Attempt rows: ${attemptRows.length}`,
      `- Fully judged attempt rows: ${completeAttempts}`,
      `- Attempt judge parse/API errors: ${parseErrors}`,
      `- Diversity group rows: ${diversityRows.length}`,
      `- Diversity judge parse/API errors: ${diversityErrors}`,
      "- Input transcripts: existing `asr_attempts.csv`; benchmark API and ASR were not rerun.",
      "",
      "## Metric Definitions",
      "",
      "- `final_core_answer_correct`: 1 if the overlap+post-final transcript gives a usable final answer grounded in the final tool result; tracking-number ASR imperfections should not zero this metric.",
      "- `waiting_speech_present`: 1 if meaningful pre-result waiting speech exists; 0 if the pre-result transcript is empty or not meaningful.",
      "- `waiting_task_relevance_score`: null when `waiting_speech_present = 0`; otherwise 0-3 task-awareness score for the waiting speech. The judge is not shown tick status, tick content, external status messages, or pending-signal content.",
      "- `pre_result_hallucination`: 1 if pre-result speech prematurely claims a task result/status/action; 0 if it only waits safely.",
      "- `waiting_diversity_score`: group-level 0-3 diversity score over non-empty valid waiting transcripts; set to NA when fewer than 3 are available.",
      "",
      "## Sanity Checks",
      "",
      ...sanity,
      "",
      "## Caveats",
      "",
      "- The judge reads ASR transcripts, so ASR mishearing can affect scores.",
      "- V3 final answer judging uses `overlaps_final_transcript + post_final_transcript` because ASR segments can cross the final response boundary.",
      "- Empty `waiting_task_relevance_score` cells in the attempts CSV represent JSON null / not applicable, not a zero quality score.",
      "- Rates in `llm_judge_v3_summary.csv` are over valid attempts unless otherwise named.",
      "",
    ].join("\n"),
    "utf-8",
  );

  writeFileSync(resolve(resultDir, `${OUTPUT_PREFIX}_interpretation.md`), buildInterpretation(summaryRows), "utf-8");
}

function buildInterpretation(summaryRows: SummaryRow[]): string {
  function row(condition: string, latency: number): SummaryRow | undefined {
    return summaryRows.find((item) => item.condition === condition && Number(item.latency_ms) === latency);
  }
  function value(condition: string, latency: number, key: keyof SummaryRow): number | undefined {
    return asNumber(row(condition, latency)?.[key]);
  }
  const lines = ["# LLM Judge V3 Interpretation", ""];
  lines.push("This is a descriptive readout of the Gemma judge results, not a combined score.");
  lines.push("");
  lines.push("## Tick vs No Tick");
  lines.push("");
  for (const latency of LATENCIES) {
    const tickCoverage = value("external_single_tick", latency, "waiting_speech_coverage_rate_valid");
    const tickRel = value("external_single_tick", latency, "mean_waiting_task_relevance_score_when_spoke_valid");
    const noFinal = value("native_no_tick", latency, "final_core_answer_correct_rate_valid");
    const tickFinal = value("external_single_tick", latency, "final_core_answer_correct_rate_valid");
    const noCoverage = value("native_no_tick", latency, "waiting_speech_coverage_rate_valid");
    const noRelFixed = value("native_no_tick", latency, "mean_waiting_task_relevance_score_when_spoke_valid");
    lines.push(
      `- ${latency} ms: waiting coverage no_tick=${fmt(noCoverage)}, single_tick=${fmt(tickCoverage)}; spoken relevance no_tick=${fmt(noRelFixed)}, single_tick=${fmt(tickRel)}; final core correctness no_tick=${fmt(noFinal)}, single_tick=${fmt(tickFinal)}.`,
    );
  }
  lines.push("");
  lines.push("## Hallucination");
  lines.push("");
  for (const latency of LATENCIES) {
    const noHall = value("native_no_tick", latency, "pre_result_hallucination_rate_valid");
    const tickHall = value("external_single_tick", latency, "pre_result_hallucination_rate_valid");
    lines.push(`- ${latency} ms: hallucination rate no_tick=${fmt(noHall)}, single_tick=${fmt(tickHall)}.`);
  }
  lines.push("");
  lines.push("## Diversity");
  lines.push("");
  for (const latency of LATENCIES) {
    const noDiv = value("native_no_tick", latency, "waiting_diversity_score");
    const tickDiv = value("external_single_tick", latency, "waiting_diversity_score");
    lines.push(`- ${latency} ms: diversity no_tick=${fmt(noDiv)}, single_tick=${fmt(tickDiv)}.`);
  }
  lines.push("");
  lines.push("## Tick Timing Notes");
  lines.push("");
  lines.push("- `external_single_tick` at 3000 ms should be interpreted as tick-skipped, because the final result arrives before the scheduled 4000 ms tick.");
  lines.push("- `external_single_tick` at 5000 ms is an edge case: the tick is only about 1 second before final, so the model may have little time to produce task-aware waiting speech.");
  lines.push("");
  lines.push("## Audio Timing Consistency");
  lines.push("");
  lines.push("Compare waiting speech coverage with audio occupancy / max silent gap. Compare spoken-only relevance with qualitative ASR transcripts, because null relevance means no speech to evaluate rather than low-quality speech.");
  lines.push("");
  return `${lines.join("\n")}\n`;
}

function buildSanityChecks(
  resultDir: string,
  attemptRows: AttemptJudgeRow[],
  diversityRows: DiversityRow[],
  summaryRows: SummaryRow[],
): string[] {
  const lines: string[] = [];
  const v1 = readV1Summary(resultDir);
  lines.push("- V3 final core answer rates compared with v1 strict final answer rates:");
  for (const row of summaryRows) {
    const key = groupKey(row.condition, row.latency_ms);
    const oldRate = v1.get(key);
    lines.push(
      `  - ${row.condition} ${row.latency_ms}ms: v3=${row.final_core_answer_correct_rate_valid || "NA"}; v1=${oldRate ?? "NA"}.`,
    );
  }

  const hallucinations = attemptRows.filter((row) => row.pre_result_hallucination === "1");
  lines.push(`- Hallucination-positive attempts requiring review: ${hallucinations.length}.`);
  for (const row of hallucinations) {
    lines.push(
      `  - ${row.condition} ${row.latency_ms}ms ${row.attempt_id}: type=${row.hallucination_type}; transcript="${row.pre_result_transcript}"; reason="${row.pre_result_hallucination_reason}"`,
    );
  }

  const notEvaluable = diversityRows.filter((row) => row.diversity_evaluable !== "true");
  lines.push(`- Diversity not evaluable groups: ${notEvaluable.length}.`);
  for (const row of notEvaluable) {
    lines.push(
      `  - ${row.condition} ${row.latency_ms}ms: nonempty=${row.n_nonempty_pre_result_transcripts}; reason=${row.reason}`,
    );
  }

  const tick3000 = summaryRows.find((row) => row.condition === "external_single_tick" && row.latency_ms === "3000");
  lines.push(
    `- external_single_tick 3000ms tick status: ${tick3000?.tick_status_summary || "missing"}; expected mostly tick_skipped_final_before_tick.`,
  );
  const tick5000 = summaryRows.find((row) => row.condition === "external_single_tick" && row.latency_ms === "5000");
  lines.push(
    `- external_single_tick 5000ms edge case: ${tick5000?.tick_status_summary || "missing"}; scheduled tick is only about 1000ms before final.`,
  );
  return lines;
}

function readV1Summary(resultDir: string): Map<string, string> {
  const path = resolve(resultDir, "llm_judge_summary.csv");
  const map = new Map<string, string>();
  if (!existsSync(path)) return map;
  for (const row of parseCsv(readFileSync(path, "utf-8"))) {
    map.set(groupKey(row.condition ?? "", row.latency_ms ?? ""), row.final_answer_correct_rate ?? "");
  }
  return map;
}

function runPlots(resultDir: string): void {
  const plotScript = resolve(PROJECT_DIR, "scripts", "plot_llm_judge_formal.py");
  if (!existsSync(plotScript)) return;
  const result = spawnSync("python3", [plotScript, resultDir], {
    cwd: PROJECT_DIR,
    encoding: "utf-8",
    stdio: "inherit",
  });
  if (result.status !== 0) {
    console.warn(`Plot script failed with status ${result.status ?? "unknown"}.`);
  }
}

main().catch((error) => {
  console.error(summarizeError(error));
  process.exit(1);
});
