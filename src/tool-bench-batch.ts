import { Behavior, GoogleGenAI, Modality } from "@google/genai";
import { config as loadEnv } from "dotenv";
import { appendFileSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { PROMPT, PROMPT_VERSION, SYSTEM_INSTRUCTION } from "./prompts.js";

const PROJECT_DIR = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const DEFAULT_ENV_FILE = resolve(PROJECT_DIR, "..", "angus_bench", ".env");
const RESULT_DIR = resolve(PROJECT_DIR, "result");
const TOOL_NAME = "external_answer_tool";
const EXPECTED_FINAL_ANSWER = "Angus";
const TOOL_RESPONSE_SCHEMA = "single_final_answer_field";
const DEFAULT_LATENCIES = [3000, 5000, 8000, 12000];
const DEFAULT_RUNS_PER_LATENCY = 1;
const DEFAULT_TARGET_VALID_RUNS = 10;
const DEFAULT_MAX_ATTEMPTS_PER_LATENCY = 40;
const DEFAULT_CONCURRENCY = 1;
const MAX_RUN_MS = 45_000;

loadEnv({ path: process.env.GEMINI_ENV_FILE || DEFAULT_ENV_FILE });

type FunctionCall = {
  id?: string;
  name?: string;
  args?: Record<string, unknown>;
};

type LiveMessage = {
  setupComplete?: unknown;
  serverContent?: {
    modelTurn?: {
      parts?: Array<{
        text?: string;
        inlineData?: { mimeType?: string; data?: string };
      }>;
    };
    outputTranscription?: { text?: string };
    turnComplete?: boolean;
  };
  toolCall?: {
    functionCalls?: FunctionCall[];
  };
  toolCallCancellation?: {
    ids?: string[];
  };
  usageMetadata?: unknown;
  goAway?: unknown;
  sessionResumptionUpdate?: unknown;
  voiceActivity?: unknown;
  voiceActivityDetectionSignal?: unknown;
};

type Session = {
  sendClientContent(params: { turns?: string; turnComplete?: boolean }): void;
  sendToolResponse(params: {
    functionResponses: {
      id?: string;
      name?: string;
      response: Record<string, unknown>;
    }[];
  }): void;
  close(): void;
};

type BatchConfig = {
  batch_id: string;
  model: string;
  latencies_ms: number[];
  runs_per_latency?: number;
  target_valid_runs: number;
  max_attempts_per_latency: number;
  concurrency: number;
  prompt_version: string;
  tool_response_schema: string;
  expected_final_answer: string;
  max_run_ms: number;
  mode: "target_valid_runs";
  cli_note: string;
};

type RunPlan = {
  run_id: string;
  latency_ms: number;
  latency_index: number;
  run_index: number;
  run_dir: string;
};

type RunMetrics = {
  run_id: string;
  latency_ms: number;
  valid: boolean;
  session_valid: boolean;
  error_type: string | null;
  behavior_hard_fail: boolean;
  behavior_error_type: string | null;
  premature_answer: "not_available" | boolean | null;
  prompt_version: string;
  model: string;
  prompt_end_time_ms: number;
  first_assistant_output_time_ms: number | null;
  first_response_latency_ms: number | null;
  tool_call_time_ms: number | null;
  tool_response_time_ms: number | null;
  waiting_interval_ms: number | null;
  assistant_output_count_before_tool_response: number;
  assistant_output_count_after_tool_response: number;
  final_answer_expected: string;
  final_answer_observed: string | null;
  final_exact_match: boolean | null;
  server_1008_error: boolean;
  close_code: number | null;
  close_reason: string | null;
  tool_call_count: number;
  tool_response_sent: boolean;
  tool_response_skipped_after_close: boolean;
  non_tool_response_outbound_during_tool_pending: boolean;
  notes?: string[];
};

type LatencySummary = {
  latency_ms: number;
  completed: boolean;
  target_valid_runs: number;
  max_attempts: number;
  attempted_runs: number;
  valid_runs: number;
  invalid_runs: number;
  server_1008_errors: number;
  server_1008_error_rate_by_attempts: number;
  other_errors: number;
  valid_run_rate: number;
  attempts_per_valid_run: number | null;
  behavior_hard_fails: number;
  behavior_hard_fail_rate: number | null;
  behavior_metrics: {
    avg_first_response_latency_ms: number | null;
    median_first_response_latency_ms: number | null;
    avg_waiting_interval_ms: number | null;
    avg_assistant_output_count_before_tool_response: number | null;
    final_exact_match_rate: number | null;
    premature_answer_rate: number | null;
  };
  stability_metrics: {
    server_1008_error_rate: number;
    other_error_rate: number;
    valid_run_rate: number;
  };
};

type BatchSummary = {
  batch_id: string;
  created_at: string;
  config: BatchConfig;
  latencies: LatencySummary[];
  overall: Omit<LatencySummary, "latency_ms" | "completed">;
};

type RunState = {
  openedAt?: number;
  promptSentAt?: number;
  firstAssistantOutputAt?: number;
  toolCallAt?: number;
  toolResponseAt?: number;
  closeCode: number | null;
  closeReason: string | null;
  sessionClosed: boolean;
  toolCallPending: boolean;
  primaryToolCallReceived: boolean;
  primaryToolResponseSent: boolean;
  toolResponseSkippedAfterClose: boolean;
  nonToolResponseOutboundDuringToolPending: boolean;
  assistantOutputCountBeforeToolResponse: number;
  assistantOutputCountAfterToolResponse: number;
  toolCallCount: number;
  observedTextAfterToolResponse: string[];
  errors: string[];
};

function requireEnv(name: string): string {
  const value = process.env[name]?.trim();
  if (!value) throw new Error(`Missing ${name}. Set it in /user_data/angus_bench/.env or GEMINI_ENV_FILE.`);
  return value;
}

function timestampForPath(date = new Date()): string {
  return date.toISOString().replace("T", "_").replace(/\..+$/, "").replace(/:/g, "-");
}

function runTimestamp(date = new Date()): string {
  return timestampForPath(date);
}

function parseCsvNumbers(value: string): number[] {
  return value
    .split(",")
    .map((part) => Number(part.trim()))
    .filter((value) => Number.isFinite(value) && value > 0);
}

function parseArgs(argv: string[]): {
  latencies: number[];
  runsPerLatency?: number;
  targetValidRuns: number;
  maxAttemptsPerLatency: number;
  concurrency: number;
} {
  const args = new Map<string, string>();
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (!arg.startsWith("--")) continue;
    const [key, inlineValue] = arg.slice(2).split("=", 2);
    const value = inlineValue ?? argv[index + 1];
    if (inlineValue === undefined) index += 1;
    args.set(key, value);
  }

  const targetValidRuns = Number(args.get("target-valid-runs") ?? DEFAULT_TARGET_VALID_RUNS);
  const runsPerLatency = args.has("runs-per-latency") ? Number(args.get("runs-per-latency")) : undefined;
  return {
    latencies: args.has("latencies") ? parseCsvNumbers(args.get("latencies") ?? "") : DEFAULT_LATENCIES,
    runsPerLatency,
    targetValidRuns: Number.isFinite(targetValidRuns)
      ? Math.max(1, targetValidRuns)
      : runsPerLatency ?? DEFAULT_TARGET_VALID_RUNS,
    maxAttemptsPerLatency: Math.max(
      1,
      Number(args.get("max-attempts-per-latency") ?? DEFAULT_MAX_ATTEMPTS_PER_LATENCY),
    ),
    concurrency: Number(args.get("concurrency") ?? DEFAULT_CONCURRENCY),
  };
}

function writeJson(path: string, value: unknown): void {
  writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

function appendEventLog(path: string, event: Record<string, unknown>): void {
  appendFileSync(path, `${JSON.stringify({ ts: new Date().toISOString(), ...event })}\n`, "utf8");
}

function eventTypes(message: LiveMessage): string[] {
  return [
    message.setupComplete ? "setupComplete" : undefined,
    message.serverContent ? "serverContent" : undefined,
    message.toolCall ? "toolCall" : undefined,
    message.toolCallCancellation ? "toolCallCancellation" : undefined,
    message.usageMetadata ? "usageMetadata" : undefined,
    message.goAway ? "goAway" : undefined,
    message.sessionResumptionUpdate ? "sessionResumptionUpdate" : undefined,
    message.voiceActivity ? "voiceActivity" : undefined,
    message.voiceActivityDetectionSignal ? "voiceActivityDetectionSignal" : undefined,
  ].filter((type): type is string => Boolean(type));
}

function sanitizeMessage(message: LiveMessage): unknown {
  return {
    ...message,
    serverContent: message.serverContent
      ? {
          ...message.serverContent,
          modelTurn: message.serverContent.modelTurn
            ? {
                ...message.serverContent.modelTurn,
                parts: message.serverContent.modelTurn.parts?.map((part) => {
                  if (!part.inlineData) return part;
                  return {
                    inlineData: {
                      mimeType: part.inlineData.mimeType,
                      bytes: part.inlineData.data ? Buffer.from(part.inlineData.data, "base64").length : 0,
                    },
                  };
                }),
              }
            : undefined,
        }
      : undefined,
  };
}

function summarizeError(error: unknown): string {
  if (error instanceof Error) return `${error.name}: ${error.message}`;
  if (typeof error !== "object" || error === null) return String(error);
  const record = error as Record<string, unknown>;
  return ["message", "type", "code", "reason", "error"]
    .map((key) => (record[key] ? `${key}: ${String(record[key])}` : undefined))
    .filter(Boolean)
    .join(", ") || Object.prototype.toString.call(error);
}

function isServer1008Error(metrics: Pick<RunMetrics, "close_code" | "close_reason" | "error_type">): boolean {
  return metrics.close_code === 1008 || metrics.error_type === "server_1008_error" || Boolean(metrics.close_reason?.includes("1008"));
}

function classifySessionError(state: RunState): string | null {
  if (state.closeCode === 1008) return "server_1008_error";
  if (state.errors.length > 0) return "session_error";
  return null;
}

function classifyBehaviorError(state: RunState): string | null {
  if (!state.primaryToolCallReceived) return "missing_tool_call";
  if (!state.primaryToolResponseSent) return "missing_tool_response";
  return null;
}

function computeRunMetrics(runId: string, latencyMs: number, model: string, state: RunState): RunMetrics {
  const errorType = classifySessionError(state);
  const behaviorErrorType = errorType === null ? classifyBehaviorError(state) : null;
  const valid = errorType === null;
  const observed = state.observedTextAfterToolResponse.join("").trim() || null;
  return {
    run_id: runId,
    latency_ms: latencyMs,
    valid,
    session_valid: valid,
    error_type: errorType,
    behavior_hard_fail: behaviorErrorType !== null,
    behavior_error_type: behaviorErrorType,
    premature_answer: "not_available",
    prompt_version: PROMPT_VERSION,
    model,
    prompt_end_time_ms: 0,
    first_assistant_output_time_ms: state.promptSentAt && state.firstAssistantOutputAt
      ? state.firstAssistantOutputAt - state.promptSentAt
      : null,
    first_response_latency_ms: state.promptSentAt && state.firstAssistantOutputAt
      ? state.firstAssistantOutputAt - state.promptSentAt
      : null,
    tool_call_time_ms: state.promptSentAt && state.toolCallAt ? state.toolCallAt - state.promptSentAt : null,
    tool_response_time_ms: state.promptSentAt && state.toolResponseAt ? state.toolResponseAt - state.promptSentAt : null,
    waiting_interval_ms: state.toolCallAt && state.toolResponseAt ? state.toolResponseAt - state.toolCallAt : null,
    assistant_output_count_before_tool_response: state.assistantOutputCountBeforeToolResponse,
    assistant_output_count_after_tool_response: state.assistantOutputCountAfterToolResponse,
    final_answer_expected: EXPECTED_FINAL_ANSWER,
    final_answer_observed: observed,
    final_exact_match: observed === null ? null : observed === EXPECTED_FINAL_ANSWER,
    server_1008_error: errorType === "server_1008_error",
    close_code: state.closeCode,
    close_reason: state.closeReason,
    tool_call_count: state.toolCallCount,
    tool_response_sent: state.primaryToolResponseSent,
    tool_response_skipped_after_close: state.toolResponseSkippedAfterClose,
    non_tool_response_outbound_during_tool_pending: state.nonToolResponseOutboundDuringToolPending,
    notes: [
      observed === null ? "final_answer_observed is null when no text/outputTranscription is available after tool response. No ASR or LLM judge is used." : undefined,
      "premature_answer is not_available because no ASR/LLM judge is used in this runner.",
    ].filter((note): note is string => Boolean(note)),
  };
}

async function runOne(ai: GoogleGenAI, model: string, plan: RunPlan): Promise<RunMetrics> {
  mkdirSync(plan.run_dir, { recursive: true });
  const eventsPath = resolve(plan.run_dir, "events.jsonl");
  const response = { final_answer: EXPECTED_FINAL_ANSWER };
  const config = {
    run_id: plan.run_id,
    latency_ms: plan.latency_ms,
    prompt_version: PROMPT_VERSION,
    model,
    tool_name: TOOL_NAME,
    tool_response_schema: TOOL_RESPONSE_SCHEMA,
    expected_final_answer: EXPECTED_FINAL_ANSWER,
  };
  writeJson(resolve(plan.run_dir, "config.json"), config);

  const state: RunState = {
    closeCode: null,
    closeReason: null,
    sessionClosed: false,
    toolCallPending: false,
    primaryToolCallReceived: false,
    primaryToolResponseSent: false,
    toolResponseSkippedAfterClose: false,
    nonToolResponseOutboundDuringToolPending: false,
    assistantOutputCountBeforeToolResponse: 0,
    assistantOutputCountAfterToolResponse: 0,
    toolCallCount: 0,
    observedTextAfterToolResponse: [],
    errors: [],
  };

  let session: Session | undefined;
  let done = false;
  let pendingTimer: ReturnType<typeof setTimeout> | undefined;

  const outbound = (outboundType: string, reason: string): boolean => {
    const allowed = !state.sessionClosed && (outboundType === "sendToolResponse" || !state.toolCallPending);
    appendEventLog(eventsPath, {
      type: "outbound_event",
      outbound_type: outboundType,
      toolCallPending: state.toolCallPending,
      sessionClosed: state.sessionClosed,
      allowed,
      reason,
    });
    if (!allowed && state.toolCallPending && outboundType !== "sendToolResponse") {
      state.nonToolResponseOutboundDuringToolPending = true;
      appendEventLog(eventsPath, { type: "blocked_outbound_during_tool_pending", outbound_type: outboundType, reason });
    }
    return allowed;
  };

  const finish = (resolveRun: () => void) => {
    if (done) return;
    done = true;
    state.toolCallPending = false;
    if (pendingTimer) clearTimeout(pendingTimer);
    if (!state.sessionClosed && outbound("close", "run finished")) {
      try {
        session?.close();
      } catch (error) {
        state.errors.push(summarizeError(error));
      }
    }
    resolveRun();
  };

  await new Promise<void>(async (resolveRun) => {
    const hardTimer = setTimeout(() => {
      state.errors.push(`max run timeout ${MAX_RUN_MS}ms`);
      finish(resolveRun);
    }, MAX_RUN_MS);

    try {
      session = (await ai.live.connect({
        model,
        config: {
          responseModalities: [Modality.AUDIO],
          outputAudioTranscription: {},
          systemInstruction: SYSTEM_INSTRUCTION,
          tools: [
            {
              functionDeclarations: [
                {
                  name: TOOL_NAME,
                  description: "Gets the final answer from an external information source.",
                  behavior: Behavior.NON_BLOCKING,
                  parametersJsonSchema: {
                    type: "object",
                    properties: {
                      question: { type: "string" },
                    },
                    required: ["question"],
                  },
                },
              ],
            },
          ],
        },
        callbacks: {
          onopen: () => {
            state.openedAt = Date.now();
            appendEventLog(eventsPath, { type: "session_opened", model, nonBlockingToolConfigured: true });
          },
          onmessage: (message: LiveMessage) => {
            const now = Date.now();
            appendEventLog(eventsPath, {
              type: "server_event",
              eventTypes: eventTypes(message),
              event_ms: state.promptSentAt ? now - state.promptSentAt : null,
              message: sanitizeMessage(message),
            });

            const hasAssistantOutput = Boolean(
              message.serverContent?.outputTranscription?.text ||
                message.serverContent?.modelTurn?.parts?.some((part) => part.text || part.inlineData?.data),
            );
            if (hasAssistantOutput) {
              state.firstAssistantOutputAt ??= now;
              if (state.toolResponseAt) state.assistantOutputCountAfterToolResponse += 1;
              else state.assistantOutputCountBeforeToolResponse += 1;
            }

            const outputText = [
              ...(message.serverContent?.modelTurn?.parts?.map((part) => part.text).filter((text): text is string => Boolean(text)) ?? []),
              message.serverContent?.outputTranscription?.text,
            ].filter((text): text is string => Boolean(text));
            if (state.toolResponseAt && outputText.length > 0) state.observedTextAfterToolResponse.push(...outputText);

            if (message.toolCall?.functionCalls?.length) {
              state.toolCallCount += message.toolCall.functionCalls.length;
              const call = message.toolCall.functionCalls[0];
              if (!state.primaryToolCallReceived) {
                state.primaryToolCallReceived = true;
                state.toolCallPending = true;
                state.toolCallAt = now;
                appendEventLog(eventsPath, {
                  type: "tool_call_received",
                  functionCalls: message.toolCall.functionCalls,
                });
                pendingTimer = setTimeout(() => {
                  if (state.sessionClosed || done) {
                    state.toolResponseSkippedAfterClose = state.sessionClosed;
                    appendEventLog(eventsPath, {
                      type: state.sessionClosed ? "tool_response_skipped_after_close" : "tool_response_skipped",
                      delayMs: plan.latency_ms,
                      reason: state.sessionClosed ? "session already closed" : "run already done",
                    });
                    return;
                  }
                  if (!outbound("sendToolResponse", "scheduled tool response delay elapsed")) return;
                  try {
                    session?.sendToolResponse({
                      functionResponses: [
                        {
                          id: call.id,
                          name: call.name || TOOL_NAME,
                          response,
                        },
                      ],
                    });
                    state.toolCallPending = false;
                    state.primaryToolResponseSent = true;
                    state.toolResponseAt = Date.now();
                    appendEventLog(eventsPath, {
                      type: "tool_response_sent",
                      functionCallId: call.id,
                      functionName: call.name,
                      delayMs: plan.latency_ms,
                      response,
                    });
                  } catch (error) {
                    state.errors.push(summarizeError(error));
                    appendEventLog(eventsPath, { type: "tool_response_send_error", error: summarizeError(error) });
                  }
                }, plan.latency_ms);
              } else {
                appendEventLog(eventsPath, {
                  type: "extra_tool_call_received",
                  functionCalls: message.toolCall.functionCalls,
                });
              }
            }

            if (state.primaryToolResponseSent && message.serverContent?.turnComplete) {
              clearTimeout(hardTimer);
              finish(resolveRun);
            }
          },
          onerror: (error) => {
            const summary = summarizeError(error);
            state.errors.push(summary);
            appendEventLog(eventsPath, { type: "socket_error", error: summary });
          },
          onclose: (event: { code?: number; reason?: string }) => {
            state.sessionClosed = true;
            state.closeCode = event.code ?? null;
            state.closeReason = event.reason || null;
            state.toolCallPending = false;
            if (pendingTimer && !state.primaryToolResponseSent) {
              clearTimeout(pendingTimer);
              state.toolResponseSkippedAfterClose = true;
              appendEventLog(eventsPath, {
                type: "tool_response_skipped_after_close",
                delayMs: plan.latency_ms,
                reason: "session closed before scheduled tool response",
              });
            }
            appendEventLog(eventsPath, { type: "session_closed", code: event.code, reason: event.reason });
            clearTimeout(hardTimer);
            resolveRun();
          },
        },
      })) as Session;

      state.openedAt ??= Date.now();
      if (outbound("sendClientContent", "initial user prompt")) {
        session.sendClientContent({ turns: PROMPT, turnComplete: true });
        state.promptSentAt = Date.now();
        appendEventLog(eventsPath, { type: "user_message_sent", prompt_version: PROMPT_VERSION });
      }
    } catch (error) {
      state.errors.push(summarizeError(error));
      appendEventLog(eventsPath, { type: "run_error", error: summarizeError(error) });
      clearTimeout(hardTimer);
      resolveRun();
    }
  });

  const metrics = computeRunMetrics(plan.run_id, plan.latency_ms, model, state);
  writeJson(resolve(plan.run_dir, "metrics.json"), metrics);
  writeJson(resolve(plan.run_dir, "raw_result.json"), { metrics, prompt_version: PROMPT_VERSION });
  if (!metrics.valid) {
    writeJson(resolve(plan.run_dir, "error.json"), {
      error_type: metrics.error_type,
      server_1008_error: isServer1008Error(metrics),
      close_code: metrics.close_code,
      close_reason: metrics.close_reason,
      errors: state.errors,
    });
  }
  return metrics;
}

function average(values: number[]): number | null {
  if (values.length === 0) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function median(values: number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle];
}

function summarizeLatency(
  latencyMs: number,
  runs: RunMetrics[],
  targetValidRuns: number,
  maxAttempts: number,
): LatencySummary {
  const valid = runs.filter((run) => run.valid);
  const server1008 = runs.filter((run) => run.server_1008_error).length;
  const exactComparable = valid.filter((run) => run.final_exact_match !== null);
  const prematureComparable = valid.filter((run) => typeof run.premature_answer === "boolean");
  const behaviorHardFails = valid.filter((run) => run.behavior_hard_fail).length;
  const attemptedRuns = runs.length;
  const validRunRate = attemptedRuns === 0 ? 0 : valid.length / attemptedRuns;
  return {
    latency_ms: latencyMs,
    completed: valid.length >= targetValidRuns,
    target_valid_runs: targetValidRuns,
    max_attempts: maxAttempts,
    attempted_runs: attemptedRuns,
    valid_runs: valid.length,
    invalid_runs: attemptedRuns - valid.length,
    server_1008_errors: server1008,
    server_1008_error_rate_by_attempts: attemptedRuns === 0 ? 0 : server1008 / attemptedRuns,
    other_errors: runs.filter((run) => !run.valid && !run.server_1008_error).length,
    valid_run_rate: validRunRate,
    attempts_per_valid_run: valid.length === 0 ? null : attemptedRuns / valid.length,
    behavior_hard_fails: behaviorHardFails,
    behavior_hard_fail_rate: valid.length === 0 ? null : behaviorHardFails / valid.length,
    behavior_metrics: {
      avg_first_response_latency_ms: average(valid.map((run) => run.first_response_latency_ms).filter((value): value is number => value !== null)),
      median_first_response_latency_ms: median(valid.map((run) => run.first_response_latency_ms).filter((value): value is number => value !== null)),
      avg_waiting_interval_ms: average(valid.map((run) => run.waiting_interval_ms).filter((value): value is number => value !== null)),
      avg_assistant_output_count_before_tool_response: average(valid.map((run) => run.assistant_output_count_before_tool_response)),
      final_exact_match_rate: exactComparable.length === 0
        ? null
        : exactComparable.filter((run) => run.final_exact_match).length / exactComparable.length,
      premature_answer_rate: prematureComparable.length === 0
        ? null
        : prematureComparable.filter((run) => run.premature_answer === true).length / prematureComparable.length,
    },
    stability_metrics: {
      server_1008_error_rate: attemptedRuns === 0 ? 0 : server1008 / attemptedRuns,
      other_error_rate: attemptedRuns === 0
        ? 0
        : runs.filter((run) => !run.valid && !run.server_1008_error).length / attemptedRuns,
      valid_run_rate: validRunRate,
    },
  };
}

function csvEscape(value: unknown): string {
  if (value === null || value === undefined) return "";
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function writeSummaryCsv(path: string, rows: LatencySummary[]): void {
  const headers = [
    "latency_ms",
    "completed",
    "target_valid_runs",
    "max_attempts",
    "attempted_runs",
    "valid_runs",
    "invalid_runs",
    "server_1008_errors",
    "server_1008_error_rate_by_attempts",
    "other_errors",
    "valid_run_rate",
    "attempts_per_valid_run",
    "behavior_hard_fails",
    "behavior_hard_fail_rate",
    "behavior_avg_first_response_latency_ms",
    "behavior_median_first_response_latency_ms",
    "behavior_avg_waiting_interval_ms",
    "behavior_avg_assistant_output_count_before_tool_response",
    "behavior_final_exact_match_rate",
    "behavior_premature_answer_rate",
    "stability_server_1008_error_rate",
    "stability_other_error_rate",
    "stability_valid_run_rate",
  ];
  const lines = [headers.join(",")];
  for (const row of rows) {
    const flat: Record<string, unknown> = {
      latency_ms: row.latency_ms,
      completed: row.completed,
      target_valid_runs: row.target_valid_runs,
      max_attempts: row.max_attempts,
      attempted_runs: row.attempted_runs,
      valid_runs: row.valid_runs,
      invalid_runs: row.invalid_runs,
      server_1008_errors: row.server_1008_errors,
      server_1008_error_rate_by_attempts: row.server_1008_error_rate_by_attempts,
      other_errors: row.other_errors,
      valid_run_rate: row.valid_run_rate,
      attempts_per_valid_run: row.attempts_per_valid_run,
      behavior_hard_fails: row.behavior_hard_fails,
      behavior_hard_fail_rate: row.behavior_hard_fail_rate,
      behavior_avg_first_response_latency_ms: row.behavior_metrics.avg_first_response_latency_ms,
      behavior_median_first_response_latency_ms: row.behavior_metrics.median_first_response_latency_ms,
      behavior_avg_waiting_interval_ms: row.behavior_metrics.avg_waiting_interval_ms,
      behavior_avg_assistant_output_count_before_tool_response: row.behavior_metrics.avg_assistant_output_count_before_tool_response,
      behavior_final_exact_match_rate: row.behavior_metrics.final_exact_match_rate,
      behavior_premature_answer_rate: row.behavior_metrics.premature_answer_rate,
      stability_server_1008_error_rate: row.stability_metrics.server_1008_error_rate,
      stability_other_error_rate: row.stability_metrics.other_error_rate,
      stability_valid_run_rate: row.stability_metrics.valid_run_rate,
    };
    lines.push(headers.map((header) => csvEscape(flat[header])).join(","));
  }
  writeFileSync(path, `${lines.join("\n")}\n`, "utf8");
}

async function runWithConcurrency<T, R>(
  items: T[],
  concurrency: number,
  worker: (item: T) => Promise<R>,
): Promise<R[]> {
  const results: R[] = [];
  let nextIndex = 0;
  async function runWorker(): Promise<void> {
    while (nextIndex < items.length) {
      const currentIndex = nextIndex;
      nextIndex += 1;
      results[currentIndex] = await worker(items[currentIndex]);
    }
  }
  await Promise.all(Array.from({ length: Math.max(1, concurrency) }, () => runWorker()));
  return results;
}

async function main(): Promise<void> {
  const parsed = parseArgs(process.argv.slice(2));
  const apiKey = requireEnv("GEMINI_API_KEY");
  const model = requireEnv("GEMINI_LIVE_MODEL");
  const batchId = `batch_${timestampForPath()}`;
  const batchDir = resolve(RESULT_DIR, batchId);
  mkdirSync(batchDir, { recursive: true });

  const config: BatchConfig = {
    batch_id: batchId,
    model,
    latencies_ms: parsed.latencies.length > 0 ? parsed.latencies : DEFAULT_LATENCIES,
    runs_per_latency: parsed.runsPerLatency,
    target_valid_runs: parsed.targetValidRuns,
    max_attempts_per_latency: Math.max(parsed.maxAttemptsPerLatency, parsed.targetValidRuns),
    concurrency: Math.max(1, parsed.concurrency || DEFAULT_CONCURRENCY),
    prompt_version: PROMPT_VERSION,
    tool_response_schema: TOOL_RESPONSE_SCHEMA,
    expected_final_answer: EXPECTED_FINAL_ANSWER,
    max_run_ms: MAX_RUN_MS,
    mode: "target_valid_runs",
    cli_note: "--target-valid-runs is the formal benchmark mode and takes priority. --runs-per-latency is kept only for backward compatibility/documentation.",
  };
  writeJson(resolve(batchDir, "config.json"), {
    ...config,
    prompt: PROMPT,
    system_instruction: SYSTEM_INSTRUCTION,
  });

  const ai = new GoogleGenAI({ apiKey });
  const allMetrics: RunMetrics[] = [];
  const latencySummaries: LatencySummary[] = [];

  for (const [latencyIndex, latencyMs] of config.latencies_ms.entries()) {
    const latencyDir = resolve(batchDir, `latency_${latencyMs}ms`);
    mkdirSync(latencyDir, { recursive: true });

    const latencyRuns: RunMetrics[] = [];
    let attempted = 0;
    let validCount = 0;

    console.log("");
    console.log(`[latency ${latencyMs}ms] target valid runs: ${config.target_valid_runs}; max attempts: ${config.max_attempts_per_latency}`);

    while (validCount < config.target_valid_runs && attempted < config.max_attempts_per_latency) {
      const remainingAttempts = config.max_attempts_per_latency - attempted;
      const remainingValid = config.target_valid_runs - validCount;
      const batchSize = Math.min(config.concurrency, remainingAttempts, remainingValid);
      const plans: RunPlan[] = [];

      for (let index = 0; index < batchSize; index += 1) {
        attempted += 1;
        const runId = `latency_${latencyMs}ms_attempt_${String(attempted).padStart(4, "0")}`;
        plans.push({
          run_id: runId,
          latency_ms: latencyMs,
          latency_index: latencyIndex,
          run_index: attempted,
          run_dir: resolve(latencyDir, `run_${String(attempted).padStart(4, "0")}_${runTimestamp()}`),
        });
      }

      const attemptResults = await runWithConcurrency(plans, config.concurrency, async (plan) => {
        console.log(`[${plan.run_id}] start`);
        const result = await runOne(ai, model, plan);
        console.log(`[${plan.run_id}] ${result.valid ? "valid" : result.error_type ?? result.behavior_error_type ?? "behavior_hard_fail"}`);
        return result;
      });

      latencyRuns.push(...attemptResults);
      allMetrics.push(...attemptResults);
      validCount = latencyRuns.filter((run) => run.valid).length;
      console.log(`[latency ${latencyMs}ms] valid ${validCount}/${config.target_valid_runs}; attempts ${attempted}/${config.max_attempts_per_latency}`);
    }

    const latencySummary = summarizeLatency(
      latencyMs,
      latencyRuns,
      config.target_valid_runs,
      config.max_attempts_per_latency,
    );
    latencySummaries.push(latencySummary);
    writeJson(resolve(latencyDir, "summary.json"), { ...latencySummary, runs: latencyRuns });
    writeSummaryCsv(resolve(latencyDir, "summary.csv"), [latencySummary]);
  }

  console.log(`Batch directory: ${relative(PROJECT_DIR, batchDir)}`);
  console.log(`Latencies: ${config.latencies_ms.join(", ")}`);
  console.log(`Target valid runs per latency: ${config.target_valid_runs}`);
  console.log(`Max attempts per latency: ${config.max_attempts_per_latency}`);
  console.log(`Concurrency: ${config.concurrency}`);
  console.log(`Prompt version: ${PROMPT_VERSION}`);

  const overall = summarizeLatency(
    0,
    allMetrics,
    config.target_valid_runs * config.latencies_ms.length,
    config.max_attempts_per_latency * config.latencies_ms.length,
  );
  const summary: BatchSummary = {
    batch_id: batchId,
    created_at: new Date().toISOString(),
    config,
    latencies: latencySummaries,
    overall: {
      target_valid_runs: overall.target_valid_runs,
      max_attempts: overall.max_attempts,
      attempted_runs: overall.attempted_runs,
      valid_runs: overall.valid_runs,
      invalid_runs: overall.invalid_runs,
      server_1008_errors: overall.server_1008_errors,
      server_1008_error_rate_by_attempts: overall.server_1008_error_rate_by_attempts,
      other_errors: overall.other_errors,
      valid_run_rate: overall.valid_run_rate,
      attempts_per_valid_run: overall.attempts_per_valid_run,
      behavior_hard_fails: overall.behavior_hard_fails,
      behavior_hard_fail_rate: overall.behavior_hard_fail_rate,
      behavior_metrics: overall.behavior_metrics,
      stability_metrics: overall.stability_metrics,
    },
  };
  writeJson(resolve(batchDir, "summary.json"), summary);
  writeSummaryCsv(resolve(batchDir, "summary.csv"), latencySummaries);

  console.log(`Summary: ${relative(PROJECT_DIR, resolve(batchDir, "summary.json"))}`);
}

main().catch((error) => {
  console.error("tool-bench-batch failed");
  console.error(summarizeError(error));
  process.exitCode = 1;
});
