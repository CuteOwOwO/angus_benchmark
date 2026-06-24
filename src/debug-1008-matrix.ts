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
const DEFAULT_LATENCIES = [3000, 5000, 8000, 12000];
const DEFAULT_RUNS_PER_CELL = 5;
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

type Condition = {
  id: string;
  label: string;
  toolsEnabled: boolean;
  responseModality: Modality;
  latencies: number[];
  timingVariant: "current" | "delayed_after_tool_call" | "no_tools" | "text_only";
  notes: string;
};

type RunPlan = {
  condition: Condition;
  latencyMs: number;
  runIndex: number;
  runId: string;
  runDir: string;
};

type RunState = {
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
  errors: string[];
};

type RunMetrics = {
  condition: string;
  run_id: string;
  latency_ms: number;
  valid: boolean;
  error_type: string | null;
  server_1008_error: boolean;
  close_code: number | null;
  close_reason: string | null;
  prompt_version: string;
  model: string;
  response_modality: string;
  tools_enabled: boolean;
  timing_variant: string;
  first_response_latency_ms: number | null;
  tool_call_time_ms: number | null;
  tool_response_time_ms: number | null;
  waiting_interval_ms: number | null;
  assistant_output_count_before_tool_response: number;
  assistant_output_count_after_tool_response: number;
  tool_call_count: number;
  tool_response_sent: boolean;
  tool_response_skipped_after_close: boolean;
  non_tool_response_outbound_during_tool_pending: boolean;
  notes: string;
};

type SummaryRow = {
  condition: string;
  latency_ms: number;
  total_runs: number;
  valid_runs: number;
  server_1008_errors: number;
  server_1008_error_rate: number;
  other_errors: number;
  common_error_messages: string[];
  close_reasons: string[];
  notes: string;
};

function requireEnv(name: string): string {
  const value = process.env[name]?.trim();
  if (!value) throw new Error(`Missing ${name}. Set it in /user_data/angus_bench/.env or GEMINI_ENV_FILE.`);
  return value;
}

function timestampForPath(date = new Date()): string {
  return date.toISOString().replace("T", "_").replace(/\..+$/, "").replace(/:/g, "-");
}

function parseCsvNumbers(value: string): number[] {
  return value
    .split(",")
    .map((part) => Number(part.trim()))
    .filter((value) => Number.isFinite(value) && value >= 0);
}

function parseArgs(argv: string[]): { runsPerCell: number; concurrency: number; latencies: number[] } {
  const args = new Map<string, string>();
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (!arg.startsWith("--")) continue;
    const [key, inlineValue] = arg.slice(2).split("=", 2);
    const value = inlineValue ?? argv[index + 1];
    if (inlineValue === undefined) index += 1;
    args.set(key, value);
  }
  return {
    runsPerCell: Math.max(1, Number(args.get("runs-per-cell") ?? DEFAULT_RUNS_PER_CELL)),
    concurrency: Math.max(1, Number(args.get("concurrency") ?? DEFAULT_CONCURRENCY)),
    latencies: args.has("latencies") ? parseCsvNumbers(args.get("latencies") ?? "") : DEFAULT_LATENCIES,
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

function conditionSet(latencies: number[]): Condition[] {
  return [
    {
      id: "baseline",
      label: "baseline",
      toolsEnabled: true,
      responseModality: Modality.AUDIO,
      latencies,
      timingVariant: "current",
      notes: "Audio output, tools enabled, current model, current tool response flow, concurrency=1 by default.",
    },
    {
      id: "delayed_after_tool_call",
      label: "delayed_after_tool_call",
      toolsEnabled: true,
      responseModality: Modality.AUDIO,
      latencies,
      timingVariant: "delayed_after_tool_call",
      notes: "Explicitly waits latencyMs after primary toolCall before sendToolResponse. This is equivalent to the current harness timing and is kept as a labeled timing variant.",
    },
    {
      id: "no_tools",
      label: "no_tools",
      toolsEnabled: false,
      responseModality: Modality.AUDIO,
      latencies: [0],
      timingVariant: "no_tools",
      notes: "Audio output with tools disabled. Latency is not applicable and is recorded as 0.",
    },
    {
      id: "text_only_tools",
      label: "text_only_tools",
      toolsEnabled: true,
      responseModality: Modality.TEXT,
      latencies,
      timingVariant: "text_only",
      notes: "Text response modality with tools enabled. Included only because SDK exposes Modality.TEXT; runtime may still reject it for this Live model.",
    },
  ];
}

function classifyError(state: RunState, condition: Condition): string | null {
  if (state.closeCode === 1008) return "server_1008_error";
  if (state.errors.length > 0) return "session_error";
  if (condition.toolsEnabled && !state.primaryToolCallReceived) return "missing_tool_call";
  if (condition.toolsEnabled && !state.primaryToolResponseSent) return "missing_tool_response";
  return null;
}

function computeMetrics(plan: RunPlan, model: string, state: RunState): RunMetrics {
  const errorType = classifyError(state, plan.condition);
  return {
    condition: plan.condition.id,
    run_id: plan.runId,
    latency_ms: plan.latencyMs,
    valid: errorType === null,
    error_type: errorType,
    server_1008_error: errorType === "server_1008_error",
    close_code: state.closeCode,
    close_reason: state.closeReason,
    prompt_version: PROMPT_VERSION,
    model,
    response_modality: plan.condition.responseModality,
    tools_enabled: plan.condition.toolsEnabled,
    timing_variant: plan.condition.timingVariant,
    first_response_latency_ms: state.promptSentAt && state.firstAssistantOutputAt
      ? state.firstAssistantOutputAt - state.promptSentAt
      : null,
    tool_call_time_ms: state.promptSentAt && state.toolCallAt ? state.toolCallAt - state.promptSentAt : null,
    tool_response_time_ms: state.promptSentAt && state.toolResponseAt ? state.toolResponseAt - state.promptSentAt : null,
    waiting_interval_ms: state.toolCallAt && state.toolResponseAt ? state.toolResponseAt - state.toolCallAt : null,
    assistant_output_count_before_tool_response: state.assistantOutputCountBeforeToolResponse,
    assistant_output_count_after_tool_response: state.assistantOutputCountAfterToolResponse,
    tool_call_count: state.toolCallCount,
    tool_response_sent: state.primaryToolResponseSent,
    tool_response_skipped_after_close: state.toolResponseSkippedAfterClose,
    non_tool_response_outbound_during_tool_pending: state.nonToolResponseOutboundDuringToolPending,
    notes: plan.condition.notes,
  };
}

async function runOne(ai: GoogleGenAI, model: string, plan: RunPlan): Promise<RunMetrics> {
  mkdirSync(plan.runDir, { recursive: true });
  const eventsPath = resolve(plan.runDir, "events.jsonl");
  const response = { final_answer: EXPECTED_FINAL_ANSWER };
  writeJson(resolve(plan.runDir, "config.json"), {
    condition: plan.condition.id,
    run_id: plan.runId,
    latency_ms: plan.latencyMs,
    response_modality: plan.condition.responseModality,
    tools_enabled: plan.condition.toolsEnabled,
    timing_variant: plan.condition.timingVariant,
    prompt_version: PROMPT_VERSION,
    model,
  });

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
          responseModalities: [plan.condition.responseModality],
          outputAudioTranscription: plan.condition.responseModality === Modality.AUDIO ? {} : undefined,
          systemInstruction: SYSTEM_INSTRUCTION,
          tools: plan.condition.toolsEnabled
            ? [
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
              ]
            : undefined,
        },
        callbacks: {
          onopen: () => {
            appendEventLog(eventsPath, {
              type: "session_opened",
              model,
              condition: plan.condition.id,
              response_modality: plan.condition.responseModality,
              tools_enabled: plan.condition.toolsEnabled,
            });
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

            if (message.toolCall?.functionCalls?.length) {
              state.toolCallCount += message.toolCall.functionCalls.length;
              const call = message.toolCall.functionCalls[0];
              appendEventLog(eventsPath, { type: "tool_call_received", functionCalls: message.toolCall.functionCalls });
              if (!state.primaryToolCallReceived) {
                state.primaryToolCallReceived = true;
                state.toolCallPending = true;
                state.toolCallAt = now;
                pendingTimer = setTimeout(() => {
                  if (state.sessionClosed || done) {
                    state.toolResponseSkippedAfterClose = state.sessionClosed;
                    appendEventLog(eventsPath, {
                      type: state.sessionClosed ? "tool_response_skipped_after_close" : "tool_response_skipped",
                      delayMs: plan.latencyMs,
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
                      delayMs: plan.latencyMs,
                      response,
                    });
                  } catch (error) {
                    state.errors.push(summarizeError(error));
                    appendEventLog(eventsPath, { type: "tool_response_send_error", error: summarizeError(error) });
                  }
                }, plan.latencyMs);
              }
            }

            if (plan.condition.toolsEnabled && state.primaryToolResponseSent && message.serverContent?.turnComplete) {
              clearTimeout(hardTimer);
              finish(resolveRun);
            }

            if (!plan.condition.toolsEnabled && message.serverContent?.turnComplete) {
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
                delayMs: plan.latencyMs,
                reason: "session closed before scheduled tool response",
              });
            }
            appendEventLog(eventsPath, { type: "session_closed", code: event.code, reason: event.reason });
            clearTimeout(hardTimer);
            resolveRun();
          },
        },
      })) as Session;

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

  const metrics = computeMetrics(plan, model, state);
  writeJson(resolve(plan.runDir, "metrics.json"), metrics);
  if (!metrics.valid) {
    writeJson(resolve(plan.runDir, "error.json"), {
      error_type: metrics.error_type,
      server_1008_error: metrics.server_1008_error,
      close_code: metrics.close_code,
      close_reason: metrics.close_reason,
      errors: state.errors,
    });
  }
  return metrics;
}

function summarizeRows(condition: Condition, latencyMs: number, runs: RunMetrics[]): SummaryRow {
  const server1008 = runs.filter((run) => run.server_1008_error).length;
  const closeReasons = [...new Set(runs.map((run) => run.close_reason).filter((reason): reason is string => Boolean(reason)))];
  const errors = [...new Set(runs.map((run) => run.error_type).filter((error): error is string => Boolean(error)))];
  return {
    condition: condition.id,
    latency_ms: latencyMs,
    total_runs: runs.length,
    valid_runs: runs.filter((run) => run.valid).length,
    server_1008_errors: server1008,
    server_1008_error_rate: runs.length === 0 ? 0 : server1008 / runs.length,
    other_errors: runs.filter((run) => !run.valid && !run.server_1008_error).length,
    common_error_messages: errors,
    close_reasons: closeReasons,
    notes: condition.notes,
  };
}

function csvEscape(value: unknown): string {
  if (value === null || value === undefined) return "";
  const text = Array.isArray(value) ? value.join(" | ") : String(value);
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function writeCsv(path: string, rows: SummaryRow[]): void {
  const headers = [
    "condition",
    "latency_ms",
    "total_runs",
    "valid_runs",
    "server_1008_errors",
    "server_1008_error_rate",
    "other_errors",
    "common_error_messages",
    "close_reasons",
    "notes",
  ];
  const lines = [headers.join(",")];
  for (const row of rows) lines.push(headers.map((header) => csvEscape(row[header as keyof SummaryRow])).join(","));
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

function printTable(rows: SummaryRow[]): void {
  const headers = ["condition", "latency_ms", "total_runs", "valid_runs", "server_1008_errors", "server_1008_error_rate", "other_errors"];
  const widths = headers.map((header) => header.length);
  const cells = rows.map((row) =>
    headers.map((header) => {
      const value = row[header as keyof SummaryRow];
      return typeof value === "number" && header === "server_1008_error_rate" ? value.toFixed(2) : String(value);
    }),
  );
  for (const row of cells) row.forEach((cell, index) => (widths[index] = Math.max(widths[index], cell.length)));
  console.log(headers.map((header, index) => header.padEnd(widths[index])).join(" | "));
  console.log(widths.map((width) => "-".repeat(width)).join("-+-"));
  for (const row of cells) console.log(row.map((cell, index) => cell.padEnd(widths[index])).join(" | "));
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const apiKey = requireEnv("GEMINI_API_KEY");
  const model = requireEnv("GEMINI_LIVE_MODEL");
  const debugId = `debug_1008_${timestampForPath()}`;
  const debugDir = resolve(RESULT_DIR, debugId);
  mkdirSync(debugDir, { recursive: true });

  const conditions = conditionSet(args.latencies);
  writeJson(resolve(debugDir, "config.json"), {
    debug_id: debugId,
    model,
    prompt_version: PROMPT_VERSION,
    prompt: PROMPT,
    system_instruction: SYSTEM_INSTRUCTION,
    runs_per_cell: args.runsPerCell,
    concurrency: args.concurrency,
    latencies_ms: args.latencies,
    conditions: conditions.map((condition) => ({
      id: condition.id,
      response_modality: condition.responseModality,
      tools_enabled: condition.toolsEnabled,
      latencies_ms: condition.latencies,
      timing_variant: condition.timingVariant,
      notes: condition.notes,
    })),
    unavailable_conditions: [
      {
        id: "safer_timing",
        reason: "Not implemented: current Live event log does not expose a reliable generationComplete/audio-finished signal before tool response. turnComplete normally arrives after tool response, so waiting for it would deadlock the tool response path.",
      },
    ],
  });

  const plans: RunPlan[] = [];
  for (const condition of conditions) {
    const conditionDir = resolve(debugDir, `condition_${condition.id}`);
    mkdirSync(conditionDir, { recursive: true });
    for (const latencyMs of condition.latencies) {
      const latencyDir = resolve(conditionDir, `latency_${latencyMs}ms`);
      mkdirSync(latencyDir, { recursive: true });
      for (let runIndex = 1; runIndex <= args.runsPerCell; runIndex += 1) {
        const runId = `${condition.id}_latency_${latencyMs}ms_run_${String(runIndex).padStart(4, "0")}`;
        plans.push({
          condition,
          latencyMs,
          runIndex,
          runId,
          runDir: resolve(latencyDir, `run_${String(runIndex).padStart(4, "0")}_${timestampForPath()}`),
        });
      }
    }
  }

  console.log(`Debug directory: ${relative(PROJECT_DIR, debugDir)}`);
  console.log(`Runs: ${plans.length}; runs per cell: ${args.runsPerCell}; concurrency: ${args.concurrency}`);
  const ai = new GoogleGenAI({ apiKey });
  const metrics = await runWithConcurrency(plans, args.concurrency, async (plan) => {
    console.log(`[${plan.runId}] start`);
    const result = await runOne(ai, model, plan);
    console.log(`[${plan.runId}] ${result.valid ? "valid" : result.error_type}`);
    return result;
  });

  const rows: SummaryRow[] = [];
  for (const condition of conditions) {
    const conditionRows: SummaryRow[] = [];
    for (const latencyMs of condition.latencies) {
      const runs = metrics.filter((run) => run.condition === condition.id && run.latency_ms === latencyMs);
      const row = summarizeRows(condition, latencyMs, runs);
      rows.push(row);
      conditionRows.push(row);
      const latencyDir = resolve(debugDir, `condition_${condition.id}`, `latency_${latencyMs}ms`);
      writeJson(resolve(latencyDir, "summary.json"), { ...row, runs });
      writeCsv(resolve(latencyDir, "summary.csv"), [row]);
    }
    const conditionDir = resolve(debugDir, `condition_${condition.id}`);
    writeJson(resolve(conditionDir, "summary.json"), {
      condition: condition.id,
      rows: conditionRows,
      runs: metrics.filter((run) => run.condition === condition.id),
    });
    writeCsv(resolve(conditionDir, "summary.csv"), conditionRows);
  }

  writeJson(resolve(debugDir, "summary.json"), {
    debug_id: debugId,
    model,
    prompt_version: PROMPT_VERSION,
    rows,
    runs: metrics,
  });
  writeCsv(resolve(debugDir, "summary.csv"), rows);

  printTable(rows);
  console.log(`Summary: ${relative(PROJECT_DIR, resolve(debugDir, "summary.json"))}`);
}

main().catch((error) => {
  console.error("debug-1008-matrix failed");
  console.error(summarizeError(error));
  process.exitCode = 1;
});
