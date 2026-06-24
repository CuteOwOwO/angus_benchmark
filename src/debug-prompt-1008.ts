import { Behavior, GoogleGenAI, Modality } from "@google/genai";
import { config as loadEnv } from "dotenv";
import { appendFileSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { PROMPT, PROMPT_VERSION, SYSTEM_INSTRUCTION } from "./prompts.js";
import {
  PROMPT as TRY_PROMPT,
  PROMPT_VERSION as TRY_PROMPT_VERSION,
  SYSTEM_INSTRUCTION as TRY_SYSTEM_INSTRUCTION,
} from "./prompt_try.js";

const PROJECT_DIR = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const DEFAULT_ENV_FILE = resolve(PROJECT_DIR, "..", "angus_bench", ".env");
const RESULT_DIR = resolve(PROJECT_DIR, "result");
const TOOL_NAME = "external_answer_tool";
const EXPECTED_FINAL_ANSWER = "Angus";
const DEFAULT_LATENCY_MS = 3000;
const DEFAULT_RUNS = 10;
const DEFAULT_CONCURRENCY = 1;
const MAX_RUN_MS = 45_000;

loadEnv({ path: process.env.GEMINI_ENV_FILE || DEFAULT_ENV_FILE });

type FunctionCall = { id?: string; name?: string; args?: Record<string, unknown> };

type LiveMessage = {
  setupComplete?: unknown;
  serverContent?: {
    modelTurn?: { parts?: Array<{ text?: string; inlineData?: { mimeType?: string; data?: string } }> };
    outputTranscription?: { text?: string };
    turnComplete?: boolean;
  };
  toolCall?: { functionCalls?: FunctionCall[] };
  toolCallCancellation?: { ids?: string[] };
  usageMetadata?: unknown;
  goAway?: unknown;
  sessionResumptionUpdate?: unknown;
  voiceActivity?: unknown;
  voiceActivityDetectionSignal?: unknown;
};

type Session = {
  sendClientContent(params: { turns?: string; turnComplete?: boolean }): void;
  sendToolResponse(params: {
    functionResponses: { id?: string; name?: string; response: Record<string, unknown> }[];
  }): void;
  close(): void;
};

type PromptCondition = {
  id: string;
  promptVersion: string;
  prompt: string;
  systemInstruction: string;
  notes: string;
};

type RunPlan = {
  condition: PromptCondition;
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
  session_valid: boolean;
  error_type: string | null;
  server_1008_error: boolean;
  close_code: number | null;
  close_reason: string | null;
  prompt_version: string;
  model: string;
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
  valid_run_rate: number;
  avg_tool_call_time_ms: number | null;
  avg_first_response_latency_ms: number | null;
  assistant_output_count_before_tool_response: number | null;
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

function parseArgs(argv: string[]): { latencyMs: number; runs: number; concurrency: number; conditions?: string[] } {
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
    latencyMs: Math.max(0, Number(args.get("latency") ?? DEFAULT_LATENCY_MS)),
    runs: Math.max(1, Number(args.get("runs") ?? DEFAULT_RUNS)),
    concurrency: Math.max(1, Number(args.get("concurrency") ?? DEFAULT_CONCURRENCY)),
    conditions: args.get("conditions")?.split(",").map((condition) => condition.trim()).filter(Boolean),
  };
}

function promptConditions(): PromptCondition[] {
  return [
    {
      id: "current_prompt",
      promptVersion: PROMPT_VERSION,
      prompt: PROMPT,
      systemInstruction: SYSTEM_INSTRUCTION,
      notes: "Current formal prompt loaded from src/prompts.ts.",
    },
    {
      id: "minimal_tool_prompt",
      promptVersion: "debug_minimal_tool_prompt_v1",
      prompt: `Please answer this question:

What is the final answer?

You must use external_answer_tool.
Call external_answer_tool exactly once.
Do not answer before the tool result arrives.
After the tool result arrives, answer exactly with the value of final_answer.`,
      systemInstruction: `You are a helpful voice assistant.

The final answer must come from external_answer_tool.
Call external_answer_tool exactly once before answering.
Before the tool result arrives, do not answer, guess, or mention any possible answer.
After the tool result arrives, answer exactly with final_answer and nothing else.`,
      notes: "Minimal tool-use prompt. It removes the explicit pre-tool reassurance sentence.",
    },
    {
      id: "no_pre_speech_prompt",
      promptVersion: "debug_no_pre_speech_prompt_v1",
      prompt: `Please answer this question:

What is the final answer?

You must use external_answer_tool.

First, immediately call external_answer_tool exactly once.
Do not say anything to the user before the tool call request has been made.

While waiting for the tool result, you may give short waiting updates, but do not guess or mention any possible answer.

After the tool result arrives, answer exactly with the value of final_answer.`,
      systemInstruction: `You are a helpful voice assistant.

The final answer must come from external_answer_tool.
Your first action must be the tool call request.
Do not produce user-facing speech before the tool call request.
Before the tool result arrives, do not answer, guess, infer, or mention any possible answer.
After the tool result arrives, answer exactly with final_answer and nothing else.`,
      notes: "Tool-first prompt designed to avoid initial audio output before tool call.",
    },
    {
      id: "prompt_try",
      promptVersion: TRY_PROMPT_VERSION,
      prompt: TRY_PROMPT,
      systemInstruction: TRY_SYSTEM_INSTRUCTION,
      notes: "Prompt candidate loaded from src/prompt_try.ts.",
    },
    {
      id: "good_2026_06_15_prompt",
      promptVersion: "single_final_answer_field_v2_good_2026_06_15",
      prompt: `Please answer this question:

What is the final answer?

You must use external_answer_tool.

Follow this order strictly:

1. Say one short sentence aloud to the user to let them know you are checking.

2. Immediately emit a tool call request to external_answer_tool.
   This tool call is mandatory and must happen exactly once.
   Do not continue toward the final answer until the tool call request has actually been made.

3. While waiting for the tool result, you may briefly reassure the user.
   Waiting updates must be short and must not contain any possible answer.

4. Before the tool result arrives, do not give, guess, infer, summarize, or mention any possible final answer.
   Time passing is not a reason to answer.
   User-facing waiting speech is not a substitute for receiving the tool result.

5. After the tool result arrives, ignore everything except the field named final_answer.

6. Your final answer must be exactly the value of final_answer from the tool result.
   Do not add explanations, prefixes, suffixes, confidence, status, tool names, or extra words.`,
      systemInstruction: `You are a helpful voice assistant.

For this task, the final answer must come from external_answer_tool.

You must emit exactly one tool call request to external_answer_tool before giving the final answer.

The tool call is a hard requirement:
- You are not allowed to produce the final answer before the tool result is received.
- You are not allowed to guess or infer the answer from context.
- You are not allowed to treat waiting time, uncertainty, or partial speech as permission to answer.
- If the tool result has not arrived yet, only give short waiting updates or remain silent.

Before the tool result arrives, you may only say brief waiting updates to the user.
Do not mention any candidate answer.

The tool result contains a field named final_answer.

After the tool result arrives:
- Read only the final_answer field.
- Answer with exactly that value.
- Do not answer with the tool status, tool name, confidence, metadata, or any other field.
- Do not add any extra words before or after the final answer.`,
      notes: "Historical best prompt copied from result/2026-06-15T03-49-10-009Z(good)/prompts.txt.",
    },
  ];
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

function classifyError(state: RunState): string | null {
  if (state.closeCode === 1008) return "server_1008_error";
  if (state.errors.length > 0) return "session_error";
  if (!state.primaryToolCallReceived) return "missing_tool_call";
  if (!state.primaryToolResponseSent) return "missing_tool_response";
  return null;
}

function computeMetrics(plan: RunPlan, model: string, state: RunState): RunMetrics {
  const errorType = classifyError(state);
  return {
    condition: plan.condition.id,
    run_id: plan.runId,
    latency_ms: plan.latencyMs,
    valid: errorType === null,
    session_valid: errorType === null,
    error_type: errorType,
    server_1008_error: errorType === "server_1008_error",
    close_code: state.closeCode,
    close_reason: state.closeReason,
    prompt_version: plan.condition.promptVersion,
    model,
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
    prompt_version: plan.condition.promptVersion,
    prompt: plan.condition.prompt,
    system_instruction: plan.condition.systemInstruction,
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
    appendEventLog(eventsPath, { type: "outbound_event", outbound_type: outboundType, toolCallPending: state.toolCallPending, sessionClosed: state.sessionClosed, allowed, reason });
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
          systemInstruction: plan.condition.systemInstruction,
          tools: [
            {
              functionDeclarations: [
                {
                  name: TOOL_NAME,
                  description: "Gets the final answer from an external information source.",
                  behavior: Behavior.NON_BLOCKING,
                  parametersJsonSchema: {
                    type: "object",
                    properties: { question: { type: "string" } },
                    required: ["question"],
                  },
                },
              ],
            },
          ],
        },
        callbacks: {
          onopen: () => appendEventLog(eventsPath, { type: "session_opened", model, condition: plan.condition.id }),
          onmessage: (message: LiveMessage) => {
            const now = Date.now();
            appendEventLog(eventsPath, { type: "server_event", eventTypes: eventTypes(message), event_ms: state.promptSentAt ? now - state.promptSentAt : null, message: sanitizeMessage(message) });

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
              appendEventLog(eventsPath, { type: "tool_call_received", functionCalls: message.toolCall.functionCalls });
              const call = message.toolCall.functionCalls[0];
              if (!state.primaryToolCallReceived) {
                state.primaryToolCallReceived = true;
                state.toolCallPending = true;
                state.toolCallAt = now;
                pendingTimer = setTimeout(() => {
                  if (state.sessionClosed || done) {
                    state.toolResponseSkippedAfterClose = state.sessionClosed;
                    appendEventLog(eventsPath, { type: state.sessionClosed ? "tool_response_skipped_after_close" : "tool_response_skipped", delayMs: plan.latencyMs });
                    return;
                  }
                  if (!outbound("sendToolResponse", "scheduled tool response delay elapsed")) return;
                  try {
                    session?.sendToolResponse({
                      functionResponses: [{ id: call.id, name: call.name || TOOL_NAME, response }],
                    });
                    state.toolCallPending = false;
                    state.primaryToolResponseSent = true;
                    state.toolResponseAt = Date.now();
                    appendEventLog(eventsPath, { type: "tool_response_sent", functionCallId: call.id, functionName: call.name, delayMs: plan.latencyMs, response });
                  } catch (error) {
                    state.errors.push(summarizeError(error));
                    appendEventLog(eventsPath, { type: "tool_response_send_error", error: summarizeError(error) });
                  }
                }, plan.latencyMs);
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
              appendEventLog(eventsPath, { type: "tool_response_skipped_after_close", delayMs: plan.latencyMs, reason: "session closed before scheduled tool response" });
            }
            appendEventLog(eventsPath, { type: "session_closed", code: event.code, reason: event.reason });
            clearTimeout(hardTimer);
            resolveRun();
          },
        },
      })) as Session;

      if (outbound("sendClientContent", "initial user prompt")) {
        session.sendClientContent({ turns: plan.condition.prompt, turnComplete: true });
        state.promptSentAt = Date.now();
        appendEventLog(eventsPath, { type: "user_message_sent", prompt_version: plan.condition.promptVersion });
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
  writeJson(resolve(plan.runDir, "raw_result.json"), { metrics });
  if (!metrics.valid) {
    writeJson(resolve(plan.runDir, "error.json"), { error_type: metrics.error_type, server_1008_error: metrics.server_1008_error, close_code: metrics.close_code, close_reason: metrics.close_reason, errors: state.errors });
  }
  return metrics;
}

function average(values: number[]): number | null {
  if (values.length === 0) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function summarizeCondition(condition: PromptCondition, latencyMs: number, runs: RunMetrics[]): SummaryRow {
  const valid = runs.filter((run) => run.valid);
  const server1008 = runs.filter((run) => run.server_1008_error).length;
  return {
    condition: condition.id,
    latency_ms: latencyMs,
    total_runs: runs.length,
    valid_runs: valid.length,
    server_1008_errors: server1008,
    server_1008_error_rate: runs.length === 0 ? 0 : server1008 / runs.length,
    other_errors: runs.filter((run) => !run.valid && !run.server_1008_error).length,
    valid_run_rate: runs.length === 0 ? 0 : valid.length / runs.length,
    avg_tool_call_time_ms: average(valid.map((run) => run.tool_call_time_ms).filter((value): value is number => value !== null)),
    avg_first_response_latency_ms: average(valid.map((run) => run.first_response_latency_ms).filter((value): value is number => value !== null)),
    assistant_output_count_before_tool_response: average(valid.map((run) => run.assistant_output_count_before_tool_response)),
    notes: condition.notes,
  };
}

function csvEscape(value: unknown): string {
  if (value === null || value === undefined) return "";
  const text = String(value);
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
    "valid_run_rate",
    "avg_tool_call_time_ms",
    "avg_first_response_latency_ms",
    "assistant_output_count_before_tool_response",
    "notes",
  ];
  writeFileSync(path, `${headers.join(",")}\n${rows.map((row) => headers.map((header) => csvEscape(row[header as keyof SummaryRow])).join(",")).join("\n")}\n`, "utf8");
}

async function runWithConcurrency<T, R>(items: T[], concurrency: number, worker: (item: T) => Promise<R>): Promise<R[]> {
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
  const cells = rows.map((row) => headers.map((header) => {
    const value = row[header as keyof SummaryRow];
    return typeof value === "number" && header === "server_1008_error_rate" ? value.toFixed(2) : String(value);
  }));
  for (const row of cells) row.forEach((cell, index) => (widths[index] = Math.max(widths[index], cell.length)));
  console.log(headers.map((header, index) => header.padEnd(widths[index])).join(" | "));
  console.log(widths.map((width) => "-".repeat(width)).join("-+-"));
  for (const row of cells) console.log(row.map((cell, index) => cell.padEnd(widths[index])).join(" | "));
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const apiKey = requireEnv("GEMINI_API_KEY");
  const model = requireEnv("GEMINI_LIVE_MODEL");
  const debugId = `debug_prompt_1008_${timestampForPath()}`;
  const debugDir = resolve(RESULT_DIR, debugId);
  mkdirSync(debugDir, { recursive: true });

  const requestedConditions = new Set(args.conditions);
  const conditions = promptConditions().filter((condition) =>
    requestedConditions.size === 0 || requestedConditions.has(condition.id),
  );
  if (conditions.length === 0) {
    throw new Error(`No matching prompt conditions. Requested: ${args.conditions?.join(", ")}`);
  }
  writeJson(resolve(debugDir, "config.json"), {
    debug_id: debugId,
    model,
    latency_ms: args.latencyMs,
    runs: args.runs,
    concurrency: args.concurrency,
    requested_conditions: args.conditions,
    formal_prompt_version: PROMPT_VERSION,
    conditions: conditions.map((condition) => ({
      id: condition.id,
      prompt_version: condition.promptVersion,
      prompt: condition.prompt,
      system_instruction: condition.systemInstruction,
      notes: condition.notes,
    })),
  });

  const plans: RunPlan[] = [];
  for (const condition of conditions) {
    const conditionDir = resolve(debugDir, `condition_${condition.id}`);
    mkdirSync(conditionDir, { recursive: true });
    for (let runIndex = 1; runIndex <= args.runs; runIndex += 1) {
      plans.push({
        condition,
        latencyMs: args.latencyMs,
        runIndex,
        runId: `${condition.id}_run_${String(runIndex).padStart(4, "0")}`,
        runDir: resolve(conditionDir, `run_${String(runIndex).padStart(4, "0")}_${timestampForPath()}`),
      });
    }
  }

  console.log(`Debug directory: ${relative(PROJECT_DIR, debugDir)}`);
  console.log(`Latency: ${args.latencyMs}ms; runs per condition: ${args.runs}; concurrency: ${args.concurrency}`);
  const ai = new GoogleGenAI({ apiKey });
  const metrics = await runWithConcurrency(plans, args.concurrency, async (plan) => {
    console.log(`[${plan.runId}] start`);
    const result = await runOne(ai, model, plan);
    console.log(`[${plan.runId}] ${result.valid ? "valid" : result.error_type}`);
    return result;
  });

  const rows: SummaryRow[] = [];
  for (const condition of conditions) {
    const conditionDir = resolve(debugDir, `condition_${condition.id}`);
    const conditionRuns = metrics.filter((run) => run.condition === condition.id);
    const row = summarizeCondition(condition, args.latencyMs, conditionRuns);
    rows.push(row);
    writeJson(resolve(conditionDir, "summary.json"), { ...row, runs: conditionRuns });
    writeCsv(resolve(conditionDir, "summary.csv"), [row]);
  }
  writeJson(resolve(debugDir, "summary.json"), { debug_id: debugId, model, prompt_version: PROMPT_VERSION, latency_ms: args.latencyMs, rows, runs: metrics });
  writeCsv(resolve(debugDir, "summary.csv"), rows);
  printTable(rows);
  console.log(`Summary: ${relative(PROJECT_DIR, resolve(debugDir, "summary.json"))}`);
}

main().catch((error) => {
  console.error("debug-prompt-1008 failed");
  console.error(summarizeError(error));
  process.exitCode = 1;
});
