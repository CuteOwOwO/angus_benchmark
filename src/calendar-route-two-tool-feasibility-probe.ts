import { GoogleGenAI, Modality } from "@google/genai";
import { config as loadEnv } from "dotenv";
import { appendFileSync, copyFileSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { getTwoStepTask, listTwoStepTaskIds, makeToolDeclarations, type TwoStepTaskSpec } from "./tasks/two-step-tasks.js";
import {
  PROMPT_NAME as CALENDAR_NO_TICK_PROMPT_NAME,
  SYSTEM_INSTRUCTION as CALENDAR_NO_TICK_SYSTEM_INSTRUCTION,
  USER_PROMPT as CALENDAR_NO_TICK_USER_PROMPT,
} from "./prompts/tow-2step-calendar-route-no-tick.js";
import {
  PROMPT_NAME as CALENDAR_WITH_TICK_PROMPT_NAME,
  SYSTEM_INSTRUCTION as CALENDAR_WITH_TICK_SYSTEM_INSTRUCTION,
  USER_PROMPT as CALENDAR_WITH_TICK_USER_PROMPT,
} from "./prompts/tow-2step-calendar-route-with-tick.js";

const PROJECT_DIR = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const DEFAULT_ENV_FILE = resolve(PROJECT_DIR, "..", "angus_bench", ".env");
const RESULT_DIR = process.env.GEMINI_LIVE_CHECK_RESULT_DIR
  ? resolve(process.env.GEMINI_LIVE_CHECK_RESULT_DIR)
  : resolve(PROJECT_DIR, "result");

const DEFAULT_TASK_ID = "calendar_route_leave_time";
const DEFAULT_ATTEMPTS = 20;
const DEFAULT_TOOL_LATENCY_MS = 3000;
const MAX_ATTEMPT_MS = 45_000;
const POST_FINAL_MIN_WAIT_MS = 15_000;
const POST_FINAL_IDLE_MS = 3000;
const PCM_BYTES_PER_SECOND = 48_000;
const TICK_EVERY_MS = 4000;
const PENDING_TICK_MESSAGE = "The tool call is still running. No final result is available yet.";
const DEFAULT_CONDITIONS: Condition[] = ["no_tick"];

type Condition = "no_tick" | "periodic_tick_4s";

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
};

type Session = {
  sendClientContent(params: { turns?: string; turnComplete?: boolean }): void;
  sendRealtimeInput(params: { text?: string }): void;
  sendToolResponse(params: {
    functionResponses: {
      id?: string;
      name?: string;
      response: Record<string, unknown>;
    }[];
  }): void;
  close(): void;
};

type Args = {
  taskId: string;
  condition: Condition;
  attempts: number;
  toolLatencyMs: number;
  latenciesMs: number[];
  conditions: Condition[];
  postFinalMinWaitMs: number;
  postFinalIdleMs: number;
  quietTerminalText: boolean;
};

type ToolCallRecord = {
  event_ms: number | null;
  name?: string;
  args?: Record<string, unknown>;
  id?: string;
};

type ToolResponseRecord = {
  event_ms: number | null;
  name?: string;
  response: Record<string, unknown>;
  send_error?: string;
};

type AudioInterval = {
  event_ms: number;
  start_ms: number;
  end_ms: number;
  bytes: number;
};

type TextEvent = {
  event_ms: number;
  text: string;
  phase: string;
};

type AttemptSummary = {
  task_id: string;
  condition: Condition;
  attempt_index: number;
  latency_ms: number;
  valid_run: boolean;
  called_any_tool: boolean;
  called_get_next_calendar_event: boolean;
  called_get_route_eta: boolean;
  tool_order: string;
  tool1_call_time_ms: number | null;
  tool1_result_sent_time_ms: number | null;
  tool2_call_time_ms: number | null;
  tool2_result_sent_time_ms: number | null;
  tool2_called_after_tool1_result: boolean;
  tool2_arg_from_tool1_result: boolean;
  parallel_tool_call_detected: boolean;
  sequential_two_tool_success: boolean;
  both_tools_called: boolean;
  calendar_call_time_ms: number | null;
  calendar_result_sent_time_ms: number | null;
  route_call_time_ms: number | null;
  route_result_sent_time_ms: number | null;
  route_called_after_calendar_result: boolean;
  route_arg_from_calendar_result: boolean;
  stage1_calendar_waiting_window_ms: number | null;
  stage1_audio_occupancy_ratio: number | null;
  stage1_max_silence_gap_ms: number | null;
  stage1_spoken_segment_count: number;
  stage1_waiting_speech_present: boolean;
  stage2_route_waiting_window_ms: number | null;
  stage2_audio_occupancy_ratio: number | null;
  stage2_max_silence_gap_ms: number | null;
  stage2_spoken_segment_count: number;
  stage2_waiting_speech_present: boolean;
  route_result_to_final_audio_latency_ms: number | null;
  final_answer_mentions_2_15: boolean;
  final_uses_calendar_time: boolean;
  final_uses_route_eta: boolean;
  final_core_answer_correct: boolean;
  premature_answer_before_tools: boolean;
  premature_answer_before_required_tool_result: boolean;
  close_1008: boolean;
  close_1011: boolean;
  close_1006: boolean;
  close_code: number | null;
  close_reason: string | null;
  first_audio_time_ms: number | null;
  last_audio_time_ms: number | null;
  session_close_time_ms: number | null;
  time_from_tool2_result_to_session_close_ms: number | null;
  server_audio_or_text_after_tool2_result: boolean;
  close_trigger: string | null;
  tool1_result_to_tool2_call_latency_ms: number | null;
  post_final_min_wait_ms: number;
  post_final_idle_ms: number;
  pending_tick_every_ms: number | null;
  pending_tick_count: number;
  pending_tick_times_ms: number[];
  stage1_pending_tick_times_ms: number[];
  stage2_pending_tick_times_ms: number[];
  send_error_count: number;
  tool_call_cancellation_count: number;
  raw_event_count: number;
  text_before_tools: string[];
  text_after_tools: string[];
  tool_calls: ToolCallRecord[];
  tool_responses: ToolResponseRecord[];
  errors: string[];
  result_dir: string;
};

type AggregateSummary = {
  attempts: number;
  any_tool_call_rate: number;
  calendar_tool_call_rate: number;
  route_tool_call_rate: number;
  both_tools_called_rate: number;
  sequential_two_tool_success_rate: number;
  parallel_tool_call_rate: number;
  tool2_arg_from_tool1_result_rate: number;
  final_answer_correct_rate: number;
  premature_answer_rate: number;
  "1008_error_count": number;
  "1011_error_count": number;
  "1006_error_count": number;
  send_error_count: number;
};

type LatencySummaryRow = {
  condition: Condition;
  latency_ms: number;
  attempts: number;
  sequential_two_tool_success_rate: number;
  parallel_tool_call_rate: number;
  final_core_answer_correct_rate: number;
  premature_answer_rate: number;
  stage1_waiting_speech_present_rate: number;
  stage1_audio_occupancy_ratio_mean: number | null;
  stage1_max_silence_gap_ms_mean: number | null;
  stage2_waiting_speech_present_rate: number;
  stage2_audio_occupancy_ratio_mean: number | null;
  stage2_max_silence_gap_ms_mean: number | null;
  pending_tick_count_mean: number | null;
  stage1_pending_tick_count_mean: number | null;
  stage2_pending_tick_count_mean: number | null;
  min_tool2_result_to_session_close_ms: number | null;
  "1008_error_count": number;
  "1011_error_count": number;
  "1006_error_count": number;
  send_error_count: number;
};

type AttemptState = {
  sessionOpenedAt?: number;
  setupCompleteAt?: number;
  promptSentAt?: number;
  allToolResponsesAt?: number;
  sessionClosed: boolean;
  closeCode: number | null;
  closeReason: string | null;
  rawEventCount: number;
  toolCalls: ToolCallRecord[];
  toolResponses: ToolResponseRecord[];
  textBeforeTools: string[];
  textAfterTools: string[];
  textEvents: TextEvent[];
  cancelledToolCallIds: string[];
  sendErrors: string[];
  errors: string[];
  audioIntervals: AudioInterval[];
  audioPlaybackCursorMs: number;
  lastAssistantEventAt?: number;
  sessionCloseTimeMs: number | null;
  closeTrigger: string | null;
  pendingTicks: Array<{ event_ms: number | null; tool_name?: string; call_id?: string; message: string }>;
};

loadEnv({ path: process.env.GEMINI_ENV_FILE || DEFAULT_ENV_FILE });

function usage(): string {
  return [
    "Usage: two-step-native-no-tick-benchmark [options]",
    "",
    "Options:",
    `  --task <id>             Two-step task id. Default: ${DEFAULT_TASK_ID}. Available: ${listTwoStepTaskIds().join(", ")}.`,
    "  --attempts <n>          Number of attempts. Default: 20.",
    "  --tool-latency-ms <n>   Mocked latency for each tool response. Default: 3000.",
    "  --latencies-ms <csv>    Comma-separated latency sweep. Overrides --tool-latency-ms.",
    "  --conditions <csv>      Comma-separated conditions: no_tick, periodic_tick_4s.",
    "  --post-final-min-wait-ms <n>",
    `                          Minimum time to keep the session alive after the final tool result. Default: ${POST_FINAL_MIN_WAIT_MS}.`,
    "  --post-final-idle-ms <n>",
    "                          Assistant audio/text idle window before closing after the minimum wait. Default: 3000.",
    "  --quiet-terminal-text   Suppress transcript text in terminal.",
    "  --help                  Show this help.",
  ].join("\n");
}

function parseArgs(argv: string[]): Args {
  const args: Args = {
    taskId: process.env.TWO_STEP_TASK_ID || DEFAULT_TASK_ID,
    condition: "no_tick",
    attempts: Number(process.env.TWO_STEP_ATTEMPTS ?? process.env.CALENDAR_ROUTE_ATTEMPTS ?? DEFAULT_ATTEMPTS),
    toolLatencyMs: Number(process.env.TWO_STEP_TOOL_LATENCY_MS ?? process.env.CALENDAR_ROUTE_TOOL_LATENCY_MS ?? DEFAULT_TOOL_LATENCY_MS),
    latenciesMs: [],
    conditions: String(process.env.TWO_STEP_CONDITIONS ?? DEFAULT_CONDITIONS.join(","))
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean) as Condition[],
    postFinalMinWaitMs: Number(process.env.TWO_STEP_POST_FINAL_MIN_WAIT_MS ?? process.env.CALENDAR_ROUTE_POST_FINAL_MIN_WAIT_MS ?? POST_FINAL_MIN_WAIT_MS),
    postFinalIdleMs: Number(process.env.TWO_STEP_POST_FINAL_IDLE_MS ?? process.env.CALENDAR_ROUTE_POST_FINAL_IDLE_MS ?? POST_FINAL_IDLE_MS),
    quietTerminalText: false,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") {
      console.log(usage());
      process.exit(0);
    }
    if (arg === "--quiet-terminal-text") {
      args.quietTerminalText = true;
      continue;
    }
    if (arg === "--task") {
      const value = argv[index + 1];
      if (!value) throw new Error("--task requires a value");
      args.taskId = value;
      index += 1;
      continue;
    }
    if (arg === "--attempts") {
      const value = argv[index + 1];
      if (!value) throw new Error("--attempts requires a value");
      args.attempts = Number(value);
      index += 1;
      continue;
    }
    if (arg === "--tool-latency-ms") {
      const value = argv[index + 1];
      if (!value) throw new Error("--tool-latency-ms requires a value");
      args.toolLatencyMs = Number(value);
      index += 1;
      continue;
    }
    if (arg === "--latencies-ms") {
      const value = argv[index + 1];
      if (!value) throw new Error("--latencies-ms requires a value");
      args.latenciesMs = value.split(",").map((item) => Number(item.trim())).filter((value) => !Number.isNaN(value));
      index += 1;
      continue;
    }
    if (arg === "--conditions") {
      const value = argv[index + 1];
      if (!value) throw new Error("--conditions requires a value");
      args.conditions = value.split(",").map((item) => item.trim()).filter(Boolean) as Condition[];
      index += 1;
      continue;
    }
    if (arg === "--post-final-min-wait-ms") {
      const value = argv[index + 1];
      if (!value) throw new Error("--post-final-min-wait-ms requires a value");
      args.postFinalMinWaitMs = Number(value);
      index += 1;
      continue;
    }
    if (arg === "--post-final-idle-ms") {
      const value = argv[index + 1];
      if (!value) throw new Error("--post-final-idle-ms requires a value");
      args.postFinalIdleMs = Number(value);
      index += 1;
      continue;
    }
    const inline = arg.match(/^--attempts=(\d+)$/);
    if (inline) {
      args.attempts = Number(inline[1]);
      continue;
    }
    const taskInline = arg.match(/^--task=([A-Za-z0-9_-]+)$/);
    if (taskInline) {
      args.taskId = taskInline[1];
      continue;
    }
    const toolLatencyInline = arg.match(/^--tool-latency-ms=(\d+)$/);
    if (toolLatencyInline) {
      args.toolLatencyMs = Number(toolLatencyInline[1]);
      continue;
    }
    const latenciesInline = arg.match(/^--latencies-ms=([0-9,]+)$/);
    if (latenciesInline) {
      args.latenciesMs = latenciesInline[1].split(",").map((item) => Number(item.trim())).filter((value) => !Number.isNaN(value));
      continue;
    }
    const conditionsInline = arg.match(/^--conditions=([A-Za-z0-9_,]+)$/);
    if (conditionsInline) {
      args.conditions = conditionsInline[1].split(",").map((item) => item.trim()).filter(Boolean) as Condition[];
      continue;
    }
    const postFinalMinWaitInline = arg.match(/^--post-final-min-wait-ms=(\d+)$/);
    if (postFinalMinWaitInline) {
      args.postFinalMinWaitMs = Number(postFinalMinWaitInline[1]);
      continue;
    }
    const postFinalIdleInline = arg.match(/^--post-final-idle-ms=(\d+)$/);
    if (postFinalIdleInline) {
      args.postFinalIdleMs = Number(postFinalIdleInline[1]);
      continue;
    }
    throw new Error(`Unknown argument: ${arg}\n${usage()}`);
  }
  if (!Number.isInteger(args.attempts) || args.attempts < 1) throw new Error(`Invalid --attempts: ${args.attempts}`);
  getTwoStepTask(args.taskId);
  if (!Number.isFinite(args.toolLatencyMs) || args.toolLatencyMs < 0) {
    throw new Error(`Invalid --tool-latency-ms: ${args.toolLatencyMs}`);
  }
  if (!args.latenciesMs.length) args.latenciesMs = [args.toolLatencyMs];
  for (const latency of args.latenciesMs) {
    if (!Number.isFinite(latency) || latency < 0) throw new Error(`Invalid --latencies-ms item: ${latency}`);
  }
  for (const condition of args.conditions) {
    if (!["no_tick", "periodic_tick_4s"].includes(condition)) throw new Error(`Invalid --conditions item: ${condition}`);
  }
  if (!Number.isFinite(args.postFinalMinWaitMs) || args.postFinalMinWaitMs < 0) {
    throw new Error(`Invalid --post-final-min-wait-ms: ${args.postFinalMinWaitMs}`);
  }
  if (!Number.isFinite(args.postFinalIdleMs) || args.postFinalIdleMs < 0) {
    throw new Error(`Invalid --post-final-idle-ms: ${args.postFinalIdleMs}`);
  }
  return args;
}

function requireEnv(name: string): string {
  const value = process.env[name]?.trim();
  if (!value) throw new Error(`Missing ${name}. Set it in your shell or .env.`);
  return value;
}

function timestampForPath(date = new Date()): string {
  const [datePart, timePart] = date.toISOString().split("T");
  return `${datePart}_${timePart.replace("Z", "").replace(/\./g, "-").replace(/:/g, "-")}`;
}

function appendJsonl(path: string, value: unknown): void {
  appendFileSync(path, `${JSON.stringify(value)}\n`, "utf8");
}

function writeJson(path: string, value: unknown): void {
  writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

function csvEscape(value: unknown): string {
  if (value === null || value === undefined) return "";
  if (Array.isArray(value) || typeof value === "object") return csvEscape(JSON.stringify(value));
  const text = String(value);
  if (!/[",\n]/.test(text)) return text;
  return `"${text.replace(/"/g, '""')}"`;
}

function writeCsv(path: string, rows: Array<Record<string, unknown>>): void {
  if (!rows.length) {
    writeFileSync(path, "", "utf8");
    return;
  }
  const headers = Object.keys(rows[0]);
  const lines = [headers.join(",")];
  for (const row of rows) lines.push(headers.map((header) => csvEscape(row[header])).join(","));
  writeFileSync(path, `${lines.join("\n")}\n`, "utf8");
}

function rate(count: number, total: number): number {
  return total ? Math.round((count / total) * 1000) / 1000 : 0;
}

function mean(values: Array<number | null | undefined>): number | null {
  const numeric = values.filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  if (!numeric.length) return null;
  return Math.round((numeric.reduce((sum, value) => sum + value, 0) / numeric.length) * 1000) / 1000;
}

function summarizeError(error: unknown): string {
  if (error instanceof Error) return `${error.name}: ${error.message}`;
  if (typeof error !== "object" || error === null) return String(error);
  const record = error as Record<string, unknown>;
  const fields = ["message", "type", "code", "reason", "name", "error"]
    .map((key) => (record[key] ? `${key}: ${String(record[key])}` : undefined))
    .filter(Boolean);
  return fields.length ? fields.join(", ") : Object.prototype.toString.call(error);
}

function eventTypes(message: LiveMessage): string[] {
  return [
    message.setupComplete ? "setupComplete" : undefined,
    message.serverContent ? "serverContent" : undefined,
    message.toolCall ? "toolCall" : undefined,
    message.toolCallCancellation ? "toolCallCancellation" : undefined,
  ].filter((value): value is string => Boolean(value));
}

function sanitizeMessage(message: LiveMessage): LiveMessage {
  return {
    ...message,
    serverContent: message.serverContent
      ? {
          ...message.serverContent,
          modelTurn: message.serverContent.modelTurn
            ? {
                parts: message.serverContent.modelTurn.parts?.map((part) =>
                  part.inlineData?.data
                    ? { ...part, inlineData: { ...part.inlineData, data: `<base64 ${part.inlineData.data.length} chars>` } }
                    : part,
                ),
              }
            : undefined,
        }
      : undefined,
  };
}

function responseForTool(task: TwoStepTaskSpec, name?: string): Record<string, unknown> {
  if (name === task.step1.toolName) return task.step1.mockedResult;
  if (name === task.step2.toolName) return task.step2.mockedResult;
  return { error: `unknown tool ${name || "unknown"}` };
}

function taskForCondition(task: TwoStepTaskSpec, condition: Condition): TwoStepTaskSpec {
  if (!task.id.startsWith("calendar_route_leave_time")) return task;
  if (condition === "periodic_tick_4s") {
    return {
      ...task,
      resultSlug: task.resultSlug.replace(/_no_tick/g, "_periodic_tick_4s"),
      promptName: CALENDAR_WITH_TICK_PROMPT_NAME,
      systemInstruction: CALENDAR_WITH_TICK_SYSTEM_INSTRUCTION,
      userPrompt: CALENDAR_WITH_TICK_USER_PROMPT,
    };
  }
  return {
    ...task,
    promptName: CALENDAR_NO_TICK_PROMPT_NAME,
    systemInstruction: CALENDAR_NO_TICK_SYSTEM_INSTRUCTION,
    userPrompt: CALENDAR_NO_TICK_USER_PROMPT,
  };
}

function finalMentionsTargetAnswer(task: TwoStepTaskSpec, text: string): boolean {
  return task.finalAnswerChecks.mentionsTargetAnswer.test(text);
}

function usesStep1Result(task: TwoStepTaskSpec, text: string): boolean {
  return task.finalAnswerChecks.usesStep1Result.test(text);
}

function usesStep2Result(task: TwoStepTaskSpec, text: string): boolean {
  return task.finalAnswerChecks.usesStep2Result.test(text);
}

function prematureAnswer(task: TwoStepTaskSpec, text: string): boolean {
  return task.finalAnswerChecks.prematureAnswer.test(text);
}

function round3(value: number): number {
  return Math.round(value * 1000) / 1000;
}

function overlapMs(startA: number, endA: number, startB: number, endB: number): number {
  return Math.max(0, Math.min(endA, endB) - Math.max(startA, startB));
}

function mergeStageSegments(intervals: AudioInterval[], windowStart: number, windowEnd: number): Array<{ start_ms: number; end_ms: number }> {
  const clipped = intervals
    .map((interval) => ({
      start_ms: Math.max(windowStart, interval.start_ms),
      end_ms: Math.min(windowEnd, interval.end_ms),
    }))
    .filter((segment) => segment.end_ms > segment.start_ms)
    .sort((left, right) => left.start_ms - right.start_ms);
  const merged: Array<{ start_ms: number; end_ms: number }> = [];
  for (const segment of clipped) {
    if (!merged.length || segment.start_ms - merged[merged.length - 1].end_ms > 250) {
      merged.push({ ...segment });
      continue;
    }
    merged[merged.length - 1].end_ms = Math.max(merged[merged.length - 1].end_ms, segment.end_ms);
  }
  return merged;
}

function stageAudioMetrics(
  intervals: AudioInterval[],
  windowStart: number | null,
  windowEnd: number | null,
): {
  waiting_window_ms: number | null;
  audio_occupancy_ratio: number | null;
  max_silence_gap_ms: number | null;
  spoken_segment_count: number;
  waiting_speech_present: boolean;
} {
  if (typeof windowStart !== "number" || typeof windowEnd !== "number" || windowEnd <= windowStart) {
    return {
      waiting_window_ms: null,
      audio_occupancy_ratio: null,
      max_silence_gap_ms: null,
      spoken_segment_count: 0,
      waiting_speech_present: false,
    };
  }
  const segments = mergeStageSegments(intervals, windowStart, windowEnd);
  const windowMs = windowEnd - windowStart;
  const audioMs = segments.reduce((sum, segment) => sum + overlapMs(segment.start_ms, segment.end_ms, windowStart, windowEnd), 0);
  let maxSilence = windowMs;
  if (segments.length) {
    maxSilence = Math.max(0, segments[0].start_ms - windowStart, windowEnd - segments[segments.length - 1].end_ms);
    for (let index = 1; index < segments.length; index += 1) {
      maxSilence = Math.max(maxSilence, segments[index].start_ms - segments[index - 1].end_ms);
    }
  }
  return {
    waiting_window_ms: windowMs,
    audio_occupancy_ratio: round3(audioMs / windowMs),
    max_silence_gap_ms: Math.round(maxSilence),
    spoken_segment_count: segments.length,
    waiting_speech_present: segments.length > 0,
  };
}

function firstAudioAtOrAfter(intervals: AudioInterval[], timeMs: number | null): number | null {
  if (typeof timeMs !== "number") return null;
  const starts = intervals.map((interval) => interval.start_ms).filter((start) => start >= timeMs).sort((a, b) => a - b);
  return starts[0] ?? null;
}

function makeSummary(task: TwoStepTaskSpec, attemptIndex: number, attemptDir: string, state: AttemptState, args: Args): AttemptSummary {
  const step1Calls = state.toolCalls.filter((call) => call.name === task.step1.toolName);
  const step2Calls = state.toolCalls.filter((call) => call.name === task.step2.toolName);
  const tool1Call = step1Calls[0];
  const tool2Call = step2Calls[0];
  const tool1Response = state.toolResponses.find((response) => response.name === task.step1.toolName);
  const tool2Response = state.toolResponses.find((response) => response.name === task.step2.toolName);
  const tool1CallTime = tool1Call?.event_ms ?? null;
  const tool1ResultTime = tool1Response?.event_ms ?? null;
  const tool2CallTime = tool2Call?.event_ms ?? null;
  const tool2ResultTime = tool2Response?.event_ms ?? null;
  const tool2CalledAfterTool1Result =
    typeof tool1ResultTime === "number" && typeof tool2CallTime === "number" && tool2CallTime > tool1ResultTime;
  const expectedStep2Arg = task.step1.mockedResult[task.step2.dependency.fromStep1ResultField];
  const acceptedStep2ArgNames = task.step2.dependency.acceptedArgNames ?? [task.step2.dependency.argName];
  const tool2ArgFromTool1Result = acceptedStep2ArgNames.some((argName) => tool2Call?.args?.[argName] === expectedStep2Arg);
  const parallelToolCallDetected =
    typeof tool1ResultTime === "number" && typeof tool2CallTime === "number" && tool2CallTime < tool1ResultTime;
  const sequentialTwoToolSuccess =
    Boolean(tool1Call) &&
    Boolean(tool2Call) &&
    tool2CalledAfterTool1Result &&
    tool2ArgFromTool1Result &&
    !parallelToolCallDetected;
  const finalText = state.textAfterTools.join(" ");
  const beforeText = state.textBeforeTools.join(" ");
  const beforeRequiredToolResultText = [
    ...state.textEvents
      .filter((event) => typeof tool2ResultTime !== "number" || event.event_ms < tool2ResultTime)
      .map((event) => event.text),
  ].join(" ");
  const stage1 = stageAudioMetrics(state.audioIntervals, tool1CallTime, tool1ResultTime);
  const stage2 = stageAudioMetrics(state.audioIntervals, tool2CallTime, tool2ResultTime);
  const stage1Ticks = state.pendingTicks
    .filter((tick) => tick.tool_name === task.step1.toolName)
    .map((tick) => tick.event_ms)
    .filter((time): time is number => typeof time === "number");
  const stage2Ticks = state.pendingTicks
    .filter((tick) => tick.tool_name === task.step2.toolName)
    .map((tick) => tick.event_ms)
    .filter((time): time is number => typeof time === "number");
  const firstFinalAudio = firstAudioAtOrAfter(state.audioIntervals, tool2ResultTime);
  const finalCoreAnswerCorrect = finalMentionsTargetAnswer(task, finalText) && usesStep1Result(task, finalText) && usesStep2Result(task, finalText);
  const firstAudioTime = state.audioIntervals.length ? Math.min(...state.audioIntervals.map((interval) => interval.start_ms)) : null;
  const lastAudioTime = state.audioIntervals.length ? Math.max(...state.audioIntervals.map((interval) => interval.end_ms)) : null;
  const serverAudioAfterTool2 =
    typeof tool2ResultTime === "number" && state.audioIntervals.some((interval) => interval.event_ms > tool2ResultTime);
  const serverTextAfterTool2 =
    typeof tool2ResultTime === "number" && state.textEvents.some((event) => event.event_ms > tool2ResultTime);
  const timeFromTool2ResultToSessionClose =
    typeof tool2ResultTime === "number" && typeof state.sessionCloseTimeMs === "number"
      ? state.sessionCloseTimeMs - tool2ResultTime
      : null;
  const tool1ResultToTool2CallLatency =
    typeof tool1ResultTime === "number" && typeof tool2CallTime === "number" ? tool2CallTime - tool1ResultTime : null;
  const close1008 = state.closeCode === 1008 || state.closeReason?.includes("1008") === true;
  const close1011 = state.closeCode === 1011 || state.closeReason?.includes("1011") === true;
  const close1006 = state.closeCode === 1006 || state.closeReason?.includes("1006") === true;
  return {
    task_id: task.id,
    condition: args.condition,
    attempt_index: attemptIndex,
    latency_ms: args.toolLatencyMs,
    valid_run: !close1008 && !close1011 && !close1006 && state.sendErrors.length === 0,
    called_any_tool: state.toolCalls.length > 0,
    called_get_next_calendar_event: step1Calls.length > 0,
    called_get_route_eta: step2Calls.length > 0,
    tool_order: state.toolCalls.map((call) => call.name || "unknown").join(" -> "),
    tool1_call_time_ms: tool1CallTime,
    tool1_result_sent_time_ms: tool1ResultTime,
    tool2_call_time_ms: tool2CallTime,
    tool2_result_sent_time_ms: tool2ResultTime,
    tool2_called_after_tool1_result: tool2CalledAfterTool1Result,
    tool2_arg_from_tool1_result: tool2ArgFromTool1Result,
    parallel_tool_call_detected: parallelToolCallDetected,
    sequential_two_tool_success: sequentialTwoToolSuccess,
    both_tools_called: Boolean(tool1Call) && Boolean(tool2Call),
    calendar_call_time_ms: tool1CallTime,
    calendar_result_sent_time_ms: tool1ResultTime,
    route_call_time_ms: tool2CallTime,
    route_result_sent_time_ms: tool2ResultTime,
    route_called_after_calendar_result: tool2CalledAfterTool1Result,
    route_arg_from_calendar_result: tool2ArgFromTool1Result,
    stage1_calendar_waiting_window_ms: stage1.waiting_window_ms,
    stage1_audio_occupancy_ratio: stage1.audio_occupancy_ratio,
    stage1_max_silence_gap_ms: stage1.max_silence_gap_ms,
    stage1_spoken_segment_count: stage1.spoken_segment_count,
    stage1_waiting_speech_present: stage1.waiting_speech_present,
    stage2_route_waiting_window_ms: stage2.waiting_window_ms,
    stage2_audio_occupancy_ratio: stage2.audio_occupancy_ratio,
    stage2_max_silence_gap_ms: stage2.max_silence_gap_ms,
    stage2_spoken_segment_count: stage2.spoken_segment_count,
    stage2_waiting_speech_present: stage2.waiting_speech_present,
    route_result_to_final_audio_latency_ms:
      typeof firstFinalAudio === "number" && typeof tool2ResultTime === "number" ? firstFinalAudio - tool2ResultTime : null,
    final_answer_mentions_2_15: finalMentionsTargetAnswer(task, finalText),
    final_uses_calendar_time: usesStep1Result(task, finalText),
    final_uses_route_eta: usesStep2Result(task, finalText),
    final_core_answer_correct: finalCoreAnswerCorrect,
    premature_answer_before_tools: prematureAnswer(task, beforeText),
    premature_answer_before_required_tool_result: prematureAnswer(task, beforeRequiredToolResultText),
    close_1008: close1008,
    close_1011: close1011,
    close_1006: close1006,
    close_code: state.closeCode,
    close_reason: state.closeReason,
    first_audio_time_ms: firstAudioTime,
    last_audio_time_ms: lastAudioTime,
    session_close_time_ms: state.sessionCloseTimeMs,
    time_from_tool2_result_to_session_close_ms: timeFromTool2ResultToSessionClose,
    server_audio_or_text_after_tool2_result: serverAudioAfterTool2 || serverTextAfterTool2,
    close_trigger: state.closeTrigger,
    tool1_result_to_tool2_call_latency_ms: tool1ResultToTool2CallLatency,
    post_final_min_wait_ms: args.postFinalMinWaitMs,
    post_final_idle_ms: args.postFinalIdleMs,
    pending_tick_every_ms: args.condition === "periodic_tick_4s" ? TICK_EVERY_MS : null,
    pending_tick_count: state.pendingTicks.length,
    pending_tick_times_ms: state.pendingTicks.map((tick) => tick.event_ms).filter((time): time is number => typeof time === "number"),
    stage1_pending_tick_times_ms: stage1Ticks,
    stage2_pending_tick_times_ms: stage2Ticks,
    send_error_count: state.sendErrors.length,
    tool_call_cancellation_count: state.cancelledToolCallIds.length,
    raw_event_count: state.rawEventCount,
    text_before_tools: state.textBeforeTools,
    text_after_tools: state.textAfterTools,
    tool_calls: state.toolCalls,
    tool_responses: state.toolResponses,
    errors: state.errors,
    result_dir: attemptDir,
  };
}

function aggregate(attempts: AttemptSummary[]): AggregateSummary {
  return {
    attempts: attempts.length,
    any_tool_call_rate: rate(attempts.filter((attempt) => attempt.called_any_tool).length, attempts.length),
    calendar_tool_call_rate: rate(attempts.filter((attempt) => attempt.called_get_next_calendar_event).length, attempts.length),
    route_tool_call_rate: rate(attempts.filter((attempt) => attempt.called_get_route_eta).length, attempts.length),
    both_tools_called_rate: rate(attempts.filter((attempt) => attempt.called_get_next_calendar_event && attempt.called_get_route_eta).length, attempts.length),
    sequential_two_tool_success_rate: rate(attempts.filter((attempt) => attempt.sequential_two_tool_success).length, attempts.length),
    parallel_tool_call_rate: rate(attempts.filter((attempt) => attempt.parallel_tool_call_detected).length, attempts.length),
    tool2_arg_from_tool1_result_rate: rate(attempts.filter((attempt) => attempt.tool2_arg_from_tool1_result).length, attempts.length),
    final_answer_correct_rate: rate(
      attempts.filter(
        (attempt) =>
          attempt.final_core_answer_correct,
      ).length,
      attempts.length,
    ),
    premature_answer_rate: rate(attempts.filter((attempt) => attempt.premature_answer_before_tools).length, attempts.length),
    "1008_error_count": attempts.filter((attempt) => attempt.close_1008).length,
    "1011_error_count": attempts.filter((attempt) => attempt.close_1011).length,
    "1006_error_count": attempts.filter((attempt) => attempt.close_1006).length,
    send_error_count: attempts.reduce((sum, attempt) => sum + attempt.send_error_count, 0),
  };
}

function summarizeByLatency(attempts: AttemptSummary[]): LatencySummaryRow[] {
  const groups = Array.from(new Set(attempts.map((attempt) => `${attempt.condition}::${attempt.latency_ms}`))).sort((left, right) => {
    const [leftCondition, leftLatency] = left.split("::");
    const [rightCondition, rightLatency] = right.split("::");
    const latencyDiff = Number(leftLatency) - Number(rightLatency);
    if (latencyDiff !== 0) return latencyDiff;
    return leftCondition.localeCompare(rightCondition);
  });
  return groups.map((group) => {
    const [condition, latencyText] = group.split("::") as [Condition, string];
    const latency = Number(latencyText);
    const rows = attempts.filter((attempt) => attempt.condition === condition && attempt.latency_ms === latency);
    const closeDeltas = rows.map((attempt) => attempt.time_from_tool2_result_to_session_close_ms).filter((value): value is number => typeof value === "number");
    return {
      condition,
      latency_ms: latency,
      attempts: rows.length,
      sequential_two_tool_success_rate: rate(rows.filter((attempt) => attempt.sequential_two_tool_success).length, rows.length),
      parallel_tool_call_rate: rate(rows.filter((attempt) => attempt.parallel_tool_call_detected).length, rows.length),
      final_core_answer_correct_rate: rate(rows.filter((attempt) => attempt.final_core_answer_correct).length, rows.length),
      premature_answer_rate: rate(rows.filter((attempt) => attempt.premature_answer_before_required_tool_result).length, rows.length),
      stage1_waiting_speech_present_rate: rate(rows.filter((attempt) => attempt.stage1_waiting_speech_present).length, rows.length),
      stage1_audio_occupancy_ratio_mean: mean(rows.map((attempt) => attempt.stage1_audio_occupancy_ratio)),
      stage1_max_silence_gap_ms_mean: mean(rows.map((attempt) => attempt.stage1_max_silence_gap_ms)),
      stage2_waiting_speech_present_rate: rate(rows.filter((attempt) => attempt.stage2_waiting_speech_present).length, rows.length),
      stage2_audio_occupancy_ratio_mean: mean(rows.map((attempt) => attempt.stage2_audio_occupancy_ratio)),
      stage2_max_silence_gap_ms_mean: mean(rows.map((attempt) => attempt.stage2_max_silence_gap_ms)),
      pending_tick_count_mean: mean(rows.map((attempt) => attempt.pending_tick_count)),
      stage1_pending_tick_count_mean: mean(rows.map((attempt) => attempt.stage1_pending_tick_times_ms.length)),
      stage2_pending_tick_count_mean: mean(rows.map((attempt) => attempt.stage2_pending_tick_times_ms.length)),
      min_tool2_result_to_session_close_ms: closeDeltas.length ? Math.min(...closeDeltas) : null,
      "1008_error_count": rows.filter((attempt) => attempt.close_1008).length,
      "1011_error_count": rows.filter((attempt) => attempt.close_1011).length,
      "1006_error_count": rows.filter((attempt) => attempt.close_1006).length,
      send_error_count: rows.reduce((sum, attempt) => sum + attempt.send_error_count, 0),
    };
  });
}

function wavHeader(dataLength: number, sampleRate = 24_000, channels = 1, bitDepth = 16): Buffer {
  const header = Buffer.alloc(44);
  const byteRate = (sampleRate * channels * bitDepth) / 8;
  const blockAlign = (channels * bitDepth) / 8;
  header.write("RIFF", 0);
  header.writeUInt32LE(36 + dataLength, 4);
  header.write("WAVE", 8);
  header.write("fmt ", 12);
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20);
  header.writeUInt16LE(channels, 22);
  header.writeUInt32LE(sampleRate, 24);
  header.writeUInt32LE(byteRate, 28);
  header.writeUInt16LE(blockAlign, 32);
  header.writeUInt16LE(bitDepth, 34);
  header.write("data", 36);
  header.writeUInt32LE(dataLength, 40);
  return header;
}

function writeAudioFile(path: string, chunks: Buffer[]): string | null {
  if (!chunks.length) return null;
  const audio = Buffer.concat(chunks);
  writeFileSync(path, Buffer.concat([wavHeader(audio.length), audio]));
  return path;
}

function writeTimelineAudioFile(path: string, chunks: Buffer[], intervals: AudioInterval[]): string | null {
  if (!chunks.length || !intervals.length) return null;
  const pieces: Buffer[] = [];
  let timelineBytes = 0;
  for (let index = 0; index < intervals.length; index += 1) {
    const interval = intervals[index];
    const startBytes = Math.max(0, Math.round((interval.start_ms / 1000) * PCM_BYTES_PER_SECOND));
    if (startBytes > timelineBytes) {
      pieces.push(Buffer.alloc(startBytes - timelineBytes));
      timelineBytes = startBytes;
    }
    const chunk = chunks[index];
    if (!chunk) break;
    pieces.push(chunk);
    timelineBytes += chunk.length;
  }
  const audio = Buffer.concat(pieces);
  writeFileSync(path, Buffer.concat([wavHeader(audio.length), audio]));
  return path;
}

async function runOne(
  task: TwoStepTaskSpec,
  ai: GoogleGenAI,
  model: string,
  attemptIndex: number,
  attemptDir: string,
  organizedDirs: Record<string, string>,
  args: Args,
): Promise<AttemptSummary> {
  mkdirSync(attemptDir, { recursive: true });
  const prefix = `${task.resultSlug}__latency${args.toolLatencyMs}__attempt${String(attemptIndex).padStart(3, "0")}`;
  const rawLogPath = resolve(attemptDir, `${prefix}.raw_log.jsonl`);
  const timelinePath = resolve(attemptDir, `${prefix}.timeline.jsonl`);
  const audioPath = resolve(attemptDir, `${prefix}.assistant.wav`);
  const timelineAudioPath = resolve(attemptDir, `${prefix}.assistant_timeline.wav`);
  const audioChunks: Buffer[] = [];

  writeJson(resolve(attemptDir, "config.json"), {
    attempt_index: attemptIndex,
    task_id: task.id,
    model,
    prompt_name: task.promptName,
    tool_latency_ms: args.toolLatencyMs,
    pending_tick: args.condition,
    pending_tick_every_ms: args.condition === "periodic_tick_4s" ? TICK_EVERY_MS : null,
    post_final_min_wait_ms: args.postFinalMinWaitMs,
    post_final_idle_ms: args.postFinalIdleMs,
    user_prompt: task.userPrompt,
    system_instruction: task.systemInstruction,
    tools: [task.step1.toolName, task.step2.toolName],
    mocked_results: {
      [task.step1.toolName]: task.step1.mockedResult,
      [task.step2.toolName]: task.step2.mockedResult,
    },
    expected_final_answer: task.expectedFinalAnswer,
  });

  const state: AttemptState = {
    sessionClosed: false,
    closeCode: null,
    closeReason: null,
    rawEventCount: 0,
    toolCalls: [],
    toolResponses: [],
    textBeforeTools: [],
    textAfterTools: [],
    textEvents: [],
    cancelledToolCallIds: [],
    sendErrors: [],
    errors: [],
    audioIntervals: [],
    audioPlaybackCursorMs: 0,
    sessionCloseTimeMs: null,
    closeTrigger: null,
    pendingTicks: [],
  };

  let session: Session | undefined;
  let done = false;
  let initialPromptSent = false;
  let finalObservationScheduled = false;
  const timers: ReturnType<typeof setTimeout>[] = [];
  const pendingTickTimers = new Map<string, ReturnType<typeof setInterval>>();

  const eventMs = () => (state.promptSentAt ? Date.now() - state.promptSentAt : null);
  const appendTimeline = (type: string, extra: Record<string, unknown> = {}) => appendJsonl(timelinePath, { type, event_ms: eventMs(), ...extra });

  const noteSendError = (type: string, error: unknown) => {
    const summary = summarizeError(error);
    state.sendErrors.push(`${type}: ${summary}`);
    appendTimeline("send_error", { send_type: type, error: summary });
    appendJsonl(rawLogPath, { type: "send_error", event_ms: eventMs(), send_type: type, error: summary });
    return summary;
  };

  const finish = (resolveRun: () => void, trigger = "unknown") => {
    if (done) return;
    done = true;
    state.closeTrigger ??= trigger;
    for (const timer of timers) clearTimeout(timer);
    try {
      if (!state.sessionClosed) {
        state.sessionCloseTimeMs = eventMs();
        appendTimeline("session_close_requested", { trigger });
        session?.close();
      }
    } catch (error) {
      state.errors.push(summarizeError(error));
    }
    writeAudioFile(audioPath, audioChunks);
    writeTimelineAudioFile(timelineAudioPath, audioChunks, state.audioIntervals);
    resolveRun();
  };

  const sendInitialUserPrompt = () => {
    if (initialPromptSent || done || state.sessionClosed || !session) return;
    initialPromptSent = true;
    session.sendClientContent({ turns: task.userPrompt, turnComplete: true });
    state.promptSentAt = Date.now();
    appendTimeline("user_message_sent", { prompt: task.userPrompt });
    appendJsonl(rawLogPath, { type: "user_message_sent", prompt: task.userPrompt });
  };

  const scheduleFinalObservation = (resolveRun: () => void) => {
    if (finalObservationScheduled) return;
    finalObservationScheduled = true;
    appendTimeline("post_final_observation_started", {
      min_wait_ms: args.postFinalMinWaitMs,
      idle_ms: args.postFinalIdleMs,
    });
    timers.push(
      setInterval(() => {
        if (!state.allToolResponsesAt) return;
        const now = Date.now();
        const elapsedAfterFinalTool = now - state.allToolResponsesAt;
        const lastAssistantAt = state.lastAssistantEventAt ?? state.allToolResponsesAt;
        const assistantIdle = now - lastAssistantAt;
        if (elapsedAfterFinalTool < args.postFinalMinWaitMs || assistantIdle < args.postFinalIdleMs) return;
        appendTimeline("post_final_observation_elapsed", {
          elapsed_after_final_tool_ms: elapsedAfterFinalTool,
          assistant_idle_ms: assistantIdle,
        });
        finish(resolveRun, "post_final_min_wait_and_idle");
      }, 250),
    );
  };

  const maybeMarkAllToolsComplete = () => {
    const step1Done = state.toolResponses.some((response) => response.name === task.step1.toolName && !response.send_error);
    const step2Done = state.toolResponses.some((response) => response.name === task.step2.toolName && !response.send_error);
    if (step1Done && step2Done) state.allToolResponsesAt ??= Date.now();
  };

  const sendToolResponse = (call: FunctionCall): boolean => {
    const pendingTickTimer = call.id ? pendingTickTimers.get(call.id) : undefined;
    if (pendingTickTimer) {
      clearInterval(pendingTickTimer);
      pendingTickTimers.delete(call.id || "");
    }
    const response = responseForTool(task, call.name);
    const record: ToolResponseRecord = { event_ms: eventMs(), name: call.name, response };
    try {
      session?.sendToolResponse({
        functionResponses: [{ id: call.id, name: call.name, response }],
      });
      appendTimeline("tool_response_sent", { function_call_id: call.id, function_name: call.name, response });
      appendJsonl(rawLogPath, {
        type: "tool_response_sent",
        event_ms: eventMs(),
        function_call_id: call.id,
        function_name: call.name,
        response,
      });
    } catch (error) {
      record.send_error = noteSendError("tool_response", error);
    }
    state.toolResponses.push(record);
    maybeMarkAllToolsComplete();
    return !record.send_error;
  };

  const schedulePendingTicks = (call: FunctionCall): void => {
    if (args.condition !== "periodic_tick_4s" || !call.id) return;
    const timer = setInterval(() => {
      if (done || state.sessionClosed) return;
      const responseAlreadySent = state.toolResponses.some((response) => response.name === call.name);
      if (responseAlreadySent) {
        clearInterval(timer);
        pendingTickTimers.delete(call.id || "");
        return;
      }
      const sentAt = eventMs();
      try {
        session?.sendRealtimeInput({ text: PENDING_TICK_MESSAGE });
        state.pendingTicks.push({ event_ms: sentAt, tool_name: call.name, call_id: call.id, message: PENDING_TICK_MESSAGE });
        appendTimeline("pending_tick_sent", { function_call_id: call.id, function_name: call.name, message: PENDING_TICK_MESSAGE });
        appendJsonl(rawLogPath, {
          type: "pending_tick_sent",
          event_ms: sentAt,
          function_call_id: call.id,
          function_name: call.name,
          message: PENDING_TICK_MESSAGE,
        });
      } catch (error) {
        noteSendError("pending_tick", error);
      }
    }, TICK_EVERY_MS);
    timers.push(timer);
    pendingTickTimers.set(call.id, timer);
  };

  await new Promise<void>(async (resolveRun) => {
    const maxAttemptMs = Math.max(MAX_ATTEMPT_MS, args.toolLatencyMs * 2 + args.postFinalMinWaitMs + args.postFinalIdleMs + 30_000);
    timers.push(
      setTimeout(() => {
        state.errors.push(`max attempt timeout ${maxAttemptMs}ms`);
        appendTimeline("max_attempt_timeout", { max_attempt_ms: maxAttemptMs });
        finish(resolveRun, "max_attempt_timeout");
      }, maxAttemptMs),
    );

    try {
      session = (await ai.live.connect({
        model,
        config: {
          responseModalities: [Modality.AUDIO],
          outputAudioTranscription: {},
          systemInstruction: task.systemInstruction,
          tools: makeToolDeclarations(task) as any,
        },
        callbacks: {
          onopen: () => {
            state.sessionOpenedAt = Date.now();
            appendTimeline("session_opened", { model });
            appendJsonl(rawLogPath, { type: "session_opened", model });
          },
          onmessage: (message: LiveMessage) => {
            state.rawEventCount += 1;
            appendJsonl(rawLogPath, { type: "server_event", event_ms: eventMs(), event_types: eventTypes(message), message: sanitizeMessage(message) });
            appendTimeline("server_event", { event_types: eventTypes(message) });

            if (message.setupComplete) {
              state.setupCompleteAt ??= Date.now();
              appendTimeline("setup_complete");
              try {
                sendInitialUserPrompt();
              } catch (error) {
                noteSendError("initial_user_prompt", error);
                finish(resolveRun, "initial_user_prompt_send_error");
              }
              return;
            }

            const parts = message.serverContent?.modelTurn?.parts ?? [];
            const textPieces = [
              ...parts.map((part) => part.text).filter((text): text is string => Boolean(text)),
              message.serverContent?.outputTranscription?.text,
            ].filter((text): text is string => Boolean(text));
            for (const text of textPieces) {
              const currentEventMs = eventMs();
              if (currentEventMs !== null) {
                state.textEvents.push({
                  event_ms: currentEventMs,
                  text,
                  phase: state.allToolResponsesAt ? "after_all_tools" : "before_all_tools",
                });
              }
              state.lastAssistantEventAt = Date.now();
              if (state.allToolResponsesAt) state.textAfterTools.push(text);
              else state.textBeforeTools.push(text);
              if (!args.quietTerminalText) console.log(`[attempt ${attemptIndex} ${eventMs() ?? "?"}ms] ${text.replace(/\s+/g, " ").trim()}`);
              appendTimeline("text_output", { text, phase: state.allToolResponsesAt ? "after_all_tools" : "before_all_tools" });
            }

            for (const part of parts) {
              if (!part.inlineData?.data) continue;
              const chunk = Buffer.from(part.inlineData.data, "base64");
              audioChunks.push(chunk);
              const currentEventMs = eventMs();
              if (currentEventMs !== null) {
                const durationMs = (chunk.length / PCM_BYTES_PER_SECOND) * 1000;
                const startMs = Math.max(currentEventMs, state.audioPlaybackCursorMs);
                const endMs = startMs + durationMs;
                state.audioPlaybackCursorMs = endMs;
                state.audioIntervals.push({ event_ms: currentEventMs, start_ms: startMs, end_ms: endMs, bytes: chunk.length });
                state.lastAssistantEventAt = Date.now();
                appendTimeline("audio_output", { bytes: chunk.length, start_ms: startMs, end_ms: endMs });
              }
            }

            if (message.toolCall?.functionCalls?.length) {
              appendTimeline("tool_call_received", { function_calls: message.toolCall.functionCalls });
              appendJsonl(rawLogPath, { type: "tool_call_received", event_ms: eventMs(), function_calls: message.toolCall.functionCalls });
              for (const call of message.toolCall.functionCalls) {
                state.toolCalls.push({ event_ms: eventMs(), name: call.name, args: call.args, id: call.id });
                schedulePendingTicks(call);
                timers.push(
                  setTimeout(() => {
                    if (done || state.sessionClosed) return;
                    sendToolResponse(call);
                    if (state.allToolResponsesAt) scheduleFinalObservation(resolveRun);
                  }, args.toolLatencyMs),
                );
              }
            }

            if (message.toolCallCancellation?.ids?.length) {
              for (const id of message.toolCallCancellation.ids) {
                if (!state.cancelledToolCallIds.includes(id)) state.cancelledToolCallIds.push(id);
              }
              appendTimeline("tool_call_cancellation_received", { ids: message.toolCallCancellation.ids });
              appendJsonl(rawLogPath, { type: "tool_call_cancellation_received", event_ms: eventMs(), ids: message.toolCallCancellation.ids });
            }

            if (message.serverContent?.turnComplete) appendTimeline("turn_complete");
          },
          onerror: (error) => {
            const summary = summarizeError(error);
            state.errors.push(summary);
            appendTimeline("socket_error", { error: summary });
            appendJsonl(rawLogPath, { type: "socket_error", event_ms: eventMs(), error: summary });
          },
          onclose: (event: { code?: number; reason?: string }) => {
            state.sessionClosed = true;
            state.closeCode = event.code ?? null;
            state.closeReason = event.reason || null;
            state.sessionCloseTimeMs ??= eventMs();
            state.closeTrigger ??= "server_onclose";
            appendTimeline("session_closed", { code: event.code, reason: event.reason });
            appendJsonl(rawLogPath, { type: "session_closed", event_ms: eventMs(), code: event.code, reason: event.reason });
            finish(resolveRun, state.closeTrigger);
          },
        },
      })) as Session;
    } catch (error) {
      state.errors.push(summarizeError(error));
      appendTimeline("connect_error", { error: summarizeError(error) });
      finish(resolveRun, "connect_error");
    }
  });

  const summary = makeSummary(task, attemptIndex, attemptDir, state, args);
  writeJson(resolve(attemptDir, "attempt_summary.json"), summary);
  writeJson(resolve(attemptDir, "audio_intervals.json"), state.audioIntervals);
  copyFileSync(rawLogPath, resolve(organizedDirs.rawLogs, `${prefix}.raw_log.jsonl`));
  copyFileSync(timelinePath, resolve(organizedDirs.timelines, `${prefix}.timeline.jsonl`));
  try {
    copyFileSync(audioPath, resolve(organizedDirs.audio, `${prefix}.assistant.wav`));
    copyFileSync(timelineAudioPath, resolve(organizedDirs.audio, `${prefix}.assistant_timeline.wav`));
  } catch {
    // Failed attempts can be silent.
  }
  return summary;
}

function writeReadme(path: string, resultDir: string, organizedDir: string, aggregateSummary: AggregateSummary, args: Args, task: TwoStepTaskSpec): void {
  writeFileSync(
    path,
    [
      `# ${task.id} native two-tool benchmark`,
      "",
      `Task: ${task.id}`,
      "",
      "- mode: native tool-call",
      `- conditions: ${args.conditions.join(", ")}`,
      `- attempts: ${args.attempts} per latency, ${aggregateSummary.attempts} total`,
      "- concurrency: 1",
      `- tool response latencies: ${args.latenciesMs.join(", ")} ms per tool call`,
      `- periodic tick interval: ${TICK_EVERY_MS} ms for periodic_tick_4s only`,
      `- post-final observation: at least ${args.postFinalMinWaitMs} ms after final tool result and ${args.postFinalIdleMs} ms assistant audio/text idle`,
      "- shutdown rule: do not close immediately after two-tool flow; close only after final tool result plus post-final min-wait and assistant idle window.",
      "",
      "Short latency conditions such as 3s and 5s provide less opportunity for waiting speech, so waiting-activity metrics should be interpreted with this caveat. The metrics themselves are still computed consistently across all latencies.",
      "",
      `Core success: ${task.step1.toolName} result arrives before ${task.step2.toolName} is called, and ${task.step2.toolName} uses ${task.step2.dependency.argName} from ${task.step1.toolName}.${task.step2.dependency.fromStep1ResultField}.`,
      "",
      `Result folder: ${resultDir}`,
      `Organized folder: ${organizedDir}`,
      "",
      "## Aggregate",
      "",
      "```json",
      JSON.stringify(aggregateSummary, null, 2),
      "```",
      "",
    ].join("\n"),
    "utf8",
  );
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const baseTask = getTwoStepTask(args.taskId);
  const apiKey = requireEnv("GEMINI_API_KEY");
  const model = requireEnv("GEMINI_LIVE_MODEL");
  const ai = new GoogleGenAI({ apiKey });

  const sweepMode = args.latenciesMs.length > 1 || args.conditions.length > 1;
  const resultDir = resolve(
    RESULT_DIR,
    `${timestampForPath()}_${baseTask.resultSlug.replace(/_no_tick/g, "")}_${sweepMode ? "tick_vs_no_tick_latency_sweep" : "feasibility_probe"}`,
  );
  const attemptsDir = resolve(resultDir, "attempts");
  const organizedDir = resolve(resultDir, "organized");
  const organizedDirs = {
    rawLogs: resolve(organizedDir, "raw_logs"),
    audio: resolve(organizedDir, "audio"),
    timelines: resolve(organizedDir, "timelines"),
    perAttempt: resolve(organizedDir, "per_attempt"),
  };
  mkdirSync(attemptsDir, { recursive: true });
  for (const dir of [organizedDir, ...Object.values(organizedDirs)]) mkdirSync(dir, { recursive: true });

  console.log(`Running ${baseTask.id} native two-tool benchmark for model: ${model}`);
  console.log(`Conditions: ${args.conditions.join(", ")}`);
  console.log(`Attempts per latency: ${args.attempts}`);
  console.log(`Tool latencies: ${args.latenciesMs.join(", ")} ms`);
  console.log(`Periodic tick interval: ${TICK_EVERY_MS} ms for periodic_tick_4s`);
  console.log(`Post-final min wait: ${args.postFinalMinWaitMs} ms`);
  console.log(`Post-final idle window: ${args.postFinalIdleMs} ms`);
  console.log(`Result directory: ${relative(PROJECT_DIR, resultDir)}`);

  const attempts: AttemptSummary[] = [];
  for (const condition of args.conditions) {
    const task = taskForCondition(baseTask, condition);
    for (const latencyMs of args.latenciesMs) {
      const latencyArgs = { ...args, condition, toolLatencyMs: latencyMs };
      for (let index = 1; index <= args.attempts; index += 1) {
        const attemptDir = resolve(attemptsDir, condition, `latency_${latencyMs}ms`, `attempt_${String(index).padStart(4, "0")}`);
        console.log(`[${condition} latency ${latencyMs}ms attempt ${index}/${args.attempts}] start`);
        const summary = await runOne(task, ai, model, index, attemptDir, organizedDirs, latencyArgs);
        attempts.push(summary);
        writeJson(resolve(organizedDirs.perAttempt, `${task.resultSlug}__latency${latencyMs}__attempt${String(index).padStart(3, "0")}.summary.json`), summary);
        console.log(
          `[${condition} latency ${latencyMs}ms attempt ${index}/${args.attempts}] sequential=${summary.sequential_two_tool_success} parallel=${summary.parallel_tool_call_detected} ticks=${summary.pending_tick_count} close=${summary.close_code ?? "none"}`,
        );
      }
    }
  }

  const aggregateSummary = aggregate(attempts);
  const metricsByLatency = summarizeByLatency(attempts);
  const summaryJson = {
    task: baseTask.id,
    condition: sweepMode ? "native_two_tool_sequential_tick_vs_no_tick_latency_sweep" : `native_two_tool_sequential_latency${args.toolLatencyMs}_${args.conditions[0]}`,
    conditions: args.conditions,
    prompt_names_by_condition: Object.fromEntries(args.conditions.map((condition) => [condition, taskForCondition(baseTask, condition).promptName])),
    user_prompt: baseTask.userPrompt,
    model,
    step1_tool_name: baseTask.step1.toolName,
    step2_tool_name: baseTask.step2.toolName,
    step2_dependency: baseTask.step2.dependency,
    mocked_results: {
      [baseTask.step1.toolName]: baseTask.step1.mockedResult,
      [baseTask.step2.toolName]: baseTask.step2.mockedResult,
    },
    tool_latencies_ms: args.latenciesMs,
    pending_tick: args.conditions.includes("periodic_tick_4s") ? `periodic ${TICK_EVERY_MS}ms in periodic_tick_4s condition` : "none",
    pending_tick_message: PENDING_TICK_MESSAGE,
    post_final_min_wait_ms: args.postFinalMinWaitMs,
    post_final_idle_ms: args.postFinalIdleMs,
    short_latency_caveat:
      "Short latency conditions such as 3s and 5s provide less opportunity for waiting speech, so waiting-activity metrics should be interpreted with this caveat. The metrics themselves are still computed consistently across all latencies.",
    aggregate: aggregateSummary,
    metrics_by_latency: metricsByLatency,
    attempts,
  };
  writeJson(resolve(resultDir, "summary.json"), summaryJson);
  writeCsv(resolve(resultDir, "summary.csv"), attempts as unknown as Array<Record<string, unknown>>);
  writeJson(resolve(organizedDir, "summary.json"), summaryJson);
  writeCsv(resolve(organizedDir, "summary.csv"), attempts as unknown as Array<Record<string, unknown>>);
  writeCsv(resolve(organizedDir, "metrics_by_attempt.csv"), attempts as unknown as Array<Record<string, unknown>>);
  writeCsv(resolve(organizedDir, "metrics_by_latency.csv"), metricsByLatency as unknown as Array<Record<string, unknown>>);
  writeJson(resolve(organizedDir, "metrics_by_latency.json"), metricsByLatency);
  writeReadme(resolve(resultDir, "README.md"), resultDir, organizedDir, aggregateSummary, args, baseTask);
  writeReadme(resolve(organizedDir, "README.md"), resultDir, organizedDir, aggregateSummary, args, baseTask);

  console.log("Summary:");
  console.log(JSON.stringify(aggregateSummary, null, 2));
  console.log(`Organized directory: ${relative(PROJECT_DIR, organizedDir)}`);
}

main().catch((error) => {
  console.error("calendar-route-two-tool-feasibility-probe failed");
  console.error(summarizeError(error));
  process.exitCode = 1;
});
