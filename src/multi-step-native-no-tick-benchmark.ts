import { GoogleGenAI, Modality } from "@google/genai";
import { config as loadEnv } from "dotenv";
import { appendFileSync, copyFileSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  getMultiStepTaskVariant,
  makeMultiStepToolDeclarations,
  type MultiStepTaskVariant,
  type MultiStepTaskStepSpec,
} from "./tasks/multi-step-task-catalog.js";

const PROJECT_DIR = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const DEFAULT_ENV_FILE = resolve(PROJECT_DIR, "..", "angus_bench", ".env");
const RESULT_DIR = process.env.GEMINI_LIVE_CHECK_RESULT_DIR
  ? resolve(process.env.GEMINI_LIVE_CHECK_RESULT_DIR)
  : resolve(PROJECT_DIR, "result");

const DEFAULT_TASK_BASES = ["calendar_route_leave_time", "airline_rebook_option"];
const DEFAULT_STEP_COUNTS = [2, 3, 4, 5];
const DEFAULT_ATTEMPTS = 3;
const DEFAULT_TOOL_LATENCY_MS = 8000;
const DEFAULT_POST_FINAL_MIN_WAIT_MS = 15_000;
const DEFAULT_POST_FINAL_IDLE_MS = 3000;
const PCM_BYTES_PER_SECOND = 48_000;

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
  taskBases: string[];
  stepCounts: number[];
  attempts: number;
  toolLatencyMs: number;
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

type AttemptState = {
  promptSentAt?: number;
  allToolResponsesAt?: number;
  sessionClosed: boolean;
  closeCode: number | null;
  closeReason: string | null;
  rawEventCount: number;
  toolCalls: ToolCallRecord[];
  toolResponses: ToolResponseRecord[];
  textBeforeAllTools: string[];
  textAfterAllTools: string[];
  textEvents: TextEvent[];
  cancelledToolCallIds: string[];
  sendErrors: string[];
  errors: string[];
  audioIntervals: AudioInterval[];
  audioPlaybackCursorMs: number;
  lastAssistantEventAt?: number;
  sessionCloseTimeMs: number | null;
  closeTrigger: string | null;
};

type StageMetric = {
  step_index: number;
  label: string;
  tool_name: string;
  call_time_ms: number | null;
  result_sent_time_ms: number | null;
  called_after_previous_result: boolean | null;
  dependency_arg_correct: boolean | null;
  waiting_window_ms: number | null;
  audio_occupancy_ratio: number | null;
  max_silence_gap_ms: number | null;
  spoken_segment_count: number;
  waiting_speech_present: boolean;
};

type AttemptSummary = {
  task_id: string;
  base_id: string;
  step_count: number;
  attempt_index: number;
  latency_ms: number;
  valid_run: boolean;
  expected_tool_count: number;
  called_expected_tool_count: number;
  all_expected_tools_called: boolean;
  tool_order: string;
  sequential_step_success: boolean;
  parallel_tool_call_detected: boolean;
  dependency_args_correct_count: number;
  dependency_args_correct_rate: number | null;
  final_answer_mentions_target: boolean;
  final_uses_step1_result: boolean;
  final_uses_final_step_result: boolean;
  final_core_answer_correct: boolean;
  premature_answer_before_required_tool_result: boolean;
  route_result_to_final_audio_latency_ms: number | null;
  close_1008: boolean;
  close_1011: boolean;
  close_1006: boolean;
  close_code: number | null;
  close_reason: string | null;
  send_error_count: number;
  tool_call_cancellation_count: number;
  raw_event_count: number;
  stage_metrics: StageMetric[];
  tool_calls: ToolCallRecord[];
  tool_responses: ToolResponseRecord[];
  errors: string[];
  result_dir: string;
};

type AggregateRow = {
  task_id: string;
  base_id: string;
  step_count: number;
  attempts: number;
  valid_run_rate: number;
  sequential_step_success_rate: number;
  all_expected_tools_called_rate: number;
  parallel_tool_call_rate: number;
  final_core_answer_correct_rate: number;
  premature_answer_rate: number;
  dependency_args_correct_rate_mean: number | null;
  stage_waiting_speech_present_rate_mean: number | null;
  stage_audio_occupancy_ratio_mean: number | null;
  stage_max_silence_gap_ms_mean: number | null;
  "1008_error_count": number;
  "1011_error_count": number;
  "1006_error_count": number;
  send_error_count: number;
};

loadEnv({ path: process.env.GEMINI_ENV_FILE || DEFAULT_ENV_FILE });

function usage(): string {
  return [
    "Usage: multi-step-native-no-tick-benchmark [options]",
    "",
    "Options:",
    `  --tasks <csv>          Base task ids. Default: ${DEFAULT_TASK_BASES.join(",")}.`,
    `  --step-counts <csv>    Step counts. Default: ${DEFAULT_STEP_COUNTS.join(",")}.`,
    "  --attempts <n>         Attempts per task x step count. Default: 3.",
    "  --tool-latency-ms <n>  Mocked latency for each tool response. Default: 8000.",
    "  --post-final-min-wait-ms <n>  Minimum post-final observation. Default: 15000.",
    "  --post-final-idle-ms <n>      Assistant idle window. Default: 3000.",
    "  --quiet-terminal-text  Suppress transcript text in terminal.",
    "  --help                 Show this help.",
  ].join("\n");
}

function parseCsvNumbers(value: string): number[] {
  return value.split(",").map((item) => Number(item.trim())).filter((item) => Number.isFinite(item));
}

function parseCsvStrings(value: string): string[] {
  return value.split(",").map((item) => item.trim()).filter(Boolean);
}

function parseArgs(argv: string[]): Args {
  const args: Args = {
    taskBases: process.env.MULTISTEP_TASKS ? parseCsvStrings(process.env.MULTISTEP_TASKS) : DEFAULT_TASK_BASES,
    stepCounts: process.env.MULTISTEP_STEP_COUNTS ? parseCsvNumbers(process.env.MULTISTEP_STEP_COUNTS) : DEFAULT_STEP_COUNTS,
    attempts: Number(process.env.MULTISTEP_ATTEMPTS ?? DEFAULT_ATTEMPTS),
    toolLatencyMs: Number(process.env.MULTISTEP_TOOL_LATENCY_MS ?? DEFAULT_TOOL_LATENCY_MS),
    postFinalMinWaitMs: Number(process.env.MULTISTEP_POST_FINAL_MIN_WAIT_MS ?? DEFAULT_POST_FINAL_MIN_WAIT_MS),
    postFinalIdleMs: Number(process.env.MULTISTEP_POST_FINAL_IDLE_MS ?? DEFAULT_POST_FINAL_IDLE_MS),
    quietTerminalText: false,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const next = argv[index + 1];
    if (arg === "--help" || arg === "-h") {
      console.log(usage());
      process.exit(0);
    } else if (arg === "--quiet-terminal-text") {
      args.quietTerminalText = true;
    } else if (arg === "--tasks" && next) {
      args.taskBases = parseCsvStrings(next);
      index += 1;
    } else if (arg === "--step-counts" && next) {
      args.stepCounts = parseCsvNumbers(next);
      index += 1;
    } else if (arg === "--attempts" && next) {
      args.attempts = Number(next);
      index += 1;
    } else if (arg === "--tool-latency-ms" && next) {
      args.toolLatencyMs = Number(next);
      index += 1;
    } else if (arg === "--post-final-min-wait-ms" && next) {
      args.postFinalMinWaitMs = Number(next);
      index += 1;
    } else if (arg === "--post-final-idle-ms" && next) {
      args.postFinalIdleMs = Number(next);
      index += 1;
    } else {
      const tasks = arg.match(/^--tasks=(.+)$/);
      const stepCounts = arg.match(/^--step-counts=([0-9,]+)$/);
      const attempts = arg.match(/^--attempts=(\d+)$/);
      const latency = arg.match(/^--tool-latency-ms=(\d+)$/);
      if (tasks) args.taskBases = parseCsvStrings(tasks[1]);
      else if (stepCounts) args.stepCounts = parseCsvNumbers(stepCounts[1]);
      else if (attempts) args.attempts = Number(attempts[1]);
      else if (latency) args.toolLatencyMs = Number(latency[1]);
      else throw new Error(`Unknown or incomplete argument: ${arg}\n${usage()}`);
    }
  }
  if (!args.taskBases.length) throw new Error("At least one task is required.");
  if (!args.stepCounts.every((count) => [2, 3, 4, 5].includes(count))) throw new Error(`Invalid step counts: ${args.stepCounts.join(",")}`);
  if (!Number.isInteger(args.attempts) || args.attempts < 1) throw new Error(`Invalid attempts: ${args.attempts}`);
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

function summarizeError(error: unknown): string {
  if (error instanceof Error) return `${error.name}: ${error.message}`;
  if (typeof error !== "object" || error === null) return String(error);
  const record = error as Record<string, unknown>;
  return ["message", "type", "code", "reason", "name", "error"]
    .map((key) => (record[key] ? `${key}: ${String(record[key])}` : undefined))
    .filter(Boolean)
    .join(", ") || Object.prototype.toString.call(error);
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

function rate(count: number, total: number): number {
  return total ? Math.round((count / total) * 1000) / 1000 : 0;
}

function mean(values: Array<number | null | undefined>): number | null {
  const nums = values.filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  if (!nums.length) return null;
  return Math.round((nums.reduce((sum, value) => sum + value, 0) / nums.length) * 1000) / 1000;
}

function round3(value: number): number {
  return Math.round(value * 1000) / 1000;
}

function mergeStageSegments(intervals: AudioInterval[], windowStart: number, windowEnd: number): Array<{ start_ms: number; end_ms: number }> {
  const clipped = intervals
    .map((interval) => ({ start_ms: Math.max(windowStart, interval.start_ms), end_ms: Math.min(windowEnd, interval.end_ms) }))
    .filter((segment) => segment.end_ms > segment.start_ms)
    .sort((left, right) => left.start_ms - right.start_ms);
  const merged: Array<{ start_ms: number; end_ms: number }> = [];
  for (const segment of clipped) {
    if (!merged.length || segment.start_ms - merged[merged.length - 1].end_ms > 250) merged.push({ ...segment });
    else merged[merged.length - 1].end_ms = Math.max(merged[merged.length - 1].end_ms, segment.end_ms);
  }
  return merged;
}

function stageAudioMetrics(intervals: AudioInterval[], windowStart: number | null, windowEnd: number | null): Pick<StageMetric, "waiting_window_ms" | "audio_occupancy_ratio" | "max_silence_gap_ms" | "spoken_segment_count" | "waiting_speech_present"> {
  if (typeof windowStart !== "number" || typeof windowEnd !== "number" || windowEnd <= windowStart) {
    return { waiting_window_ms: null, audio_occupancy_ratio: null, max_silence_gap_ms: null, spoken_segment_count: 0, waiting_speech_present: false };
  }
  const segments = mergeStageSegments(intervals, windowStart, windowEnd);
  const windowMs = windowEnd - windowStart;
  const audioMs = segments.reduce((sum, segment) => sum + (segment.end_ms - segment.start_ms), 0);
  let maxSilence = windowMs;
  if (segments.length) {
    maxSilence = Math.max(0, segments[0].start_ms - windowStart, windowEnd - segments[segments.length - 1].end_ms);
    for (let index = 1; index < segments.length; index += 1) maxSilence = Math.max(maxSilence, segments[index].start_ms - segments[index - 1].end_ms);
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
  return intervals.map((interval) => interval.start_ms).filter((time) => time >= timeMs).sort((a, b) => a - b)[0] ?? null;
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

function writeAudioFile(path: string, chunks: Buffer[]): void {
  if (!chunks.length) return;
  const audio = Buffer.concat(chunks);
  writeFileSync(path, Buffer.concat([wavHeader(audio.length), audio]));
}

function writeTimelineAudioFile(path: string, chunks: Buffer[], intervals: AudioInterval[]): void {
  if (!chunks.length || !intervals.length) return;
  const pieces: Buffer[] = [];
  let timelineBytes = 0;
  for (let index = 0; index < intervals.length; index += 1) {
    const startBytes = Math.max(0, Math.round((intervals[index].start_ms / 1000) * PCM_BYTES_PER_SECOND));
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
}

function responseForTool(task: MultiStepTaskVariant, name?: string): Record<string, unknown> {
  return task.steps.find((step) => step.toolName === name)?.mockedResult ?? { error: `unknown tool ${name || "unknown"}` };
}

function stepCall(state: AttemptState, step: MultiStepTaskStepSpec): ToolCallRecord | undefined {
  return state.toolCalls.find((call) => call.name === step.toolName);
}

function stepResponse(state: AttemptState, step: MultiStepTaskStepSpec): ToolResponseRecord | undefined {
  return state.toolResponses.find((response) => response.name === step.toolName && !response.send_error);
}

function dependencyCorrect(task: MultiStepTaskVariant, state: AttemptState, stepIndex: number): boolean | null {
  const step = task.steps[stepIndex];
  if (!step.dependency) return null;
  const call = stepCall(state, step);
  if (!call) return false;
  const sourceStep = task.steps[step.dependency.fromStepIndex - 1];
  const expected = sourceStep.mockedResult[step.dependency.fromResultField];
  const argNames = step.dependency.acceptedArgNames ?? [step.dependency.argName];
  return argNames.some((argName) => call.args?.[argName] === expected);
}

function makeSummary(task: MultiStepTaskVariant, attemptIndex: number, attemptDir: string, state: AttemptState, args: Args): AttemptSummary {
  const expectedTools = task.steps.map((step) => step.toolName);
  const expectedCalls = task.steps.map((step) => stepCall(state, step));
  const expectedResponses = task.steps.map((step) => stepResponse(state, step));
  const allExpectedToolsCalled = expectedCalls.every(Boolean);
  const stageMetrics: StageMetric[] = task.steps.map((step, index) => {
    const call = expectedCalls[index];
    const response = expectedResponses[index];
    const previousResponse = index > 0 ? expectedResponses[index - 1] : undefined;
    const callTime = call?.event_ms ?? null;
    const resultTime = response?.event_ms ?? null;
    return {
      step_index: index + 1,
      label: step.label,
      tool_name: step.toolName,
      call_time_ms: callTime,
      result_sent_time_ms: resultTime,
      called_after_previous_result:
        index === 0 ? null : typeof callTime === "number" && typeof previousResponse?.event_ms === "number" ? callTime > previousResponse.event_ms : false,
      dependency_arg_correct: dependencyCorrect(task, state, index),
      ...stageAudioMetrics(state.audioIntervals, callTime, resultTime),
    };
  });

  const parallelToolCallDetected = stageMetrics.some((stage) => stage.called_after_previous_result === false);
  const dependencyStages = stageMetrics.filter((stage) => stage.dependency_arg_correct !== null);
  const dependencyArgsCorrectCount = dependencyStages.filter((stage) => stage.dependency_arg_correct).length;
  const sequentialStepSuccess =
    allExpectedToolsCalled &&
    !parallelToolCallDetected &&
    dependencyStages.every((stage) => stage.dependency_arg_correct === true);
  const finalText = state.textAfterAllTools.join(" ");
  const finalToolResultTime = expectedResponses[expectedResponses.length - 1]?.event_ms ?? null;
  const firstFinalAudio = firstAudioAtOrAfter(state.audioIntervals, finalToolResultTime);
  const beforeRequiredToolResultText = state.textEvents
    .filter((event) => typeof finalToolResultTime !== "number" || event.event_ms < finalToolResultTime)
    .map((event) => event.text)
    .join(" ");
  const finalAnswerMentionsTarget = task.finalAnswerChecks.mentionsTargetAnswer.test(finalText);
  const finalUsesStep1Result = task.finalAnswerChecks.usesStep1Result.test(finalText);
  const finalUsesFinalStepResult = task.finalAnswerChecks.usesFinalStepResult.test(finalText);
  const close1008 = state.closeCode === 1008 || state.closeReason?.includes("1008") === true;
  const close1011 = state.closeCode === 1011 || state.closeReason?.includes("1011") === true;
  const close1006 = state.closeCode === 1006 || state.closeReason?.includes("1006") === true;

  return {
    task_id: task.id,
    base_id: task.baseId,
    step_count: task.stepCount,
    attempt_index: attemptIndex,
    latency_ms: args.toolLatencyMs,
    valid_run: !close1008 && !close1011 && !close1006 && state.sendErrors.length === 0,
    expected_tool_count: expectedTools.length,
    called_expected_tool_count: expectedCalls.filter(Boolean).length,
    all_expected_tools_called: allExpectedToolsCalled,
    tool_order: state.toolCalls.map((call) => call.name || "unknown").join(" -> "),
    sequential_step_success: sequentialStepSuccess,
    parallel_tool_call_detected: parallelToolCallDetected,
    dependency_args_correct_count: dependencyArgsCorrectCount,
    dependency_args_correct_rate: dependencyStages.length ? round3(dependencyArgsCorrectCount / dependencyStages.length) : null,
    final_answer_mentions_target: finalAnswerMentionsTarget,
    final_uses_step1_result: finalUsesStep1Result,
    final_uses_final_step_result: finalUsesFinalStepResult,
    final_core_answer_correct: finalAnswerMentionsTarget && finalUsesStep1Result && finalUsesFinalStepResult,
    premature_answer_before_required_tool_result: task.finalAnswerChecks.prematureAnswer.test(beforeRequiredToolResultText),
    route_result_to_final_audio_latency_ms:
      typeof firstFinalAudio === "number" && typeof finalToolResultTime === "number" ? firstFinalAudio - finalToolResultTime : null,
    close_1008: close1008,
    close_1011: close1011,
    close_1006: close1006,
    close_code: state.closeCode,
    close_reason: state.closeReason,
    send_error_count: state.sendErrors.length,
    tool_call_cancellation_count: state.cancelledToolCallIds.length,
    raw_event_count: state.rawEventCount,
    stage_metrics: stageMetrics,
    tool_calls: state.toolCalls,
    tool_responses: state.toolResponses,
    errors: state.errors,
    result_dir: attemptDir,
  };
}

function summarizeGroups(attempts: AttemptSummary[]): AggregateRow[] {
  const keys = Array.from(new Set(attempts.map((attempt) => `${attempt.task_id}`))).sort();
  return keys.map((taskId) => {
    const rows = attempts.filter((attempt) => attempt.task_id === taskId);
    const stageRows = rows.flatMap((row) => row.stage_metrics);
    return {
      task_id: taskId,
      base_id: rows[0]?.base_id ?? "",
      step_count: rows[0]?.step_count ?? 0,
      attempts: rows.length,
      valid_run_rate: rate(rows.filter((row) => row.valid_run).length, rows.length),
      sequential_step_success_rate: rate(rows.filter((row) => row.sequential_step_success).length, rows.length),
      all_expected_tools_called_rate: rate(rows.filter((row) => row.all_expected_tools_called).length, rows.length),
      parallel_tool_call_rate: rate(rows.filter((row) => row.parallel_tool_call_detected).length, rows.length),
      final_core_answer_correct_rate: rate(rows.filter((row) => row.final_core_answer_correct).length, rows.length),
      premature_answer_rate: rate(rows.filter((row) => row.premature_answer_before_required_tool_result).length, rows.length),
      dependency_args_correct_rate_mean: mean(rows.map((row) => row.dependency_args_correct_rate)),
      stage_waiting_speech_present_rate_mean: mean(stageRows.map((row) => (row.waiting_speech_present ? 1 : 0))),
      stage_audio_occupancy_ratio_mean: mean(stageRows.map((row) => row.audio_occupancy_ratio)),
      stage_max_silence_gap_ms_mean: mean(stageRows.map((row) => row.max_silence_gap_ms)),
      "1008_error_count": rows.filter((row) => row.close_1008).length,
      "1011_error_count": rows.filter((row) => row.close_1011).length,
      "1006_error_count": rows.filter((row) => row.close_1006).length,
      send_error_count: rows.reduce((sum, row) => sum + row.send_error_count, 0),
    };
  });
}

async function runOne(
  task: MultiStepTaskVariant,
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
  const state: AttemptState = {
    sessionClosed: false,
    closeCode: null,
    closeReason: null,
    rawEventCount: 0,
    toolCalls: [],
    toolResponses: [],
    textBeforeAllTools: [],
    textAfterAllTools: [],
    textEvents: [],
    cancelledToolCallIds: [],
    sendErrors: [],
    errors: [],
    audioIntervals: [],
    audioPlaybackCursorMs: 0,
    sessionCloseTimeMs: null,
    closeTrigger: null,
  };
  writeJson(resolve(attemptDir, "config.json"), {
    task_id: task.id,
    model,
    prompt_name: task.promptName,
    tool_latency_ms: args.toolLatencyMs,
    pending_tick: "none",
    post_final_min_wait_ms: args.postFinalMinWaitMs,
    post_final_idle_ms: args.postFinalIdleMs,
    user_prompt: task.userPrompt,
    system_instruction: task.systemInstruction,
    tools: task.steps.map((step) => step.toolName),
    mocked_results: Object.fromEntries(task.steps.map((step) => [step.toolName, step.mockedResult])),
    expected_final_answer: task.expectedFinalAnswer,
  });

  let session: Session | undefined;
  let done = false;
  let initialPromptSent = false;
  let finalObservationScheduled = false;
  const timers: Array<ReturnType<typeof setTimeout> | ReturnType<typeof setInterval>> = [];
  const eventMs = () => (state.promptSentAt ? Date.now() - state.promptSentAt : null);
  const appendTimeline = (type: string, extra: Record<string, unknown> = {}) => appendJsonl(timelinePath, { type, event_ms: eventMs(), ...extra });

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

  const noteSendError = (type: string, error: unknown) => {
    const summary = summarizeError(error);
    state.sendErrors.push(`${type}: ${summary}`);
    appendTimeline("send_error", { send_type: type, error: summary });
    appendJsonl(rawLogPath, { type: "send_error", event_ms: eventMs(), send_type: type, error: summary });
    return summary;
  };

  const allExpectedResponsesDone = () => task.steps.every((step) => state.toolResponses.some((response) => response.name === step.toolName && !response.send_error));

  const scheduleFinalObservation = (resolveRun: () => void) => {
    if (finalObservationScheduled) return;
    finalObservationScheduled = true;
    appendTimeline("post_final_observation_started", { min_wait_ms: args.postFinalMinWaitMs, idle_ms: args.postFinalIdleMs });
    timers.push(
      setInterval(() => {
        if (!state.allToolResponsesAt) return;
        const now = Date.now();
        const elapsedAfterFinalTool = now - state.allToolResponsesAt;
        const lastAssistantAt = state.lastAssistantEventAt ?? state.allToolResponsesAt;
        const assistantIdle = now - lastAssistantAt;
        if (elapsedAfterFinalTool < args.postFinalMinWaitMs || assistantIdle < args.postFinalIdleMs) return;
        appendTimeline("post_final_observation_elapsed", { elapsed_after_final_tool_ms: elapsedAfterFinalTool, assistant_idle_ms: assistantIdle });
        finish(resolveRun, "post_final_min_wait_and_idle");
      }, 250),
    );
  };

  const sendInitialUserPrompt = () => {
    if (initialPromptSent || done || state.sessionClosed || !session) return;
    initialPromptSent = true;
    session.sendClientContent({ turns: task.userPrompt, turnComplete: true });
    state.promptSentAt = Date.now();
    appendTimeline("user_message_sent", { prompt: task.userPrompt });
    appendJsonl(rawLogPath, { type: "user_message_sent", prompt: task.userPrompt });
  };

  const sendToolResponse = (call: FunctionCall): void => {
    const response = responseForTool(task, call.name);
    const record: ToolResponseRecord = { event_ms: eventMs(), name: call.name, response };
    try {
      session?.sendToolResponse({ functionResponses: [{ id: call.id, name: call.name, response }] });
      appendTimeline("tool_response_sent", { function_call_id: call.id, function_name: call.name, response });
      appendJsonl(rawLogPath, { type: "tool_response_sent", event_ms: eventMs(), function_call_id: call.id, function_name: call.name, response });
    } catch (error) {
      record.send_error = noteSendError("tool_response", error);
    }
    state.toolResponses.push(record);
    if (allExpectedResponsesDone()) state.allToolResponsesAt ??= Date.now();
  };

  await new Promise<void>(async (resolveRun) => {
    const maxAttemptMs = Math.max(60_000, args.toolLatencyMs * task.stepCount + args.postFinalMinWaitMs + args.postFinalIdleMs + 35_000);
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
          tools: makeMultiStepToolDeclarations(task) as any,
        },
        callbacks: {
          onopen: () => {
            appendTimeline("session_opened", { model });
            appendJsonl(rawLogPath, { type: "session_opened", model });
          },
          onmessage: (message: LiveMessage) => {
            state.rawEventCount += 1;
            appendJsonl(rawLogPath, { type: "server_event", event_ms: eventMs(), event_types: eventTypes(message), message: sanitizeMessage(message) });
            appendTimeline("server_event", { event_types: eventTypes(message) });
            if (message.setupComplete) {
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
              if (currentEventMs !== null) state.textEvents.push({ event_ms: currentEventMs, text, phase: state.allToolResponsesAt ? "after_all_tools" : "before_all_tools" });
              state.lastAssistantEventAt = Date.now();
              if (state.allToolResponsesAt) state.textAfterAllTools.push(text);
              else state.textBeforeAllTools.push(text);
              if (!args.quietTerminalText) console.log(`[${task.id} attempt ${attemptIndex} ${eventMs() ?? "?"}ms] ${text.replace(/\s+/g, " ").trim()}`);
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
              for (const id of message.toolCallCancellation.ids) if (!state.cancelledToolCallIds.includes(id)) state.cancelledToolCallIds.push(id);
              appendTimeline("tool_call_cancellation_received", { ids: message.toolCallCancellation.ids });
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

function writeReadme(path: string, resultDir: string, organizedDir: string, args: Args, aggregateRows: AggregateRow[]): void {
  writeFileSync(
    path,
    [
      "# Multi-step native no-tick pilot",
      "",
      "- mode: native tool-call",
      "- condition: no_tick",
      `- task bases: ${args.taskBases.join(", ")}`,
      `- step counts: ${args.stepCounts.join(", ")}`,
      `- attempts per task x step count: ${args.attempts}`,
      `- tool latency: ${args.toolLatencyMs} ms per tool result`,
      `- post-final observation: ${args.postFinalMinWaitMs} ms min and ${args.postFinalIdleMs} ms idle`,
      "",
      `Result folder: ${resultDir}`,
      `Organized folder: ${organizedDir}`,
      "",
      "## Aggregate Rows",
      "",
      "```json",
      JSON.stringify(aggregateRows, null, 2),
      "```",
      "",
    ].join("\n"),
    "utf8",
  );
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const apiKey = requireEnv("GEMINI_API_KEY");
  const model = requireEnv("GEMINI_LIVE_MODEL");
  const ai = new GoogleGenAI({ apiKey });
  const resultDir = resolve(RESULT_DIR, `${timestampForPath()}_multi_step_no_tick_pilot`);
  const attemptsDir = resolve(resultDir, "attempts");
  const organizedDir = resolve(resultDir, "organized");
  const organizedDirs = {
    rawLogs: resolve(organizedDir, "raw_logs"),
    audio: resolve(organizedDir, "audio"),
    timelines: resolve(organizedDir, "timelines"),
    perAttempt: resolve(organizedDir, "per_attempt"),
  };
  mkdirSync(attemptsDir, { recursive: true });
  for (const dir of [organizedDir, ...Object.values(organizedDirs), resolve(organizedDir, "visualizations")]) mkdirSync(dir, { recursive: true });

  console.log(`Running multi-step native no-tick pilot for model: ${model}`);
  console.log(`Task bases: ${args.taskBases.join(", ")}`);
  console.log(`Step counts: ${args.stepCounts.join(", ")}`);
  console.log(`Attempts: ${args.attempts}`);
  console.log(`Tool latency: ${args.toolLatencyMs} ms`);
  console.log(`Result directory: ${relative(PROJECT_DIR, resultDir)}`);

  const attempts: AttemptSummary[] = [];
  for (const base of args.taskBases) {
    for (const stepCount of args.stepCounts) {
      const task = getMultiStepTaskVariant(`${base}_${stepCount}step`);
      for (let index = 1; index <= args.attempts; index += 1) {
        const attemptDir = resolve(attemptsDir, task.id, `attempt_${String(index).padStart(4, "0")}`);
        console.log(`[${task.id} attempt ${index}/${args.attempts}] start`);
        const summary = await runOne(task, ai, model, index, attemptDir, organizedDirs, args);
        attempts.push(summary);
        writeJson(resolve(organizedDirs.perAttempt, `${task.resultSlug}__attempt${String(index).padStart(3, "0")}.summary.json`), summary);
        console.log(
          `[${task.id} attempt ${index}/${args.attempts}] sequential=${summary.sequential_step_success} all_tools=${summary.all_expected_tools_called} parallel=${summary.parallel_tool_call_detected} close=${summary.close_code ?? "none"} order=${summary.tool_order || "none"}`,
        );
      }
    }
  }

  const aggregateRows = summarizeGroups(attempts);
  const summaryJson = {
    condition: "native_multi_step_no_tick",
    model,
    tool_latency_ms: args.toolLatencyMs,
    post_final_min_wait_ms: args.postFinalMinWaitMs,
    post_final_idle_ms: args.postFinalIdleMs,
    aggregate_rows: aggregateRows,
    attempts,
  };
  writeJson(resolve(resultDir, "summary.json"), summaryJson);
  writeCsv(resolve(resultDir, "summary.csv"), attempts as unknown as Array<Record<string, unknown>>);
  writeJson(resolve(organizedDir, "summary.json"), summaryJson);
  writeCsv(resolve(organizedDir, "summary.csv"), attempts as unknown as Array<Record<string, unknown>>);
  writeCsv(resolve(organizedDir, "metrics_by_attempt.csv"), attempts as unknown as Array<Record<string, unknown>>);
  writeCsv(resolve(organizedDir, "metrics_by_task_step.csv"), aggregateRows as unknown as Array<Record<string, unknown>>);
  writeJson(resolve(organizedDir, "metrics_by_task_step.json"), aggregateRows);
  writeReadme(resolve(resultDir, "README.md"), resultDir, organizedDir, args, aggregateRows);
  writeReadme(resolve(organizedDir, "README.md"), resultDir, organizedDir, args, aggregateRows);

  console.log("Aggregate:");
  console.log(JSON.stringify(aggregateRows, null, 2));
  console.log(`Organized directory: ${relative(PROJECT_DIR, organizedDir)}`);
}

main().catch((error) => {
  console.error("multi-step-native-no-tick-benchmark failed");
  console.error(summarizeError(error));
  process.exitCode = 1;
});
