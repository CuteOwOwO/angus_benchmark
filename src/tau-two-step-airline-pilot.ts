import { Behavior, GoogleGenAI, Modality } from "@google/genai";
import { config as loadEnv } from "dotenv";
import { appendFileSync, copyFileSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  PROMPT_NAME,
  SYSTEM_INSTRUCTION as BASE_SYSTEM_INSTRUCTION,
  USER_PROMPT,
} from "./prompts/tow-2step-airline-no-tick.js";

const PROJECT_DIR = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const DEFAULT_ENV_FILE = resolve(PROJECT_DIR, "..", "angus_bench", ".env");
const RESULT_DIR = process.env.GEMINI_LIVE_CHECK_RESULT_DIR
  ? resolve(process.env.GEMINI_LIVE_CHECK_RESULT_DIR)
  : resolve(PROJECT_DIR, "result");

const RESERVATION_TOOL = "get_reservation_details";
const USER_TOOL = "get_user_details";
const DEFAULT_ATTEMPTS = 5;
const TOOL_LATENCY_MS = 8000;
const TICK_EVERY_MS = 4000;
const MAX_ATTEMPT_MS = 45_000;
const POST_FINAL_WAIT_MS = 7000;
const PCM_BYTES_PER_SECOND = 48_000;

const SYSTEM_INSTRUCTION = `${BASE_SYSTEM_INSTRUCTION}

For this airline baggage task, use this task rule after the tools return:
- economy + silver membership = 2 checked suitcases per passenger.
- The final answer should be a number with a brief explanation.`;

const RESERVATION_RESULT = {
  reservation_id: "JMO1MG",
  cabin: "economy",
  passenger_count: 2,
  route: "DEN -> MIA",
  trip_type: "one_way",
  total_baggages: 1,
};

const USER_RESULT = {
  user_id: "anya_garcia_5901",
  name: "Anya Garcia",
  membership: "silver",
};

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
  attempts: number;
  quietTerminalText: boolean;
};

type AudioInterval = {
  event_ms: number;
  start_ms: number;
  end_ms: number;
  bytes: number;
  phase: string;
};

type TextEvent = {
  event_ms: number | null;
  text: string;
};

type ToolCallRecord = {
  event_ms: number | null;
  name?: string;
  args?: Record<string, unknown>;
};

type ToolResponseRecord = {
  event_ms: number | null;
  name?: string;
  response: Record<string, unknown>;
  send_error?: string;
};

type StageRecord = {
  index: number;
  tool_name?: string;
  call_id?: string;
  args?: Record<string, unknown>;
  call_ms: number;
  response_ms?: number;
  pending_tick_times_ms: number[];
  text_events: TextEvent[];
};

type AttemptSummary = {
  attempt_index: number;
  valid_run: boolean;
  called_any_tool: boolean;
  called_get_reservation_details: boolean;
  get_reservation_details_args_correct: boolean;
  called_get_user_details: boolean;
  get_user_details_args_correct: boolean;
  both_tools_called: boolean;
  both_tool_args_correct: boolean;
  tool_order: string;
  completed_two_tool_flow: boolean;
  stage_1_tool_name: string | null;
  stage_1_latency_ms: number | null;
  stage_1_pending_tick_times_ms: number[];
  stage_1_waiting_speech_present: boolean;
  stage_1_audio_output_count_before_tool_result: number;
  stage_1_max_silence_gap_before_tool_result_ms: number | null;
  stage_1_waiting_task_relevance_score: number;
  stage_1_pre_result_hallucination: boolean;
  stage_2_tool_name: string | null;
  stage_2_latency_ms: number | null;
  stage_2_pending_tick_times_ms: number[];
  stage_2_waiting_speech_present: boolean;
  stage_2_audio_output_count_before_tool_result: number;
  stage_2_max_silence_gap_before_tool_result_ms: number | null;
  stage_2_waiting_task_relevance_score: number;
  stage_2_progress_consistency_score: number;
  stage_2_pre_result_hallucination: boolean;
  final_answer_mentions_4: boolean;
  final_uses_reservation_result: boolean;
  final_uses_user_membership_result: boolean;
  final_core_answer_correct: boolean;
  post_final_answer_latency_ms: number | null;
  close_1008: boolean;
  close_1011: boolean;
  close_1006: boolean;
  close_code: number | null;
  close_reason: string | null;
  send_error_count: number;
  tool_call_cancellation_count: number;
  raw_event_count: number;
  final_transcript: string;
  errors: string[];
  result_dir: string;
};

type AggregateSummary = {
  attempt_count: number;
  retry_count: number;
  valid_run_rate: number;
  completed_two_tool_flow_rate: number;
  final_core_answer_correct_rate: number;
  "1008_error_count": number;
  "1011_error_count": number;
  "1006_error_count": number;
  send_error_count: number;
  tool_call_cancellation_count: number;
};

type AttemptState = {
  sessionOpenedAt?: number;
  setupCompleteAt?: number;
  promptSentAt?: number;
  allRelevantToolResponsesAt?: number;
  firstFinalTextAt?: number;
  sessionClosed: boolean;
  closeCode: number | null;
  closeReason: string | null;
  rawEventCount: number;
  toolCalls: ToolCallRecord[];
  toolResponses: ToolResponseRecord[];
  stages: StageRecord[];
  audioIntervals: AudioInterval[];
  textBeforeAllTools: TextEvent[];
  textAfterAllTools: TextEvent[];
  cancelledToolCallIds: string[];
  sendErrors: string[];
  errors: string[];
  audioPlaybackCursorMs: number;
};

loadEnv({ path: process.env.GEMINI_ENV_FILE || DEFAULT_ENV_FILE });

function usage(): string {
  return [
    "Usage: tau-two-step-airline-pilot [options]",
    "",
    "Options:",
    "  --attempts <n>          Number of attempts. Default: 5.",
    "  --quiet-terminal-text   Suppress transcript text in terminal.",
    "  --help                  Show this help.",
  ].join("\n");
}

function parseArgs(argv: string[]): Args {
  const args: Args = {
    attempts: Number(process.env.TWO_STEP_AIRLINE_ATTEMPTS ?? DEFAULT_ATTEMPTS),
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
    if (arg === "--attempts") {
      const value = argv[index + 1];
      if (!value) throw new Error("--attempts requires a value");
      args.attempts = Number(value);
      index += 1;
      continue;
    }
    const inline = arg.match(/^--attempts=(\d+)$/);
    if (inline) {
      args.attempts = Number(inline[1]);
      continue;
    }
    throw new Error(`Unknown argument: ${arg}\n${usage()}`);
  }
  if (!Number.isInteger(args.attempts) || args.attempts < 1) throw new Error(`Invalid --attempts: ${args.attempts}`);
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

function makeTools(): unknown[] {
  return [
    {
      functionDeclarations: [
        {
          name: RESERVATION_TOOL,
          description: "Get the cabin, passenger count, route, trip type, and baggage information for an airline reservation.",
          behavior: Behavior.NON_BLOCKING,
          parametersJsonSchema: {
            type: "object",
            properties: {
              reservation_id: { type: "string", description: "Reservation id, such as JMO1MG." },
            },
            required: ["reservation_id"],
          },
        },
        {
          name: USER_TOOL,
          description: "Get the user's profile details, including airline membership tier.",
          behavior: Behavior.NON_BLOCKING,
          parametersJsonSchema: {
            type: "object",
            properties: {
              user_id: { type: "string", description: "User id, such as anya_garcia_5901." },
            },
            required: ["user_id"],
          },
        },
      ],
    },
  ];
}

function responseForTool(name?: string): Record<string, unknown> {
  if (name === RESERVATION_TOOL) return RESERVATION_RESULT;
  if (name === USER_TOOL) return USER_RESULT;
  return { error: `unknown tool ${name || "unknown"}` };
}

function argsCorrect(call: ToolCallRecord): boolean {
  if (call.name === RESERVATION_TOOL) return call.args?.reservation_id === "JMO1MG";
  if (call.name === USER_TOOL) return call.args?.user_id === "anya_garcia_5901";
  return false;
}

function compactText(text: string): string {
  return text.replace(/\s+/g, " ").trim();
}

function finalMentions4(text: string): boolean {
  return /(^|[^0-9])4([^0-9]|$)|four/i.test(text);
}

function usesReservationResult(text: string): boolean {
  return /economy|2 passenger|two passenger|passenger count|reservation|JMO1MG|DEN|MIA/i.test(text);
}

function usesMembershipResult(text: string): boolean {
  return /silver|member|membership/i.test(text);
}

function preResultHallucination(text: string): boolean {
  return finalMentions4(text) || /has (?:returned|completed)|result is|I found|I see .*silver|I see .*economy/i.test(text);
}

function waitingTaskRelevanceScore(text: string, toolName: string | null): number {
  const lower = text.toLowerCase();
  const hits = [
    /reservation|jmo1mg|cabin|passenger/.test(lower),
    /user|member|membership|silver|anya/.test(lower),
    /baggage|bag|suitcase|checked/.test(lower),
    Boolean(toolName && lower.includes(toolName.toLowerCase())),
  ].filter(Boolean).length;
  if (hits >= 2) return 1;
  if (hits === 1) return 0.5;
  return 0;
}

function progressConsistencyScore(stage: StageRecord | undefined, previousStage: StageRecord | undefined): number {
  if (!stage) return 0;
  const text = stage.text_events.map((event) => event.text).join(" ").toLowerCase();
  if (!text.trim()) return 0;
  const mentionsCurrent = stage.tool_name === USER_TOOL
    ? /user|member|membership|profile|anya/.test(text)
    : /reservation|jmo1mg|cabin|passenger/.test(text);
  const mentionsPrevious = previousStage?.tool_name === RESERVATION_TOOL
    ? /reservation|economy|passenger|cabin/.test(text)
    : previousStage?.tool_name === USER_TOOL
      ? /silver|member|membership|user/.test(text)
      : true;
  if (mentionsCurrent && mentionsPrevious) return 1;
  if (mentionsCurrent || mentionsPrevious) return 0.5;
  return 0;
}

function audioCountBeforeResult(stage: StageRecord | undefined, audioIntervals: AudioInterval[]): number {
  if (!stage?.response_ms) return 0;
  const responseMs = stage.response_ms;
  return audioIntervals.filter((audio) => audio.event_ms >= stage.call_ms && audio.event_ms < responseMs).length;
}

function maxSilenceGap(startMs: number, endMs: number, intervals: AudioInterval[]): number | null {
  if (endMs <= startMs) return null;
  const relevant = intervals
    .filter((interval) => interval.end_ms > startMs && interval.start_ms < endMs)
    .map((interval) => ({
      start_ms: Math.max(interval.start_ms, startMs),
      end_ms: Math.min(interval.end_ms, endMs),
    }))
    .sort((a, b) => a.start_ms - b.start_ms);
  let cursor = startMs;
  let maxGap = 0;
  for (const interval of relevant) {
    if (interval.start_ms > cursor) maxGap = Math.max(maxGap, interval.start_ms - cursor);
    cursor = Math.max(cursor, interval.end_ms);
  }
  if (endMs > cursor) maxGap = Math.max(maxGap, endMs - cursor);
  return Math.round(maxGap);
}

function maybeMarkAllRelevantToolsComplete(state: AttemptState): void {
  const respondedReservation = state.toolResponses.some((response) => response.name === RESERVATION_TOOL && !response.send_error);
  const respondedUser = state.toolResponses.some((response) => response.name === USER_TOOL && !response.send_error);
  if (respondedReservation && respondedUser) state.allRelevantToolResponsesAt ??= Date.now();
}

function makeSummary(attemptIndex: number, attemptDir: string, state: AttemptState): AttemptSummary {
  const calls = state.toolCalls;
  const reservationCalls = calls.filter((call) => call.name === RESERVATION_TOOL);
  const userCalls = calls.filter((call) => call.name === USER_TOOL);
  const calledReservation = reservationCalls.length > 0;
  const calledUser = userCalls.length > 0;
  const reservationArgsOk = reservationCalls.some(argsCorrect);
  const userArgsOk = userCalls.some(argsCorrect);
  const bothToolsCalled = calledReservation && calledUser;
  const bothToolArgsCorrect = reservationArgsOk && userArgsOk;
  const finalText = state.textAfterAllTools.map((event) => event.text).join(" ");
  const close1008 = state.closeCode === 1008 || state.closeReason?.includes("1008") === true;
  const close1011 = state.closeCode === 1011 || state.closeReason?.includes("1011") === true;
  const close1006 = state.closeCode === 1006 || state.closeReason?.includes("1006") === true;
  const stage1 = state.stages[0];
  const stage2 = state.stages[1];
  const stage1Text = stage1?.text_events.map((event) => event.text).join(" ") ?? "";
  const stage2Text = stage2?.text_events.map((event) => event.text).join(" ") ?? "";
  const finalCoreAnswerCorrect = finalMentions4(finalText) && usesReservationResult(finalText) && usesMembershipResult(finalText);
  return {
    attempt_index: attemptIndex,
    valid_run: !close1008 && !close1011 && !close1006 && state.sendErrors.length === 0,
    called_any_tool: calls.length > 0,
    called_get_reservation_details: calledReservation,
    get_reservation_details_args_correct: reservationArgsOk,
    called_get_user_details: calledUser,
    get_user_details_args_correct: userArgsOk,
    both_tools_called: bothToolsCalled,
    both_tool_args_correct: bothToolArgsCorrect,
    tool_order: calls.map((call) => call.name || "unknown").join(" -> "),
    completed_two_tool_flow: bothToolsCalled,
    stage_1_tool_name: stage1?.tool_name ?? null,
    stage_1_latency_ms: stage1?.response_ms ? stage1.response_ms - stage1.call_ms : null,
    stage_1_pending_tick_times_ms: stage1?.pending_tick_times_ms ?? [],
    stage_1_waiting_speech_present: Boolean(stage1Text.trim()),
    stage_1_audio_output_count_before_tool_result: audioCountBeforeResult(stage1, state.audioIntervals),
    stage_1_max_silence_gap_before_tool_result_ms: stage1?.response_ms ? maxSilenceGap(stage1.call_ms, stage1.response_ms, state.audioIntervals) : null,
    stage_1_waiting_task_relevance_score: waitingTaskRelevanceScore(stage1Text, stage1?.tool_name ?? null),
    stage_1_pre_result_hallucination: preResultHallucination(stage1Text),
    stage_2_tool_name: stage2?.tool_name ?? null,
    stage_2_latency_ms: stage2?.response_ms ? stage2.response_ms - stage2.call_ms : null,
    stage_2_pending_tick_times_ms: stage2?.pending_tick_times_ms ?? [],
    stage_2_waiting_speech_present: Boolean(stage2Text.trim()),
    stage_2_audio_output_count_before_tool_result: audioCountBeforeResult(stage2, state.audioIntervals),
    stage_2_max_silence_gap_before_tool_result_ms: stage2?.response_ms ? maxSilenceGap(stage2.call_ms, stage2.response_ms, state.audioIntervals) : null,
    stage_2_waiting_task_relevance_score: waitingTaskRelevanceScore(stage2Text, stage2?.tool_name ?? null),
    stage_2_progress_consistency_score: progressConsistencyScore(stage2, stage1),
    stage_2_pre_result_hallucination: preResultHallucination(stage2Text),
    final_answer_mentions_4: finalMentions4(finalText),
    final_uses_reservation_result: usesReservationResult(finalText),
    final_uses_user_membership_result: usesMembershipResult(finalText),
    final_core_answer_correct: finalCoreAnswerCorrect,
    post_final_answer_latency_ms:
      state.allRelevantToolResponsesAt && state.firstFinalTextAt && state.firstFinalTextAt >= state.allRelevantToolResponsesAt
        ? state.firstFinalTextAt - state.allRelevantToolResponsesAt
        : null,
    close_1008: close1008,
    close_1011: close1011,
    close_1006: close1006,
    close_code: state.closeCode,
    close_reason: state.closeReason,
    send_error_count: state.sendErrors.length,
    tool_call_cancellation_count: state.cancelledToolCallIds.length,
    raw_event_count: state.rawEventCount,
    final_transcript: compactText(finalText),
    errors: state.errors,
    result_dir: attemptDir,
  };
}

function aggregate(attempts: AttemptSummary[]): AggregateSummary {
  return {
    attempt_count: attempts.length,
    retry_count: 0,
    valid_run_rate: rate(attempts.filter((attempt) => attempt.valid_run).length, attempts.length),
    completed_two_tool_flow_rate: rate(attempts.filter((attempt) => attempt.completed_two_tool_flow).length, attempts.length),
    final_core_answer_correct_rate: rate(attempts.filter((attempt) => attempt.final_core_answer_correct).length, attempts.length),
    "1008_error_count": attempts.filter((attempt) => attempt.close_1008).length,
    "1011_error_count": attempts.filter((attempt) => attempt.close_1011).length,
    "1006_error_count": attempts.filter((attempt) => attempt.close_1006).length,
    send_error_count: attempts.reduce((sum, attempt) => sum + attempt.send_error_count, 0),
    tool_call_cancellation_count: attempts.reduce((sum, attempt) => sum + attempt.tool_call_cancellation_count, 0),
  };
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
  let cursorBytes = 0;
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
    cursorBytes += chunk.length;
    timelineBytes += chunk.length;
  }
  if (cursorBytes === 0) return null;
  const audio = Buffer.concat(pieces);
  writeFileSync(path, Buffer.concat([wavHeader(audio.length), audio]));
  return path;
}

async function runOne(
  ai: GoogleGenAI,
  model: string,
  attemptIndex: number,
  attemptDir: string,
  organizedDirs: Record<string, string>,
  args: Args,
): Promise<AttemptSummary> {
  mkdirSync(attemptDir, { recursive: true });
  const rawLogPath = resolve(attemptDir, `airline_suitcase_2step__tick4s__latency8000__attempt${String(attemptIndex).padStart(3, "0")}.raw_log.jsonl`);
  const timelinePath = resolve(attemptDir, `airline_suitcase_2step__tick4s__latency8000__attempt${String(attemptIndex).padStart(3, "0")}.timeline.jsonl`);
  const audioPath = resolve(attemptDir, `airline_suitcase_2step__tick4s__latency8000__attempt${String(attemptIndex).padStart(3, "0")}.assistant.wav`);
  const timelineAudioPath = resolve(attemptDir, `airline_suitcase_2step__tick4s__latency8000__attempt${String(attemptIndex).padStart(3, "0")}.assistant_timeline.wav`);
  const audioChunks: Buffer[] = [];

  writeJson(resolve(attemptDir, "config.json"), {
    attempt_index: attemptIndex,
    model,
    prompt_name: PROMPT_NAME,
    tool_latency_ms: TOOL_LATENCY_MS,
    tick_every_ms: TICK_EVERY_MS,
    user_prompt: USER_PROMPT,
    system_instruction: SYSTEM_INSTRUCTION,
    tools: [RESERVATION_TOOL, USER_TOOL],
    expected_final_answer: "4 checked suitcases total",
  });

  const state: AttemptState = {
    sessionClosed: false,
    closeCode: null,
    closeReason: null,
    rawEventCount: 0,
    toolCalls: [],
    toolResponses: [],
    stages: [],
    audioIntervals: [],
    textBeforeAllTools: [],
    textAfterAllTools: [],
    cancelledToolCallIds: [],
    sendErrors: [],
    errors: [],
    audioPlaybackCursorMs: 0,
  };

  let session: Session | undefined;
  let done = false;
  let initialPromptSent = false;
  let finalObservationScheduled = false;
  const timers: ReturnType<typeof setTimeout>[] = [];

  const eventMs = () => (state.promptSentAt ? Date.now() - state.promptSentAt : null);
  const appendTimeline = (type: string, extra: Record<string, unknown> = {}) => appendJsonl(timelinePath, { type, event_ms: eventMs(), ...extra });

  const noteSendError = (type: string, error: unknown) => {
    const summary = summarizeError(error);
    state.sendErrors.push(`${type}: ${summary}`);
    appendTimeline("send_error", { send_type: type, error: summary });
    appendJsonl(rawLogPath, { type: "send_error", event_ms: eventMs(), send_type: type, error: summary });
    return summary;
  };

  const finish = (resolveRun: () => void) => {
    if (done) return;
    done = true;
    for (const timer of timers) clearTimeout(timer);
    try {
      if (!state.sessionClosed) session?.close();
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
    session.sendClientContent({ turns: USER_PROMPT, turnComplete: true });
    state.promptSentAt = Date.now();
    appendTimeline("user_message_sent", { prompt: USER_PROMPT });
    appendJsonl(rawLogPath, { type: "user_message_sent", prompt: USER_PROMPT });
  };

  const scheduleFinalObservation = (resolveRun: () => void) => {
    if (finalObservationScheduled) return;
    finalObservationScheduled = true;
    appendTimeline("post_final_observation_started", { wait_ms: POST_FINAL_WAIT_MS });
    timers.push(
      setTimeout(() => {
        appendTimeline("post_final_observation_elapsed", { wait_ms: POST_FINAL_WAIT_MS });
        finish(resolveRun);
      }, POST_FINAL_WAIT_MS),
    );
  };

  const sendToolResponse = (stage: StageRecord, call: FunctionCall): boolean => {
    const response = responseForTool(call.name);
    const record: ToolResponseRecord = { event_ms: eventMs(), name: call.name, response };
    try {
      session?.sendToolResponse({
        functionResponses: [{ id: call.id, name: call.name, response }],
      });
      stage.response_ms = eventMs() ?? undefined;
      appendTimeline("tool_response_sent", { stage_index: stage.index, function_call_id: call.id, function_name: call.name, response });
      appendJsonl(rawLogPath, {
        type: "tool_response_sent",
        event_ms: eventMs(),
        stage_index: stage.index,
        function_call_id: call.id,
        function_name: call.name,
        response,
      });
    } catch (error) {
      record.send_error = noteSendError("tool_response", error);
    }
    state.toolResponses.push(record);
    maybeMarkAllRelevantToolsComplete(state);
    return !record.send_error;
  };

  const schedulePendingTick = (stage: StageRecord, call: FunctionCall) => {
    timers.push(
      setTimeout(() => {
        if (done || state.sessionClosed || stage.response_ms !== undefined) return;
        const sentAt = eventMs();
        if (sentAt === null) return;
        const message = `Still waiting for ${call.name || "the current tool"} to finish. I do not have that result yet.`;
        try {
          session?.sendRealtimeInput({ text: message });
          stage.pending_tick_times_ms.push(sentAt);
          appendTimeline("pending_tick_sent", { stage_index: stage.index, function_name: call.name, message });
          appendJsonl(rawLogPath, { type: "pending_tick_sent", event_ms: sentAt, stage_index: stage.index, function_name: call.name, message });
        } catch (error) {
          noteSendError("pending_tick", error);
        }
      }, TICK_EVERY_MS),
    );
  };

  await new Promise<void>(async (resolveRun) => {
    timers.push(
      setTimeout(() => {
        state.errors.push(`max attempt timeout ${MAX_ATTEMPT_MS}ms`);
        appendTimeline("max_attempt_timeout", { max_attempt_ms: MAX_ATTEMPT_MS });
        finish(resolveRun);
      }, MAX_ATTEMPT_MS),
    );

    try {
      session = (await ai.live.connect({
        model,
        config: {
          responseModalities: [Modality.AUDIO],
          outputAudioTranscription: {},
          systemInstruction: SYSTEM_INSTRUCTION,
          tools: makeTools() as any,
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
                finish(resolveRun);
              }
              return;
            }

            const parts = message.serverContent?.modelTurn?.parts ?? [];
            const textPieces = [
              ...parts.map((part) => part.text).filter((text): text is string => Boolean(text)),
              message.serverContent?.outputTranscription?.text,
            ].filter((text): text is string => Boolean(text));

            for (const text of textPieces) {
              const record: TextEvent = { event_ms: eventMs(), text };
              const pendingStages = state.stages.filter((stage) => stage.response_ms === undefined);
              for (const stage of pendingStages) stage.text_events.push(record);
              if (state.allRelevantToolResponsesAt) {
                state.textAfterAllTools.push(record);
                state.firstFinalTextAt ??= Date.now();
              } else {
                state.textBeforeAllTools.push(record);
              }
              if (!args.quietTerminalText) console.log(`[attempt ${attemptIndex} ${eventMs() ?? "?"}ms] ${compactText(text)}`);
              appendTimeline("text_output", { text });
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
                state.audioIntervals.push({
                  event_ms: currentEventMs,
                  start_ms: startMs,
                  end_ms: endMs,
                  bytes: chunk.length,
                  phase: state.allRelevantToolResponsesAt ? "after_all_tools" : "before_all_tools",
                });
                appendTimeline("audio_output", { bytes: chunk.length, start_ms: startMs, end_ms: endMs });
              }
            }

            if (message.toolCall?.functionCalls?.length) {
              appendTimeline("tool_call_received", { function_calls: message.toolCall.functionCalls });
              appendJsonl(rawLogPath, { type: "tool_call_received", event_ms: eventMs(), function_calls: message.toolCall.functionCalls });
              for (const call of message.toolCall.functionCalls) {
                const callMs = eventMs() ?? 0;
                const stage: StageRecord = {
                  index: state.stages.length + 1,
                  tool_name: call.name,
                  call_id: call.id,
                  args: call.args,
                  call_ms: callMs,
                  pending_tick_times_ms: [],
                  text_events: [],
                };
                state.stages.push(stage);
                state.toolCalls.push({ event_ms: callMs, name: call.name, args: call.args });
                schedulePendingTick(stage, call);
                timers.push(
                  setTimeout(() => {
                    if (done || state.sessionClosed || stage.response_ms !== undefined) return;
                    sendToolResponse(stage, call);
                    if (state.allRelevantToolResponsesAt) scheduleFinalObservation(resolveRun);
                  }, TOOL_LATENCY_MS),
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
            appendTimeline("session_closed", { code: event.code, reason: event.reason });
            appendJsonl(rawLogPath, { type: "session_closed", event_ms: eventMs(), code: event.code, reason: event.reason });
            finish(resolveRun);
          },
        },
      })) as Session;
    } catch (error) {
      state.errors.push(summarizeError(error));
      appendTimeline("connect_error", { error: summarizeError(error) });
      finish(resolveRun);
    }
  });

  const summary = makeSummary(attemptIndex, attemptDir, state);
  writeJson(resolve(attemptDir, "attempt_summary.json"), summary);
  writeJson(resolve(attemptDir, "stage_records.json"), state.stages);
  writeJson(resolve(attemptDir, "audio_intervals.json"), state.audioIntervals);
  copyFileSync(rawLogPath, resolve(organizedDirs.rawLogs, `airline_suitcase_2step__tick4s__latency8000__attempt${String(attemptIndex).padStart(3, "0")}.raw_log.jsonl`));
  copyFileSync(timelinePath, resolve(organizedDirs.timelines, `airline_suitcase_2step__tick4s__latency8000__attempt${String(attemptIndex).padStart(3, "0")}.timeline.jsonl`));
  try {
    copyFileSync(audioPath, resolve(organizedDirs.audio, `airline_suitcase_2step__tick4s__latency8000__attempt${String(attemptIndex).padStart(3, "0")}.assistant.wav`));
  } catch {
    // Some failed attempts may not produce audio.
  }
  try {
    copyFileSync(timelineAudioPath, resolve(organizedDirs.audio, `airline_suitcase_2step__tick4s__latency8000__attempt${String(attemptIndex).padStart(3, "0")}.assistant_timeline.wav`));
  } catch {
    // Some failed attempts may not produce timeline audio.
  }
  return summary;
}

function writeTimelineSvg(path: string, attempts: AttemptSummary[]): void {
  const width = 1200;
  const rowHeight = 42;
  const height = 60 + attempts.length * rowHeight;
  const maxMs = Math.max(
    1,
    ...attempts.flatMap((attempt) => [
      attempt.stage_1_latency_ms ?? 0,
      attempt.stage_2_latency_ms ?? 0,
      ...(attempt.stage_1_pending_tick_times_ms || []),
      ...(attempt.stage_2_pending_tick_times_ms || []),
    ]),
  );
  const scale = (ms: number) => 160 + (ms / Math.max(maxMs, TOOL_LATENCY_MS)) * 900;
  const lines = [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`,
    `<rect width="100%" height="100%" fill="#ffffff"/>`,
    `<text x="24" y="30" font-family="Arial" font-size="18" fill="#111">Two-step airline pilot, tick4s, latency8000</text>`,
  ];
  attempts.forEach((attempt, index) => {
    const y = 58 + index * rowHeight;
    lines.push(`<text x="24" y="${y + 16}" font-family="Arial" font-size="12" fill="#222">attempt ${String(attempt.attempt_index).padStart(3, "0")}</text>`);
    lines.push(`<line x1="${scale(0)}" y1="${y + 10}" x2="${scale(TOOL_LATENCY_MS)}" y2="${y + 10}" stroke="#2563eb" stroke-width="6"/>`);
    lines.push(`<line x1="${scale(0)}" y1="${y + 26}" x2="${scale(TOOL_LATENCY_MS)}" y2="${y + 26}" stroke="#059669" stroke-width="6"/>`);
    for (const tick of attempt.stage_1_pending_tick_times_ms) {
      lines.push(`<circle cx="${scale(tick - (attempt.stage_1_pending_tick_times_ms[0] ? attempt.stage_1_pending_tick_times_ms[0] - TICK_EVERY_MS : 0))}" cy="${y + 10}" r="5" fill="#f59e0b"/>`);
    }
    for (const tick of attempt.stage_2_pending_tick_times_ms) {
      lines.push(`<circle cx="${scale(tick - (attempt.stage_2_pending_tick_times_ms[0] ? attempt.stage_2_pending_tick_times_ms[0] - TICK_EVERY_MS : 0))}" cy="${y + 26}" r="5" fill="#f59e0b"/>`);
    }
  });
  lines.push("</svg>");
  writeFileSync(path, `${lines.join("\n")}\n`, "utf8");
}

function writeReadme(path: string, resultDir: string, organizedDir: string, summary: AggregateSummary): void {
  writeFileSync(
    path,
    [
      "# Two-step airline native tool-wait pilot",
      "",
      "Task: Airline Suitcase Allowance Check",
      "",
      "- mode: native tool-call",
      "- condition: periodic pending tick every 4000 ms",
      "- tool response latency: 8000 ms per tool call",
      "- attempts: 5",
      "- concurrency: 1",
      "",
      `Result folder: ${resultDir}`,
      `Organized folder: ${organizedDir}`,
      "",
      "## Aggregate",
      "",
      "```json",
      JSON.stringify(summary, null, 2),
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

  const resultDir = resolve(RESULT_DIR, `${timestampForPath()}_two_step_airline_8s_tick4s_pilot`);
  const attemptsDir = resolve(resultDir, "attempts");
  const organizedDir = resolve(resultDir, "organized");
  const organizedDirs = {
    rawLogs: resolve(organizedDir, "raw_logs"),
    audio: resolve(organizedDir, "audio"),
    asr: resolve(organizedDir, "asr"),
    timelines: resolve(organizedDir, "timelines"),
    visualizations: resolve(organizedDir, "visualizations"),
    judgeOutputs: resolve(organizedDir, "judge_outputs"),
    perAttempt: resolve(organizedDir, "per_attempt"),
  };
  mkdirSync(attemptsDir, { recursive: true });
  for (const dir of [organizedDir, ...Object.values(organizedDirs)]) mkdirSync(dir, { recursive: true });

  console.log(`Running two-step airline native tool-wait pilot for model: ${model}`);
  console.log(`Attempts: ${args.attempts}`);
  console.log(`Tool latency: ${TOOL_LATENCY_MS} ms`);
  console.log(`Pending tick: every ${TICK_EVERY_MS} ms while each tool is pending`);
  console.log(`Result directory: ${relative(PROJECT_DIR, resultDir)}`);

  const attempts: AttemptSummary[] = [];
  for (let index = 1; index <= args.attempts; index += 1) {
    const attemptDir = resolve(attemptsDir, `attempt_${String(index).padStart(4, "0")}`);
    console.log(`[attempt ${index}/${args.attempts}] start`);
    const summary = await runOne(ai, model, index, attemptDir, organizedDirs, args);
    attempts.push(summary);
    writeJson(resolve(organizedDirs.perAttempt, `airline_suitcase_2step__tick4s__latency8000__attempt${String(index).padStart(3, "0")}.summary.json`), summary);
    console.log(
      `[attempt ${index}/${args.attempts}] tools=${summary.both_tools_called} final=${summary.final_core_answer_correct} ticks=${summary.stage_1_pending_tick_times_ms.length}/${summary.stage_2_pending_tick_times_ms.length} close=${summary.close_code ?? "none"}`,
    );
  }

  const aggregateSummary = aggregate(attempts);
  const summaryJson = {
    task: "airline_suitcase_allowance_check",
    condition: "native_two_step_tick4s_latency8000",
    prompt_name: PROMPT_NAME,
    user_prompt: USER_PROMPT,
    model,
    tool_latency_ms: TOOL_LATENCY_MS,
    pending_tick_every_ms: TICK_EVERY_MS,
    aggregate: aggregateSummary,
    attempts,
  };

  writeJson(resolve(resultDir, "summary.json"), summaryJson);
  writeCsv(resolve(resultDir, "summary.csv"), attempts as unknown as Array<Record<string, unknown>>);
  writeJson(resolve(organizedDir, "summary.json"), summaryJson);
  writeCsv(resolve(organizedDir, "summary.csv"), attempts as unknown as Array<Record<string, unknown>>);
  writeCsv(resolve(organizedDir, "metrics_by_attempt.csv"), attempts as unknown as Array<Record<string, unknown>>);
  writeTimelineSvg(resolve(organizedDirs.visualizations, "airline_suitcase_2step__tick4s__latency8000.timeline.svg"), attempts);
  writeReadme(resolve(organizedDir, "README.md"), resultDir, organizedDir, aggregateSummary);
  writeReadme(resolve(resultDir, "README.md"), resultDir, organizedDir, aggregateSummary);

  console.log("Summary:");
  console.log(JSON.stringify(aggregateSummary, null, 2));
  console.log(`Organized directory: ${relative(PROJECT_DIR, organizedDir)}`);
}

main().catch((error) => {
  console.error("tau-two-step-airline-pilot failed");
  console.error(summarizeError(error));
  process.exitCode = 1;
});
