import { Behavior, GoogleGenAI, Modality } from "@google/genai";
import { config as loadEnv } from "dotenv";
import { spawnSync } from "node:child_process";
import { randomUUID } from "node:crypto";
import { appendFileSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  PROMPT_NAME as NO_TICK_PROMPT_NAME,
  SYSTEM_INSTRUCTION as NO_TICK_SYSTEM_INSTRUCTION,
  USER_PROMPT as NO_TICK_USER_PROMPT,
} from "./prompts/taw-no-tick.js";
import {
  PROMPT_NAME as WITH_TICK_PROMPT_NAME,
  SYSTEM_INSTRUCTION as WITH_TICK_SYSTEM_INSTRUCTION,
  USER_PROMPT as WITH_TICK_USER_PROMPT,
} from "./prompts/taw-with-tick.js";
import {
  PROMPT_NAME as EXTERNAL_TICK_PROMPT_NAME,
  SYSTEM_INSTRUCTION as EXTERNAL_TICK_SYSTEM_INSTRUCTION,
  USER_PROMPT as EXTERNAL_TICK_USER_PROMPT,
} from "./prompts/taw-external-tick.js";

const PROJECT_DIR = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const DEFAULT_ENV_FILE = resolve(PROJECT_DIR, "..", "angus_bench", ".env");
const RESULT_DIR = process.env.GEMINI_LIVE_CHECK_RESULT_DIR
  ? resolve(process.env.GEMINI_LIVE_CHECK_RESULT_DIR)
  : resolve(PROJECT_DIR, "result");
const MAIN_TOOL_NAME = "get_order_details";
const CHECK_STATUS_TOOL_NAME = "check_status";
const DEFAULT_LATENCY_MS = 8000;
const DEFAULT_ATTEMPTS_PER_CONDITION = 10;
const MAX_ATTEMPT_MS = 45_000;
const POST_FINAL_WAIT_MS = 8000;
const TICK_TIMES_MS = [4000];
const AUDIO_IDLE_BOUNDARY_MS = 300;
const REPEAT_BOUNDARY_MIN_GAP_MS = 2500;

const FINAL_ORDER_RESPONSE = {
  order_id: "#A123",
  status: "shipped",
  estimated_delivery: "tomorrow",
  carrier: "UPS",
  tracking_number: "1Z999AA10123456784",
};

function finalAnswerText(): string {
  return "Order #A123 has shipped. The carrier is UPS, the tracking number is 1Z999AA10123456784, and the estimated delivery is tomorrow.";
}

function pendingPayload(): Record<string, unknown> {
  return {
    event_type: "TOOL_PENDING_STATUS",
    phase: "pending",
    has_final_answer: false,
    answer_now: false,
    tool_name: MAIN_TOOL_NAME,
    message_to_assistant: "The tool is still running. Do not provide the final order status yet.",
  };
}

function clientPendingMessage(): string {
  return "The lookup is still running. No final result is available yet.";
}

function finalPayload(): Record<string, unknown> {
  return {
    event_type: "TOOL_FINAL_RESULT",
    phase: "final",
    has_final_answer: true,
    answer_now: true,
    tool_name: MAIN_TOOL_NAME,
    final_answer: finalAnswerText(),
    result: FINAL_ORDER_RESPONSE,
    message_to_assistant: "This is the final tool result. Stop waiting and answer the user now based only on this result.",
  };
}

function finalToolResponsePayload(tickMode: TickMode): Record<string, unknown> {
  if (
    tickMode === "client_status_tick_3000ms" ||
    tickMode === "periodic_tick_4s" ||
    tickMode === "tick_after_utterance_0s" ||
    tickMode === "tick_after_utterance_1s" ||
    tickMode === "tick_after_audio_idle_0s" ||
    tickMode === "tick_after_audio_idle_1s" ||
    tickMode === "tick_after_audio_idle_repeat_0s" ||
    tickMode === "tick_after_audio_idle_repeat_1s" ||
    tickMode === "native_no_tick"
  ) {
    return {
      event_type: "TOOL_RESULT",
      phase: "final",
      has_final_answer: true,
      answer_now: true,
      tool_name: MAIN_TOOL_NAME,
      message_to_assistant: "This is the tool result. Answer the user now based only on this tool result.",
      ...FINAL_ORDER_RESPONSE,
    };
  }
  return finalPayload();
}

type TickMode =
  | "native_no_tick"
  | "client_status_tick_3000ms"
  | "periodic_tick_4s"
  | "tick_after_utterance_0s"
  | "tick_after_utterance_1s"
  | "tick_after_audio_idle_0s"
  | "tick_after_audio_idle_1s"
  | "tick_after_audio_idle_repeat_0s"
  | "tick_after_audio_idle_repeat_1s"
  | "same_call_pending_function_response_3000ms"
  | "async_tool_polling";

const TICK_MODES: TickMode[] = [
  "native_no_tick",
  "client_status_tick_3000ms",
  "periodic_tick_4s",
  "tick_after_utterance_0s",
  "tick_after_utterance_1s",
  "tick_after_audio_idle_0s",
  "tick_after_audio_idle_1s",
  "tick_after_audio_idle_repeat_0s",
  "tick_after_audio_idle_repeat_1s",
  "same_call_pending_function_response_3000ms",
  "async_tool_polling",
];
const DEFAULT_TICK_MODES: TickMode[] = [
  "native_no_tick",
  "same_call_pending_function_response_3000ms",
  "client_status_tick_3000ms",
];
const TICK_MODE_ALIASES: Record<string, TickMode> = {
  no_tick: "native_no_tick",
  native_no_tick: "native_no_tick",
  with_tick_tool: "same_call_pending_function_response_3000ms",
  tool_tick: "same_call_pending_function_response_3000ms",
  same_call_pending_function_response_3000ms: "same_call_pending_function_response_3000ms",
  with_tick_external: "client_status_tick_3000ms",
  external_tick: "client_status_tick_3000ms",
  client_status_tick_3000ms: "client_status_tick_3000ms",
  fixed_single_tick_4s: "client_status_tick_3000ms",
  periodic_tick_4s: "periodic_tick_4s",
  fixed_periodic_tick_4s: "periodic_tick_4s",
  tick_after_utterance_0s: "tick_after_utterance_0s",
  boundary_tick_0s: "tick_after_utterance_0s",
  tick_after_utterance_1s: "tick_after_utterance_1s",
  boundary_tick_1s: "tick_after_utterance_1s",
  tick_after_audio_idle_0s: "tick_after_audio_idle_0s",
  audio_idle_tick_0s: "tick_after_audio_idle_0s",
  tick_after_audio_idle_1s: "tick_after_audio_idle_1s",
  audio_idle_tick_1s: "tick_after_audio_idle_1s",
  tick_after_audio_idle_repeat_0s: "tick_after_audio_idle_repeat_0s",
  audio_idle_repeat_tick_0s: "tick_after_audio_idle_repeat_0s",
  tick_after_audio_idle_repeat_1s: "tick_after_audio_idle_repeat_1s",
  audio_idle_repeat_tick_1s: "tick_after_audio_idle_repeat_1s",
  async_tool_polling: "async_tool_polling",
};
const TICK_MODE_LABELS: Record<TickMode, string> = {
  native_no_tick: "no_tick",
  same_call_pending_function_response_3000ms: "with_tick_tool_response",
  client_status_tick_3000ms: "with_tick_client_content",
  periodic_tick_4s: "periodic_tick_4s",
  tick_after_utterance_0s: "tick_after_utterance_0s",
  tick_after_utterance_1s: "tick_after_utterance_1s",
  tick_after_audio_idle_0s: "tick_after_audio_idle_0s",
  tick_after_audio_idle_1s: "tick_after_audio_idle_1s",
  tick_after_audio_idle_repeat_0s: "tick_after_audio_idle_repeat_0s",
  tick_after_audio_idle_repeat_1s: "tick_after_audio_idle_repeat_1s",
  async_tool_polling: "async_tool_polling",
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
  usageMetadata?: unknown;
  goAway?: unknown;
  sessionResumptionUpdate?: unknown;
  voiceActivity?: unknown;
  voiceActivityDetectionSignal?: unknown;
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
  latencyMs: number;
  attemptsPerCondition: number;
  tickModes: TickMode[];
  quietTerminalText: boolean;
};

type AttemptPlan = {
  tickMode: TickMode;
  latencyMs: number;
  attemptIndex: number;
  attemptDir: string;
  quietTerminalText?: boolean;
};

type PromptProfile = {
  promptName: string;
  systemInstruction: string;
  userPrompt: string;
};

type AudioSegment = {
  offsetMs: number;
  chunk: Buffer;
  mimeType?: string;
};

type TickRecord = {
  kind: "client_status" | "boundary_client_status" | "interim_function_response" | "check_status_pending";
  sentAtMs: number;
  sendError?: string;
};

type CheckStatusRecord = {
  callTimeMs: number;
  responseTimeMs?: number;
  status?: "pending" | "ready";
  sendError?: string;
};

type AttemptState = {
  setupCompleteAt?: number;
  promptSentAt?: number;
  closeCode: number | null;
  closeReason: string | null;
  sessionClosed: boolean;
  mainToolCall?: FunctionCall;
  mainToolCallAt?: number;
  toolCallCount: number;
  mainToolResponseSentAt?: number;
  finalToolResponseSentAt?: number;
  firstAudioAt?: number;
  firstAudioAfterFinalAt?: number;
  lastAudioBeforeFinalAt?: number;
  sessionOpenedAt?: number;
  turnCompleteAt?: number;
  rawEventCount: number;
  audioTimesMs: number[];
  outputTimesMs: number[];
  audioEventsBeforeFinal: number;
  audioEventsAfterFinal: number;
  textBeforeFinal: string[];
  textAfterFinal: string[];
  ticks: TickRecord[];
  boundaryDetectedTimesMs: number[];
  boundaryTickTimesMs: number[];
  boundaryTickSkippedFinalReadyCount: number;
  boundaryTickScheduled: boolean;
  checkStatusCalls: CheckStatusRecord[];
  cancelledToolCallIds: string[];
  sendErrors: string[];
  errors: string[];
};

type AttemptSummary = {
  condition: TickMode;
  latency_ms: number;
  attempt_index: number;
  session_valid: boolean;
  close_1008: boolean;
  close_1011: boolean;
  close_code: number | null;
  close_reason: string | null;
  client_send_error_count: number;
  tool_call_success: boolean;
  tool_call_time_ms: number | null;
  final_tool_response_sent: boolean;
  final_tool_response_sent_time_ms: number | null;
  tick_send_success_count: number;
  waiting_audio_before_final_tool_result: boolean;
  audio_after_tick: boolean;
  premature_final_answer: boolean;
  post_tool_final_answer: boolean;
  post_tool_final_latency_ms: number | null;
  first_audio_time_ms: number | null;
  turnComplete_time_ms: number | null;
  setupComplete_time_ms: number | null;
  setupComplete_after_session_open_ms: number | null;
  setup_complete_before_prompt: boolean;
  last_audio_before_final_tool_response_ms: number | null;
  pending_tick_times_ms: number[];
  boundary_detected_times_ms: number[];
  boundary_tick_times_ms: number[];
  has_output_after_boundary_tick: boolean;
  pending_tick_skipped_final_ready_count: number;
  raw_event_count: number;
  text_before_final: string[];
  text_after_final: string[];
  check_status_calls: CheckStatusRecord[];
  cancelled_tool_call_ids: string[];
  send_errors: string[];
  errors: string[];
  result_dir: string;
};

type ConditionSummary = {
  condition: TickMode;
  attempts: number;
  valid_attempts: number;
  close_1008_count: number;
  close_1011_count: number;
  client_send_error_count: number;
  tool_call_success_count: number;
  final_tool_response_sent_count: number;
  tick_send_success_count: number;
  waiting_audio_before_final_tool_result_count: number;
  audio_after_tick_count: number;
  premature_final_answer_count: number;
  post_tool_final_answer_count: number;
  avg_post_tool_final_latency_ms: number | null;
  avg_first_audio_time_ms: number | null;
  avg_turnComplete_time_ms: number | null;
  avg_tool_call_time_ms: number | null;
  avg_final_tool_response_sent_time_ms: number | null;
  avg_last_audio_before_final_tool_response_ms: number | null;
};

loadEnv({ path: process.env.GEMINI_ENV_FILE || DEFAULT_ENV_FILE });

function requireEnv(name: string): string {
  const value = process.env[name]?.trim();
  if (!value) throw new Error(`Missing ${name}. Set it in /user_data/angus_bench/.env or GEMINI_ENV_FILE.`);
  return value;
}

function timestampForPath(date = new Date()): string {
  const [datePart, timePart] = date.toISOString().split("T");
  return `${datePart}_${timePart.replace("Z", "").replace(/\./g, "-").replace(/:/g, "-")}`;
}

function parseCsvTickModes(value: string | undefined): TickMode[] {
  if (!value) return DEFAULT_TICK_MODES;
  const modes = value.split(",").map((part) => part.trim()).filter(Boolean);
  return modes.map((mode) => {
    const resolved = TICK_MODE_ALIASES[mode];
    if (!resolved || !TICK_MODES.includes(resolved)) throw new Error(`Unknown tick mode: ${mode}`);
    return resolved;
  });
}

function parseArgs(argv: string[]): Args {
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
    latencyMs: Math.max(1, Number(args.get("latency-ms") ?? DEFAULT_LATENCY_MS)),
    attemptsPerCondition: Math.max(
      1,
      Number(args.get("attempts-per-condition") ?? args.get("attempts") ?? DEFAULT_ATTEMPTS_PER_CONDITION),
    ),
    tickModes: parseCsvTickModes(args.get("tick-modes")),
    quietTerminalText: args.has("quiet-terminal-text"),
  };
}

function promptForTickMode(tickMode: TickMode): PromptProfile {
  if (tickMode === "native_no_tick") {
    return {
      promptName: NO_TICK_PROMPT_NAME,
      systemInstruction: NO_TICK_SYSTEM_INSTRUCTION,
      userPrompt: NO_TICK_USER_PROMPT,
    };
  }
  if (
    tickMode === "client_status_tick_3000ms" ||
    tickMode === "periodic_tick_4s" ||
    tickMode === "tick_after_utterance_0s" ||
    tickMode === "tick_after_utterance_1s" ||
    tickMode === "tick_after_audio_idle_0s" ||
    tickMode === "tick_after_audio_idle_1s" ||
    tickMode === "tick_after_audio_idle_repeat_0s" ||
    tickMode === "tick_after_audio_idle_repeat_1s"
  ) {
    return {
      promptName: EXTERNAL_TICK_PROMPT_NAME,
      systemInstruction: EXTERNAL_TICK_SYSTEM_INSTRUCTION,
      userPrompt: EXTERNAL_TICK_USER_PROMPT,
    };
  }
  return {
    promptName: WITH_TICK_PROMPT_NAME,
    systemInstruction: WITH_TICK_SYSTEM_INSTRUCTION,
    userPrompt: WITH_TICK_USER_PROMPT,
  };
}

function writeJson(path: string, value: unknown): void {
  writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

function appendJsonl(path: string, value: Record<string, unknown>): void {
  appendFileSync(path, `${JSON.stringify({ ts: new Date().toISOString(), ...value })}\n`, "utf8");
}

function summarizeError(error: unknown): string {
  if (error instanceof Error) return `${error.name}: ${error.message}${error.cause ? ` Cause: ${summarizeError(error.cause)}` : ""}`;
  if (typeof error !== "object" || error === null) return String(error);
  const record = error as Record<string, unknown>;
  return ["message", "type", "code", "reason", "name", "error"]
    .map((key) => (record[key] ? `${key}: ${String(record[key])}` : undefined))
    .filter(Boolean)
    .join(", ") || Object.prototype.toString.call(error);
}

function consoleText(text: string, maxLength = 260): string {
  const cleaned = text.replace(/\s+/g, " ").trim();
  if (cleaned.length <= maxLength) return cleaned;
  return `${cleaned.slice(0, maxLength - 1)}...`;
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

function parsePcmMime(mimeType: string | undefined): { sampleRate?: number; channels?: number; bitDepth?: number } {
  if (!mimeType?.startsWith("audio/pcm")) return {};
  const rate = mimeType.match(/rate=(\d+)/)?.[1];
  const channels = mimeType.match(/channels=(\d+)/)?.[1];
  return { sampleRate: rate ? Number(rate) : undefined, channels: channels ? Number(channels) : 1, bitDepth: 16 };
}

function wavHeader(dataLength: number, sampleRate: number, channels: number, bitDepth: number): Buffer {
  const byteRate = (sampleRate * channels * bitDepth) / 8;
  const blockAlign = (channels * bitDepth) / 8;
  const header = Buffer.alloc(44);
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

function writeAudioFile(audioDir: string, segments: AudioSegment[]): string | null {
  if (segments.length === 0) return null;
  mkdirSync(audioDir, { recursive: true });
  const mimeType = segments.find((segment) => segment.mimeType)?.mimeType;
  const audio = Buffer.concat(segments.map((segment) => segment.chunk));
  const pcm = parsePcmMime(mimeType);
  const path = resolve(audioDir, "assistant_output.wav");
  if (pcm.sampleRate && pcm.channels && pcm.bitDepth) {
    writeFileSync(path, Buffer.concat([wavHeader(audio.length, pcm.sampleRate, pcm.channels, pcm.bitDepth), audio]));
    return path;
  }
  const rawPath = resolve(audioDir, "assistant_output.bin");
  writeFileSync(rawPath, audio);
  return rawPath;
}

function msDelta(start: number | undefined, value: number | undefined): number | null {
  if (!start || !value) return null;
  return value - start;
}

function isFinalAnswerText(texts: string[]): boolean {
  const text = texts.join(" ").toLowerCase();
  return /shipped/.test(text) && /tomorrow/.test(text) && /#?a123/.test(text);
}

function makeTools(mode: TickMode): unknown[] {
  const mainTool = {
    name: MAIN_TOOL_NAME,
    description:
      mode === "async_tool_polling"
        ? "Start an asynchronous retail order status lookup. Returns a job_id if the lookup is still pending."
        : "Get the status and details of a retail order.",
    behavior: Behavior.NON_BLOCKING,
    parametersJsonSchema: {
      type: "object",
      properties: {
        order_id: {
          type: "string",
          description: "The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id.",
        },
      },
      required: ["order_id"],
    },
  };
  const declarations: any[] = [mainTool];
  if (mode === "async_tool_polling") {
    declarations.push({
      name: CHECK_STATUS_TOOL_NAME,
      description: "Poll an asynchronous order lookup job. Call this with the job_id until status is ready.",
      behavior: Behavior.NON_BLOCKING,
      parametersJsonSchema: {
        type: "object",
        properties: {
          job_id: { type: "string", description: "The job id returned by get_order_details." },
        },
        required: ["job_id"],
      },
    });
  }
  return [{ functionDeclarations: declarations }];
}

function computeSummary(plan: AttemptPlan, state: AttemptState): AttemptSummary {
  const close1008 = state.closeCode === 1008 || state.closeReason?.includes("1008") === true;
  const close1011 = state.closeCode === 1011 || state.closeReason?.includes("1011") === true;
  const finalSent = Boolean(state.finalToolResponseSentAt);
  const postFinalAnswer = isFinalAnswerText(state.textAfterFinal);
  const prematureFinalAnswer = isFinalAnswerText(state.textBeforeFinal);
  const audioAfterTick = state.ticks.some(
    (tick) =>
      !tick.sendError &&
      state.audioTimesMs.some((audioMs) => audioMs >= tick.sentAtMs && audioMs <= tick.sentAtMs + 3000),
  );
  const hasOutputAfterBoundaryTick = state.boundaryTickTimesMs.some((tickMs) =>
    state.audioTimesMs.some((audioMs) => audioMs >= tickMs && audioMs <= tickMs + 3000) ||
    state.outputTimesMs.some((outputMs) => outputMs >= tickMs && outputMs <= tickMs + 3000),
  );
  return {
    condition: plan.tickMode,
    latency_ms: plan.latencyMs,
    attempt_index: plan.attemptIndex,
    session_valid: !close1008 && !close1011 && finalSent && postFinalAnswer,
    close_1008: close1008,
    close_1011: close1011,
    close_code: state.closeCode,
    close_reason: state.closeReason,
    client_send_error_count: state.sendErrors.length,
    tool_call_success: Boolean(state.mainToolCallAt),
    tool_call_time_ms: msDelta(state.promptSentAt, state.mainToolCallAt),
    final_tool_response_sent: finalSent,
    final_tool_response_sent_time_ms: msDelta(state.promptSentAt, state.finalToolResponseSentAt),
    tick_send_success_count: state.ticks.filter((tick) => !tick.sendError).length,
    waiting_audio_before_final_tool_result: state.audioEventsBeforeFinal > 0,
    audio_after_tick: audioAfterTick,
    premature_final_answer: prematureFinalAnswer,
    post_tool_final_answer: postFinalAnswer,
    post_tool_final_latency_ms:
      state.finalToolResponseSentAt && state.firstAudioAfterFinalAt && state.firstAudioAfterFinalAt >= state.finalToolResponseSentAt
        ? state.firstAudioAfterFinalAt - state.finalToolResponseSentAt
        : null,
    first_audio_time_ms: msDelta(state.promptSentAt, state.firstAudioAt),
    turnComplete_time_ms: msDelta(state.promptSentAt, state.turnCompleteAt),
    setupComplete_time_ms: msDelta(state.sessionOpenedAt, state.setupCompleteAt),
    setupComplete_after_session_open_ms: msDelta(state.sessionOpenedAt, state.setupCompleteAt),
    setup_complete_before_prompt: Boolean(state.setupCompleteAt && state.promptSentAt && state.setupCompleteAt <= state.promptSentAt),
    last_audio_before_final_tool_response_ms: msDelta(state.promptSentAt, state.lastAudioBeforeFinalAt),
    pending_tick_times_ms: state.ticks.filter((tick) => !tick.sendError).map((tick) => tick.sentAtMs),
    boundary_detected_times_ms: state.boundaryDetectedTimesMs,
    boundary_tick_times_ms: state.boundaryTickTimesMs,
    has_output_after_boundary_tick: hasOutputAfterBoundaryTick,
    pending_tick_skipped_final_ready_count: state.boundaryTickSkippedFinalReadyCount,
    raw_event_count: state.rawEventCount,
    text_before_final: state.textBeforeFinal,
    text_after_final: state.textAfterFinal,
    check_status_calls: state.checkStatusCalls,
    cancelled_tool_call_ids: state.cancelledToolCallIds,
    send_errors: state.sendErrors,
    errors: state.errors,
    result_dir: plan.attemptDir,
  };
}

async function runOne(ai: GoogleGenAI, model: string, plan: AttemptPlan): Promise<AttemptSummary> {
  mkdirSync(plan.attemptDir, { recursive: true });
  const audioDir = resolve(plan.attemptDir, "audio");
  const timelineDir = resolve(plan.attemptDir, "timeline");
  mkdirSync(audioDir, { recursive: true });
  mkdirSync(timelineDir, { recursive: true });
  const rawLogPath = resolve(plan.attemptDir, "raw_log.jsonl");
  const timelinePath = resolve(timelineDir, "events.jsonl");
  const audioSegments: AudioSegment[] = [];
  const prompt = promptForTickMode(plan.tickMode);
  const state: AttemptState = {
    closeCode: null,
    closeReason: null,
    sessionClosed: false,
    toolCallCount: 0,
    rawEventCount: 0,
    audioTimesMs: [],
    outputTimesMs: [],
    audioEventsBeforeFinal: 0,
    audioEventsAfterFinal: 0,
    textBeforeFinal: [],
    textAfterFinal: [],
    ticks: [],
    boundaryDetectedTimesMs: [],
    boundaryTickTimesMs: [],
    boundaryTickSkippedFinalReadyCount: 0,
    boundaryTickScheduled: false,
    checkStatusCalls: [],
    cancelledToolCallIds: [],
    sendErrors: [],
    errors: [],
  };
  let session: Session | undefined;
  let done = false;
  let resolveRunRef: (() => void) | undefined;
  let postFinalObservationScheduled = false;
  let initialPromptSent = false;
  let audioIdleBoundaryTimer: ReturnType<typeof setTimeout> | undefined;
  let audioPlaybackCursorMs = 0;
  let repeatBoundaryTickPending = false;
  let lastBoundaryTickSentAtMs: number | null = null;
  const timers: ReturnType<typeof setTimeout>[] = [];
  const jobId = `job_${plan.tickMode}_${plan.attemptIndex}_${Date.now()}`;

  writeJson(resolve(plan.attemptDir, "config.json"), {
    condition: plan.tickMode,
    latency_ms: plan.latencyMs,
    attempt_index: plan.attemptIndex,
    close_strategy: "post_final_observation_or_timeout",
    prompt_name: prompt.promptName,
    system_instruction: prompt.systemInstruction,
    user_prompt: prompt.userPrompt,
    model,
  });

  const appendTimeline = (type: string, extra: Record<string, unknown> = {}) => {
    appendJsonl(timelinePath, { type, event_ms: state.promptSentAt ? Date.now() - state.promptSentAt : null, ...extra });
  };

  const noteSendError = (type: string, error: unknown) => {
    const summary = summarizeError(error);
    state.sendErrors.push(`${type}: ${summary}`);
    appendTimeline("send_error", { send_type: type, error: summary });
    appendJsonl(rawLogPath, { type: "send_error", send_type: type, error: summary });
    return summary;
  };

  const isCancelledCall = (call: FunctionCall): boolean => Boolean(call.id && state.cancelledToolCallIds.includes(call.id));

  const sendInitialUserPrompt = () => {
    if (initialPromptSent || done || state.sessionClosed || !session) return;
    initialPromptSent = true;
    session.sendClientContent({ turns: prompt.userPrompt, turnComplete: true });
    state.promptSentAt = Date.now();
    appendTimeline("user_message_sent", { prompt_name: prompt.promptName, prompt: prompt.userPrompt, sent_after_setup_complete: Boolean(state.setupCompleteAt) });
    appendJsonl(rawLogPath, {
      type: "user_message_sent",
      prompt_name: prompt.promptName,
      prompt: prompt.userPrompt,
      sent_after_setup_complete: Boolean(state.setupCompleteAt),
    });
  };

  const schedulePostFinalObservation = () => {
    if (!resolveRunRef || postFinalObservationScheduled) return;
    postFinalObservationScheduled = true;
    appendTimeline("post_final_observation_started", { wait_ms: POST_FINAL_WAIT_MS });
    appendJsonl(rawLogPath, {
      type: "post_final_observation_started",
      event_ms: state.promptSentAt ? Date.now() - state.promptSentAt : null,
      wait_ms: POST_FINAL_WAIT_MS,
    });
    const timer = setTimeout(() => {
      appendTimeline("post_final_observation_elapsed", { wait_ms: POST_FINAL_WAIT_MS });
      appendJsonl(rawLogPath, {
        type: "post_final_observation_elapsed",
        event_ms: state.promptSentAt ? Date.now() - state.promptSentAt : null,
        wait_ms: POST_FINAL_WAIT_MS,
      });
      if (resolveRunRef) finish(resolveRunRef);
    }, POST_FINAL_WAIT_MS);
    timers.push(timer);
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
    writeAudioFile(audioDir, audioSegments);
    resolveRun();
  };

  const sendToolResponse = (call: FunctionCall, response: Record<string, unknown>, responseKind: string): boolean => {
    if (isCancelledCall(call)) {
      const summary = `skipped ${responseKind}: tool call ${call.id} was cancelled by server`;
      state.sendErrors.push(summary);
      appendTimeline("tool_response_skipped_cancelled_call", { function_call_id: call.id, function_name: call.name, response_kind: responseKind });
      appendJsonl(rawLogPath, {
        type: "tool_response_skipped_cancelled_call",
        event_ms: state.promptSentAt ? Date.now() - state.promptSentAt : null,
        function_call_id: call.id,
        function_name: call.name,
        response_kind: responseKind,
      });
      return false;
    }
    try {
      session?.sendToolResponse({
        functionResponses: [{ id: call.id, name: call.name || MAIN_TOOL_NAME, response }],
      });
      appendTimeline(responseKind, { function_call_id: call.id, function_name: call.name, response });
      appendJsonl(rawLogPath, {
        type: responseKind,
        event_ms: state.promptSentAt ? Date.now() - state.promptSentAt : null,
        function_call_id: call.id,
        function_name: call.name,
        response,
      });
      return true;
    } catch (error) {
      noteSendError(responseKind, error);
      return false;
    }
  };

  const scheduleFinalMainResponse = (call: FunctionCall) => {
    timers.push(
      setTimeout(() => {
        if (done || state.sessionClosed) return;
        if (sendToolResponse(call, finalToolResponsePayload(plan.tickMode), "final_tool_response_sent")) {
          state.finalToolResponseSentAt = Date.now();
          schedulePostFinalObservation();
        }
      }, plan.latencyMs),
    );
  };

  const clientTickTimes = (): number[] => {
    if (plan.tickMode === "client_status_tick_3000ms") return TICK_TIMES_MS.filter((tickMs) => tickMs < plan.latencyMs);
    if (plan.tickMode !== "periodic_tick_4s") return [];
    const times: number[] = [];
    for (let tickMs = 4000; tickMs < plan.latencyMs; tickMs += 4000) times.push(tickMs);
    return times;
  };

  const scheduleClientTicks = () => {
    for (const tickMs of clientTickTimes()) {
      timers.push(
        setTimeout(() => {
          if (done || state.sessionClosed) return;
          const sentAtMs = Date.now() - (state.promptSentAt ?? Date.now());
          const tick: TickRecord = { kind: "client_status", sentAtMs };
          try {
            session?.sendRealtimeInput({ text: clientPendingMessage() });
            appendTimeline("client_status_tick_sent", {
              tick_ms: tickMs,
              send_method: "sendRealtimeInput",
              message: clientPendingMessage(),
            });
            appendJsonl(rawLogPath, {
              type: "client_status_tick_sent",
              event_ms: sentAtMs,
              tick_ms: tickMs,
              send_method: "sendRealtimeInput",
              message: clientPendingMessage(),
            });
          } catch (error) {
            tick.sendError = noteSendError("client_status_tick", error);
          }
          state.ticks.push(tick);
        }, tickMs),
      );
    }
  };

  const boundaryTickDelayMs = (): number | null => {
    if (plan.tickMode === "tick_after_utterance_0s") return 0;
    if (plan.tickMode === "tick_after_utterance_1s") return 1000;
    if (plan.tickMode === "tick_after_audio_idle_0s") return 0;
    if (plan.tickMode === "tick_after_audio_idle_1s") return 1000;
    if (plan.tickMode === "tick_after_audio_idle_repeat_0s") return 0;
    if (plan.tickMode === "tick_after_audio_idle_repeat_1s") return 1000;
    return null;
  };

  const usesAudioIdleBoundary = (): boolean =>
    plan.tickMode === "tick_after_audio_idle_0s" ||
    plan.tickMode === "tick_after_audio_idle_1s" ||
    plan.tickMode === "tick_after_audio_idle_repeat_0s" ||
    plan.tickMode === "tick_after_audio_idle_repeat_1s";

  const usesRepeatAudioIdleBoundary = (): boolean =>
    plan.tickMode === "tick_after_audio_idle_repeat_0s" || plan.tickMode === "tick_after_audio_idle_repeat_1s";

  const scheduleAudioIdleBoundary = (audioStartMs: number, audioDurationMs: number) => {
    if (!usesAudioIdleBoundary()) return;
    if (!usesRepeatAudioIdleBoundary() && state.boundaryTickScheduled) return;
    if (repeatBoundaryTickPending || done || state.sessionClosed || !state.promptSentAt) return;
    if (!state.mainToolCallAt || state.finalToolResponseSentAt) return;
    if (audioIdleBoundaryTimer) clearTimeout(audioIdleBoundaryTimer);
    audioPlaybackCursorMs = Math.max(audioPlaybackCursorMs, audioStartMs) + audioDurationMs;
    const waitMs = Math.max(0, Math.ceil(audioPlaybackCursorMs + AUDIO_IDLE_BOUNDARY_MS - audioStartMs));
    audioIdleBoundaryTimer = setTimeout(() => {
      maybeScheduleBoundaryTick(`audio_playback_idle_${AUDIO_IDLE_BOUNDARY_MS}ms`);
    }, waitMs);
    timers.push(audioIdleBoundaryTimer);
  };

  const maybeScheduleBoundaryTick = (detectionMethod = "turn_complete") => {
    const delayMs = boundaryTickDelayMs();
    if (delayMs === null) return;
    const repeatMode = usesRepeatAudioIdleBoundary();
    if ((!repeatMode && state.boundaryTickScheduled) || repeatBoundaryTickPending || done || state.sessionClosed || !state.promptSentAt) return;
    if (!state.mainToolCallAt) return;
    const detectedAtMs = Date.now() - state.promptSentAt;
    if (repeatMode && lastBoundaryTickSentAtMs !== null && detectedAtMs - lastBoundaryTickSentAtMs < REPEAT_BOUNDARY_MIN_GAP_MS) {
      appendTimeline("boundary_tick_skipped_cooldown", {
        detection_method: detectionMethod,
        min_gap_ms: REPEAT_BOUNDARY_MIN_GAP_MS,
        ms_since_last_tick: detectedAtMs - lastBoundaryTickSentAtMs,
      });
      appendJsonl(rawLogPath, {
        type: "boundary_tick_skipped_cooldown",
        event_ms: detectedAtMs,
        detection_method: detectionMethod,
        min_gap_ms: REPEAT_BOUNDARY_MIN_GAP_MS,
        ms_since_last_tick: detectedAtMs - lastBoundaryTickSentAtMs,
      });
      return;
    }
    if (state.finalToolResponseSentAt) {
      state.boundaryTickSkippedFinalReadyCount += 1;
      appendTimeline("boundary_tick_skipped_final_ready", { delay_ms: delayMs });
      appendJsonl(rawLogPath, {
        type: "boundary_tick_skipped_final_ready",
        event_ms: Date.now() - state.promptSentAt,
        delay_ms: delayMs,
      });
      return;
    }
    if (repeatMode) repeatBoundaryTickPending = true;
    else state.boundaryTickScheduled = true;
    state.boundaryDetectedTimesMs.push(detectedAtMs);
    appendTimeline("speech_boundary_detected", { detection_method: detectionMethod, delay_ms: delayMs });
    appendJsonl(rawLogPath, {
      type: "speech_boundary_detected",
      event_ms: detectedAtMs,
      detection_method: detectionMethod,
      delay_ms: delayMs,
    });
    timers.push(
      setTimeout(() => {
        if (done || state.sessionClosed || !state.promptSentAt) return;
        if (state.finalToolResponseSentAt) {
          state.boundaryTickSkippedFinalReadyCount += 1;
          if (repeatMode) repeatBoundaryTickPending = false;
          appendTimeline("boundary_tick_skipped_final_ready", { delay_ms: delayMs });
          appendJsonl(rawLogPath, {
            type: "boundary_tick_skipped_final_ready",
            event_ms: Date.now() - state.promptSentAt,
            delay_ms: delayMs,
          });
          return;
        }
        const sentAtMs = Date.now() - state.promptSentAt;
        const tick: TickRecord = { kind: "boundary_client_status", sentAtMs };
        try {
          session?.sendRealtimeInput({ text: clientPendingMessage() });
          state.boundaryTickTimesMs.push(sentAtMs);
          lastBoundaryTickSentAtMs = sentAtMs;
          appendTimeline("boundary_client_status_tick_sent", {
            send_method: "sendRealtimeInput",
            delay_ms: delayMs,
            message: clientPendingMessage(),
          });
          appendJsonl(rawLogPath, {
            type: "boundary_client_status_tick_sent",
            event_ms: sentAtMs,
            send_method: "sendRealtimeInput",
            delay_ms: delayMs,
            message: clientPendingMessage(),
          });
        } catch (error) {
          tick.sendError = noteSendError("boundary_client_status_tick", error);
        }
        state.ticks.push(tick);
        if (repeatMode) repeatBoundaryTickPending = false;
      }, delayMs),
    );
  };

  const scheduleInterimResponses = (call: FunctionCall) => {
    for (const tickMs of TICK_TIMES_MS.filter((tickMs) => tickMs < plan.latencyMs)) {
      timers.push(
        setTimeout(() => {
          if (done || state.sessionClosed) return;
          const sentAtMs = Date.now() - (state.promptSentAt ?? Date.now());
          const tick: TickRecord = { kind: "interim_function_response", sentAtMs };
          const ok = sendToolResponse(
            call,
            pendingPayload(),
            "interim_function_response_sent",
          );
          if (!ok) tick.sendError = state.sendErrors.at(-1);
          state.ticks.push(tick);
        }, tickMs),
      );
    }
  };

  await new Promise<void>(async (resolveRun) => {
    resolveRunRef = resolveRun;
    const hardTimer = setTimeout(() => {
      state.errors.push(`max attempt timeout ${MAX_ATTEMPT_MS}ms`);
      appendTimeline("max_attempt_timeout", { max_attempt_ms: MAX_ATTEMPT_MS });
      finish(resolveRun);
    }, MAX_ATTEMPT_MS);
    timers.push(hardTimer);

    try {
      session = (await ai.live.connect({
        model,
        config: {
          responseModalities: [Modality.AUDIO],
          outputAudioTranscription: {},
          systemInstruction: prompt.systemInstruction,
          tools: makeTools(plan.tickMode) as any,
        },
        callbacks: {
          onopen: () => {
            state.sessionOpenedAt = Date.now();
            appendTimeline("session_opened", { model });
            appendJsonl(rawLogPath, { type: "session_opened", model });
          },
          onmessage: (message: LiveMessage) => {
            const now = Date.now();
            state.rawEventCount += 1;
            const eventMs = state.promptSentAt ? now - state.promptSentAt : null;
            appendJsonl(rawLogPath, { type: "server_event", event_ms: eventMs, event_types: eventTypes(message), message: sanitizeMessage(message) });
            appendTimeline("server_event", { event_types: eventTypes(message) });

            if (message.setupComplete) {
              state.setupCompleteAt ??= now;
              appendTimeline("setup_complete", { setup_complete_before_prompt: !state.promptSentAt });
              appendJsonl(rawLogPath, {
                type: "setup_complete",
                event_ms: state.promptSentAt ? now - state.promptSentAt : null,
                setup_complete_before_prompt: !state.promptSentAt,
              });
              try {
                sendInitialUserPrompt();
              } catch (error) {
                noteSendError("initial_user_prompt", error);
                finish(resolveRunRef ?? (() => undefined));
              }
              return;
            }

            const parts = message.serverContent?.modelTurn?.parts ?? [];
            const textPieces = [
              ...parts.map((part) => part.text).filter((text): text is string => Boolean(text)),
              message.serverContent?.outputTranscription?.text,
            ].filter((text): text is string => Boolean(text));
            for (const text of textPieces) {
              const isTranscription = message.serverContent?.outputTranscription?.text === text;
              const phase = state.finalToolResponseSentAt ? "after_final" : "before_final";
              const label = isTranscription ? "transcript" : "text";
              if (!plan.quietTerminalText) {
                console.log(
                  `[${plan.tickMode} attempt ${plan.attemptIndex} ${eventMs ?? "?"}ms ${phase} ${label}] ${consoleText(text)}`,
                );
              }
              if (state.finalToolResponseSentAt) state.textAfterFinal.push(text);
              else state.textBeforeFinal.push(text);
              if (eventMs !== null) state.outputTimesMs.push(eventMs);
              appendTimeline(isTranscription ? "output_transcription" : "text_output", { text });
            }
            for (const part of parts) {
              if (!part.inlineData?.data) continue;
              const chunk = Buffer.from(part.inlineData.data, "base64");
              const audioDurationMs = (chunk.length / 48_000) * 1000;
              audioSegments.push({ offsetMs: eventMs ?? 0, chunk, mimeType: part.inlineData.mimeType });
              if (eventMs !== null) state.audioTimesMs.push(eventMs);
              if (eventMs !== null) state.outputTimesMs.push(eventMs);
              state.firstAudioAt ??= now;
              if (state.finalToolResponseSentAt) {
                state.audioEventsAfterFinal += 1;
                state.firstAudioAfterFinalAt ??= now;
              } else {
                state.audioEventsBeforeFinal += 1;
                state.lastAudioBeforeFinalAt = now;
              }
              appendTimeline("audio_output", {
                bytes: chunk.length,
                mime_type: part.inlineData.mimeType,
                phase: state.finalToolResponseSentAt ? "after_tool_response" : "before_tool_response",
              });
              if (!state.finalToolResponseSentAt && eventMs !== null) scheduleAudioIdleBoundary(eventMs, audioDurationMs);
            }

            if (message.toolCall?.functionCalls?.length) {
              state.toolCallCount += message.toolCall.functionCalls.length;
              appendTimeline("tool_call_received", { function_calls: message.toolCall.functionCalls });
              appendJsonl(rawLogPath, { type: "tool_call_received", function_calls: message.toolCall.functionCalls });
              for (const call of message.toolCall.functionCalls) {
                if (call.name === CHECK_STATUS_TOOL_NAME) {
                  const callTimeMs = Date.now() - (state.promptSentAt ?? Date.now());
                  const record: CheckStatusRecord = { callTimeMs };
                  const ready = callTimeMs >= plan.latencyMs;
                  const response = ready ? finalPayload() : pendingPayload();
                  const ok = sendToolResponse(call, response, ready ? "final_tool_response_sent" : "check_status_pending_response_sent");
                  record.responseTimeMs = Date.now() - (state.promptSentAt ?? Date.now());
                  record.status = ready ? "ready" : "pending";
                  if (!ok) record.sendError = state.sendErrors.at(-1);
                  state.checkStatusCalls.push(record);
                  if (ready && ok) state.finalToolResponseSentAt = Date.now();
                  if (ready && ok) schedulePostFinalObservation();
                  if (!ready) state.ticks.push({ kind: "check_status_pending", sentAtMs: record.responseTimeMs, sendError: record.sendError });
                  continue;
                }

                if (call.name === MAIN_TOOL_NAME || !state.mainToolCall) {
                  state.mainToolCall ??= call;
                  state.mainToolCallAt ??= now;
                  if (plan.tickMode === "async_tool_polling") {
                    const ok = sendToolResponse(
                      call,
                      {
                        status: "pending",
                        event_type: "TOOL_PENDING_STATUS",
                        phase: "pending",
                        has_final_answer: false,
                        answer_now: false,
                        job_id: jobId,
                        instruction: "Call check_status with this job_id until ready.",
                        message_to_assistant: "The tool is still running. Do not provide the final order status yet.",
                      },
                      "main_pending_tool_response_sent",
                    );
                    if (ok) state.mainToolResponseSentAt = Date.now();
                    continue;
                  }
                  if (plan.tickMode === "client_status_tick_3000ms" || plan.tickMode === "periodic_tick_4s") scheduleClientTicks();
                  if (plan.tickMode === "tick_after_utterance_0s" || plan.tickMode === "tick_after_utterance_1s") {
                    // Boundary ticks are scheduled from the first pre-final turnComplete after this tool call.
                  }
                  if (plan.tickMode === "same_call_pending_function_response_3000ms") scheduleInterimResponses(call);
                  scheduleFinalMainResponse(call);
                }
              }
            }

            if (message.toolCallCancellation?.ids?.length) {
              for (const id of message.toolCallCancellation.ids) {
                if (!state.cancelledToolCallIds.includes(id)) state.cancelledToolCallIds.push(id);
              }
              appendTimeline("tool_call_cancellation_received", { ids: message.toolCallCancellation.ids });
              appendJsonl(rawLogPath, {
                type: "tool_call_cancellation_received",
                event_ms: state.promptSentAt ? Date.now() - state.promptSentAt : null,
                ids: message.toolCallCancellation.ids,
              });
            }

            if (message.serverContent?.turnComplete) {
              state.turnCompleteAt ??= now;
              appendTimeline("turn_complete");
              if (plan.tickMode === "tick_after_utterance_0s" || plan.tickMode === "tick_after_utterance_1s") {
                maybeScheduleBoundaryTick("turn_complete");
              }
            }
          },
          onerror: (error) => {
            const summary = summarizeError(error);
            state.errors.push(summary);
            appendTimeline("socket_error", { error: summary });
            appendJsonl(rawLogPath, { type: "socket_error", error: summary });
          },
          onclose: (event: { code?: number; reason?: string }) => {
            state.sessionClosed = true;
            state.closeCode = event.code ?? null;
            state.closeReason = event.reason || null;
            appendTimeline("session_closed", { code: event.code, reason: event.reason });
            appendJsonl(rawLogPath, { type: "session_closed", code: event.code, reason: event.reason });
            clearTimeout(hardTimer);
            writeAudioFile(audioDir, audioSegments);
            resolveRun();
          },
        },
      })) as Session;
    } catch (error) {
      state.errors.push(summarizeError(error));
      appendTimeline("run_error", { error: summarizeError(error) });
      appendJsonl(rawLogPath, { type: "run_error", error: summarizeError(error) });
      clearTimeout(hardTimer);
      writeAudioFile(audioDir, audioSegments);
      resolveRun();
    }
  });

  const summary = computeSummary(plan, state);
  writeJson(resolve(plan.attemptDir, "summary.json"), summary);
  return summary;
}

function average(values: Array<number | null>): number | null {
  const nums = values.filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  if (nums.length === 0) return null;
  return Math.round(nums.reduce((sum, value) => sum + value, 0) / nums.length);
}

function summarizeCondition(condition: TickMode, attempts: AttemptSummary[]): ConditionSummary {
  return {
    condition,
    attempts: attempts.length,
    valid_attempts: attempts.filter((attempt) => attempt.session_valid).length,
    close_1008_count: attempts.filter((attempt) => attempt.close_1008).length,
    close_1011_count: attempts.filter((attempt) => attempt.close_1011).length,
    client_send_error_count: attempts.reduce((sum, attempt) => sum + attempt.client_send_error_count, 0),
    tool_call_success_count: attempts.filter((attempt) => attempt.tool_call_success).length,
    final_tool_response_sent_count: attempts.filter((attempt) => attempt.final_tool_response_sent).length,
    tick_send_success_count: attempts.reduce((sum, attempt) => sum + attempt.tick_send_success_count, 0),
    waiting_audio_before_final_tool_result_count: attempts.filter((attempt) => attempt.waiting_audio_before_final_tool_result).length,
    audio_after_tick_count: attempts.filter((attempt) => attempt.audio_after_tick).length,
    premature_final_answer_count: attempts.filter((attempt) => attempt.premature_final_answer).length,
    post_tool_final_answer_count: attempts.filter((attempt) => attempt.post_tool_final_answer).length,
    avg_post_tool_final_latency_ms: average(attempts.map((attempt) => attempt.post_tool_final_latency_ms)),
    avg_first_audio_time_ms: average(attempts.map((attempt) => attempt.first_audio_time_ms)),
    avg_turnComplete_time_ms: average(attempts.map((attempt) => attempt.turnComplete_time_ms)),
    avg_tool_call_time_ms: average(attempts.map((attempt) => attempt.tool_call_time_ms)),
    avg_final_tool_response_sent_time_ms: average(attempts.map((attempt) => attempt.final_tool_response_sent_time_ms)),
    avg_last_audio_before_final_tool_response_ms: average(attempts.map((attempt) => attempt.last_audio_before_final_tool_response_ms)),
  };
}

function csvEscape(value: unknown): string {
  if (value === null || value === undefined) return "";
  const text = Array.isArray(value) ? value.join(" | ") : String(value);
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function writeCsv(path: string, rows: Record<string, unknown>[]): void {
  if (rows.length === 0) return writeFileSync(path, "\n", "utf8");
  const headers = Object.keys(rows[0]);
  const lines = [headers.join(",")];
  for (const row of rows) lines.push(headers.map((header) => csvEscape(row[header])).join(","));
  writeFileSync(path, `${lines.join("\n")}\n`, "utf8");
}

function writeFinalMarkdown(path: string, rows: ConditionSummary[], resultDir: string): void {
  const lines = [
    "# Gemini Live native tool-call tick factor probe",
    "",
    `Result folder: ${resultDir}`,
    "",
    "| condition | attempts | 1008 | 1011 | send_error | waiting_audio | audio_after_tick | post_tool_final | premature_final |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ...rows.map(
      (row) =>
        `| ${row.condition} | ${row.attempts} | ${row.close_1008_count} | ${row.close_1011_count} | ${row.client_send_error_count} | ${row.waiting_audio_before_final_tool_result_count} | ${row.audio_after_tick_count} | ${row.post_tool_final_answer_count} | ${row.premature_final_answer_count} |`,
    ),
    "",
    "Interpretation:",
    "",
    "This is a feasibility probe, not a benchmark. Compare conditions by whether pending/tick injection was accepted, whether the model produced audio after ticks, and whether 1008 closes increased. Treat duplicate/interim function responses as a protocol stress test; send errors there are expected evidence about feasibility, not necessarily model failures.",
    "",
  ];
  writeFileSync(path, `${lines.join("\n")}\n`, "utf8");
}

async function runWithConcurrency<T, R>(items: T[], worker: (item: T) => Promise<R>): Promise<R[]> {
  const results: R[] = [];
  for (let index = 0; index < items.length; index += 1) results[index] = await worker(items[index]);
  return results;
}

function printTable(rows: ConditionSummary[]): void {
  const headers = ["condition", "attempts", "valid_attempts", "close_1008_count", "close_1011_count", "client_send_error_count"];
  const cells = rows.map((row) => headers.map((header) => String(row[header as keyof ConditionSummary])));
  const widths = headers.map((header, index) => Math.max(header.length, ...cells.map((row) => row[index].length)));
  console.log(headers.map((header, index) => header.padEnd(widths[index])).join(" | "));
  console.log(widths.map((width) => "-".repeat(width)).join("-+-"));
  for (const row of cells) console.log(row.map((cell, index) => cell.padEnd(widths[index])).join(" | "));
}

function runPostprocess(resultDir: string): void {
  const scriptPath = resolve(PROJECT_DIR, "scripts", "postprocess_tau_live_tool_tick_factor_probe.py");
  const result = spawnSync("python3", [scriptPath, resultDir], {
    cwd: PROJECT_DIR,
    encoding: "utf8",
    stdio: "inherit",
  });
  if (result.error) {
    console.warn(`Postprocess skipped: ${summarizeError(result.error)}`);
    return;
  }
  if (result.status !== 0) {
    console.warn(`Postprocess exited with status ${result.status ?? "unknown"}.`);
  }
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const apiKey = requireEnv("GEMINI_API_KEY");
  const model = requireEnv("GEMINI_LIVE_MODEL");
  const uniqueRunSuffix = `${process.pid}_${randomUUID().slice(0, 8)}`;
  const resultId = `${timestampForPath()}_${uniqueRunSuffix}_tau_live_tool_tick_factor_probe`;
  const resultDir = resolve(RESULT_DIR, resultId);
  mkdirSync(resultDir, { recursive: true });

  writeJson(resolve(resultDir, "config.json"), {
    result_id: resultId,
    model,
    latency_ms: args.latencyMs,
    attempts_per_condition: args.attemptsPerCondition,
    concurrency: 1,
    close_strategy: "post_final_observation_or_timeout",
    tick_modes: args.tickModes,
    tick_mode_labels: Object.fromEntries(args.tickModes.map((mode) => [mode, TICK_MODE_LABELS[mode]])),
    default_tick_modes_note:
      "Default conditions intentionally exclude async_tool_polling: no tick, tick via same-call tool response, and tick via client content.",
    prompt_routing: {
      native_no_tick: NO_TICK_PROMPT_NAME,
      client_status_tick_3000ms: EXTERNAL_TICK_PROMPT_NAME,
      periodic_tick_4s: EXTERNAL_TICK_PROMPT_NAME,
      tick_after_utterance_0s: EXTERNAL_TICK_PROMPT_NAME,
      tick_after_utterance_1s: EXTERNAL_TICK_PROMPT_NAME,
      tick_after_audio_idle_0s: EXTERNAL_TICK_PROMPT_NAME,
      tick_after_audio_idle_1s: EXTERNAL_TICK_PROMPT_NAME,
      tick_after_audio_idle_repeat_0s: EXTERNAL_TICK_PROMPT_NAME,
      tick_after_audio_idle_repeat_1s: EXTERNAL_TICK_PROMPT_NAME,
      same_call_pending_function_response_3000ms: WITH_TICK_PROMPT_NAME,
      async_tool_polling: WITH_TICK_PROMPT_NAME,
    },
    prompts: {
      [NO_TICK_PROMPT_NAME]: {
        system_instruction: NO_TICK_SYSTEM_INSTRUCTION,
        user_prompt: NO_TICK_USER_PROMPT,
      },
      [EXTERNAL_TICK_PROMPT_NAME]: {
        system_instruction: EXTERNAL_TICK_SYSTEM_INSTRUCTION,
        user_prompt: EXTERNAL_TICK_USER_PROMPT,
      },
      [WITH_TICK_PROMPT_NAME]: {
        system_instruction: WITH_TICK_SYSTEM_INSTRUCTION,
        user_prompt: WITH_TICK_USER_PROMPT,
      },
    },
  });

  const plans: AttemptPlan[] = [];
  for (const tickMode of args.tickModes) {
    const conditionDir = resolve(resultDir, `condition_${tickMode}`);
    mkdirSync(conditionDir, { recursive: true });
    for (let attemptIndex = 1; attemptIndex <= args.attemptsPerCondition; attemptIndex += 1) {
      plans.push({
        tickMode,
        latencyMs: args.latencyMs,
        attemptIndex,
        attemptDir: resolve(conditionDir, `attempt_${String(attemptIndex).padStart(4, "0")}`),
        quietTerminalText: args.quietTerminalText,
      });
    }
  }

  console.log(`Result directory: ${relative(PROJECT_DIR, resultDir)}`);
  console.log(`Attempts: ${plans.length}; concurrency: 1`);
  const ai = new GoogleGenAI({ apiKey });
  const attempts = await runWithConcurrency(plans, async (plan) => {
    console.log(`[${plan.tickMode} attempt ${plan.attemptIndex}] start`);
    const result = await runOne(ai, model, plan);
    console.log(
      `[${plan.tickMode} attempt ${plan.attemptIndex}] ${
        result.session_valid ? "valid" : result.close_1008 ? "1008" : result.close_1011 ? "1011" : "not_valid"
      }`,
    );
    return result;
  });

  const rows = args.tickModes.map((mode) => summarizeCondition(mode, attempts.filter((attempt) => attempt.condition === mode)));
  for (const row of rows) {
    const conditionDir = resolve(resultDir, `condition_${row.condition}`);
    const conditionAttempts = attempts.filter((attempt) => attempt.condition === row.condition);
    writeJson(resolve(conditionDir, "summary.json"), { ...row, attempts: conditionAttempts });
    writeCsv(resolve(conditionDir, "summary.csv"), [row as unknown as Record<string, unknown>]);
  }
  writeJson(resolve(resultDir, "summary.json"), { result_id: resultId, model, rows, attempts });
  writeCsv(resolve(resultDir, "summary.csv"), rows as unknown as Record<string, unknown>[]);
  writeFinalMarkdown(resolve(resultDir, "final_comparison.md"), rows, relative(PROJECT_DIR, resultDir));
  runPostprocess(resultDir);
  printTable(rows);
  console.log(`Summary: ${relative(PROJECT_DIR, resolve(resultDir, "summary.json"))}`);
  console.log(`Final comparison: ${relative(PROJECT_DIR, resolve(resultDir, "final_comparison.md"))}`);
}

main().catch((error) => {
  console.error("tau-live-tool-tick-factor-probe failed");
  console.error(summarizeError(error));
  process.exitCode = 1;
});
