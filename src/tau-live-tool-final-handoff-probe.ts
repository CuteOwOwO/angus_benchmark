import { Behavior, GoogleGenAI, Modality } from "@google/genai";
import { config as loadEnv } from "dotenv";
import { spawnSync } from "node:child_process";
import { appendFileSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  PROMPT_NAME as WITH_TICK_PROMPT_NAME,
  SYSTEM_INSTRUCTION as WITH_TICK_SYSTEM_INSTRUCTION,
  USER_PROMPT as WITH_TICK_USER_PROMPT,
} from "./prompts/taw-with-tick.js";

const PROJECT_DIR = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const DEFAULT_ENV_FILE = resolve(PROJECT_DIR, "..", "angus_bench", ".env");
const RESULT_DIR = resolve(PROJECT_DIR, "result");
const MAIN_TOOL_NAME = "get_order_details";
const DEFAULT_LATENCY_MS = 8000;
const DEFAULT_ATTEMPTS_PER_VARIANT = 5;
const MAX_ATTEMPT_MS = 45_000;
const TICK_TIMES_MS = [3000, 6000];
const FINAL_READY_CLIENT_DELAY_MS = 300;
const AUDIO_IDLE_BEFORE_FINAL_MS = 700;
const BOUNDARY_FINAL_MAX_EXTRA_WAIT_MS = 10_000;

const FINAL_ORDER_RESPONSE = {
  order_id: "#A123",
  status: "shipped",
  estimated_delivery: "tomorrow",
  carrier: "UPS",
  tracking_number: "1Z999AA10123456784",
};

type FinalHandoffVariant =
  | "final_function_response_only"
  | "final_function_response_plus_ready_client_message"
  | "boundary_final_function_response";

const DEFAULT_VARIANTS: FinalHandoffVariant[] = [
  "final_function_response_only",
  "final_function_response_plus_ready_client_message",
  "boundary_final_function_response",
];

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

type AudioSegment = {
  offsetMs: number;
  chunk: Buffer;
  mimeType?: string;
};

type Args = {
  latencyMs: number;
  attemptsPerVariant: number;
  variants: FinalHandoffVariant[];
};

type AttemptPlan = {
  variant: FinalHandoffVariant;
  latencyMs: number;
  attemptIndex: number;
  attemptDir: string;
};

type AttemptState = {
  promptSentAt?: number;
  closeCode: number | null;
  closeReason: string | null;
  sessionClosed: boolean;
  mainToolCall?: FunctionCall;
  mainToolCallAt?: number;
  finalFunctionResponseSentAt?: number;
  finalReadyClientMessageSentAt?: number;
  firstServerEventAfterFinalAt?: number;
  firstAudioAfterFinalAt?: number;
  firstTextOrTranscriptionAfterFinalAt?: number;
  firstAudioAt?: number;
  lastAudioAt?: number;
  lastAudioBeforeFinalAt?: number;
  turnCompleteAt?: number;
  turnCompleteBeforeFinal: boolean;
  turnCompleteAfterFinal: boolean;
  anyModelEventAfterFinal: boolean;
  sessionClosesBeforeAnyPostFinalModelEvent: boolean;
  pendingEventSentTimesMs: number[];
  rawEventCount: number;
  audioTimesMs: number[];
  textBeforeFinal: string[];
  textAfterFinal: string[];
  sendErrors: string[];
  errors: string[];
};

type AttemptSummary = {
  condition: FinalHandoffVariant;
  variant: FinalHandoffVariant;
  latency_ms: number;
  attempt_index: number;
  session_valid: boolean;
  close_1008: boolean;
  close_1011: boolean;
  close_code: number | null;
  close_reason: string | null;
  tool_call_success: boolean;
  tool_call_time_ms: number | null;
  pending_event_sent_times_ms: number[];
  final_function_response_sent: boolean;
  final_function_response_sent_time_ms: number | null;
  final_ready_client_message_sent_time_ms: number | null;
  first_server_event_after_final_time_ms: number | null;
  first_audio_after_final_time_ms: number | null;
  first_text_or_transcription_after_final_time_ms: number | null;
  any_model_event_after_final: boolean;
  turnComplete_occurs_before_final: boolean;
  turnComplete_occurs_after_final: boolean;
  session_closes_before_any_post_final_model_event: boolean;
  post_tool_final_answer: boolean;
  text_after_final: string[];
  text_before_final: string[];
  raw_event_count: number;
  send_errors: string[];
  errors: string[];
  result_dir: string;
};

type VariantSummary = {
  condition: FinalHandoffVariant;
  attempts: number;
  valid_attempts: number;
  close_1008_count: number;
  close_1011_count: number;
  tool_call_success_count: number;
  final_function_response_sent_count: number;
  final_ready_client_message_sent_count: number;
  any_model_event_after_final_count: number;
  first_audio_after_final_count: number;
  first_text_or_transcription_after_final_count: number;
  post_tool_final_answer_count: number;
  turnComplete_before_final_count: number;
  turnComplete_after_final_count: number;
  session_closes_before_any_post_final_model_event_count: number;
  avg_first_server_event_after_final_time_ms: number | null;
  avg_first_audio_after_final_time_ms: number | null;
  avg_first_text_or_transcription_after_final_time_ms: number | null;
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

function parseCsvVariants(value: string | undefined): FinalHandoffVariant[] {
  if (!value) return DEFAULT_VARIANTS;
  const variants = value.split(",").map((part) => part.trim()).filter(Boolean);
  for (const variant of variants) {
    if (!DEFAULT_VARIANTS.includes(variant as FinalHandoffVariant)) throw new Error(`Unknown final handoff variant: ${variant}`);
  }
  return variants as FinalHandoffVariant[];
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
    attemptsPerVariant: Math.max(
      1,
      Number(args.get("attempts-per-variant") ?? args.get("attempts-per-condition") ?? args.get("attempts") ?? DEFAULT_ATTEMPTS_PER_VARIANT),
    ),
    variants: parseCsvVariants(args.get("variants")),
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

function makeTools(): unknown[] {
  return [
    {
      functionDeclarations: [
        {
          name: MAIN_TOOL_NAME,
          description: "Get the status and details of a retail order.",
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
        },
      ],
    },
  ];
}

function finalAnswerText(): string {
  return `Order #A123 has shipped. The carrier is UPS, the tracking number is 1Z999AA10123456784, and the estimated delivery is tomorrow.`;
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
  return `[TOOL_PENDING_STATUS]\n${JSON.stringify(pendingPayload(), null, 2)}`;
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

function finalReadyClientMessage(): string {
  return [
    "[TOOL_FINAL_RESULT_READY]",
    "The final tool result is now available. Stop waiting and answer the user now based only on the final tool result.",
  ].join("\n");
}

function isFinalAnswerText(texts: string[]): boolean {
  const text = texts.join(" ").toLowerCase().replace(/[^a-z0-9#]+/g, "");
  return /shipped/.test(text) && /tomorrow/.test(text) && /ups/.test(text);
}

function computeSummary(plan: AttemptPlan, state: AttemptState): AttemptSummary {
  const close1008 = state.closeCode === 1008 || state.closeReason?.includes("1008") === true;
  const close1011 = state.closeCode === 1011 || state.closeReason?.includes("1011") === true;
  const postToolFinalAnswer = isFinalAnswerText(state.textAfterFinal);
  return {
    condition: plan.variant,
    variant: plan.variant,
    latency_ms: plan.latencyMs,
    attempt_index: plan.attemptIndex,
    session_valid: !close1008 && !close1011 && Boolean(state.finalFunctionResponseSentAt) && postToolFinalAnswer,
    close_1008: close1008,
    close_1011: close1011,
    close_code: state.closeCode,
    close_reason: state.closeReason,
    tool_call_success: Boolean(state.mainToolCallAt),
    tool_call_time_ms: msDelta(state.promptSentAt, state.mainToolCallAt),
    pending_event_sent_times_ms: state.pendingEventSentTimesMs,
    final_function_response_sent: Boolean(state.finalFunctionResponseSentAt),
    final_function_response_sent_time_ms: msDelta(state.promptSentAt, state.finalFunctionResponseSentAt),
    final_ready_client_message_sent_time_ms: msDelta(state.promptSentAt, state.finalReadyClientMessageSentAt),
    first_server_event_after_final_time_ms: msDelta(state.promptSentAt, state.firstServerEventAfterFinalAt),
    first_audio_after_final_time_ms: msDelta(state.promptSentAt, state.firstAudioAfterFinalAt),
    first_text_or_transcription_after_final_time_ms: msDelta(state.promptSentAt, state.firstTextOrTranscriptionAfterFinalAt),
    any_model_event_after_final: state.anyModelEventAfterFinal,
    turnComplete_occurs_before_final: state.turnCompleteBeforeFinal,
    turnComplete_occurs_after_final: state.turnCompleteAfterFinal,
    session_closes_before_any_post_final_model_event: state.sessionClosesBeforeAnyPostFinalModelEvent,
    post_tool_final_answer: postToolFinalAnswer,
    text_after_final: state.textAfterFinal,
    text_before_final: state.textBeforeFinal,
    raw_event_count: state.rawEventCount,
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
  const state: AttemptState = {
    closeCode: null,
    closeReason: null,
    sessionClosed: false,
    turnCompleteBeforeFinal: false,
    turnCompleteAfterFinal: false,
    anyModelEventAfterFinal: false,
    sessionClosesBeforeAnyPostFinalModelEvent: false,
    pendingEventSentTimesMs: [],
    rawEventCount: 0,
    audioTimesMs: [],
    textBeforeFinal: [],
    textAfterFinal: [],
    sendErrors: [],
    errors: [],
  };
  let session: Session | undefined;
  let done = false;
  const timers: ReturnType<typeof setTimeout>[] = [];

  writeJson(resolve(plan.attemptDir, "config.json"), {
    condition: plan.variant,
    variant: plan.variant,
    tick_mode: "with_tick_external",
    latency_ms: plan.latencyMs,
    attempt_index: plan.attemptIndex,
    close_strategy: "turnComplete_or_timeout",
    prompt_name: WITH_TICK_PROMPT_NAME,
    system_instruction: WITH_TICK_SYSTEM_INSTRUCTION,
    user_prompt: WITH_TICK_USER_PROMPT,
    model,
    pending_payload: pendingPayload(),
    final_payload: finalPayload(),
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

  const sendFinalResponse = (call: FunctionCall) => {
    if (done || state.sessionClosed || state.finalFunctionResponseSentAt) return;
    if (sendToolResponse(call, finalPayload(), "final_tool_response_sent")) {
      state.finalFunctionResponseSentAt = Date.now();
    }
    if (plan.variant === "final_function_response_plus_ready_client_message") {
      timers.push(
        setTimeout(() => {
          if (done || state.sessionClosed || !state.finalFunctionResponseSentAt) return;
          try {
            session?.sendClientContent({ turns: finalReadyClientMessage(), turnComplete: true });
            state.finalReadyClientMessageSentAt = Date.now();
            appendTimeline("final_ready_client_message_sent", { message: finalReadyClientMessage() });
            appendJsonl(rawLogPath, {
              type: "final_ready_client_message_sent",
              event_ms: state.promptSentAt ? Date.now() - state.promptSentAt : null,
              message: finalReadyClientMessage(),
            });
          } catch (error) {
            noteSendError("final_ready_client_message", error);
          }
        }, FINAL_READY_CLIENT_DELAY_MS),
      );
    }
  };

  const scheduleFinalResponse = (call: FunctionCall) => {
    timers.push(
      setTimeout(() => {
        if (plan.variant !== "boundary_final_function_response") {
          sendFinalResponse(call);
          return;
        }
        const deadline = Date.now() + BOUNDARY_FINAL_MAX_EXTRA_WAIT_MS;
        const waitForIdle = () => {
          if (done || state.sessionClosed || state.finalFunctionResponseSentAt) return;
          const lastAudioAt = state.lastAudioAt ?? 0;
          const idleMs = lastAudioAt ? Date.now() - lastAudioAt : Number.POSITIVE_INFINITY;
          if (idleMs >= AUDIO_IDLE_BEFORE_FINAL_MS || Date.now() >= deadline) {
            appendTimeline("audio_idle_boundary_reached", { idle_ms: Number.isFinite(idleMs) ? idleMs : null });
            sendFinalResponse(call);
            return;
          }
          timers.push(setTimeout(waitForIdle, 100));
        };
        waitForIdle();
      }, plan.latencyMs),
    );
  };

  const scheduleClientPendingTicks = () => {
    for (const tickMs of TICK_TIMES_MS.filter((value) => value < plan.latencyMs)) {
      timers.push(
        setTimeout(() => {
          if (done || state.sessionClosed) return;
          const sentAtMs = Date.now() - (state.promptSentAt ?? Date.now());
          try {
            session?.sendClientContent({ turns: clientPendingMessage(), turnComplete: true });
            state.pendingEventSentTimesMs.push(sentAtMs);
            appendTimeline("client_status_tick_sent", { tick_ms: tickMs, payload: pendingPayload(), message: clientPendingMessage() });
            appendJsonl(rawLogPath, {
              type: "client_status_tick_sent",
              event_ms: sentAtMs,
              tick_ms: tickMs,
              payload: pendingPayload(),
              message: clientPendingMessage(),
            });
          } catch (error) {
            noteSendError("client_status_tick", error);
          }
        }, tickMs),
      );
    }
  };

  await new Promise<void>(async (resolveRun) => {
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
          systemInstruction: WITH_TICK_SYSTEM_INSTRUCTION,
          tools: makeTools() as any,
        },
        callbacks: {
          onopen: () => {
            appendTimeline("session_opened", { model });
            appendJsonl(rawLogPath, { type: "session_opened", model });
          },
          onmessage: (message: LiveMessage) => {
            const now = Date.now();
            state.rawEventCount += 1;
            const eventMs = state.promptSentAt ? now - state.promptSentAt : null;
            if (state.finalFunctionResponseSentAt) {
              state.anyModelEventAfterFinal = true;
              state.firstServerEventAfterFinalAt ??= now;
            }
            appendJsonl(rawLogPath, { type: "server_event", event_ms: eventMs, event_types: eventTypes(message), message: sanitizeMessage(message) });
            appendTimeline("server_event", { event_types: eventTypes(message) });

            const parts = message.serverContent?.modelTurn?.parts ?? [];
            const textPieces = [
              ...parts.map((part) => part.text).filter((text): text is string => Boolean(text)),
              message.serverContent?.outputTranscription?.text,
            ].filter((text): text is string => Boolean(text));
            for (const text of textPieces) {
              if (state.finalFunctionResponseSentAt) {
                state.textAfterFinal.push(text);
                state.firstTextOrTranscriptionAfterFinalAt ??= now;
              } else {
                state.textBeforeFinal.push(text);
              }
              appendTimeline(message.serverContent?.outputTranscription?.text === text ? "output_transcription" : "text_output", { text });
            }
            for (const part of parts) {
              if (!part.inlineData?.data) continue;
              const chunk = Buffer.from(part.inlineData.data, "base64");
              audioSegments.push({ offsetMs: eventMs ?? 0, chunk, mimeType: part.inlineData.mimeType });
              state.lastAudioAt = now;
              state.firstAudioAt ??= now;
              if (eventMs !== null) state.audioTimesMs.push(eventMs);
              if (state.finalFunctionResponseSentAt) {
                state.firstAudioAfterFinalAt ??= now;
              } else {
                state.lastAudioBeforeFinalAt = now;
              }
              appendTimeline("audio_output", {
                bytes: chunk.length,
                mime_type: part.inlineData.mimeType,
                phase: state.finalFunctionResponseSentAt ? "after_tool_response" : "before_tool_response",
              });
            }

            if (message.toolCall?.functionCalls?.length) {
              appendTimeline("tool_call_received", { function_calls: message.toolCall.functionCalls });
              appendJsonl(rawLogPath, { type: "tool_call_received", function_calls: message.toolCall.functionCalls });
              for (const call of message.toolCall.functionCalls) {
                if (call.name === MAIN_TOOL_NAME || !state.mainToolCall) {
                  state.mainToolCall ??= call;
                  state.mainToolCallAt ??= now;
                  scheduleClientPendingTicks();
                  scheduleFinalResponse(call);
                }
              }
            }

            if (message.serverContent?.turnComplete) {
              state.turnCompleteAt ??= now;
              if (state.finalFunctionResponseSentAt) state.turnCompleteAfterFinal = true;
              else state.turnCompleteBeforeFinal = true;
              appendTimeline("turn_complete", { phase: state.finalFunctionResponseSentAt ? "after_final" : "before_final" });
              if (state.finalFunctionResponseSentAt) {
                clearTimeout(hardTimer);
                finish(resolveRun);
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
            state.sessionClosesBeforeAnyPostFinalModelEvent = Boolean(state.finalFunctionResponseSentAt && !state.anyModelEventAfterFinal);
            appendTimeline("session_closed", { code: event.code, reason: event.reason });
            appendJsonl(rawLogPath, { type: "session_closed", code: event.code, reason: event.reason });
            clearTimeout(hardTimer);
            writeAudioFile(audioDir, audioSegments);
            resolveRun();
          },
        },
      })) as Session;

      session.sendClientContent({ turns: WITH_TICK_USER_PROMPT, turnComplete: true });
      state.promptSentAt = Date.now();
      appendTimeline("user_message_sent", { prompt_name: WITH_TICK_PROMPT_NAME, prompt: WITH_TICK_USER_PROMPT });
      appendJsonl(rawLogPath, { type: "user_message_sent", prompt_name: WITH_TICK_PROMPT_NAME, prompt: WITH_TICK_USER_PROMPT });
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

function summarizeVariant(condition: FinalHandoffVariant, attempts: AttemptSummary[]): VariantSummary {
  return {
    condition,
    attempts: attempts.length,
    valid_attempts: attempts.filter((attempt) => attempt.session_valid).length,
    close_1008_count: attempts.filter((attempt) => attempt.close_1008).length,
    close_1011_count: attempts.filter((attempt) => attempt.close_1011).length,
    tool_call_success_count: attempts.filter((attempt) => attempt.tool_call_success).length,
    final_function_response_sent_count: attempts.filter((attempt) => attempt.final_function_response_sent).length,
    final_ready_client_message_sent_count: attempts.filter((attempt) => attempt.final_ready_client_message_sent_time_ms !== null).length,
    any_model_event_after_final_count: attempts.filter((attempt) => attempt.any_model_event_after_final).length,
    first_audio_after_final_count: attempts.filter((attempt) => attempt.first_audio_after_final_time_ms !== null).length,
    first_text_or_transcription_after_final_count: attempts.filter((attempt) => attempt.first_text_or_transcription_after_final_time_ms !== null).length,
    post_tool_final_answer_count: attempts.filter((attempt) => attempt.post_tool_final_answer).length,
    turnComplete_before_final_count: attempts.filter((attempt) => attempt.turnComplete_occurs_before_final).length,
    turnComplete_after_final_count: attempts.filter((attempt) => attempt.turnComplete_occurs_after_final).length,
    session_closes_before_any_post_final_model_event_count: attempts.filter((attempt) => attempt.session_closes_before_any_post_final_model_event).length,
    avg_first_server_event_after_final_time_ms: average(attempts.map((attempt) => attempt.first_server_event_after_final_time_ms)),
    avg_first_audio_after_final_time_ms: average(attempts.map((attempt) => attempt.first_audio_after_final_time_ms)),
    avg_first_text_or_transcription_after_final_time_ms: average(attempts.map((attempt) => attempt.first_text_or_transcription_after_final_time_ms)),
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

function writeFinalMarkdown(path: string, rows: VariantSummary[], resultDir: string): void {
  const lines = [
    "# Gemini Live final handoff debug probe",
    "",
    `Result folder: ${resultDir}`,
    "",
    "| variant | attempts | 1008 | final_sent | post_final_model_event | post_final_audio | post_final_text | post_tool_final | close_before_post_final_event |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ...rows.map(
      (row) =>
        `| ${row.condition} | ${row.attempts} | ${row.close_1008_count} | ${row.final_function_response_sent_count} | ${row.any_model_event_after_final_count} | ${row.first_audio_after_final_count} | ${row.first_text_or_transcription_after_final_count} | ${row.post_tool_final_answer_count} | ${row.session_closes_before_any_post_final_model_event_count} |`,
    ),
    "",
    "Interpretation:",
    "",
    "This is a small handoff debug run for the native tool-call path. It isolates whether stronger pending/final payloads and final handoff variants produce any model event, audio, or text after the final tool result. Do not treat these 5-attempt cells as final stability estimates.",
    "",
  ];
  writeFileSync(path, `${lines.join("\n")}\n`, "utf8");
}

async function runWithConcurrency<T, R>(items: T[], worker: (item: T) => Promise<R>): Promise<R[]> {
  const results: R[] = [];
  for (let index = 0; index < items.length; index += 1) results[index] = await worker(items[index]);
  return results;
}

function printTable(rows: VariantSummary[]): void {
  const headers = ["condition", "attempts", "valid_attempts", "close_1008_count", "any_model_event_after_final_count", "post_tool_final_answer_count"];
  const cells = rows.map((row) => headers.map((header) => String(row[header as keyof VariantSummary])));
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
  const resultId = `${timestampForPath()}_tau_live_tool_final_handoff_probe`;
  const resultDir = resolve(RESULT_DIR, resultId);
  mkdirSync(resultDir, { recursive: true });

  writeJson(resolve(resultDir, "config.json"), {
    result_id: resultId,
    model,
    latency_ms: args.latencyMs,
    attempts_per_variant: args.attemptsPerVariant,
    concurrency: 1,
    tick_mode: "with_tick_external",
    variants: args.variants,
    prompt_name: WITH_TICK_PROMPT_NAME,
    system_instruction: WITH_TICK_SYSTEM_INSTRUCTION,
    user_prompt: WITH_TICK_USER_PROMPT,
    pending_payload: pendingPayload(),
    final_payload: finalPayload(),
  });

  const plans: AttemptPlan[] = [];
  for (const variant of args.variants) {
    const conditionDir = resolve(resultDir, `condition_${variant}`);
    mkdirSync(conditionDir, { recursive: true });
    for (let attemptIndex = 1; attemptIndex <= args.attemptsPerVariant; attemptIndex += 1) {
      plans.push({
        variant,
        latencyMs: args.latencyMs,
        attemptIndex,
        attemptDir: resolve(conditionDir, `attempt_${String(attemptIndex).padStart(4, "0")}`),
      });
    }
  }

  console.log(`Result directory: ${relative(PROJECT_DIR, resultDir)}`);
  console.log(`Attempts: ${plans.length}; concurrency: 1`);
  const ai = new GoogleGenAI({ apiKey });
  const attempts = await runWithConcurrency(plans, async (plan) => {
    console.log(`[${plan.variant} attempt ${plan.attemptIndex}] start`);
    const result = await runOne(ai, model, plan);
    console.log(
      `[${plan.variant} attempt ${plan.attemptIndex}] ${
        result.session_valid ? "valid" : result.close_1008 ? "1008" : result.close_1011 ? "1011" : "not_valid"
      }`,
    );
    return result;
  });

  const rows = args.variants.map((variant) => summarizeVariant(variant, attempts.filter((attempt) => attempt.condition === variant)));
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
  console.error("tau-live-tool-final-handoff-probe failed");
  console.error(summarizeError(error));
  process.exitCode = 1;
});
