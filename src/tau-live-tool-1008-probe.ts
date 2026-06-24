import { Behavior, GoogleGenAI, Modality } from "@google/genai";
import { config as loadEnv } from "dotenv";
import { appendFileSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const PROJECT_DIR = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const DEFAULT_ENV_FILE = resolve(PROJECT_DIR, "..", "angus_bench", ".env");
const RESULT_DIR = resolve(PROJECT_DIR, "result");
const TOOL_NAME = "get_order_details";
const DEFAULT_LATENCIES = [3000, 5000, 8000, 12000];
const DEFAULT_ATTEMPTS_PER_LATENCY = 5;
const DEFAULT_CONCURRENCY = 1;
const MAX_ATTEMPT_MS = 45_000;
const POST_TOOL_WAIT_MS = 8000;

const ORIGINAL_SYSTEM_INSTRUCTION = `You are a helpful voice assistant.

The user asks a task that requires using the provided tool.
You must call the tool to get the answer.
Before the tool result arrives, do not guess the result.
While waiting for the tool result, give short task-aware waiting updates.
After the tool result arrives, answer based only on the tool result.`;

const ORIGINAL_USER_PROMPT = "Can you check the status of my order #A123?";

const EXPLICIT_SYSTEM_INSTRUCTION = `You are a helpful voice assistant.

The user asks a task that requires using the provided tool.
You must call the tool to get the answer.
Before the tool result arrives, do not guess the result.
While waiting for the tool result, keep speaking in short task-aware updates such as "I'm checking that order now" or "I'm still waiting for the order details." Do not go silent during the wait.
After the tool result arrives, you must give a final spoken answer based only on the tool result.
The final answer should directly tell the user the order status and estimated delivery.`;

const EXPLICIT_USER_PROMPT =
  "Can you check the status of my order #A123? Please keep me briefly updated while you check, and then tell me the final status once you have it.";

const TOOL_RESPONSE = {
  order_id: "#A123",
  status: "shipped",
  estimated_delivery: "tomorrow",
  carrier: "UPS",
  tracking_number: "1Z999AA10123456784",
};

const TAU_INSPECTION = {
  repo_path: "/user_data/tau-bench",
  repo_url: "https://github.com/sierra-research/tau-bench.git",
  inspected: [
    "README.md",
    "tau_bench/envs/retail/env.py",
    "tau_bench/envs/retail/tools/get_order_details.py",
    "tau_bench/envs/tool.py",
    "retail and airline domain directories",
  ],
  chosen_tool_basis:
    "retail get_order_details: Get the status and details of an order; parameter order_id string.",
  schema_kind: "tau-bench-style mock tool using Gemini Live native tool declaration",
};

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
    inputTranscription?: { text?: string };
    outputTranscription?: { text?: string };
    turnComplete?: boolean;
    interrupted?: boolean;
    waitingForInput?: boolean;
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

type Args = {
  latencies: number[];
  attemptsPerLatency: number;
  concurrency: number;
  promptMode: PromptMode;
  closeMode: CloseMode;
};

type PromptMode = "original" | "explicit";
type CloseMode = "turn_complete_or_timeout" | "post_wait_only";

type PromptConfig = {
  promptMode: PromptMode;
  systemInstruction: string;
  userPrompt: string;
};

type AttemptPlan = {
  latencyMs: number;
  attemptIndex: number;
  attemptDir: string;
};

type AudioSegment = {
  offsetMs: number;
  chunk: Buffer;
  mimeType?: string;
};

type AttemptState = {
  openedAt?: number;
  promptSentAt?: number;
  firstAudioAt?: number;
  firstTextAt?: number;
  firstOutputAt?: number;
  toolCallAt?: number;
  toolResponseSentAt?: number;
  firstAudioAfterToolResponseAt?: number;
  firstOutputAfterToolResponseAt?: number;
  closeCode: number | null;
  closeReason: string | null;
  sessionClosed: boolean;
  toolCallPending: boolean;
  toolCallEmitted: boolean;
  toolResponseSent: boolean;
  toolResponseSkippedAfterClose: boolean;
  toolCallCount: number;
  audioOutputCountBeforeToolResponse: number;
  audioOutputCountAfterToolResponse: number;
  textOutputEvents: string[];
  textOutputEventsAfterToolResponse: string[];
  audioOutputEvents: number;
  rawEventCount: number;
  errors: string[];
  turnCompleteAfterToolResponse: boolean;
};

type AttemptSummary = {
  latency_ms: number;
  attempt_index: number;
  session_valid: boolean;
  server_1008_error: boolean;
  server_1011_error: boolean;
  other_error: string | null;
  close_code: number | null;
  close_reason: string | null;
  tool_call_emitted: boolean;
  tool_call_time_ms: number | null;
  tool_response_sent: boolean;
  tool_response_time_ms: number | null;
  first_audio_output_time_ms: number | null;
  has_audio_before_tool_response: boolean;
  audio_output_count_before_tool_response: number;
  has_audio_after_tool_response: boolean;
  post_tool_first_audio_latency_ms: number | null;
  has_final_answer_after_tool_response: boolean;
  text_output_events: string[];
  text_output_events_after_tool_response: string[];
  audio_output_events: number;
  raw_event_count: number;
  result_dir: string;
};

type LatencySummary = {
  latency_ms: number;
  attempts: number;
  valid: number;
  server_1008: number;
  server_1011: number;
  tool_call_emitted: number;
  tool_response_sent: number;
  waiting_audio: number;
  post_tool_audio: number;
  post_tool_answer: number;
};

function requireEnv(name: string): string {
  const value = process.env[name]?.trim();
  if (!value) throw new Error(`Missing ${name}. Set it in /user_data/angus_bench/.env or GEMINI_ENV_FILE.`);
  return value;
}

function timestampForPath(date = new Date()): string {
  const [datePart, timePart] = date.toISOString().split("T");
  return `${datePart}_${timePart.replace("Z", "").replace(/\./g, "-").replace(/:/g, "-")}`;
}

function parseCsvNumbers(value: string): number[] {
  return value
    .split(",")
    .map((part) => Number(part.trim()))
    .filter((value) => Number.isFinite(value) && value >= 0);
}

function parsePromptMode(value: string | undefined): PromptMode {
  if (value === "original" || value === "explicit") return value;
  if (value) throw new Error(`Invalid --prompt-mode ${value}. Use original or explicit.`);
  return "explicit";
}

function parseCloseMode(value: string | undefined): CloseMode {
  if (value === "turn_complete_or_timeout" || value === "post_wait_only") return value;
  if (value) throw new Error(`Invalid --close-mode ${value}. Use turn_complete_or_timeout or post_wait_only.`);
  return "post_wait_only";
}

function promptConfig(promptMode: PromptMode): PromptConfig {
  if (promptMode === "original") {
    return {
      promptMode,
      systemInstruction: ORIGINAL_SYSTEM_INSTRUCTION,
      userPrompt: ORIGINAL_USER_PROMPT,
    };
  }
  return {
    promptMode,
    systemInstruction: EXPLICIT_SYSTEM_INSTRUCTION,
    userPrompt: EXPLICIT_USER_PROMPT,
  };
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
    latencies: args.has("latencies") ? parseCsvNumbers(args.get("latencies") ?? "") : DEFAULT_LATENCIES,
    attemptsPerLatency: Math.max(
      1,
      Number(args.get("attempts-per-latency") ?? args.get("attempts") ?? DEFAULT_ATTEMPTS_PER_LATENCY),
    ),
    concurrency: Math.max(1, Number(args.get("concurrency") ?? DEFAULT_CONCURRENCY)),
    promptMode: parsePromptMode(args.get("prompt-mode")),
    closeMode: parseCloseMode(args.get("close-mode")),
  };
}

function writeJson(path: string, value: unknown): void {
  writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

function appendJsonl(path: string, value: Record<string, unknown>): void {
  appendFileSync(path, `${JSON.stringify({ ts: new Date().toISOString(), ...value })}\n`, "utf8");
}

function summarizeError(error: unknown): string {
  if (error instanceof Error) {
    const cause = error.cause ? ` Cause: ${summarizeError(error.cause)}` : "";
    return `${error.name}: ${error.message}${cause}`;
  }
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
  return {
    sampleRate: rate ? Number(rate) : undefined,
    channels: channels ? Number(channels) : 1,
    bitDepth: 16,
  };
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
  if (pcm.sampleRate && pcm.channels && pcm.bitDepth) {
    const path = resolve(audioDir, "assistant_output.wav");
    writeFileSync(path, Buffer.concat([wavHeader(audio.length, pcm.sampleRate, pcm.channels, pcm.bitDepth), audio]));
    return path;
  }
  const path = resolve(audioDir, "assistant_output.bin");
  writeFileSync(path, audio);
  return path;
}

function msDelta(start: number | undefined, value: number | undefined): number | null {
  if (!start || !value) return null;
  return value - start;
}

function computeAttemptSummary(plan: AttemptPlan, state: AttemptState): AttemptSummary {
  const server1008 = state.closeCode === 1008 || state.closeReason?.includes("1008") === true;
  const server1011 = state.closeCode === 1011 || state.closeReason?.includes("1011") === true;
  const otherError = state.errors.length > 0 ? state.errors.join(" | ") : null;
  const postToolText = state.textOutputEventsAfterToolResponse.join(" ").toLowerCase();
  const hasFinalAnswerAfterToolResponse =
    /shipped/.test(postToolText) && /tomorrow/.test(postToolText) && /#?a123/.test(postToolText);
  const sessionValid =
    !server1008 &&
    !server1011 &&
    !otherError &&
    state.toolCallEmitted &&
    state.toolResponseSent &&
    hasFinalAnswerAfterToolResponse;

  return {
    latency_ms: plan.latencyMs,
    attempt_index: plan.attemptIndex,
    session_valid: sessionValid,
    server_1008_error: server1008,
    server_1011_error: server1011,
    other_error: otherError,
    close_code: state.closeCode,
    close_reason: state.closeReason,
    tool_call_emitted: state.toolCallEmitted,
    tool_call_time_ms: msDelta(state.promptSentAt, state.toolCallAt),
    tool_response_sent: state.toolResponseSent,
    tool_response_time_ms: msDelta(state.promptSentAt, state.toolResponseSentAt),
    first_audio_output_time_ms: msDelta(state.promptSentAt, state.firstAudioAt),
    has_audio_before_tool_response: state.audioOutputCountBeforeToolResponse > 0,
    audio_output_count_before_tool_response: state.audioOutputCountBeforeToolResponse,
    has_audio_after_tool_response: state.audioOutputCountAfterToolResponse > 0,
    post_tool_first_audio_latency_ms: msDelta(state.toolResponseSentAt, state.firstAudioAfterToolResponseAt),
    has_final_answer_after_tool_response: hasFinalAnswerAfterToolResponse,
    text_output_events: state.textOutputEvents,
    text_output_events_after_tool_response: state.textOutputEventsAfterToolResponse,
    audio_output_events: state.audioOutputEvents,
    raw_event_count: state.rawEventCount,
    result_dir: plan.attemptDir,
  };
}

async function runOne(
  ai: GoogleGenAI,
  model: string,
  plan: AttemptPlan,
  prompt: PromptConfig,
  closeMode: CloseMode,
): Promise<AttemptSummary> {
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
    toolCallPending: false,
    toolCallEmitted: false,
    toolResponseSent: false,
    toolResponseSkippedAfterClose: false,
    toolCallCount: 0,
    audioOutputCountBeforeToolResponse: 0,
    audioOutputCountAfterToolResponse: 0,
    textOutputEvents: [],
    textOutputEventsAfterToolResponse: [],
    audioOutputEvents: 0,
    rawEventCount: 0,
    errors: [],
    turnCompleteAfterToolResponse: false,
  };

  writeJson(resolve(plan.attemptDir, "config.json"), {
    latency_ms: plan.latencyMs,
    attempt_index: plan.attemptIndex,
    model,
    response_modality: Modality.AUDIO,
    prompt_mode: prompt.promptMode,
    close_mode: closeMode,
    system_instruction: prompt.systemInstruction,
    user_prompt: prompt.userPrompt,
    tool_name: TOOL_NAME,
    tool_response: TOOL_RESPONSE,
    tau_inspection: TAU_INSPECTION,
  });

  let session: Session | undefined;
  let done = false;
  let pendingToolTimer: ReturnType<typeof setTimeout> | undefined;
  let postToolDoneTimer: ReturnType<typeof setTimeout> | undefined;

  const appendTimeline = (type: string, extra: Record<string, unknown> = {}) => {
    appendJsonl(timelinePath, {
      type,
      event_ms: state.promptSentAt ? Date.now() - state.promptSentAt : null,
      ...extra,
    });
  };

  const finish = (resolveRun: () => void) => {
    if (done) return;
    done = true;
    if (pendingToolTimer) clearTimeout(pendingToolTimer);
    if (postToolDoneTimer) clearTimeout(postToolDoneTimer);
    if (!state.sessionClosed) {
      try {
        session?.close();
      } catch (error) {
        state.errors.push(summarizeError(error));
      }
    }
    writeAudioFile(audioDir, audioSegments);
    resolveRun();
  };

  await new Promise<void>(async (resolveRun) => {
    const hardTimer = setTimeout(() => {
      state.errors.push(`max attempt timeout ${MAX_ATTEMPT_MS}ms`);
      appendTimeline("max_attempt_timeout", { max_attempt_ms: MAX_ATTEMPT_MS });
      finish(resolveRun);
    }, MAX_ATTEMPT_MS);

    try {
      session = (await ai.live.connect({
        model,
        config: {
          responseModalities: [Modality.AUDIO],
          outputAudioTranscription: {},
          systemInstruction: prompt.systemInstruction,
          tools: [
            {
              functionDeclarations: [
                {
                  name: TOOL_NAME,
                  description: "Get the status and details of a retail order.",
                  behavior: Behavior.NON_BLOCKING,
                  parametersJsonSchema: {
                    type: "object",
                    properties: {
                      order_id: {
                        type: "string",
                        description:
                          "The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id.",
                      },
                    },
                    required: ["order_id"],
                  },
                },
              ],
            },
          ],
        },
        callbacks: {
          onopen: () => {
            state.openedAt = Date.now();
            appendTimeline("session_opened", { model });
            appendJsonl(rawLogPath, { type: "session_opened", model });
          },
          onmessage: (message: LiveMessage) => {
            const now = Date.now();
            state.rawEventCount += 1;
            const types = eventTypes(message);
            appendJsonl(rawLogPath, {
              type: "server_event",
              event_ms: state.promptSentAt ? now - state.promptSentAt : null,
              event_types: types,
              message: sanitizeMessage(message),
            });
            appendTimeline("server_event", { event_types: types });

            const parts = message.serverContent?.modelTurn?.parts ?? [];
            const textParts = parts.map((part) => part.text).filter((text): text is string => Boolean(text));
            const transcription = message.serverContent?.outputTranscription?.text;
            if (transcription) {
              state.textOutputEvents.push(transcription);
              if (state.toolResponseSent) state.textOutputEventsAfterToolResponse.push(transcription);
              state.firstTextAt ??= now;
              state.firstOutputAt ??= now;
              appendTimeline("output_transcription", { text: transcription });
            }
            for (const text of textParts) {
              state.textOutputEvents.push(text);
              if (state.toolResponseSent) state.textOutputEventsAfterToolResponse.push(text);
              state.firstTextAt ??= now;
              state.firstOutputAt ??= now;
              appendTimeline("text_output", { text });
            }

            for (const part of parts) {
              if (!part.inlineData?.data) continue;
              const chunk = Buffer.from(part.inlineData.data, "base64");
              audioSegments.push({
                offsetMs: state.promptSentAt ? now - state.promptSentAt : 0,
                chunk,
                mimeType: part.inlineData.mimeType,
              });
              state.audioOutputEvents += 1;
              state.firstAudioAt ??= now;
              state.firstOutputAt ??= now;
              if (state.toolResponseSent) {
                state.audioOutputCountAfterToolResponse += 1;
                state.firstAudioAfterToolResponseAt ??= now;
                state.firstOutputAfterToolResponseAt ??= now;
              } else {
                state.audioOutputCountBeforeToolResponse += 1;
              }
              appendTimeline("audio_output", {
                bytes: chunk.length,
                mime_type: part.inlineData.mimeType,
                phase: state.toolResponseSent ? "after_tool_response" : "before_tool_response",
              });
            }

            if (message.toolCall?.functionCalls?.length) {
              state.toolCallCount += message.toolCall.functionCalls.length;
              state.toolCallEmitted = true;
              state.toolCallPending = true;
              state.toolCallAt ??= now;
              const call = message.toolCall.functionCalls[0];
              appendTimeline("tool_call_received", { function_calls: message.toolCall.functionCalls });
              appendJsonl(rawLogPath, { type: "tool_call_received", function_calls: message.toolCall.functionCalls });

              if (!pendingToolTimer) {
                pendingToolTimer = setTimeout(() => {
                  if (state.sessionClosed || done) {
                    state.toolResponseSkippedAfterClose = state.sessionClosed;
                    appendTimeline("tool_response_skipped", {
                      reason: state.sessionClosed ? "session closed before delay elapsed" : "attempt already done",
                    });
                    return;
                  }
                  try {
                    session?.sendToolResponse({
                      functionResponses: [
                        {
                          id: call.id,
                          name: call.name || TOOL_NAME,
                          response: TOOL_RESPONSE,
                        },
                      ],
                    });
                    state.toolCallPending = false;
                    state.toolResponseSent = true;
                    state.toolResponseSentAt = Date.now();
                    appendTimeline("tool_response_sent", {
                      delay_ms: plan.latencyMs,
                      function_call_id: call.id,
                      function_name: call.name || TOOL_NAME,
                      response: TOOL_RESPONSE,
                    });
                    appendJsonl(rawLogPath, {
                      type: "tool_response_sent",
                      event_ms: state.promptSentAt ? Date.now() - state.promptSentAt : null,
                      delay_ms: plan.latencyMs,
                      function_call_id: call.id,
                      function_name: call.name || TOOL_NAME,
                      response: TOOL_RESPONSE,
                    });
                    postToolDoneTimer = setTimeout(() => {
                      appendTimeline("post_tool_wait_elapsed", { post_tool_wait_ms: POST_TOOL_WAIT_MS });
                      clearTimeout(hardTimer);
                      finish(resolveRun);
                    }, POST_TOOL_WAIT_MS);
                  } catch (error) {
                    state.errors.push(summarizeError(error));
                    appendTimeline("tool_response_send_error", { error: summarizeError(error) });
                  }
                }, plan.latencyMs);
              }
            }

            if (state.toolResponseSent && message.serverContent?.turnComplete) {
              state.turnCompleteAfterToolResponse = true;
              appendTimeline("turn_complete_after_tool_response");
              if (closeMode === "turn_complete_or_timeout") {
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
            state.toolCallPending = false;
            if (pendingToolTimer && !state.toolResponseSent) {
              clearTimeout(pendingToolTimer);
              state.toolResponseSkippedAfterClose = true;
            }
            appendTimeline("session_closed", { code: event.code, reason: event.reason });
            appendJsonl(rawLogPath, { type: "session_closed", code: event.code, reason: event.reason });
            clearTimeout(hardTimer);
            writeAudioFile(audioDir, audioSegments);
            resolveRun();
          },
        },
      })) as Session;

      session.sendClientContent({ turns: prompt.userPrompt, turnComplete: true });
      state.promptSentAt = Date.now();
      appendTimeline("user_message_sent", { prompt: prompt.userPrompt });
      appendJsonl(rawLogPath, { type: "user_message_sent", prompt: prompt.userPrompt });
    } catch (error) {
      state.errors.push(summarizeError(error));
      appendTimeline("run_error", { error: summarizeError(error) });
      appendJsonl(rawLogPath, { type: "run_error", error: summarizeError(error) });
      clearTimeout(hardTimer);
      writeAudioFile(audioDir, audioSegments);
      resolveRun();
    }
  });

  const summary = computeAttemptSummary(plan, state);
  writeJson(resolve(plan.attemptDir, "summary.json"), summary);
  return summary;
}

function summarizeLatency(latencyMs: number, attempts: AttemptSummary[]): LatencySummary {
  return {
    latency_ms: latencyMs,
    attempts: attempts.length,
    valid: attempts.filter((attempt) => attempt.session_valid).length,
    server_1008: attempts.filter((attempt) => attempt.server_1008_error).length,
    server_1011: attempts.filter((attempt) => attempt.server_1011_error).length,
    tool_call_emitted: attempts.filter((attempt) => attempt.tool_call_emitted).length,
    tool_response_sent: attempts.filter((attempt) => attempt.tool_response_sent).length,
    waiting_audio: attempts.filter((attempt) => attempt.has_audio_before_tool_response).length,
    post_tool_audio: attempts.filter((attempt) => attempt.has_audio_after_tool_response).length,
    post_tool_answer: attempts.filter((attempt) => attempt.has_final_answer_after_tool_response).length,
  };
}

function csvEscape(value: unknown): string {
  if (value === null || value === undefined) return "";
  const text = Array.isArray(value) ? value.join(" | ") : String(value);
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function writeCsv(path: string, rows: Record<string, unknown>[]): void {
  if (rows.length === 0) {
    writeFileSync(path, "\n", "utf8");
    return;
  }
  const headers = Object.keys(rows[0]);
  const lines = [headers.join(",")];
  for (const row of rows) lines.push(headers.map((header) => csvEscape(row[header])).join(","));
  writeFileSync(path, `${lines.join("\n")}\n`, "utf8");
}

function writeAnalysis(path: string, resultDir: string, model: string, rows: LatencySummary[], attempts: AttemptSummary[]): void {
  const total1008 = attempts.filter((attempt) => attempt.server_1008_error).length;
  const total1011 = attempts.filter((attempt) => attempt.server_1011_error).length;
  const totalToolCalls = attempts.filter((attempt) => attempt.tool_call_emitted).length;
  const totalWaitingAudio = attempts.filter((attempt) => attempt.has_audio_before_tool_response).length;
  const totalPostToolAudio = attempts.filter((attempt) => attempt.has_audio_after_tool_response).length;
  const instabilityLine =
    total1008 > 0
      ? "tau-bench-style realistic tasks do not automatically avoid the native Live tool-pending instability."
      : "This small probe did not reproduce 1008, but more attempts / multi-tool tasks may still be needed.";

  const lines = [
    "# tau-bench + Gemini Live native tool calling 1008 feasibility probe",
    "",
    `Result folder: ${resultDir}`,
    `Model: ${model}`,
    "",
    "1. tau-bench clone / inspect: succeeded. Inspected README, retail/airline domains, Env wiring, Tool base class, and retail get_order_details.",
    "2. Tool schema used: tau-bench-style mock tool, based on retail get_order_details. It uses Gemini Live native tool declaration and native sendToolResponse, not the external event protocol.",
    `3. Native tool call emitted: ${totalToolCalls}/${attempts.length} attempts.`,
    `4. 1008 / 1011 totals: ${total1008} / ${total1011}.`,
    `5. Waiting audio output before tool response: ${totalWaitingAudio}/${attempts.length} attempts.`,
    `6. Audio or answer output after tool response: ${totalPostToolAudio}/${attempts.length} attempts had post-tool audio.`,
    `7. Instability interpretation: ${instabilityLine}`,
    "",
    "Per-latency summary:",
    "| latency_ms | attempts | valid | 1008 | 1011 | tool_call | tool_response | waiting_audio | post_tool_audio |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ...rows.map(
      (row) =>
        `| ${row.latency_ms} | ${row.attempts} | ${row.valid} | ${row.server_1008} | ${row.server_1011} | ${row.tool_call_emitted} | ${row.tool_response_sent} | ${row.waiting_audio} | ${row.post_tool_audio} |`,
    ),
    "",
    "Expected final answer meaning: order #A123 has shipped and is estimated to arrive tomorrow.",
    "",
  ];
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

function printTable(rows: LatencySummary[]): void {
  const headers = ["latency_ms", "attempts", "valid", "server_1008", "server_1011", "tool_call_emitted", "waiting_audio", "post_tool_audio"];
  const cells = rows.map((row) => headers.map((header) => String(row[header as keyof LatencySummary])));
  const widths = headers.map((header, index) =>
    Math.max(header.length, ...cells.map((row) => row[index].length)),
  );
  console.log(headers.map((header, index) => header.padEnd(widths[index])).join(" | "));
  console.log(widths.map((width) => "-".repeat(width)).join("-+-"));
  for (const row of cells) console.log(row.map((cell, index) => cell.padEnd(widths[index])).join(" | "));
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const apiKey = requireEnv("GEMINI_API_KEY");
  const model = requireEnv("GEMINI_LIVE_MODEL");
  const prompt = promptConfig(args.promptMode);
  const resultId = `${timestampForPath()}_tau_live_tool_1008_probe_${args.promptMode}_${args.closeMode}`;
  const resultDir = resolve(RESULT_DIR, resultId);
  mkdirSync(resultDir, { recursive: true });

  writeJson(resolve(resultDir, "config.json"), {
    result_id: resultId,
    model,
    latencies_ms: args.latencies,
    attempts_per_latency: args.attemptsPerLatency,
    concurrency: args.concurrency,
    prompt_mode: args.promptMode,
    close_mode: args.closeMode,
    max_attempt_ms: MAX_ATTEMPT_MS,
    post_tool_wait_ms: POST_TOOL_WAIT_MS,
    response_modality: Modality.AUDIO,
    system_instruction: prompt.systemInstruction,
    user_prompt: prompt.userPrompt,
    tool_name: TOOL_NAME,
    tool_response: TOOL_RESPONSE,
    tau_inspection: TAU_INSPECTION,
  });

  const plans: AttemptPlan[] = [];
  for (const latencyMs of args.latencies) {
    const latencyDir = resolve(resultDir, `latency_${latencyMs}ms`);
    mkdirSync(latencyDir, { recursive: true });
    for (let attemptIndex = 1; attemptIndex <= args.attemptsPerLatency; attemptIndex += 1) {
      plans.push({
        latencyMs,
        attemptIndex,
        attemptDir: resolve(latencyDir, `attempt_${String(attemptIndex).padStart(4, "0")}`),
      });
    }
  }

  console.log(`Result directory: ${relative(PROJECT_DIR, resultDir)}`);
  console.log(`Attempts: ${plans.length}; concurrency: ${args.concurrency}`);

  const ai = new GoogleGenAI({ apiKey });
  const attempts = await runWithConcurrency(plans, args.concurrency, async (plan) => {
    console.log(`[latency ${plan.latencyMs}ms attempt ${plan.attemptIndex}] start`);
    const summary = await runOne(ai, model, plan, prompt, args.closeMode);
    console.log(
      `[latency ${plan.latencyMs}ms attempt ${plan.attemptIndex}] ${
        summary.session_valid ? "valid" : summary.server_1008_error ? "1008" : summary.server_1011_error ? "1011" : "not_valid"
      }`,
    );
    return summary;
  });

  const rows = args.latencies.map((latencyMs) =>
    summarizeLatency(
      latencyMs,
      attempts.filter((attempt) => attempt.latency_ms === latencyMs),
    ),
  );

  for (const row of rows) {
    const latencyAttempts = attempts.filter((attempt) => attempt.latency_ms === row.latency_ms);
    const latencyDir = resolve(resultDir, `latency_${row.latency_ms}ms`);
    writeJson(resolve(latencyDir, "summary.json"), { ...row, attempts: latencyAttempts });
    writeCsv(resolve(latencyDir, "summary.csv"), [row as unknown as Record<string, unknown>]);
  }

  writeJson(resolve(resultDir, "summary.json"), {
    result_id: resultId,
    model,
    tau_inspection: TAU_INSPECTION,
    rows,
    attempts,
  });
  writeCsv(resolve(resultDir, "summary.csv"), rows as unknown as Record<string, unknown>[]);
  writeAnalysis(resolve(resultDir, "analysis.txt"), relative(PROJECT_DIR, resultDir), model, rows, attempts);

  printTable(rows);
  console.log(`Summary: ${relative(PROJECT_DIR, resolve(resultDir, "summary.json"))}`);
  console.log(`Analysis: ${relative(PROJECT_DIR, resolve(resultDir, "analysis.txt"))}`);
}

main().catch((error) => {
  console.error("tau-live-tool-1008-probe failed");
  console.error(summarizeError(error));
  process.exitCode = 1;
});
