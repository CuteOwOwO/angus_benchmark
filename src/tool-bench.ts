import { Behavior, GoogleGenAI, Modality } from "@google/genai";
import { config as loadEnv } from "dotenv";
import { copyFileSync, cpSync, createWriteStream, existsSync, mkdirSync, writeFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { dirname, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  PROMPT as DEFAULT_PROMPT,
  PROMPT_VERSION as DEFAULT_PROMPT_VERSION,
  SYSTEM_INSTRUCTION as DEFAULT_SYSTEM_INSTRUCTION,
} from "./prompts.js";
import {
  PROMPT as TRY_PROMPT,
  PROMPT_VERSION as TRY_PROMPT_VERSION,
  SYSTEM_INSTRUCTION as TRY_SYSTEM_INSTRUCTION,
} from "./prompt_try.js";

const PROJECT_DIR = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const DEFAULT_ENV_FILE = resolve(PROJECT_DIR, "..", "angus_bench", ".env");
const LOG_DIR = resolve(PROJECT_DIR, "logs");
const AUDIO_DIR = resolve(LOG_DIR, "audio");
const RESULT_DIR = resolve(PROJECT_DIR, "result");
const TOOL_NAME = "external_answer_tool";
const EXPECTED_FINAL_ANSWER = "Angus";
const TOOL_RESPONSE_SCHEMA = "single_final_answer_field";
const POST_TOOL_TIMEOUT_MS = 15_000;
const MAX_SCENARIO_MS = 45_000;

const PROMPT_SOURCE = process.env.TOOL_BENCH_PROMPT === "prompt_try" ? "prompt_try" : "default";
const PROMPT_VERSION = PROMPT_SOURCE === "prompt_try" ? TRY_PROMPT_VERSION : DEFAULT_PROMPT_VERSION;
const PROMPT = PROMPT_SOURCE === "prompt_try" ? TRY_PROMPT : DEFAULT_PROMPT;
const SYSTEM_INSTRUCTION = PROMPT_SOURCE === "prompt_try" ? TRY_SYSTEM_INSTRUCTION : DEFAULT_SYSTEM_INSTRUCTION;

loadEnv({ path: process.env.GEMINI_ENV_FILE || DEFAULT_ENV_FILE });

type Scenario = {
  id: string;
  delayMs: number;
  response: Record<string, unknown>;
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

type ScenarioMetrics = {
  openedAt?: number;
  userSentAt?: number;
  firstServerEventAt?: number;
  firstOutputAt?: number;
  firstAudioAt?: number;
  firstTextAt?: number;
  primaryToolCallId?: string;
  primaryToolCallAt?: number;
  primaryToolResponseSentAt?: number;
  lastToolCallAt?: number;
  lastToolResponseSentAt?: number;
  firstOutputAfterToolCallAt?: number;
  toolCallAt?: number;
  toolResponseSentAt?: number;
  firstAudioAfterToolCallAt?: number;
  firstAudioAfterToolResponseAt?: number;
  firstTextAfterToolCallAt?: number;
  firstOutputAfterToolResponseAt?: number;
  audioBytesBeforeToolResponse: number;
  audioBytesAfterToolResponse: number;
  eventTypesSeen: Set<string>;
  textOutputSeen: boolean;
  toolCallReceived: boolean;
  toolResponseSent: boolean;
  toolCallCount: number;
  toolResponseSentCount: number;
  extraToolCallCount: number;
  sessionClosed: boolean;
  toolCallPending: boolean;
  primaryToolCallReceived: boolean;
  primaryToolResponseSent: boolean;
  toolResponseScheduled: boolean;
  toolResponseAttempted: boolean;
  toolResponseSentBeforeClose: boolean;
  toolResponseSkippedAfterClose: boolean;
  toolResponseSendError: string | null;
  blockedOutboundDuringToolPending: number;
  nonToolResponseOutboundDuringToolPending: boolean;
};

type AudioSegment = {
  offsetMs: number;
  chunk: Buffer;
};

type ScenarioSummary = Record<string, unknown>;

type PromptMetadata = {
  promptSource: string;
  promptVersion: string;
  prompt: string;
  systemInstruction: string;
};

type PendingToolResponse = {
  call: FunctionCall;
  isPrimaryToolCall: boolean;
  timer: ReturnType<typeof setTimeout>;
  resolve: () => void;
};

type RecentIoEvent = {
  ts: string;
  direction: "inbound" | "outbound";
  type: string;
  allowed?: boolean;
  reason?: string;
  eventTypes?: string[];
};

const scenarios: Scenario[] = [
  {
    id: "slow_correct_3s",
    delayMs: 3000,
    response: {
      final_answer: EXPECTED_FINAL_ANSWER,
    },
  },
  {
    id: "slow_correct_5s",
    delayMs: 5000,
    response: {
      final_answer: EXPECTED_FINAL_ANSWER,
    },
  },
  {
    id: "slow_correct_8s",
    delayMs: 8000,
    response: {
      final_answer: EXPECTED_FINAL_ANSWER,
    },
  },
  {
    id: "slow_correct_12s",
    delayMs: 12000,
    response: {
      final_answer: EXPECTED_FINAL_ANSWER,
    },
  },
];

function requireEnv(name: string): string {
  const value = process.env[name]?.trim();
  if (!value) throw new Error(`Missing ${name}. Set it in /user_data/angus_bench/.env or GEMINI_ENV_FILE.`);
  return value;
}

function timestamp(): string {
  return new Date().toISOString().replace(/[:.]/g, "-");
}

function msSince(start: number | undefined, value: number | undefined): string {
  if (!start || !value) return "n/a";
  return `${value - start} ms`;
}

function msDelta(start: number | undefined, value: number | undefined): number | undefined {
  if (!start || !value) return undefined;
  return value - start;
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

function decodeAudio(data: string): Buffer {
  return Buffer.from(data, "base64");
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

function writeAudioFile(pathWithoutExt: string, chunks: Buffer[], mimeType: string | undefined): { path: string; format: string } {
  const audio = Buffer.concat(chunks);
  const pcm = parsePcmMime(mimeType);
  if (pcm.sampleRate && pcm.channels && pcm.bitDepth) {
    const file = `${pathWithoutExt}.wav`;
    writeFileSync(file, Buffer.concat([wavHeader(audio.length, pcm.sampleRate, pcm.channels, pcm.bitDepth), audio]));
    return { path: file, format: `wav audio/pcm rate=${pcm.sampleRate} channels=${pcm.channels} bit_depth=${pcm.bitDepth}` };
  }

  const file = `${pathWithoutExt}.pcm`;
  writeFileSync(file, audio);
  return { path: file, format: `raw_pcm sample_rate=unknown channels=unknown mime_type=${mimeType ?? "unknown"}` };
}

function writeTimelineAudioFile(
  pathWithoutExt: string,
  segments: AudioSegment[],
  mimeType: string | undefined,
): { path: string; format: string; durationMs?: number } {
  const pcm = parsePcmMime(mimeType);
  if (!pcm.sampleRate || !pcm.channels || !pcm.bitDepth) {
    return writeAudioFile(pathWithoutExt, segments.map((segment) => segment.chunk), mimeType);
  }

  const bytesPerMs = (pcm.sampleRate * pcm.channels * pcm.bitDepth) / 8 / 1000;
  const blockAlign = (pcm.channels * pcm.bitDepth) / 8;
  const parts: Buffer[] = [];
  let cursorBytes = 0;

  for (const segment of segments) {
    const requestedStart = Math.max(0, Math.round(segment.offsetMs * bytesPerMs));
    const alignedStart = requestedStart - (requestedStart % blockAlign);
    const startBytes = Math.max(alignedStart, cursorBytes);

    if (startBytes > cursorBytes) {
      parts.push(Buffer.alloc(startBytes - cursorBytes));
      cursorBytes = startBytes;
    }

    parts.push(segment.chunk);
    cursorBytes += segment.chunk.length;
  }

  const audio = Buffer.concat(parts);
  const file = `${pathWithoutExt}.wav`;
  writeFileSync(file, Buffer.concat([wavHeader(audio.length, pcm.sampleRate, pcm.channels, pcm.bitDepth), audio]));
  return {
    path: file,
    format: `timeline wav audio/pcm rate=${pcm.sampleRate} channels=${pcm.channels} bit_depth=${pcm.bitDepth}`,
    durationMs: Math.round(audio.length / bytesPerMs),
  };
}

function audioPathForMime(pathWithoutExt: string, mimeType: string | undefined): string {
  return `${pathWithoutExt}${parsePcmMime(mimeType).sampleRate ? ".wav" : ".pcm"}`;
}

function promptMetadata(): PromptMetadata {
  return {
    promptSource: PROMPT_SOURCE,
    promptVersion: PROMPT_VERSION,
    prompt: PROMPT,
    systemInstruction: SYSTEM_INSTRUCTION,
  };
}

function writePromptsFile(resultRunDir: string, metadata: PromptMetadata): void {
  const content = [
    `PROMPT_SOURCE=${metadata.promptSource}`,
    `PROMPT_VERSION=${metadata.promptVersion}`,
    "",
    "PROMPT:",
    metadata.prompt,
    "",
    "SYSTEM_INSTRUCTION:",
    metadata.systemInstruction,
    "",
  ].join("\n");
  writeFileSync(resolve(resultRunDir, "prompts.txt"), content, "utf8");
}

function runTimelineVisualization(resultRunDir: string): void {
  execFileSync("python3", [resolve(PROJECT_DIR, "scripts", "visualize_timeline.py"), resultRunDir], {
    cwd: PROJECT_DIR,
    stdio: "inherit",
  });
}

function sanitizeMessage(message: LiveMessage, pathWithoutExt: string): unknown {
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
                      bytes: part.inlineData.data ? decodeAudio(part.inlineData.data).length : 0,
                      audioFile: relative(PROJECT_DIR, audioPathForMime(pathWithoutExt, part.inlineData.mimeType)),
                    },
                  };
                }),
              }
            : undefined,
        }
      : undefined,
  };
}

async function runScenario(
  ai: GoogleGenAI,
  model: string,
  scenario: Scenario,
  log: ReturnType<typeof createWriteStream>,
  runStamp: string,
): Promise<ScenarioSummary> {
  const metrics: ScenarioMetrics = {
    audioBytesBeforeToolResponse: 0,
    audioBytesAfterToolResponse: 0,
    eventTypesSeen: new Set<string>(),
    textOutputSeen: false,
    toolCallReceived: false,
    toolResponseSent: false,
    toolCallCount: 0,
    toolResponseSentCount: 0,
    extraToolCallCount: 0,
    sessionClosed: false,
    toolCallPending: false,
    primaryToolCallReceived: false,
    primaryToolResponseSent: false,
    toolResponseScheduled: false,
    toolResponseAttempted: false,
    toolResponseSentBeforeClose: false,
    toolResponseSkippedAfterClose: false,
    toolResponseSendError: null,
    blockedOutboundDuringToolPending: 0,
    nonToolResponseOutboundDuringToolPending: false,
  };
  const audioChunks: Buffer[] = [];
  const timelineSegments: AudioSegment[] = [];
  const recentIoEvents: RecentIoEvent[] = [];
  let audioMimeType: string | undefined;
  let session: Session | undefined;
  let done = false;
  let closeReason = "";
  const pendingToolResponses = new Set<PendingToolResponse>();
  const runAudioDir = resolve(AUDIO_DIR, runStamp);
  const audioBasePath = resolve(runAudioDir, `${scenario.id}_compressed`);
  const timelineAudioBasePath = resolve(runAudioDir, `${scenario.id}_timeline`);
  mkdirSync(runAudioDir, { recursive: true });

  const logLine = (event: Record<string, unknown>) => {
    log.write(`${JSON.stringify({ ts: new Date().toISOString(), scenario: scenario.id, ...event })}\n`);
  };

  const rememberIoEvent = (event: RecentIoEvent) => {
    recentIoEvents.push(event);
    if (recentIoEvents.length > 10) recentIoEvents.shift();
  };

  const recordInboundEvent = (type: string, details: Omit<RecentIoEvent, "ts" | "direction" | "type"> = {}) => {
    rememberIoEvent({ ts: new Date().toISOString(), direction: "inbound", type, ...details });
  };

  const recordOutboundAttempt = (outboundType: string, reason: string): boolean => {
    const allowed =
      !metrics.sessionClosed &&
      (outboundType === "sendToolResponse" || !metrics.toolCallPending);
    const timestamp = new Date().toISOString();
    const event = {
      type: "outbound_event",
      timestamp,
      outbound_type: outboundType,
      toolCallPending: metrics.toolCallPending,
      sessionClosed: metrics.sessionClosed,
      allowed,
      reason,
    };
    logLine(event);
    rememberIoEvent({
      ts: timestamp,
      direction: "outbound",
      type: outboundType,
      allowed,
      reason,
    });

    if (!allowed && metrics.toolCallPending && outboundType !== "sendToolResponse") {
      metrics.blockedOutboundDuringToolPending += 1;
      metrics.nonToolResponseOutboundDuringToolPending = true;
      logLine({
        type: "blocked_outbound_during_tool_pending",
        outbound_type: outboundType,
        reason,
      });
    }

    return allowed;
  };

  const cancelPendingToolResponses = (reason: string) => {
    for (const pending of pendingToolResponses) {
      clearTimeout(pending.timer);
      pendingToolResponses.delete(pending);
      if (metrics.sessionClosed) metrics.toolResponseSkippedAfterClose = true;
      logLine({
        type: metrics.sessionClosed ? "tool_response_skipped_after_close" : "tool_response_skipped",
        functionCallId: pending.call.id,
        functionName: pending.call.name,
        delayMs: scenario.delayMs,
        reason,
        isPrimaryToolCall: pending.isPrimaryToolCall,
      });
      pending.resolve();
    }
  };

  const markDone = (skipReason: string) => {
    if (done) return false;
    done = true;
    metrics.toolCallPending = false;
    cancelPendingToolResponses(skipReason);
    return true;
  };

  const finish = () => {
    if (!markDone("scenario finished before scheduled tool response")) return;
    if (!recordOutboundAttempt("close", "finish scenario")) return;
    try {
      session?.close();
    } catch {
      // The socket may already be closed.
    }
  };

  const scheduleDelayedToolResponse = (call: FunctionCall, isPrimaryToolCall: boolean): Promise<void> => {
    return new Promise((resolve) => {
      const pending: PendingToolResponse = {
        call,
        isPrimaryToolCall,
        resolve,
        timer: setTimeout(() => {
          pendingToolResponses.delete(pending);
          if (done) {
            if (metrics.sessionClosed) metrics.toolResponseSkippedAfterClose = true;
            logLine({
              type: metrics.sessionClosed ? "tool_response_skipped_after_close" : "tool_response_skipped",
              functionCallId: call.id,
              functionName: call.name,
              delayMs: scenario.delayMs,
              reason: metrics.sessionClosed ? "session already closed" : "scenario already done",
              isPrimaryToolCall,
            });
            resolve();
            return;
          }
          metrics.toolResponseAttempted = true;
          if (!recordOutboundAttempt("sendToolResponse", "scheduled tool response delay elapsed")) {
            metrics.toolResponseSkippedAfterClose ||= metrics.sessionClosed;
            resolve();
            return;
          }
          try {
            session?.sendToolResponse({
              functionResponses: [
                {
                  id: call.id,
                  name: call.name || TOOL_NAME,
                  response: scenario.response,
                },
              ],
            });
          } catch (error) {
            metrics.toolResponseSendError = summarizeError(error);
            logLine({
              type: "tool_response_send_error",
              functionCallId: call.id,
              functionName: call.name,
              delayMs: scenario.delayMs,
              error: metrics.toolResponseSendError,
              isPrimaryToolCall,
            });
            resolve();
            return;
          }
          const sentAt = Date.now();
          metrics.toolResponseSentCount += 1;
          metrics.lastToolResponseSentAt = sentAt;
          metrics.toolResponseSentBeforeClose = !metrics.sessionClosed;
          if (isPrimaryToolCall) {
            metrics.toolResponseSent = true;
            metrics.toolCallPending = false;
            metrics.primaryToolResponseSent = true;
            metrics.primaryToolResponseSentAt = sentAt;
            metrics.toolResponseSentAt = sentAt;
          }
          logLine({
            type: "tool_response_sent",
            functionCallId: call.id,
            functionName: call.name,
            delayMs: scenario.delayMs,
            response: scenario.response,
            isPrimaryToolCall,
          });
          resolve();
        }, scenario.delayMs),
      };
      metrics.toolResponseScheduled = true;
      pendingToolResponses.add(pending);
    });
  };

  await new Promise<void>(async (resolvePromise) => {
    const hardTimer = setTimeout(() => {
      closeReason = `max scenario timeout ${MAX_SCENARIO_MS}ms`;
      finish();
    }, MAX_SCENARIO_MS);

    try {
      console.log("");
      console.log(`[${scenario.id}] PROMPT_VERSION: ${PROMPT_VERSION}`);
      console.log(`[${scenario.id}] PROMPT:`);
      console.log(PROMPT);
      console.log(`[${scenario.id}] SYSTEM_INSTRUCTION:`);
      console.log(SYSTEM_INSTRUCTION);

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
            metrics.openedAt = Date.now();
            logLine({ type: "session_opened", model, nonBlockingToolConfigured: true });
          },
          onmessage: (message: LiveMessage) => {
            const now = Date.now();
            metrics.firstServerEventAt ??= now;
            const inboundTypes = eventTypes(message);
            for (const type of inboundTypes) metrics.eventTypesSeen.add(type);
            recordInboundEvent("server_event", { eventTypes: inboundTypes });

            for (const part of message.serverContent?.modelTurn?.parts ?? []) {
              if (part.text) {
                metrics.textOutputSeen = true;
                metrics.firstTextAt ??= now;
                console.log(`[${scenario.id}] text: ${part.text}`);
                if (metrics.toolCallAt && !metrics.toolResponseSentAt) metrics.firstTextAfterToolCallAt ??= now;
                if (metrics.toolResponseSentAt) metrics.firstOutputAfterToolResponseAt ??= now;
              }

              if (part.inlineData?.data) {
                const chunk = decodeAudio(part.inlineData.data);
                metrics.firstOutputAt ??= now;
                if (metrics.toolCallAt) metrics.firstOutputAfterToolCallAt ??= now;
                metrics.firstAudioAt ??= now;
                audioMimeType ??= part.inlineData.mimeType;
                audioChunks.push(chunk);
                timelineSegments.push({
                  offsetMs: Math.max(0, now - (metrics.userSentAt ?? metrics.openedAt ?? now)),
                  chunk,
                });
                if (metrics.toolResponseSentAt) {
                  metrics.audioBytesAfterToolResponse += chunk.length;
                  metrics.firstAudioAfterToolResponseAt ??= now;
                  metrics.firstOutputAfterToolResponseAt ??= now;
                } else {
                  metrics.audioBytesBeforeToolResponse += chunk.length;
                }
                if (metrics.toolCallAt && !metrics.toolResponseSentAt) metrics.firstAudioAfterToolCallAt ??= now;
              }
            }

            if (message.toolCall?.functionCalls?.length) {
              metrics.toolCallReceived = true;
              const scheduledToolResponses: Array<{ call: FunctionCall; isPrimaryToolCall: boolean }> = [];
              for (const call of message.toolCall.functionCalls) {
                metrics.toolCallCount += 1;
                metrics.lastToolCallAt = now;
                let isPrimaryToolCall = false;
                if (!metrics.primaryToolCallAt) {
                  metrics.primaryToolCallId = call.id;
                  metrics.primaryToolCallAt = now;
                  metrics.toolCallAt = now;
                  metrics.toolCallPending = true;
                  metrics.primaryToolCallReceived = true;
                  isPrimaryToolCall = true;
                } else {
                  metrics.extraToolCallCount += 1;
                }
                scheduledToolResponses.push({ call, isPrimaryToolCall });
              }
              logLine({
                type: "tool_call_received",
                functionCalls: scheduledToolResponses.map(({ call, isPrimaryToolCall }) => ({
                  id: call.id,
                  name: call.name,
                  args: call.args,
                  isPrimaryToolCall,
                })),
              });
              for (const { call, isPrimaryToolCall } of scheduledToolResponses) {
                void scheduleDelayedToolResponse(call, isPrimaryToolCall);
              }
            }

            logLine({
              type: "server_event",
              eventTypes: eventTypes(message),
              eventMs: metrics.openedAt ? now - metrics.openedAt : undefined,
              message: sanitizeMessage(message, audioBasePath),
            });

            if (metrics.toolResponseSentAt && message.serverContent?.turnComplete) {
              finish();
            }
          },
          onerror: (error) => {
            recordInboundEvent("socket_error");
            logLine({ type: "socket_error", error: summarizeError(error) });
          },
          onclose: (event: { code?: number; reason?: string }) => {
            closeReason ||= `socket closed code=${event.code ?? "unknown"} reason=${event.reason || "none"}`;
            metrics.sessionClosed = true;
            recordInboundEvent("session_closed", { reason: closeReason });
            logLine({ type: "session_closed", code: event.code, reason: event.reason });
            markDone("session closed before scheduled tool response");
            clearTimeout(hardTimer);
            resolvePromise();
          },
        },
      })) as Session;

      metrics.openedAt ??= Date.now();
      if (recordOutboundAttempt("sendClientContent", "initial user prompt")) {
        session.sendClientContent({ turns: PROMPT, turnComplete: true });
      }
      metrics.userSentAt = Date.now();
      logLine({ type: "user_message_sent", prompt: PROMPT });
    } catch (error) {
      closeReason = summarizeError(error);
      logLine({ type: "scenario_error", error: closeReason });
      markDone("scenario error before scheduled tool response");
      clearTimeout(hardTimer);
      resolvePromise();
    }
  });

  let audioFile = "none";
  let audioFormat = "none";
  let timelineAudioFile = "none";
  let timelineAudioFormat = "none";
  let timelineAudioDurationMs: number | undefined;
  if (audioChunks.length > 0) {
    const written = writeAudioFile(audioBasePath, audioChunks, audioMimeType);
    audioFile = relative(PROJECT_DIR, written.path);
    audioFormat = written.format;

    const timeline = writeTimelineAudioFile(timelineAudioBasePath, timelineSegments, audioMimeType);
    timelineAudioFile = relative(PROJECT_DIR, timeline.path);
    timelineAudioFormat = timeline.format;
    timelineAudioDurationMs = timeline.durationMs;
  }

  const firstOutputWhileWaitingAt = metrics.firstAudioAfterToolCallAt ?? metrics.firstTextAfterToolCallAt;
  const modelOutputWhileWaiting =
    Boolean(firstOutputWhileWaitingAt && metrics.toolResponseSentAt && firstOutputWhileWaitingAt < metrics.toolResponseSentAt);
  const result = metrics.toolCallReceived && metrics.toolResponseSent ? "PASS" : "FAIL";
  const timingStartAt = metrics.userSentAt ?? metrics.openedAt;
  const timeToPrimaryToolCallMs = msDelta(timingStartAt, metrics.primaryToolCallAt);
  const primaryToolResponseDelayMs = msDelta(metrics.primaryToolCallAt, metrics.primaryToolResponseSentAt);
  const timeToPrimaryToolResponseMs = msDelta(timingStartAt, metrics.primaryToolResponseSentAt);
  const timeToFirstOutputMs = msDelta(timingStartAt, metrics.firstOutputAt);
  const timeToFirstAudioMs = msDelta(timingStartAt, metrics.firstAudioAt);
  const timeToFirstTextMs = msDelta(timingStartAt, metrics.firstTextAt);
  const timeFromToolResponseToFirstAudioMs = msDelta(metrics.primaryToolResponseSentAt, metrics.firstAudioAfterToolResponseAt);
  const primaryToolCallToFirstOutputMs = msDelta(metrics.primaryToolCallAt, metrics.firstOutputAfterToolCallAt);
  const primaryToolCallToFirstAudioMs = msDelta(metrics.primaryToolCallAt, metrics.firstAudioAt);
  const serverClosedBeforeToolResponse = Boolean(
    metrics.sessionClosed && metrics.primaryToolCallReceived && !metrics.primaryToolResponseSent,
  );
  const closeWas1008 = closeReason.includes("code=1008");
  const likelyServerSideLiveApiInstability = Boolean(
    closeWas1008 &&
      serverClosedBeforeToolResponse &&
      !metrics.nonToolResponseOutboundDuringToolPending,
  );
  const resultCode = serverClosedBeforeToolResponse
    ? "INVALID_SERVER_CLOSED_BEFORE_TOOL_RESPONSE"
    : result === "PASS"
    ? "PASS"
    : "FAIL";

  const summary: ScenarioSummary = {
    type: "scenario_summary",
    scenario: scenario.id,
    delayMs: scenario.delayMs,
    expected_final_answer: EXPECTED_FINAL_ANSWER,
    final_answer_exact_match: null,
    tool_response_schema: TOOL_RESPONSE_SCHEMA,
    closeReason,
    toolCallReceived: metrics.toolCallReceived,
    toolResponseSent: metrics.toolResponseSent,
    primaryToolCallId: metrics.primaryToolCallId,
    toolCallCount: metrics.toolCallCount,
    extraToolCallCount: metrics.extraToolCallCount,
    toolResponseSentCount: metrics.toolResponseSentCount,
    sessionClosed: metrics.sessionClosed,
    toolCallPending: metrics.toolCallPending,
    primaryToolCallReceived: metrics.primaryToolCallReceived,
    primaryToolResponseSent: metrics.primaryToolResponseSent,
    toolResponseScheduled: metrics.toolResponseScheduled,
    toolResponseAttempted: metrics.toolResponseAttempted,
    toolResponseSentBeforeClose: metrics.toolResponseSentBeforeClose,
    toolResponseSkippedAfterClose: metrics.toolResponseSkippedAfterClose,
    toolResponseSendError: metrics.toolResponseSendError,
    serverClosedBeforeToolResponse,
    nonToolResponseOutboundDuringToolPending: metrics.nonToolResponseOutboundDuringToolPending,
    blockedOutboundDuringToolPending: metrics.blockedOutboundDuringToolPending,
    likely_server_side_live_api_instability: likelyServerSideLiveApiInstability,
    invalidReason: resultCode === "INVALID_SERVER_CLOSED_BEFORE_TOOL_RESPONSE"
      ? "server closed the session after primary tool call and before primary tool response was sent"
      : undefined,
    resultCode,
    recentIoEventsBeforeClose: closeWas1008 ? recentIoEvents : undefined,
    modelOutputWhileWaiting,
    audioFile,
    audioFormat,
    timelineAudioFile,
    timelineAudioFormat,
    timelineAudioDurationMs,
    eventTypesSeen: [...metrics.eventTypesSeen],
    timings: {
      sessionOpenedAt: metrics.openedAt,
      userMessageSentAt: metrics.userSentAt,
      firstServerEventAt: metrics.firstServerEventAt,
      firstOutputAt: metrics.firstOutputAt,
      firstAudioAt: metrics.firstAudioAt,
      firstTextAt: metrics.firstTextAt,
      primaryToolCallAt: metrics.primaryToolCallAt,
      primaryToolResponseSentAt: metrics.primaryToolResponseSentAt,
      lastToolCallAt: metrics.lastToolCallAt,
      lastToolResponseSentAt: metrics.lastToolResponseSentAt,
      toolCallAt: metrics.primaryToolCallAt,
      toolResponseSentAt: metrics.primaryToolResponseSentAt,
      timeToPrimaryToolCallMs,
      primaryToolResponseDelayMs,
      timeToPrimaryToolResponseMs,
      timeToFirstOutputMs,
      timeToFirstAudioMs,
      timeToFirstTextMs,
      timeFromToolResponseToFirstAudioMs,
      primaryToolCallToFirstOutputMs,
      primaryToolCallToFirstAudioMs,
      firstOutputAfterToolCallAt: metrics.firstOutputAfterToolCallAt,
      firstAudioAfterToolCallAt: metrics.firstAudioAfterToolCallAt,
      firstAudioAfterToolResponseAt: metrics.firstAudioAfterToolResponseAt,
      firstTextAfterToolCallAt: metrics.firstTextAfterToolCallAt,
      firstOutputAfterToolResponseAt: metrics.firstOutputAfterToolResponseAt,
    },
    audioBytesBeforeToolResponse: metrics.audioBytesBeforeToolResponse,
    audioBytesAfterToolResponse: metrics.audioBytesAfterToolResponse,
  };
  logLine(summary);

  console.log("");
  console.log(`Scenario: ${scenario.id}`);
  console.log(`Delay: ${scenario.delayMs} ms`);
  console.log(`Tool call received: ${metrics.toolCallReceived ? "yes" : "no"}`);
  console.log(`Tool response sent: ${metrics.toolResponseSent ? "yes" : "no"}`);
  console.log(`Tool calls received: ${metrics.toolCallCount}`);
  console.log(`Extra tool calls: ${metrics.extraToolCallCount}`);
  console.log(`Tool responses sent: ${metrics.toolResponseSentCount}`);
  console.log(`Primary tool call time: ${timeToPrimaryToolCallMs === undefined ? "n/a" : `${timeToPrimaryToolCallMs} ms`}`);
  console.log(`Tool response received time: ${timeToPrimaryToolResponseMs === undefined ? "n/a" : `${timeToPrimaryToolResponseMs} ms`}`);
  console.log(`Primary tool response delay: ${primaryToolResponseDelayMs === undefined ? "n/a" : `${primaryToolResponseDelayMs} ms`}`);
  console.log(`First audio output time: ${timeToFirstOutputMs === undefined ? "n/a" : `${timeToFirstOutputMs} ms`}`);
  console.log(`First audio after tool response: ${timeFromToolResponseToFirstAudioMs === undefined ? "n/a" : `${timeFromToolResponseToFirstAudioMs} ms`}`);
  console.log(`First text time: ${timeToFirstTextMs === undefined ? "n/a" : `${timeToFirstTextMs} ms`}`);
  console.log(`First audio time: ${msSince(metrics.openedAt, metrics.firstAudioAt)}`);
  console.log(`Model output while waiting: ${modelOutputWhileWaiting ? "yes" : "no"}`);
  console.log(`Audio bytes before tool response: ${metrics.audioBytesBeforeToolResponse}`);
  console.log(`Audio bytes after tool response: ${metrics.audioBytesAfterToolResponse}`);
  console.log(`Audio file: ${audioFile}`);
  console.log(`Timeline audio file: ${timelineAudioFile}`);
  console.log(`Text output seen: ${metrics.textOutputSeen ? "yes" : "no"}`);
  console.log(`Event types seen: ${JSON.stringify([...metrics.eventTypesSeen])}`);
  console.log(`Result: ${result}`);
  console.log(`Result code: ${resultCode}`);
  console.log(`Server closed before tool response: ${serverClosedBeforeToolResponse ? "yes" : "no"}`);
  console.log(`Non-tool outbound while pending: ${metrics.nonToolResponseOutboundDuringToolPending ? "yes" : "no"}`);
  console.log(`Likely server-side Live API instability: ${likelyServerSideLiveApiInstability ? "yes" : "no"}`);
  if (result === "FAIL") console.log(`Close/error: ${closeReason}`);

  return summary;
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

async function main(): Promise<void> {
  mkdirSync(AUDIO_DIR, { recursive: true });
  mkdirSync(RESULT_DIR, { recursive: true });
  const apiKey = requireEnv("GEMINI_API_KEY");
  const model = requireEnv("GEMINI_LIVE_MODEL");
  const runStamp = timestamp();
  const logPath = resolve(LOG_DIR, `tool-bench-${runStamp}.jsonl`);
  const resultRunDir = resolve(RESULT_DIR, runStamp);
  const runAudioDir = resolve(AUDIO_DIR, runStamp);
  mkdirSync(resultRunDir, { recursive: true });
  const prompts = promptMetadata();
  writePromptsFile(resultRunDir, prompts);
  const log = createWriteStream(logPath, { flags: "a" });

  console.log(`Running tool bench for model: ${model}`);
  console.log(`Log file: ${relative(PROJECT_DIR, logPath)}`);
  console.log(`Result directory: ${relative(PROJECT_DIR, resultRunDir)}`);
  console.log("Function behavior: NON_BLOCKING");

  log.write(
    `${JSON.stringify({
      ts: new Date().toISOString(),
      type: "prompt_metadata",
      ...prompts,
    })}\n`,
  );

  const ai = new GoogleGenAI({ apiKey });
  const summaries: ScenarioSummary[] = [];

  for (const scenario of scenarios) {
    summaries.push(await runScenario(ai, model, scenario, log, runStamp));
  }

  writeFileSync(
    resolve(resultRunDir, "summary.json"),
    `${JSON.stringify(
      {
        runTimestamp: runStamp,
        runStamp,
        prompt_version: prompts.promptVersion,
        expected_final_answer: EXPECTED_FINAL_ANSWER,
        tool_response_schema: TOOL_RESPONSE_SCHEMA,
        model,
        logFile: relative(PROJECT_DIR, logPath),
        rawLogFile: "raw_log.jsonl",
        prompts,
        scenarios: summaries,
      },
      null,
      2,
    )}\n`,
    "utf8",
  );

  await new Promise<void>((resolvePromise) => log.end(resolvePromise));
  copyFileSync(logPath, resolve(resultRunDir, "raw_log.jsonl"));
  if (existsSync(runAudioDir)) {
    cpSync(runAudioDir, resolve(resultRunDir, "audio"), { recursive: true });
  }
  runTimelineVisualization(resultRunDir);
  console.log("");
  console.log(`Detailed log: ${relative(PROJECT_DIR, logPath)}`);
  console.log(`Audio directory: ${relative(PROJECT_DIR, AUDIO_DIR)}`);
  console.log(`Prompt file: ${relative(PROJECT_DIR, resolve(resultRunDir, "prompts.txt"))}`);
  console.log(`Summary file: ${relative(PROJECT_DIR, resolve(resultRunDir, "summary.json"))}`);
  console.log(`Timeline directory: ${relative(PROJECT_DIR, resolve(resultRunDir, "timeline"))}`);
}

main().catch((error) => {
  console.error("tool-bench failed");
  console.error(summarizeError(error));
  process.exitCode = 1;
});
