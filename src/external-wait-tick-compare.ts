import { GoogleGenAI, Modality } from "@google/genai";
import { config as loadEnv } from "dotenv";
import { createWriteStream, mkdirSync, writeFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { dirname, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const PROJECT_DIR = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const DEFAULT_ENV_FILE = resolve(PROJECT_DIR, "..", "angus_bench", ".env");
const RESULT_DIR = resolve(PROJECT_DIR, "result");
const EXPECTED_FINAL_ANSWER = "Angus";
const PROMPT_VERSION = "controlled_external_result_tick_v3_3000ms";
const MAX_SCENARIO_MS = 45_000;
const DEFAULT_POST_EXTERNAL_WAIT_MS = 6000;

const PROMPT = `Please answer this question:

What is the final answer?

The final answer is not available yet.

The environment may send EXTERNAL_EVENT messages.

There are two possible EXTERNAL_EVENT types:

1. type: pending
This means the external result is not available yet.
If you receive a pending event, do not answer the question.
You may give a short waiting update, but do not guess or mention any possible final answer.

2. type: ready
This means the external result has arrived.
A ready event contains final_answer.
If you receive a ready event, immediately answer with exactly the value of final_answer.

Before receiving an EXTERNAL_EVENT with type: ready, do not answer, guess, infer, calculate, or mention any possible final answer.

After receiving an EXTERNAL_EVENT with type: ready, do not continue waiting.
Answer exactly with final_answer and nothing else.`;

const SYSTEM_INSTRUCTION = `You are a helpful voice assistant.

For this task, the final answer is controlled by EXTERNAL_EVENT messages from the environment.

Only an EXTERNAL_EVENT with type: ready contains the final answer.

If EXTERNAL_EVENT type is pending:
- The final answer is not available.
- Do not answer the question.
- Do not guess, infer, calculate, summarize, imply, or mention any possible answer.
- You may briefly reassure the user.

If EXTERNAL_EVENT type is ready:
- The final answer is available.
- Read only the final_answer field.
- Immediately answer with exactly the value of final_answer.
- Do not continue waiting.
- Do not say that the result is unavailable.
- Do not add any extra words.`;

const EXTERNAL_STATUS = `EXTERNAL_EVENT
type: pending
has_final_answer: false
message: external result is not available yet`;

loadEnv({ path: process.env.GEMINI_ENV_FILE || DEFAULT_ENV_FILE });

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
  usageMetadata?: unknown;
  goAway?: unknown;
  sessionResumptionUpdate?: unknown;
  voiceActivity?: unknown;
  voiceActivityDetectionSignal?: unknown;
};

type Session = {
  sendClientContent(params: { turns?: string; turnComplete?: boolean }): void;
  close(): void;
};

type Scenario = {
  id: string;
  latencyMs: number;
};

type Condition = {
  id: string;
  tickMode: "none" | "scheduled" | "every_3000ms";
};

type AudioSegment = {
  offsetMs: number;
  chunk: Buffer;
};

type ScenarioMetrics = {
  openedAt?: number;
  promptSentAt?: number;
  externalResultInjectedAt?: number;
  firstAssistantOutputAt?: number;
  firstTextAt?: number;
  firstAudioAt?: number;
  firstOutputAfterExternalResultAt?: number;
  firstAudioAfterExternalResultAt?: number;
  assistantOutputCountBeforeExternalResult: number;
  assistantOutputCountAfterExternalResult: number;
  pendingTickTimes: number[];
  assistantOutputTimes: number[];
  sessionClosed: boolean;
  closeCode?: number;
  closeReason?: string;
  errors: string[];
  eventTypesSeen: Set<string>;
};

type ScenarioSummary = Record<string, unknown>;

const DEFAULT_SCENARIOS: Scenario[] = [
  { id: "slow_correct_3s", latencyMs: 3000 },
  { id: "slow_correct_5s", latencyMs: 5000 },
  { id: "slow_correct_8s", latencyMs: 8000 },
  { id: "slow_correct_12s", latencyMs: 12000 },
];

function requireEnv(name: string): string {
  const value = process.env[name]?.trim();
  if (!value) throw new Error(`Missing ${name}. Set it in /user_data/angus_bench/.env or GEMINI_ENV_FILE.`);
  return value;
}

function timestampForPath(): string {
  return new Date().toISOString().replace(/[:.]/g, "-");
}

function parseArgs(argv: string[]): { scenarios: Scenario[]; conditions: string[]; postExternalWaitMs: number } {
  const args = new Map<string, string>();
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (!arg.startsWith("--")) continue;
    const [key, inlineValue] = arg.slice(2).split("=", 2);
    const value = inlineValue ?? argv[index + 1];
    if (inlineValue === undefined) index += 1;
    args.set(key, value);
  }
  const latencies = args.get("latencies")
    ? args
        .get("latencies")!
        .split(",")
        .map((item) => Number(item.trim()))
        .filter((value) => Number.isFinite(value) && value >= 0)
        .map((latencyMs) => ({ id: `slow_correct_${Math.round(latencyMs / 1000)}s`, latencyMs }))
    : DEFAULT_SCENARIOS;
  return {
    scenarios: latencies.length ? latencies : DEFAULT_SCENARIOS,
    conditions: (args.get("conditions") ?? "no_tick,tick_every_3000ms")
      .split(",")
      .map((condition) => condition.trim())
      .filter(Boolean),
    postExternalWaitMs: Math.max(0, Number(args.get("post-external-wait-ms") ?? DEFAULT_POST_EXTERNAL_WAIT_MS)),
  };
}

function scheduledTickTimesForScenario(scenario: Scenario, condition: Condition): number[] {
  if (condition.tickMode === "none") return [];
  if (condition.tickMode === "every_3000ms") {
    const ticks: number[] = [];
    for (let tick = 3000; tick < scenario.latencyMs; tick += 3000) ticks.push(tick);
    return ticks;
  }
  if (scenario.latencyMs <= 3000) return [1500].filter((tick) => tick < scenario.latencyMs);
  if (scenario.latencyMs <= 5000) return [2000].filter((tick) => tick < scenario.latencyMs);
  if (scenario.latencyMs <= 8000) return [3000, 6000].filter((tick) => tick < scenario.latencyMs);
  if (scenario.latencyMs <= 12000) return [3000, 6000, 9000].filter((tick) => tick < scenario.latencyMs);
  return [3000, 6000, 9000].filter((tick) => tick < scenario.latencyMs);
}

function msDelta(start: number | undefined, value: number | undefined): number | null {
  if (!start || !value) return null;
  return value - start;
}

function eventTypes(message: LiveMessage): string[] {
  return [
    message.setupComplete ? "setupComplete" : undefined,
    message.serverContent ? "serverContent" : undefined,
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

function writeAudioFile(pathWithoutExt: string, chunks: Buffer[], mimeType: string | undefined): { path: string; format: string } | null {
  if (chunks.length === 0) return null;
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

function writeTimelineAudioFile(pathWithoutExt: string, segments: AudioSegment[], mimeType: string | undefined): { path: string; format: string; durationMs?: number } | null {
  if (segments.length === 0) return null;
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

function summarizeError(error: unknown): string {
  if (error instanceof Error) return `${error.name}: ${error.message}`;
  if (typeof error !== "object" || error === null) return String(error);
  const record = error as Record<string, unknown>;
  return ["message", "type", "code", "reason", "error"]
    .map((key) => (record[key] ? `${key}: ${String(record[key])}` : undefined))
    .filter(Boolean)
    .join(", ") || Object.prototype.toString.call(error);
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
                      bytes: part.inlineData.data ? decodeAudio(part.inlineData.data).length : 0,
                    },
                  };
                }),
              }
            : undefined,
        }
      : undefined,
  };
}

function writeJson(path: string, value: unknown): void {
  writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

function writePromptsFile(conditionDir: string, condition: Condition): void {
  writeFileSync(
    resolve(conditionDir, "prompts.txt"),
    [
      `PROMPT_VERSION=${PROMPT_VERSION}`,
      `CONDITION=${condition.id}`,
      `TICK_MODE=${condition.tickMode}`,
      "",
      "PROMPT:",
      PROMPT,
      "",
      "SYSTEM_INSTRUCTION:",
      SYSTEM_INSTRUCTION,
      "",
    ].join("\n"),
    "utf8",
  );
}

function finalAnswerFromTexts(texts: string[]): string | null {
  const normalized = texts
    .map((text) => text.trim())
    .filter(Boolean)
    .join(" ")
    .trim();
  return normalized || null;
}

function tickResponseStats(metrics: ScenarioMetrics, scenario: Scenario): { tickResponseCount: number | null; tickResponseRate: number | null } {
  if (metrics.pendingTickTimes.length === 0) return { tickResponseCount: null, tickResponseRate: null };
  let count = 0;
  for (let index = 0; index < metrics.pendingTickTimes.length; index += 1) {
    const start = metrics.pendingTickTimes[index];
    const end = metrics.pendingTickTimes[index + 1] ?? metrics.externalResultInjectedAt ?? (metrics.promptSentAt ?? 0) + scenario.latencyMs;
    if (metrics.assistantOutputTimes.some((time) => time >= start && time < end)) count += 1;
  }
  return { tickResponseCount: count, tickResponseRate: count / metrics.pendingTickTimes.length };
}

async function runScenario(
  ai: GoogleGenAI,
  model: string,
  condition: Condition,
  scenario: Scenario,
  conditionDir: string,
  log: ReturnType<typeof createWriteStream>,
  postExternalWaitMs: number,
): Promise<ScenarioSummary> {
  const audioDir = resolve(conditionDir, "audio");
  mkdirSync(audioDir, { recursive: true });

  const metrics: ScenarioMetrics = {
    assistantOutputCountBeforeExternalResult: 0,
    assistantOutputCountAfterExternalResult: 0,
    pendingTickTimes: [],
    assistantOutputTimes: [],
    sessionClosed: false,
    errors: [],
    eventTypesSeen: new Set<string>(),
  };
  const audioChunks: Buffer[] = [];
  const timelineSegments: AudioSegment[] = [];
  const textOutputs: string[] = [];
  const transcriptionOutputs: string[] = [];
  const timers: ReturnType<typeof setTimeout>[] = [];
  let audioMimeType: string | undefined;
  let session: Session | undefined;
  let done = false;
  let closeReason = "";

  const relativeTime = () => (metrics.promptSentAt ? Date.now() - metrics.promptSentAt : 0);
  const logLine = (eventType: string, data: Record<string, unknown> = {}) => {
    log.write(
      `${JSON.stringify({
        ts: new Date().toISOString(),
        condition: condition.id,
        scenario_id: scenario.id,
        relative_time_ms: relativeTime(),
        event_type: eventType,
        data,
      })}\n`,
    );
  };
  const clearTimers = () => {
    while (timers.length) clearTimeout(timers.pop()!);
  };
  const finish = (reason: string, resolveScenario: () => void) => {
    if (done) return;
    done = true;
    clearTimers();
    logLine("scenario_finished", { reason });
    if (!metrics.sessionClosed) {
      try {
        session?.close();
      } catch {
        // Socket may already be closed.
      }
    }
    resolveScenario();
  };

  await new Promise<void>(async (resolveScenario) => {
    const hardTimer = setTimeout(() => {
      metrics.errors.push(`max scenario timeout ${MAX_SCENARIO_MS}ms`);
      finish("max scenario timeout", resolveScenario);
    }, MAX_SCENARIO_MS);
    timers.push(hardTimer);

    try {
      console.log("");
      console.log(`[${condition.id}/${scenario.id}] latency: ${scenario.latencyMs} ms`);

      session = (await ai.live.connect({
        model,
        config: {
          responseModalities: [Modality.AUDIO],
          outputAudioTranscription: {},
          systemInstruction: SYSTEM_INSTRUCTION,
        },
        callbacks: {
          onopen: () => {
            metrics.openedAt = Date.now();
            logLine("session_opened", { model, tools_enabled: false });
          },
          onmessage: (message: LiveMessage) => {
            const now = Date.now();
            const types = eventTypes(message);
            for (const type of types) metrics.eventTypesSeen.add(type);
            logLine("server_event", { event_types: types, message: sanitizeMessage(message) });

            let assistantOutputSeenInEvent = false;
            for (const part of message.serverContent?.modelTurn?.parts ?? []) {
              if (part.text) {
                assistantOutputSeenInEvent = true;
                metrics.firstTextAt ??= now;
                textOutputs.push(part.text);
                console.log(`[${condition.id}/${scenario.id}] text: ${part.text}`);
              }
              if (part.inlineData?.data) {
                assistantOutputSeenInEvent = true;
                const chunk = decodeAudio(part.inlineData.data);
                metrics.firstAudioAt ??= now;
                audioMimeType ??= part.inlineData.mimeType;
                audioChunks.push(chunk);
                timelineSegments.push({
                  offsetMs: Math.max(0, now - (metrics.promptSentAt ?? now)),
                  chunk,
                });
                if (metrics.externalResultInjectedAt) metrics.firstAudioAfterExternalResultAt ??= now;
              }
            }

            const transcription = message.serverContent?.outputTranscription?.text;
            if (transcription) {
              assistantOutputSeenInEvent = true;
              transcriptionOutputs.push(transcription);
              logLine("output_transcription", { text: transcription });
            }

            if (assistantOutputSeenInEvent) {
              metrics.firstAssistantOutputAt ??= now;
              metrics.assistantOutputTimes.push(now);
              if (metrics.externalResultInjectedAt) {
                metrics.assistantOutputCountAfterExternalResult += 1;
                metrics.firstOutputAfterExternalResultAt ??= now;
              } else {
                metrics.assistantOutputCountBeforeExternalResult += 1;
              }
            }

          },
          onerror: (error) => {
            const summary = summarizeError(error);
            metrics.errors.push(summary);
            logLine("socket_error", { error: summary });
          },
          onclose: (event: { code?: number; reason?: string }) => {
            metrics.sessionClosed = true;
            metrics.closeCode = event.code;
            metrics.closeReason = event.reason;
            closeReason = `socket closed code=${event.code ?? "unknown"} reason=${event.reason || "none"}`;
            logLine("session_closed", { code: event.code, reason: event.reason });
            clearTimers();
            resolveScenario();
          },
        },
      })) as Session;

      metrics.openedAt ??= Date.now();
      session.sendClientContent({ turns: PROMPT, turnComplete: true });
      metrics.promptSentAt = Date.now();
      logLine("user_prompt_sent", { prompt: PROMPT });

      if (condition.tickMode !== "none") {
        for (const tickAt of scheduledTickTimesForScenario(scenario, condition)) {
          const timer = setTimeout(() => {
            if (done || metrics.sessionClosed || metrics.externalResultInjectedAt) return;
            try {
              session?.sendClientContent({ turns: EXTERNAL_STATUS, turnComplete: true });
              metrics.pendingTickTimes.push(Date.now());
              logLine("external_status_sent", { message: EXTERNAL_STATUS, scheduled_tick_ms: tickAt });
            } catch (error) {
              const summary = summarizeError(error);
              metrics.errors.push(summary);
              logLine("external_status_send_error", { error: summary });
            }
          }, tickAt);
          timers.push(timer);
        }
      }

      const resultTimer = setTimeout(() => {
        if (done || metrics.sessionClosed) {
          logLine("external_result_skipped_after_close", { latency_ms: scenario.latencyMs });
          return;
        }
        const externalResult = `EXTERNAL_EVENT
type: ready
has_final_answer: true
final_answer: ${EXPECTED_FINAL_ANSWER}
instruction: answer_now_with_final_answer_only`;
        try {
          session?.sendClientContent({ turns: externalResult, turnComplete: true });
          metrics.externalResultInjectedAt = Date.now();
          logLine("external_result_injected", { final_answer: EXPECTED_FINAL_ANSWER, message: externalResult });
          const postWaitTimer = setTimeout(() => {
            finish("post external observation window elapsed", resolveScenario);
          }, postExternalWaitMs);
          timers.push(postWaitTimer);
        } catch (error) {
          const summary = summarizeError(error);
          metrics.errors.push(summary);
          logLine("external_result_inject_error", { error: summary });
        }
      }, scenario.latencyMs);
      timers.push(resultTimer);
    } catch (error) {
      const summary = summarizeError(error);
      metrics.errors.push(summary);
      closeReason = summary;
      logLine("scenario_error", { error: summary });
      clearTimers();
      resolveScenario();
    }
  });

  const compressed = writeAudioFile(resolve(conditionDir, "audio", `${scenario.id}_compressed`), audioChunks, audioMimeType);
  const timeline = writeTimelineAudioFile(resolve(conditionDir, "audio", `${scenario.id}_timeline`), timelineSegments, audioMimeType);
  const transcript = finalAnswerFromTexts([...textOutputs, ...transcriptionOutputs]);
  const promptSentAt = metrics.promptSentAt;
  const firstResponseLatencyMs = msDelta(promptSentAt, metrics.firstAssistantOutputAt);
  const externalResultInjectedTimeMs = metrics.externalResultInjectedAt ? msDelta(promptSentAt, metrics.externalResultInjectedAt) : null;
  const firstOutputAfterExternalResultTimeMs = metrics.firstOutputAfterExternalResultAt ? msDelta(promptSentAt, metrics.firstOutputAfterExternalResultAt) : null;
  const postExternalFirstResponseLatencyMs = msDelta(metrics.externalResultInjectedAt, metrics.firstOutputAfterExternalResultAt);
  const server1008 = metrics.closeCode === 1008;
  const server1011 = metrics.closeCode === 1011;
  const errorType = server1008
    ? "server_1008_error"
    : server1011
    ? "server_1011_error"
    : metrics.errors.length
    ? "session_error"
    : metrics.externalResultInjectedAt
    ? null
    : "external_result_not_injected";
  const valid = errorType === null;
  const tickStats = tickResponseStats(metrics, scenario);
  const outputRate = scenario.latencyMs > 0 ? metrics.assistantOutputCountBeforeExternalResult / (scenario.latencyMs / 1000) : null;

  const summary: ScenarioSummary = {
    condition: condition.id,
    scenario_id: scenario.id,
    scenario: scenario.id,
    latency_ms: scenario.latencyMs,
    tick_mode: condition.tickMode,
    pending_tick_schedule_ms: scheduledTickTimesForScenario(scenario, condition),
    pending_tick_count_sent: metrics.pendingTickTimes.length,
    tick_response_count: tickStats.tickResponseCount,
    tick_response_rate: tickStats.tickResponseRate,
    valid,
    error_type: errorType,
    server_1008_error: server1008,
    server_1011_error: server1011,
    close_reason: closeReason || null,
    prompt_version: PROMPT_VERSION,
    prompt_sent_time_ms: 0,
    external_result_injected_time_ms: externalResultInjectedTimeMs,
    first_assistant_output_time_ms: firstResponseLatencyMs,
    first_response_latency_ms: firstResponseLatencyMs,
    assistant_output_count_before_external_result: metrics.assistantOutputCountBeforeExternalResult,
    assistant_output_rate_before_external_result_per_sec: outputRate,
    first_output_after_external_result_time_ms: firstOutputAfterExternalResultTimeMs,
    post_external_first_response_latency_ms: postExternalFirstResponseLatencyMs,
    post_external_wait_ms: postExternalWaitMs,
    has_output_after_external_result: metrics.assistantOutputCountAfterExternalResult > 0,
    output_exists_after_external_result: metrics.assistantOutputCountAfterExternalResult > 0,
    assistant_output_count_after_external_result: metrics.assistantOutputCountAfterExternalResult,
    final_answer_expected: EXPECTED_FINAL_ANSWER,
    final_answer_observed: null,
    final_exact_match: null,
    premature_answer_detection: "not_available_without_asr",
    post_external_answer_success: "not_available_without_asr",
    output_transcript_observed: transcript,
    event_types_seen: [...metrics.eventTypesSeen],
    audio_file: compressed ? relative(PROJECT_DIR, compressed.path) : "none",
    audio_format: compressed?.format ?? "none",
    timeline_audio_file: timeline ? relative(PROJECT_DIR, timeline.path) : "none",
    timeline_audio_format: timeline?.format ?? "none",
    timeline_audio_duration_ms: timeline?.durationMs ?? null,
    timings: {
      promptSentAt: metrics.promptSentAt,
      externalResultInjectedAt: metrics.externalResultInjectedAt,
      firstAssistantOutputAt: metrics.firstAssistantOutputAt,
      firstTextAt: metrics.firstTextAt,
      firstAudioAt: metrics.firstAudioAt,
      firstOutputAfterExternalResultAt: metrics.firstOutputAfterExternalResultAt,
      firstAudioAfterExternalResultAt: metrics.firstAudioAfterExternalResultAt,
    },
  };
  logLine("scenario_summary", summary);

  console.log(`Scenario: ${condition.id}/${scenario.id}`);
  console.log(`Valid: ${valid ? "yes" : "no"}${errorType ? ` (${errorType})` : ""}`);
  console.log(`External result injected: ${externalResultInjectedTimeMs ?? "n/a"} ms`);
  console.log(`First response latency: ${firstResponseLatencyMs ?? "n/a"} ms`);
  console.log(`Outputs before external result: ${metrics.assistantOutputCountBeforeExternalResult}`);
  console.log(`Post-external first response latency: ${postExternalFirstResponseLatencyMs ?? "n/a"} ms`);
  console.log(`Pending ticks sent: ${metrics.pendingTickTimes.length}`);
  console.log(`Tick response rate: ${tickStats.tickResponseRate ?? "n/a"}`);

  return summary;
}

function fmt(value: unknown): string {
  return value === null || value === undefined ? "n/a" : String(value);
}

function writeTextEventSummary(conditionDir: string, summaries: ScenarioSummary[]): void {
  const blocks = summaries.map((scenario) =>
    [
      `Scenario: ${scenario.scenario_id}`,
      `Latency: ${scenario.latency_ms}ms`,
      `Condition: ${scenario.condition}`,
      `Pending ticks sent: ${fmt(scenario.pending_tick_count_sent)}`,
      `Pending tick schedule: ${fmt(Array.isArray(scenario.pending_tick_schedule_ms) ? scenario.pending_tick_schedule_ms.join(",") : scenario.pending_tick_schedule_ms)}`,
      `Tick response rate: ${fmt(scenario.tick_response_rate)}`,
      `First assistant output: ${fmt(scenario.first_response_latency_ms)}ms`,
      `External result injected: ${fmt(scenario.external_result_injected_time_ms)}ms`,
      `First output after external result: ${fmt(scenario.first_output_after_external_result_time_ms)}ms`,
      `Assistant output events before external result: ${fmt(scenario.assistant_output_count_before_external_result)}`,
      `Errors: ${scenario.error_type ?? "none"}`,
    ].join("\n"),
  );
  writeFileSync(resolve(conditionDir, "text_event_summary.txt"), `${blocks.join("\n\n------------------------------------------------------------\n\n")}\n`, "utf8");
}

function writeLegacyAnalysis(conditionDir: string, summaries: ScenarioSummary[]): void {
  const lines = [`Prompt version: ${PROMPT_VERSION}`, `Expected final answer: ${EXPECTED_FINAL_ANSWER}`, ""];
  for (const scenario of summaries) {
    lines.push(
      `Scenario: ${scenario.scenario_id}`,
      `Condition: ${scenario.condition}`,
      `Latency: ${scenario.latency_ms} ms`,
      `Valid: ${scenario.valid ? "yes" : "no"}`,
      `Pending ticks sent: ${fmt(scenario.pending_tick_count_sent)}`,
      `Tick response rate: ${fmt(scenario.tick_response_rate)}`,
      `First response latency: ${fmt(scenario.first_response_latency_ms)} ms`,
      `Output count before external result: ${fmt(scenario.assistant_output_count_before_external_result)}`,
      `Output rate before external result: ${fmt(scenario.assistant_output_rate_before_external_result_per_sec)} events/sec`,
      `Post external first response latency: ${fmt(scenario.post_external_first_response_latency_ms)} ms`,
      `Output exists after external result: ${scenario.output_exists_after_external_result ? "yes" : "no"}`,
      `Server 1008: ${scenario.server_1008_error ? "yes" : "no"}`,
      `Server 1011: ${scenario.server_1011_error ? "yes" : "no"}`,
      `Other error: ${scenario.error_type && !scenario.server_1008_error && !scenario.server_1011_error ? scenario.error_type : "none"}`,
      `Audio file: ${fmt(scenario.audio_file)}`,
      "",
    );
  }
  writeFileSync(resolve(conditionDir, "legacy-analysis.txt"), `${lines.join("\n")}\n`, "utf8");
}

function runExternalTimelineVisualization(conditionDir: string): void {
  execFileSync("python3", [resolve(PROJECT_DIR, "scripts", "visualize_external_wait.py"), conditionDir], {
    cwd: PROJECT_DIR,
    stdio: "inherit",
  });
}

async function runCondition(
  ai: GoogleGenAI,
  model: string,
  rootDir: string,
  condition: Condition,
  scenarios: Scenario[],
  postExternalWaitMs: number,
): Promise<ScenarioSummary[]> {
  const conditionDir = resolve(rootDir, condition.id);
  mkdirSync(conditionDir, { recursive: true });
  mkdirSync(resolve(conditionDir, "audio"), { recursive: true });
  writePromptsFile(conditionDir, condition);
  const log = createWriteStream(resolve(conditionDir, "raw_log.jsonl"), { flags: "a" });
  log.write(
    `${JSON.stringify({
      ts: new Date().toISOString(),
      condition: condition.id,
      scenario_id: null,
      relative_time_ms: 0,
      event_type: "prompt_metadata",
      data: {
        prompt_version: PROMPT_VERSION,
        prompt: PROMPT,
        system_instruction: SYSTEM_INSTRUCTION,
        tick_mode: condition.tickMode,
        post_external_wait_ms: postExternalWaitMs,
      },
    })}\n`,
  );

  const summaries: ScenarioSummary[] = [];
  for (const scenario of scenarios) {
    summaries.push(await runScenario(ai, model, condition, scenario, conditionDir, log, postExternalWaitMs));
  }
  const summary = {
    condition: condition.id,
    prompt_version: PROMPT_VERSION,
    prompt: PROMPT,
    system_instruction: SYSTEM_INSTRUCTION,
    expected_final_answer: EXPECTED_FINAL_ANSWER,
    model,
    tick_mode: condition.tickMode,
    post_external_wait_ms: postExternalWaitMs,
    raw_log_file: "raw_log.jsonl",
    scenarios: summaries,
  };
  writeJson(resolve(conditionDir, "summary.json"), summary);
  writeTextEventSummary(conditionDir, summaries);
  writeLegacyAnalysis(conditionDir, summaries);
  await new Promise<void>((resolveLog) => log.end(resolveLog));
  runExternalTimelineVisualization(conditionDir);
  return summaries;
}

function csvEscape(value: unknown): string {
  if (value === null || value === undefined) return "";
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function comparisonRows(results: Record<string, ScenarioSummary[]>): ScenarioSummary[] {
  return Object.values(results).flat();
}

function conditionFromName(name: string): Condition {
  if (name === "no_tick" || name === "condition_no_tick") return { id: "condition_no_tick", tickMode: "none" };
  if (name === "tick_every_3000ms" || name === "condition_tick_every_3000ms") {
    return { id: "condition_tick_every_3000ms", tickMode: "every_3000ms" };
  }
  if (name === "scheduled_tick" || name === "condition_scheduled_tick") {
    return { id: "condition_scheduled_tick", tickMode: "scheduled" };
  }
  throw new Error(`Unknown condition: ${name}`);
}

function writeComparisonCsv(path: string, rows: ScenarioSummary[]): void {
  const headers = [
    "condition",
    "scenario_id",
    "latency_ms",
    "valid",
    "server_1008_error",
    "server_1011_error",
    "first_response_latency_ms",
    "assistant_output_count_before_external_result",
    "assistant_output_rate_before_external_result_per_sec",
    "pending_tick_count_sent",
    "tick_response_rate",
    "post_external_first_response_latency_ms",
    "post_external_wait_ms",
    "has_output_after_external_result",
    "output_exists_after_external_result",
    "assistant_output_count_after_external_result",
  ];
  writeFileSync(
    path,
    `${headers.join(",")}\n${rows.map((row) => headers.map((header) => csvEscape(row[header])).join(",")).join("\n")}\n`,
    "utf8",
  );
}

function writeComparisonAnalysis(path: string, rows: ScenarioSummary[]): void {
  const byScenario = new Map<string, ScenarioSummary[]>();
  for (const row of rows) {
    const scenario = String(row.scenario_id);
    byScenario.set(scenario, [...(byScenario.get(scenario) ?? []), row]);
  }
  const lines = [
    "External wait tick comparison",
    "",
    "Question 1: tick 是否讓 external result 前的 assistant output 變多",
    "Question 2: tick 是否讓 external result 後比較容易回答",
    "Question 3: tick 是否降低 post external first response latency",
    "Question 4: tick 是否增加 1008 / 1011",
    "Question 5: tick condition 有沒有明顯過度輸出",
    "Question 6: 哪個 condition 比較適合作為正式 benchmark",
    "",
  ];
  for (const [scenario, scenarioRows] of byScenario) {
    const noTick = scenarioRows.find((row) => row.condition === "condition_no_tick");
    const tick =
      scenarioRows.find((row) => row.condition === "condition_tick_every_3000ms") ??
      scenarioRows.find((row) => row.condition === "condition_scheduled_tick");
    lines.push(`Scenario: ${scenario}`);
    if (noTick && tick) {
      const beforeDelta =
        Number(tick.assistant_output_count_before_external_result ?? 0) -
        Number(noTick.assistant_output_count_before_external_result ?? 0);
      const postNoTick = noTick.post_external_first_response_latency_ms;
      const postTick = tick.post_external_first_response_latency_ms;
      const postDelta =
        typeof postNoTick === "number" && typeof postTick === "number" ? postTick - postNoTick : null;
      lines.push(
        `- before-result output delta tick - no_tick: ${beforeDelta}`,
        `- post-external latency delta tick - no_tick: ${fmt(postDelta)} ms`,
        `- no_tick valid/error: ${noTick.valid ? "valid" : noTick.error_type}`,
        `- tick valid/error: ${tick.valid ? "valid" : tick.error_type}`,
        `- tick response rate: ${fmt(tick.tick_response_rate)}`,
      );
    } else {
      lines.push("- missing one condition result");
    }
    lines.push("");
  }

  const tickRows = rows.filter((row) => row.condition === "condition_tick_every_3000ms" || row.condition === "condition_scheduled_tick");
  const noTickRows = rows.filter((row) => row.condition === "condition_no_tick");
  const tickErrors = tickRows.filter((row) => row.server_1008_error || row.server_1011_error).length;
  const noTickErrors = noTickRows.filter((row) => row.server_1008_error || row.server_1011_error).length;
  const tickValid = tickRows.filter((row) => row.valid).length;
  const noTickValid = noTickRows.filter((row) => row.valid).length;
  lines.push(
    "Overall:",
    `- no_tick valid: ${noTickValid}/${noTickRows.length}`,
    `- tick valid: ${tickValid}/${tickRows.length}`,
    `- no_tick 1008/1011 count: ${noTickErrors}`,
    `- tick 1008/1011 count: ${tickErrors}`,
    "- Recommendation should be based on the per-latency table above; invalid scenarios are preserved but should not be averaged into behavior metrics.",
    "",
  );
  writeFileSync(path, `${lines.join("\n")}\n`, "utf8");
}

async function main(): Promise<void> {
  mkdirSync(RESULT_DIR, { recursive: true });
  const args = parseArgs(process.argv.slice(2));
  const apiKey = requireEnv("GEMINI_API_KEY");
  const model = requireEnv("GEMINI_LIVE_MODEL");
  const runId = `external_wait_tick_compare_${timestampForPath()}`;
  const rootDir = resolve(RESULT_DIR, runId);
  mkdirSync(rootDir, { recursive: true });
  const conditions = args.conditions.map(conditionFromName);
  writeJson(resolve(rootDir, "config.json"), {
    run_id: runId,
    prompt_version: PROMPT_VERSION,
    model,
    latencies_ms: args.scenarios.map((scenario) => scenario.latencyMs),
    post_external_wait_ms: args.postExternalWaitMs,
    scheduled_tick_rules: {
      slow_correct_3s: [1500],
      slow_correct_5s: [2000],
      slow_correct_8s: [3000, 6000],
      slow_correct_12s: [3000, 6000, 9000],
    },
    every_3000ms_tick_rules: {
      slow_correct_3s: [],
      slow_correct_5s: [3000],
      slow_correct_8s: [3000, 6000],
      slow_correct_12s: [3000, 6000, 9000],
    },
    conditions,
  });

  console.log(`Running external wait tick compare for model: ${model}`);
  console.log(`Result directory: ${relative(PROJECT_DIR, rootDir)}`);
  const ai = new GoogleGenAI({ apiKey });
  const results: Record<string, ScenarioSummary[]> = {};
  for (const condition of conditions) {
    console.log("");
    console.log(`Condition: ${condition.id}`);
    results[condition.id] = await runCondition(ai, model, rootDir, condition, args.scenarios, args.postExternalWaitMs);
  }

  const rows = comparisonRows(results);
  writeJson(resolve(rootDir, "comparison-summary.json"), {
    run_id: runId,
    prompt_version: PROMPT_VERSION,
    model,
    post_external_wait_ms: args.postExternalWaitMs,
    rows,
  });
  writeComparisonCsv(resolve(rootDir, "comparison-summary.csv"), rows);
  writeComparisonAnalysis(resolve(rootDir, "comparison-analysis.txt"), rows);

  console.log("");
  console.log(`Result directory: ${relative(PROJECT_DIR, rootDir)}`);
  console.log(`Comparison summary: ${relative(PROJECT_DIR, resolve(rootDir, "comparison-summary.json"))}`);
  console.log(`Comparison analysis: ${relative(PROJECT_DIR, resolve(rootDir, "comparison-analysis.txt"))}`);
}

main().catch((error) => {
  console.error("external-wait-tick-compare failed");
  console.error(summarizeError(error));
  process.exitCode = 1;
});
