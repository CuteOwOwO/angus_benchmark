import { GoogleGenAI, Modality } from "@google/genai";
import { config as loadEnv } from "dotenv";
import { createWriteStream, mkdirSync, writeFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { dirname, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { PROMPT, PROMPT_VERSION, SYSTEM_INSTRUCTION } from "./external_prompt.js";

const PROJECT_DIR = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const DEFAULT_ENV_FILE = resolve(PROJECT_DIR, "..", "angus_bench", ".env");
const RESULT_DIR = resolve(PROJECT_DIR, "result");
const EXPECTED_FINAL_ANSWER = "Angus";
const POST_EXTERNAL_TIMEOUT_MS = 15_000;
const MAX_SCENARIO_MS = 45_000;

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

function timestamp(): string {
  return new Date().toISOString().replace(/[:.]/g, "-");
}

function parseLatencies(argv: string[]): Scenario[] {
  const argIndex = argv.findIndex((arg) => arg === "--latencies" || arg.startsWith("--latencies="));
  if (argIndex < 0) return DEFAULT_SCENARIOS;
  const raw = argv[argIndex].includes("=") ? argv[argIndex].split("=", 2)[1] : argv[argIndex + 1];
  if (!raw) return DEFAULT_SCENARIOS;
  return raw
    .split(",")
    .map((item) => Number(item.trim()))
    .filter((value) => Number.isFinite(value) && value >= 0)
    .map((latencyMs) => ({ id: `slow_correct_${Math.round(latencyMs / 1000)}s`, latencyMs }));
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

function writeTimelineAudioFile(
  pathWithoutExt: string,
  segments: AudioSegment[],
  mimeType: string | undefined,
): { path: string; format: string; durationMs?: number } | null {
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

function writePromptsFile(resultRunDir: string): void {
  writeFileSync(
    resolve(resultRunDir, "prompts.txt"),
    [
      `PROMPT_VERSION=${PROMPT_VERSION}`,
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

async function runScenario(
  ai: GoogleGenAI,
  model: string,
  scenario: Scenario,
  resultRunDir: string,
  log: ReturnType<typeof createWriteStream>,
): Promise<ScenarioSummary> {
  const audioDir = resolve(resultRunDir, "audio");
  mkdirSync(audioDir, { recursive: true });

  const metrics: ScenarioMetrics = {
    assistantOutputCountBeforeExternalResult: 0,
    assistantOutputCountAfterExternalResult: 0,
    sessionClosed: false,
    errors: [],
    eventTypesSeen: new Set<string>(),
  };
  const audioChunks: Buffer[] = [];
  const timelineSegments: AudioSegment[] = [];
  const textOutputs: string[] = [];
  const transcriptionOutputs: string[] = [];
  let audioMimeType: string | undefined;
  let session: Session | undefined;
  let done = false;
  let externalTimer: ReturnType<typeof setTimeout> | undefined;
  let closeReason = "";

  const relativeTime = () => (metrics.promptSentAt ? Date.now() - metrics.promptSentAt : 0);
  const logLine = (eventType: string, data: Record<string, unknown> = {}) => {
    log.write(
      `${JSON.stringify({
        ts: new Date().toISOString(),
        scenario_id: scenario.id,
        relative_time_ms: relativeTime(),
        event_type: eventType,
        data,
      })}\n`,
    );
  };

  const finish = (reason: string, resolveScenario: () => void) => {
    if (done) return;
    done = true;
    if (externalTimer) clearTimeout(externalTimer);
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

    try {
      console.log("");
      console.log(`[${scenario.id}] PROMPT_VERSION: ${PROMPT_VERSION}`);
      console.log(`[${scenario.id}] latency: ${scenario.latencyMs} ms`);

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
            const parts = message.serverContent?.modelTurn?.parts ?? [];
            for (const part of parts) {
              if (part.text) {
                assistantOutputSeenInEvent = true;
                metrics.firstTextAt ??= now;
                textOutputs.push(part.text);
                console.log(`[${scenario.id}] text: ${part.text}`);
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
              if (metrics.externalResultInjectedAt) {
                metrics.assistantOutputCountAfterExternalResult += 1;
                metrics.firstOutputAfterExternalResultAt ??= now;
              } else {
                metrics.assistantOutputCountBeforeExternalResult += 1;
              }
            }

            if (metrics.externalResultInjectedAt && message.serverContent?.turnComplete) {
              clearTimeout(hardTimer);
              setTimeout(() => finish("turn complete after external result", resolveScenario), 250);
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
            clearTimeout(hardTimer);
            if (externalTimer) clearTimeout(externalTimer);
            resolveScenario();
          },
        },
      })) as Session;

      metrics.openedAt ??= Date.now();
      session.sendClientContent({ turns: PROMPT, turnComplete: true });
      metrics.promptSentAt = Date.now();
      logLine("user_prompt_sent", { prompt: PROMPT });

      externalTimer = setTimeout(() => {
        if (done || metrics.sessionClosed) {
          logLine("external_result_skipped_after_close", { latency_ms: scenario.latencyMs });
          return;
        }
        const externalResult = `EXTERNAL_RESULT:\nfinal_answer: ${EXPECTED_FINAL_ANSWER}`;
        try {
          session?.sendClientContent({ turns: externalResult, turnComplete: true });
          metrics.externalResultInjectedAt = Date.now();
          logLine("external_result_injected", { final_answer: EXPECTED_FINAL_ANSWER, message: externalResult });
        } catch (error) {
          const summary = summarizeError(error);
          metrics.errors.push(summary);
          logLine("external_result_inject_error", { error: summary });
        }
      }, scenario.latencyMs);
    } catch (error) {
      const summary = summarizeError(error);
      metrics.errors.push(summary);
      closeReason = summary;
      logLine("scenario_error", { error: summary });
      clearTimeout(hardTimer);
      resolveScenario();
    }
  });

  const compressed = writeAudioFile(resolve(audioDir, `${scenario.id}_compressed`), audioChunks, audioMimeType);
  const timeline = writeTimelineAudioFile(resolve(audioDir, `${scenario.id}_timeline`), timelineSegments, audioMimeType);
  const observed = finalAnswerFromTexts([...textOutputs, ...transcriptionOutputs]);
  const finalExactMatch = observed ? observed === EXPECTED_FINAL_ANSWER : null;
  const promptSentAt = metrics.promptSentAt;
  const firstResponseLatencyMs = msDelta(promptSentAt, metrics.firstAssistantOutputAt);
  const externalResultInjectedTimeMs = metrics.externalResultInjectedAt
    ? msDelta(promptSentAt, metrics.externalResultInjectedAt)
    : null;
  const firstOutputAfterExternalResultTimeMs = metrics.firstOutputAfterExternalResultAt
    ? msDelta(promptSentAt, metrics.firstOutputAfterExternalResultAt)
    : null;
  const postExternalFirstResponseLatencyMs = msDelta(
    metrics.externalResultInjectedAt,
    metrics.firstOutputAfterExternalResultAt,
  );
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
  const outputRate = scenario.latencyMs > 0
    ? metrics.assistantOutputCountBeforeExternalResult / (scenario.latencyMs / 1000)
    : null;

  const summary: ScenarioSummary = {
    scenario_id: scenario.id,
    scenario: scenario.id,
    latency_ms: scenario.latencyMs,
    latencyMs: scenario.latencyMs,
    valid,
    error_type: errorType,
    close_reason: closeReason || null,
    prompt_version: PROMPT_VERSION,
    prompt_sent_time_ms: 0,
    external_result_injected_time_ms: externalResultInjectedTimeMs,
    first_assistant_output_time_ms: firstResponseLatencyMs,
    first_response_latency_ms: firstResponseLatencyMs,
    first_output_after_external_result_time_ms: firstOutputAfterExternalResultTimeMs,
    post_external_first_response_latency_ms: postExternalFirstResponseLatencyMs,
    assistant_output_count_before_external_result: metrics.assistantOutputCountBeforeExternalResult,
    assistant_output_count_after_external_result: metrics.assistantOutputCountAfterExternalResult,
    assistant_output_rate_before_external_result_per_sec: outputRate,
    final_answer_expected: EXPECTED_FINAL_ANSWER,
    final_answer_observed: observed,
    final_exact_match: finalExactMatch,
    premature_answer_detection: "not_available_without_asr",
    server_1008_error: server1008,
    server_1011_error: server1011,
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

  console.log(`Scenario: ${scenario.id}`);
  console.log(`Latency: ${scenario.latencyMs} ms`);
  console.log(`External result injected: ${externalResultInjectedTimeMs ?? "n/a"} ms`);
  console.log(`First response latency: ${firstResponseLatencyMs ?? "n/a"} ms`);
  console.log(`Post-external first response latency: ${postExternalFirstResponseLatencyMs ?? "n/a"} ms`);
  console.log(`Assistant outputs before external result: ${metrics.assistantOutputCountBeforeExternalResult}`);
  console.log(`Audio file: ${summary.audio_file}`);
  console.log(`Timeline audio file: ${summary.timeline_audio_file}`);
  console.log(`Result: ${valid ? "PASS" : "FAIL"}${errorType ? ` (${errorType})` : ""}`);

  return summary;
}

function fmt(value: unknown): string {
  return value === null || value === undefined ? "n/a" : String(value);
}

function writeTextEventSummary(resultRunDir: string, summaries: ScenarioSummary[]): void {
  const blocks = summaries.map((scenario) =>
    [
      `Scenario: ${scenario.scenario_id}`,
      `Latency: ${scenario.latency_ms}ms`,
      `First assistant output: ${fmt(scenario.first_response_latency_ms)}ms`,
      `External result injected: ${fmt(scenario.external_result_injected_time_ms)}ms`,
      `First output after external result: ${fmt(scenario.first_output_after_external_result_time_ms)}ms`,
      `Assistant output events before external result: ${fmt(scenario.assistant_output_count_before_external_result)}`,
      `Errors: ${scenario.error_type ?? "none"}`,
    ].join("\n"),
  );
  writeFileSync(resolve(resultRunDir, "text_event_summary.txt"), `${blocks.join("\n\n------------------------------------------------------------\n\n")}\n`, "utf8");
}

function writeLegacyAnalysis(resultRunDir: string, summaries: ScenarioSummary[]): void {
  const lines = [
    `Prompt version: ${PROMPT_VERSION}`,
    `Expected final answer: ${EXPECTED_FINAL_ANSWER}`,
    "",
  ];
  for (const scenario of summaries) {
    lines.push(
      `Scenario: ${scenario.scenario_id}`,
      `Latency: ${scenario.latency_ms} ms`,
      `Valid: ${scenario.valid ? "yes" : "no"}`,
      `First response latency: ${fmt(scenario.first_response_latency_ms)} ms`,
      `Output count before external result: ${fmt(scenario.assistant_output_count_before_external_result)}`,
      `Output rate before external result: ${fmt(scenario.assistant_output_rate_before_external_result_per_sec)} events/sec`,
      `Post external first response latency: ${fmt(scenario.post_external_first_response_latency_ms)} ms`,
      `Server 1008: ${scenario.server_1008_error ? "yes" : "no"}`,
      `Server 1011: ${scenario.server_1011_error ? "yes" : "no"}`,
      `Other error: ${scenario.error_type && !scenario.server_1008_error && !scenario.server_1011_error ? scenario.error_type : "none"}`,
      `Audio file: ${fmt(scenario.audio_file)}`,
      "",
    );
  }
  writeFileSync(resolve(resultRunDir, "legacy-analysis.txt"), `${lines.join("\n")}\n`, "utf8");
}

function runExternalTimelineVisualization(resultRunDir: string): void {
  execFileSync("python3", [resolve(PROJECT_DIR, "scripts", "visualize_external_wait.py"), resultRunDir], {
    cwd: PROJECT_DIR,
    stdio: "inherit",
  });
}

async function main(): Promise<void> {
  mkdirSync(RESULT_DIR, { recursive: true });
  const apiKey = requireEnv("GEMINI_API_KEY");
  const model = requireEnv("GEMINI_LIVE_MODEL");
  const scenarios = parseLatencies(process.argv.slice(2));
  const runStamp = timestamp();
  const resultRunDir = resolve(RESULT_DIR, runStamp);
  mkdirSync(resultRunDir, { recursive: true });
  writePromptsFile(resultRunDir);

  const rawLogPath = resolve(resultRunDir, "raw_log.jsonl");
  const log = createWriteStream(rawLogPath, { flags: "a" });
  log.write(
    `${JSON.stringify({
      ts: new Date().toISOString(),
      scenario_id: null,
      relative_time_ms: 0,
      event_type: "prompt_metadata",
      data: {
        prompt_version: PROMPT_VERSION,
        prompt: PROMPT,
        system_instruction: SYSTEM_INSTRUCTION,
      },
    })}\n`,
  );

  console.log(`Running external wait bench for model: ${model}`);
  console.log(`Result directory: ${relative(PROJECT_DIR, resultRunDir)}`);
  console.log("Native tools: disabled");

  const ai = new GoogleGenAI({ apiKey });
  const summaries: ScenarioSummary[] = [];
  for (const scenario of scenarios) {
    summaries.push(await runScenario(ai, model, scenario, resultRunDir, log));
  }

  const summary = {
    run_timestamp: runStamp,
    runTimestamp: runStamp,
    prompt_version: PROMPT_VERSION,
    prompt: PROMPT,
    system_instruction: SYSTEM_INSTRUCTION,
    expected_final_answer: EXPECTED_FINAL_ANSWER,
    model,
    raw_log_file: "raw_log.jsonl",
    scenarios: summaries,
  };
  writeFileSync(resolve(resultRunDir, "summary.json"), `${JSON.stringify(summary, null, 2)}\n`, "utf8");
  writeTextEventSummary(resultRunDir, summaries);
  writeLegacyAnalysis(resultRunDir, summaries);
  await new Promise<void>((resolveLog) => log.end(resolveLog));
  runExternalTimelineVisualization(resultRunDir);

  console.log("");
  console.log(`Result directory: ${relative(PROJECT_DIR, resultRunDir)}`);
  console.log(`Summary: ${relative(PROJECT_DIR, resolve(resultRunDir, "summary.json"))}`);
  console.log(`Legacy analysis: ${relative(PROJECT_DIR, resolve(resultRunDir, "legacy-analysis.txt"))}`);
  console.log(`Timeline directory: ${relative(PROJECT_DIR, resolve(resultRunDir, "timeline"))}`);
}

main().catch((error) => {
  console.error("external-wait-bench failed");
  console.error(summarizeError(error));
  process.exitCode = 1;
});
