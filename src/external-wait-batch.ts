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

type Scenario = { id: string; latencyMs: number };
type Condition = { id: string; tickMode: "none" | "every_3000ms" };
type AudioSegment = { offsetMs: number; chunk: Buffer };
type RunSummary = Record<string, unknown>;

const DEFAULT_SCENARIOS: Scenario[] = [{ id: "slow_correct_3s", latencyMs: 3000 }];

function requireEnv(name: string): string {
  const value = process.env[name]?.trim();
  if (!value) throw new Error(`Missing ${name}. Set it in /user_data/angus_bench/.env or GEMINI_ENV_FILE.`);
  return value;
}

function timestampForPath(): string {
  return new Date().toISOString().replace(/[:.]/g, "-").replace("T", "_").replace("Z", "");
}

function parseArgs(argv: string[]) {
  const args = new Map<string, string>();
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (!arg.startsWith("--")) continue;
    const [key, inlineValue] = arg.slice(2).split("=", 2);
    const value = inlineValue ?? argv[index + 1];
    if (inlineValue === undefined) index += 1;
    args.set(key, value);
  }
  const scenarios = (args.get("latencies") ?? "3000")
    .split(",")
    .map((item) => Number(item.trim()))
    .filter((value) => Number.isFinite(value) && value >= 0)
    .map((latencyMs) => ({ id: `slow_correct_${Math.round(latencyMs / 1000)}s`, latencyMs }));
  return {
    scenarios: scenarios.length ? scenarios : DEFAULT_SCENARIOS,
    conditions: (args.get("conditions") ?? "no_tick,tick_every_3000ms").split(",").map((item) => item.trim()).filter(Boolean),
    targetValidRuns: Math.max(1, Number(args.get("target-valid-runs") ?? 10)),
    maxAttemptsPerCondition: Math.max(1, Number(args.get("max-attempts-per-condition") ?? 30)),
    postExternalWaitMs: Math.max(0, Number(args.get("post-external-wait-ms") ?? DEFAULT_POST_EXTERNAL_WAIT_MS)),
  };
}

function conditionFromName(name: string): Condition {
  if (name === "no_tick" || name === "condition_no_tick") return { id: "condition_no_tick", tickMode: "none" };
  if (name === "tick_every_3000ms" || name === "condition_tick_every_3000ms") return { id: "condition_tick_every_3000ms", tickMode: "every_3000ms" };
  throw new Error(`Unknown condition: ${name}`);
}

function scheduledTickTimesForScenario(scenario: Scenario, condition: Condition): number[] {
  if (condition.tickMode === "none") return [];
  const ticks: number[] = [];
  for (let tick = 3000; tick < scenario.latencyMs; tick += 3000) ticks.push(tick);
  return ticks;
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

function writeAudioFile(pathWithoutExt: string, chunks: Buffer[], mimeType: string | undefined): string {
  if (chunks.length === 0) return "none";
  const audio = Buffer.concat(chunks);
  const pcm = parsePcmMime(mimeType);
  if (pcm.sampleRate && pcm.channels && pcm.bitDepth) {
    const file = `${pathWithoutExt}.wav`;
    writeFileSync(file, Buffer.concat([wavHeader(audio.length, pcm.sampleRate, pcm.channels, pcm.bitDepth), audio]));
    return relative(PROJECT_DIR, file);
  }
  const file = `${pathWithoutExt}.pcm`;
  writeFileSync(file, audio);
  return relative(PROJECT_DIR, file);
}

function writeTimelineAudioFile(pathWithoutExt: string, segments: AudioSegment[], mimeType: string | undefined): string {
  if (segments.length === 0) return "none";
  const pcm = parsePcmMime(mimeType);
  if (!pcm.sampleRate || !pcm.channels || !pcm.bitDepth) return writeAudioFile(pathWithoutExt, segments.map((segment) => segment.chunk), mimeType);
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
  return relative(PROJECT_DIR, file);
}

function summarizeError(error: unknown): string {
  if (error instanceof Error) return `${error.name}: ${error.message}`;
  if (typeof error !== "object" || error === null) return String(error);
  const record = error as Record<string, unknown>;
  return ["message", "type", "code", "reason", "error"].map((key) => (record[key] ? `${key}: ${String(record[key])}` : undefined)).filter(Boolean).join(", ") || Object.prototype.toString.call(error);
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
                  return { inlineData: { mimeType: part.inlineData.mimeType, bytes: part.inlineData.data ? decodeAudio(part.inlineData.data).length : 0 } };
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

function writePromptsFile(runDir: string, condition: Condition): void {
  writeFileSync(resolve(runDir, "prompts.txt"), [`PROMPT_VERSION=${PROMPT_VERSION}`, `CONDITION=${condition.id}`, `TICK_MODE=${condition.tickMode}`, "", "PROMPT:", PROMPT, "", "SYSTEM_INSTRUCTION:", SYSTEM_INSTRUCTION, ""].join("\n"), "utf8");
}

async function runOne(ai: GoogleGenAI, model: string, runRoot: string, condition: Condition, scenario: Scenario, attempt: number, postExternalWaitMs: number): Promise<RunSummary> {
  const runDir = resolve(runRoot, `run_${String(attempt).padStart(4, "0")}`);
  const audioDir = resolve(runDir, "audio");
  mkdirSync(audioDir, { recursive: true });
  writePromptsFile(runDir, condition);
  const log = createWriteStream(resolve(runDir, "raw_log.jsonl"), { flags: "a" });
  const audioChunks: Buffer[] = [];
  const timelineSegments: AudioSegment[] = [];
  const timers: ReturnType<typeof setTimeout>[] = [];
  const state = {
    promptSentAt: undefined as number | undefined,
    externalResultInjectedAt: undefined as number | undefined,
    firstAssistantOutputAt: undefined as number | undefined,
    firstTextAt: undefined as number | undefined,
    firstAudioAt: undefined as number | undefined,
    firstOutputAfterExternalResultAt: undefined as number | undefined,
    beforeCount: 0,
    afterCount: 0,
    pendingTickTimes: [] as number[],
    assistantOutputTimes: [] as number[],
    closeCode: undefined as number | undefined,
    closeReason: undefined as string | undefined,
    errors: [] as string[],
    eventTypesSeen: new Set<string>(),
    sessionClosed: false,
  };
  let audioMimeType: string | undefined;
  let session: Session | undefined;
  let done = false;
  const relativeTime = () => (state.promptSentAt ? Date.now() - state.promptSentAt : 0);
  const logLine = (eventType: string, data: Record<string, unknown> = {}) => {
    log.write(`${JSON.stringify({ ts: new Date().toISOString(), condition: condition.id, scenario_id: scenario.id, relative_time_ms: relativeTime(), event_type: eventType, data })}\n`);
  };
  const clearTimers = () => {
    while (timers.length) clearTimeout(timers.pop()!);
  };
  const finish = (reason: string, resolveRun: () => void) => {
    if (done) return;
    done = true;
    clearTimers();
    logLine("scenario_finished", { reason });
    if (!state.sessionClosed) {
      try {
        session?.close();
      } catch {
        // Already closed.
      }
    }
    resolveRun();
  };

  await new Promise<void>(async (resolveRun) => {
    const hardTimer = setTimeout(() => {
      state.errors.push(`max scenario timeout ${MAX_SCENARIO_MS}ms`);
      finish("max scenario timeout", resolveRun);
    }, MAX_SCENARIO_MS);
    timers.push(hardTimer);
    try {
      session = (await ai.live.connect({
        model,
        config: { responseModalities: [Modality.AUDIO], outputAudioTranscription: {}, systemInstruction: SYSTEM_INSTRUCTION },
        callbacks: {
          onopen: () => logLine("session_opened", { model, tools_enabled: false }),
          onmessage: (message: LiveMessage) => {
            const now = Date.now();
            for (const type of eventTypes(message)) state.eventTypesSeen.add(type);
            logLine("server_event", { event_types: eventTypes(message), message: sanitizeMessage(message) });
            let output = Boolean(message.serverContent?.outputTranscription?.text);
            for (const part of message.serverContent?.modelTurn?.parts ?? []) {
              if (part.text) {
                output = true;
                state.firstTextAt ??= now;
              }
              if (part.inlineData?.data) {
                output = true;
                const chunk = decodeAudio(part.inlineData.data);
                audioMimeType ??= part.inlineData.mimeType;
                state.firstAudioAt ??= now;
                audioChunks.push(chunk);
                timelineSegments.push({ offsetMs: Math.max(0, now - (state.promptSentAt ?? now)), chunk });
              }
            }
            if (output) {
              state.firstAssistantOutputAt ??= now;
              state.assistantOutputTimes.push(now);
              if (state.externalResultInjectedAt) {
                state.afterCount += 1;
                state.firstOutputAfterExternalResultAt ??= now;
              } else {
                state.beforeCount += 1;
              }
            }
          },
          onerror: (error) => {
            const summary = summarizeError(error);
            state.errors.push(summary);
            logLine("socket_error", { error: summary });
          },
          onclose: (event: { code?: number; reason?: string }) => {
            state.sessionClosed = true;
            state.closeCode = event.code;
            state.closeReason = event.reason;
            logLine("session_closed", { code: event.code, reason: event.reason });
            clearTimers();
            resolveRun();
          },
        },
      })) as Session;
      session.sendClientContent({ turns: PROMPT, turnComplete: true });
      state.promptSentAt = Date.now();
      logLine("user_prompt_sent", { prompt: PROMPT });
      for (const tickAt of scheduledTickTimesForScenario(scenario, condition)) {
        const timer = setTimeout(() => {
          if (done || state.sessionClosed || state.externalResultInjectedAt) return;
          session?.sendClientContent({ turns: EXTERNAL_STATUS, turnComplete: true });
          state.pendingTickTimes.push(Date.now());
          logLine("external_status_sent", { message: EXTERNAL_STATUS, scheduled_tick_ms: tickAt });
        }, tickAt);
        timers.push(timer);
      }
      const resultTimer = setTimeout(() => {
        if (done || state.sessionClosed) return;
        const externalResult = `EXTERNAL_EVENT\ntype: ready\nhas_final_answer: true\nfinal_answer: ${EXPECTED_FINAL_ANSWER}\ninstruction: answer_now_with_final_answer_only`;
        session?.sendClientContent({ turns: externalResult, turnComplete: true });
        state.externalResultInjectedAt = Date.now();
        logLine("external_result_injected", { final_answer: EXPECTED_FINAL_ANSWER, message: externalResult });
        const postTimer = setTimeout(() => finish("post external observation window elapsed", resolveRun), postExternalWaitMs);
        timers.push(postTimer);
      }, scenario.latencyMs);
      timers.push(resultTimer);
    } catch (error) {
      state.errors.push(summarizeError(error));
      logLine("scenario_error", { error: summarizeError(error) });
      clearTimers();
      resolveRun();
    }
  });

  const tickResponseCount = state.pendingTickTimes.length === 0
    ? null
    : state.pendingTickTimes.filter((start, index) => {
        const end = state.pendingTickTimes[index + 1] ?? state.externalResultInjectedAt ?? (state.promptSentAt ?? 0) + scenario.latencyMs;
        return state.assistantOutputTimes.some((time) => time >= start && time < end);
      }).length;
  const tickResponseRate = tickResponseCount === null ? null : tickResponseCount / state.pendingTickTimes.length;
  const server1008 = state.closeCode === 1008;
  const server1011 = state.closeCode === 1011;
  const errorType = server1008 ? "server_1008_error" : server1011 ? "server_1011_error" : state.errors.length ? "session_error" : state.externalResultInjectedAt ? null : "external_result_not_injected";
  const sessionValid = errorType === null;
  const summary: RunSummary = {
    condition: condition.id,
    scenario_id: scenario.id,
    latency_ms: scenario.latencyMs,
    attempt,
    session_valid: sessionValid,
    valid: sessionValid,
    error_type: errorType,
    server_1008_error: server1008,
    server_1011_error: server1011,
    answer_success_asr: null,
    prompt_version: PROMPT_VERSION,
    external_result_injected_time_ms: msDelta(state.promptSentAt, state.externalResultInjectedAt),
    first_response_latency_ms: msDelta(state.promptSentAt, state.firstAssistantOutputAt),
    assistant_output_count_before_external_result: state.beforeCount,
    assistant_output_rate_before_external_result_per_sec: scenario.latencyMs > 0 ? state.beforeCount / (scenario.latencyMs / 1000) : null,
    pending_tick_count_sent: state.pendingTickTimes.length,
    tick_response_rate: tickResponseRate,
    post_external_first_response_latency_ms: msDelta(state.externalResultInjectedAt, state.firstOutputAfterExternalResultAt),
    post_external_wait_ms: postExternalWaitMs,
    has_output_after_external_result: state.afterCount > 0,
    assistant_output_count_after_external_result: state.afterCount,
    final_answer_observed: null,
    final_exact_match: null,
    post_external_answer_success: "not_available_without_asr",
    note: "assistant_output_count_before_external_result is output event/chunk count, not utterance count.",
    audio_file: writeAudioFile(resolve(audioDir, `${scenario.id}_compressed`), audioChunks, audioMimeType),
    timeline_audio_file: writeTimelineAudioFile(resolve(audioDir, `${scenario.id}_timeline`), timelineSegments, audioMimeType),
    event_types_seen: [...state.eventTypesSeen],
  };
  logLine("scenario_summary", summary);
  writeJson(resolve(runDir, "summary.json"), { prompt_version: PROMPT_VERSION, model, condition: condition.id, scenarios: [summary] });
  writeFileSync(resolve(runDir, "text_event_summary.txt"), `Scenario: ${scenario.id}\nCondition: ${condition.id}\nValid: ${sessionValid ? "yes" : "no"}\nFirst response latency: ${summary.first_response_latency_ms ?? "n/a"}ms\nOutput count before external result: ${state.beforeCount}\nHas output after external result: ${state.afterCount > 0 ? "yes" : "no"}\n`, "utf8");
  writeFileSync(resolve(runDir, "legacy-analysis.txt"), `Prompt version: ${PROMPT_VERSION}\nScenario: ${scenario.id}\nCondition: ${condition.id}\nValid: ${sessionValid ? "yes" : "no"}\nOutput count before external result: ${state.beforeCount}\nNote: output count is event/chunk count, not utterance count.\n`, "utf8");
  await new Promise<void>((resolveLog) => log.end(resolveLog));
  execFileSync("python3", [resolve(PROJECT_DIR, "scripts", "visualize_external_wait.py"), runDir], { cwd: PROJECT_DIR, stdio: "ignore" });
  return summary;
}

function numberValues(rows: RunSummary[], key: string): number[] {
  return rows.map((row) => row[key]).filter((value): value is number => typeof value === "number" && Number.isFinite(value));
}

function average(values: number[]): number | null {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
}

function median(values: number[]): number | null {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function summarizeCondition(condition: Condition, latencyMs: number, runs: RunSummary[], targetValidRuns: number, maxAttempts: number) {
  const valid = runs.filter((run) => run.session_valid);
  return {
    condition: condition.id,
    latency_ms: latencyMs,
    target_valid_runs: targetValidRuns,
    max_attempts: maxAttempts,
    attempted_runs: runs.length,
    valid_runs: valid.length,
    invalid_runs: runs.length - valid.length,
    completed: valid.length >= targetValidRuns,
    server_1008_errors: runs.filter((run) => run.server_1008_error).length,
    server_1011_errors: runs.filter((run) => run.server_1011_error).length,
    other_errors: runs.filter((run) => !run.session_valid && !run.server_1008_error && !run.server_1011_error).length,
    valid_run_rate: runs.length ? valid.length / runs.length : 0,
    avg_first_response_latency_ms: average(numberValues(valid, "first_response_latency_ms")),
    median_first_response_latency_ms: median(numberValues(valid, "first_response_latency_ms")),
    avg_assistant_output_count_before_external_result: average(numberValues(valid, "assistant_output_count_before_external_result")),
    avg_assistant_output_rate_before_external_result_per_sec: average(numberValues(valid, "assistant_output_rate_before_external_result_per_sec")),
    avg_post_external_first_response_latency_ms: average(numberValues(valid, "post_external_first_response_latency_ms")),
    has_output_after_external_result_rate: valid.length ? valid.filter((run) => run.has_output_after_external_result).length / valid.length : null,
    note: "assistant output count/rate are event/chunk counts, not utterance counts.",
  };
}

function csvEscape(value: unknown): string {
  if (value === null || value === undefined) return "";
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function writeCsv(path: string, rows: Record<string, unknown>[]): void {
  const headers = ["condition", "latency_ms", "target_valid_runs", "attempted_runs", "valid_runs", "invalid_runs", "server_1008_errors", "server_1011_errors", "other_errors", "valid_run_rate", "avg_first_response_latency_ms", "median_first_response_latency_ms", "avg_assistant_output_count_before_external_result", "avg_assistant_output_rate_before_external_result_per_sec", "avg_post_external_first_response_latency_ms", "has_output_after_external_result_rate"];
  writeFileSync(path, `${headers.join(",")}\n${rows.map((row) => headers.map((header) => csvEscape(row[header])).join(",")).join("\n")}\n`, "utf8");
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const apiKey = requireEnv("GEMINI_API_KEY");
  const model = requireEnv("GEMINI_LIVE_MODEL");
  const root = resolve(RESULT_DIR, `external_wait_batch_${timestampForPath()}`);
  mkdirSync(root, { recursive: true });
  const ai = new GoogleGenAI({ apiKey });
  const conditions = args.conditions.map(conditionFromName);
  writeJson(resolve(root, "config.json"), { prompt_version: PROMPT_VERSION, model, ...args, conditions });

  const allRuns: RunSummary[] = [];
  const summaryRows: Record<string, unknown>[] = [];
  for (const condition of conditions) {
    const conditionRoot = resolve(root, condition.id);
    mkdirSync(conditionRoot, { recursive: true });
    for (const scenario of args.scenarios) {
      const latencyRoot = resolve(conditionRoot, `latency_${scenario.latencyMs}ms`);
      mkdirSync(latencyRoot, { recursive: true });
      const runs: RunSummary[] = [];
      let attempt = 0;
      while (runs.filter((run) => run.session_valid).length < args.targetValidRuns && attempt < args.maxAttemptsPerCondition) {
        attempt += 1;
        console.log(`[${condition.id} ${scenario.latencyMs}ms] attempt ${attempt}`);
        const run = await runOne(ai, model, latencyRoot, condition, scenario, attempt, args.postExternalWaitMs);
        runs.push(run);
        allRuns.push(run);
        console.log(`[${condition.id} ${scenario.latencyMs}ms] ${run.session_valid ? "valid" : run.error_type}`);
      }
      const conditionSummary = summarizeCondition(condition, scenario.latencyMs, runs, args.targetValidRuns, args.maxAttemptsPerCondition);
      summaryRows.push(conditionSummary);
      writeJson(resolve(latencyRoot, "summary.json"), { ...conditionSummary, runs });
    }
  }
  writeJson(resolve(root, "summary.json"), { prompt_version: PROMPT_VERSION, model, rows: summaryRows, runs: allRuns });
  writeCsv(resolve(root, "summary.csv"), summaryRows);
  try {
    execFileSync("python3", [resolve(PROJECT_DIR, "scripts", "plot_external_wait_batch.py"), "--input", root], { cwd: PROJECT_DIR, stdio: "inherit" });
  } catch (error) {
    console.warn(`Warning: external-wait batch visualization failed: ${summarizeError(error)}`);
  }
  console.log(`Result directory: ${relative(PROJECT_DIR, root)}`);
  console.log(`Summary: ${relative(PROJECT_DIR, resolve(root, "summary.json"))}`);
}

main().catch((error) => {
  console.error("external-wait-batch failed");
  console.error(summarizeError(error));
  process.exitCode = 1;
});
