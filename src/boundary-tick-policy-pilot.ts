import { spawn, spawnSync } from "node:child_process";
import {
  cpSync,
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  writeFileSync,
} from "node:fs";
import { dirname, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

type Condition =
  | "fixed_single_tick_4s"
  | "periodic_tick_4s"
  | "tick_after_utterance_0s"
  | "tick_after_utterance_1s"
  | "tick_after_audio_idle_0s"
  | "tick_after_audio_idle_1s"
  | "tick_after_audio_idle_repeat_0s"
  | "tick_after_audio_idle_repeat_1s";

type ProbeMode =
  | "client_status_tick_3000ms"
  | "periodic_tick_4s"
  | "tick_after_utterance_0s"
  | "tick_after_utterance_1s"
  | "tick_after_audio_idle_0s"
  | "tick_after_audio_idle_1s"
  | "tick_after_audio_idle_repeat_0s"
  | "tick_after_audio_idle_repeat_1s";

type AttemptSummary = {
  session_valid?: boolean;
  close_1008?: boolean;
  close_1011?: boolean;
  client_send_error_count?: number;
  tool_call_success?: boolean;
  final_tool_response_sent?: boolean;
  tick_send_success_count?: number;
  waiting_audio_before_final_tool_result?: boolean;
  audio_after_tick?: boolean;
  post_tool_final_answer?: boolean;
  premature_final_answer?: boolean;
  tool_call_time_ms?: number | null;
  first_audio_time_ms?: number | null;
  final_tool_response_sent_time_ms?: number | null;
  post_tool_final_latency_ms?: number | null;
  pending_tick_times_ms?: number[];
  boundary_detected_times_ms?: number[];
  boundary_tick_times_ms?: number[];
  has_output_after_boundary_tick?: boolean;
  pending_tick_skipped_final_ready_count?: number;
  cancelled_tool_call_ids?: string[];
  send_errors?: string[];
  errors?: string[];
};

type AttemptRecord = {
  condition: Condition;
  latency_ms: number;
  run_index: number;
  source_result_dir: string;
  source_attempt_dir: string;
  organized_attempt_dir: string;
  valid: boolean;
  retry_index: number;
  first_audio_time_ms: number | null;
  max_silence_gap_before_result: number | null;
  audio_segment_count_before_result: number;
  audio_output_count_before_result: number;
  audio_occupancy_ratio_before_result: number | null;
  post_final_answer_latency_ms: number | null;
  waiting_speech_present: boolean;
  waiting_task_relevance_score_when_spoke: number | null;
  waiting_task_relevance_score_all: number;
  pre_result_hallucination: boolean;
  within_attempt_waiting_diversity_score: number | null;
  repetition_template_similarity: number | null;
  pending_tick_count: number;
  pending_tick_times_ms: number[];
  boundary_detected_times_ms: number[];
  boundary_tick_times_ms: number[];
  has_output_after_boundary_tick: boolean;
  pending_tick_skipped_final_ready: boolean;
  has_output_after_external_result: boolean;
  final_core_answer_correct: boolean;
  close_1008: boolean;
  close_1011: boolean;
  send_error_count: number;
  tool_call_cancellation_count: number;
};

type CellSummary = {
  condition: Condition;
  latency_ms: number;
  attempt_count: number;
  valid_runs: number;
  valid_run_rate: number;
  retry_count: number;
  "1008_error_count": number;
  "1011_error_count": number;
  send_error_count: number;
  tool_call_cancellation_count: number;
  pending_tick_count_avg: number | null;
  waiting_speech_present_rate: number | null;
  waiting_task_relevance_score_when_spoke_avg: number | null;
  waiting_task_relevance_score_all_avg: number | null;
  pre_result_hallucination_count: number;
  final_core_answer_correct_rate: number | null;
  first_audio_time_ms_avg: number | null;
  max_silence_gap_before_result_avg: number | null;
  audio_segment_count_before_result_avg: number | null;
  audio_output_count_before_result_avg: number | null;
  audio_occupancy_ratio_before_result_avg: number | null;
  post_final_answer_latency_ms_avg: number | null;
  has_output_after_boundary_tick_count: number;
  tick_skipped_final_ready_count: number;
};

const PROJECT_DIR = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const RESULT_DIR = resolve(PROJECT_DIR, "result");
const PROBE = resolve(PROJECT_DIR, "dist", "tau-live-tool-tick-factor-probe.js");
const ALL_CONDITIONS: Array<{ condition: Condition; probeMode: ProbeMode }> = [
  { condition: "periodic_tick_4s", probeMode: "periodic_tick_4s" },
  { condition: "tick_after_audio_idle_repeat_0s", probeMode: "tick_after_audio_idle_repeat_0s" },
  { condition: "tick_after_audio_idle_repeat_1s", probeMode: "tick_after_audio_idle_repeat_1s" },
  { condition: "tick_after_audio_idle_0s", probeMode: "tick_after_audio_idle_0s" },
  { condition: "tick_after_audio_idle_1s", probeMode: "tick_after_audio_idle_1s" },
  { condition: "fixed_single_tick_4s", probeMode: "client_status_tick_3000ms" },
  { condition: "tick_after_utterance_0s", probeMode: "tick_after_utterance_0s" },
  { condition: "tick_after_utterance_1s", probeMode: "tick_after_utterance_1s" },
];
const DEFAULT_CONDITION_NAMES = new Set<Condition>(["periodic_tick_4s", "tick_after_audio_idle_repeat_0s", "tick_after_audio_idle_repeat_1s"]);
const CONDITION_FILTER = new Set(
  String(process.env.BOUNDARY_TICK_CONDITIONS ?? "")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean),
);
const CONDITIONS = CONDITION_FILTER.size
  ? ALL_CONDITIONS.filter((item) => CONDITION_FILTER.has(item.condition))
  : ALL_CONDITIONS.filter((item) => DEFAULT_CONDITION_NAMES.has(item.condition));
const LATENCIES_MS = String(process.env.BOUNDARY_TICK_LATENCIES_MS ?? "8000,12000")
  .split(",")
  .map((item) => Number(item.trim()))
  .filter((item) => Number.isFinite(item) && item > 0);
const TARGET_VALID = Number(process.env.BOUNDARY_TICK_TARGET_VALID ?? 5);
const MAX_ATTEMPTS_PER_CELL = Number(process.env.BOUNDARY_TICK_MAX_ATTEMPTS_PER_CELL ?? 30);
const CELL_CONCURRENCY = Math.max(1, Number(process.env.BOUNDARY_TICK_CELL_CONCURRENCY ?? 5));
const PCM_BYTES_PER_SECOND = 48_000;
const AUDIO_SEGMENT_MERGE_GAP_MS = 200;

function timestampForPath(date = new Date()): string {
  const [datePart, timePart] = date.toISOString().split("T");
  return `${datePart}_${timePart.replace("Z", "").replace(/\./g, "-").replace(/:/g, "-")}`;
}

function readJson<T>(path: string): T {
  return JSON.parse(readFileSync(path, "utf8")) as T;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolveSleep) => setTimeout(resolveSleep, ms));
}

async function waitForFile(path: string, timeoutMs = 5000): Promise<void> {
  const started = Date.now();
  while (!existsSync(path)) {
    if (Date.now() - started > timeoutMs) throw new Error(`Timed out waiting for ${path}`);
    await sleep(100);
  }
}

function readJsonl(path: string): Array<Record<string, any>> {
  if (!existsSync(path)) return [];
  return readFileSync(path, "utf8")
    .trim()
    .split(/\n/)
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

function writeJson(path: string, value: unknown): void {
  writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

function csvEscape(value: unknown): string {
  if (value === null || value === undefined) return "";
  const text = Array.isArray(value) ? value.join(" | ") : String(value);
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

function average(values: Array<number | null | undefined>, digits = 3): number | null {
  const nums = values.filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  if (!nums.length) return null;
  const factor = 10 ** digits;
  return Math.round((nums.reduce((sum, value) => sum + value, 0) / nums.length) * factor) / factor;
}

function listProbeDirs(sourceRunsDir: string): Set<string> {
  if (!existsSync(sourceRunsDir)) return new Set();
  return new Set(
    readdirSync(sourceRunsDir)
      .filter((name) => name.endsWith("_tau_live_tool_tick_factor_probe"))
      .map((name) => resolve(sourceRunsDir, name)),
  );
}

function findLatestProbeDir(sourceRunsDir: string, before: Set<string>, stdout: string): string {
  const match = stdout.match(/Result directory:\s+([^\n]+)/);
  if (match) {
    const candidate = resolve(PROJECT_DIR, match[1].trim());
    if (existsSync(candidate)) return candidate;
  }
  const after = [...listProbeDirs(sourceRunsDir)].filter((path) => !before.has(path));
  after.sort().reverse();
  if (!after[0]) throw new Error("Could not find probe result directory.");
  return after[0];
}

function runProbeAttempt(sourceRunsDir: string, probeMode: ProbeMode, latencyMs: number): Promise<string> {
  const before = listProbeDirs(sourceRunsDir);
  return new Promise((resolveRun, rejectRun) => {
    const child = spawn(process.execPath, [
      PROBE,
      "--tick-modes",
      probeMode,
      "--attempts",
      "1",
      "--latency-ms",
      String(latencyMs),
      "--quiet-terminal-text",
    ], {
      cwd: PROJECT_DIR,
      env: {
        ...process.env,
        GEMINI_LIVE_CHECK_RESULT_DIR: sourceRunsDir,
      },
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => {
      stdout += chunk;
      for (const line of chunk.split(/\n/).filter(Boolean)) {
        if (/Result directory|Attempts:|\] start|\] valid|\] not_valid|\] 1008|\] 1011|Summary:|Final comparison:/.test(line)) {
          console.log(`  ${line}`);
        }
      }
    });
    child.stderr.on("data", (chunk: string) => {
      stderr += chunk;
    });
    child.on("error", rejectRun);
    child.on("close", (status) => {
      if (stderr) console.error(stderr);
      if (status !== 0) {
        rejectRun(new Error(`Probe exited with status ${status ?? "unknown"}`));
        return;
      }
      try {
        resolveRun(findLatestProbeDir(sourceRunsDir, before, stdout));
      } catch (error) {
        rejectRun(error);
      }
    });
  });
}

function firstEventMs(events: Array<Record<string, any>>, type: string): number | null {
  const event = events.find((item) => item.type === type && typeof item.event_ms === "number");
  return event ? Number(event.event_ms) : null;
}

function eventTimes(events: Array<Record<string, any>>, types: string[]): number[] {
  return events
    .filter((event) => types.includes(String(event.type)) && typeof event.event_ms === "number")
    .map((event) => Number(event.event_ms));
}

function audioIntervals(events: Array<Record<string, any>>): Array<{ startMs: number; endMs: number }> {
  const intervals: Array<{ startMs: number; endMs: number }> = [];
  let playbackCursorMs = 0;
  for (const event of events) {
    if (event.type !== "audio_output" || typeof event.event_ms !== "number" || Number(event.bytes ?? 0) <= 0) continue;
    const requestedStartMs = Number(event.event_ms);
    const startMs = Math.max(requestedStartMs, playbackCursorMs);
    const durationMs = (Number(event.bytes) / PCM_BYTES_PER_SECOND) * 1000;
    const endMs = startMs + durationMs;
    playbackCursorMs = endMs;
    intervals.push({ startMs, endMs });
  }
  return intervals;
}

function mergeIntervals(intervals: Array<{ startMs: number; endMs: number }>): Array<{ startMs: number; endMs: number }> {
  const sorted = [...intervals].sort((a, b) => a.startMs - b.startMs);
  const merged: Array<{ startMs: number; endMs: number }> = [];
  for (const interval of sorted) {
    const previous = merged.at(-1);
    if (!previous || interval.startMs > previous.endMs) merged.push({ ...interval });
    else previous.endMs = Math.max(previous.endMs, interval.endMs);
  }
  return merged;
}

function mergeAudioSegments(intervals: Array<{ startMs: number; endMs: number }>): Array<{ startMs: number; endMs: number }> {
  const sorted = [...intervals].sort((a, b) => a.startMs - b.startMs);
  const merged: Array<{ startMs: number; endMs: number }> = [];
  for (const interval of sorted) {
    const previous = merged.at(-1);
    if (!previous || interval.startMs - previous.endMs > AUDIO_SEGMENT_MERGE_GAP_MS) merged.push({ ...interval });
    else previous.endMs = Math.max(previous.endMs, interval.endMs);
  }
  return merged;
}

function clippedAudioIntervals(events: Array<Record<string, any>>, startMs: number, endMs: number | null, mergeGapMs = 0): Array<{ startMs: number; endMs: number }> {
  if (typeof endMs !== "number" || endMs <= startMs) return [];
  const intervals = audioIntervals(events)
    .map((interval) => ({ startMs: Math.max(startMs, interval.startMs), endMs: Math.min(endMs, interval.endMs) }))
    .filter((interval) => interval.endMs > interval.startMs);
  return mergeGapMs > 0 ? mergeAudioSegments(intervals) : mergeIntervals(intervals);
}

function unionDuration(intervals: Array<{ startMs: number; endMs: number }>): number {
  return Math.round(intervals.reduce((sum, interval) => sum + (interval.endMs - interval.startMs), 0));
}

function maxSilenceGap(intervals: Array<{ startMs: number; endMs: number }>, startMs: number, endMs: number | null): number | null {
  if (typeof endMs !== "number" || endMs <= startMs) return null;
  let cursor = startMs;
  let maxGap = 0;
  for (const interval of intervals) {
    maxGap = Math.max(maxGap, interval.startMs - cursor);
    cursor = Math.max(cursor, interval.endMs);
  }
  return Math.round(Math.max(maxGap, endMs - cursor));
}

function textBeforeFinal(events: Array<Record<string, any>>, finalMs: number | null): string {
  return events
    .filter((event) =>
      (event.type === "text_output" || event.type === "output_transcription") &&
      typeof event.event_ms === "number" &&
      (typeof finalMs !== "number" || event.event_ms < finalMs) &&
      event.text
    )
    .map((event) => String(event.text))
    .join(" ");
}

function scoreWaitingRelevance(text: string, hasWaitingSpeech: boolean): { whenSpoke: number | null; all: number } {
  if (!hasWaitingSpeech) return { whenSpoke: null, all: 0 };
  const cleaned = text.toLowerCase();
  if (/#?a\s?123|order/.test(cleaned) && /check|status|look|retrieve|details|processing|waiting/.test(cleaned)) {
    return { whenSpoke: 3, all: 3 };
  }
  if (/order|status|tool|lookup/.test(cleaned)) return { whenSpoke: 2, all: 2 };
  return { whenSpoke: 1, all: 1 };
}

function hallucinatedBeforeFinal(text: string): boolean {
  return /shipped|ups|tracking|tomorrow|delivered/.test(text.toLowerCase());
}

function repetitionScore(text: string): number | null {
  const tokens = text.toLowerCase().match(/[a-z0-9]+/g) ?? [];
  if (tokens.length < 6) return null;
  const bigrams = tokens.slice(0, -1).map((token, index) => `${token} ${tokens[index + 1]}`);
  if (!bigrams.length) return null;
  const unique = new Set(bigrams);
  return Math.round((1 - unique.size / bigrams.length) * 1000) / 1000;
}

async function organizeAttempt(params: {
  sourceRunDir: string;
  sourceAttemptDir: string;
  organizedDir: string;
  condition: Condition;
  latencyMs: number;
  runIndex: number;
  retryIndex: number;
}): Promise<AttemptRecord> {
  const { sourceRunDir, sourceAttemptDir, organizedDir, condition, latencyMs, runIndex, retryIndex } = params;
  const runName = `${condition}__latency_${latencyMs}__run_${String(runIndex).padStart(3, "0")}`;
  const perAttemptDir = resolve(organizedDir, "per_attempt", runName);
  mkdirSync(perAttemptDir, { recursive: true });
  await waitForFile(resolve(sourceAttemptDir, "summary.json"));
  cpSync(sourceAttemptDir, perAttemptDir, { recursive: true });

  const summary = readJson<AttemptSummary>(resolve(sourceAttemptDir, "summary.json"));
  const events = readJsonl(resolve(sourceAttemptDir, "timeline", "events.jsonl"));
  const finalMs = summary.final_tool_response_sent_time_ms ?? firstEventMs(events, "final_tool_response_sent");
  const clippedAudio = clippedAudioIntervals(events, 0, finalMs);
  const preResultAudioSegments = clippedAudioIntervals(events, 0, finalMs, AUDIO_SEGMENT_MERGE_GAP_MS);
  const audioDuration = unionDuration(clippedAudio);
  const windowMs = typeof finalMs === "number" && finalMs > 0 ? finalMs : null;
  const preText = textBeforeFinal(events, finalMs);
  const hasWaitingSpeech = Boolean(summary.waiting_audio_before_final_tool_result);
  const relevance = scoreWaitingRelevance(preText, hasWaitingSpeech);
  const outputAfterFinal = typeof finalMs === "number" && events.some((event) =>
    (event.type === "audio_output" || event.type === "text_output" || event.type === "output_transcription") &&
    typeof event.event_ms === "number" &&
    event.event_ms >= finalMs
  );
  const tickTimes = [
    ...(summary.pending_tick_times_ms ?? []),
    ...eventTimes(events, ["client_status_tick_sent", "boundary_client_status_tick_sent"]),
  ];
  const uniqueTickTimes = [...new Set(tickTimes.map((time) => Math.round(time)))].sort((a, b) => a - b);
  const boundaryTimes = [
    ...(summary.boundary_tick_times_ms ?? []),
    ...eventTimes(events, ["boundary_client_status_tick_sent"]),
  ];
  const boundaryDetected = [
    ...(summary.boundary_detected_times_ms ?? []),
    ...eventTimes(events, ["speech_boundary_detected"]),
  ];
  const record: AttemptRecord = {
    condition,
    latency_ms: latencyMs,
    run_index: runIndex,
    source_result_dir: relative(PROJECT_DIR, sourceRunDir),
    source_attempt_dir: relative(PROJECT_DIR, sourceAttemptDir),
    organized_attempt_dir: relative(PROJECT_DIR, perAttemptDir),
    valid: Boolean(summary.session_valid),
    retry_index: retryIndex,
    first_audio_time_ms: summary.first_audio_time_ms ?? null,
    max_silence_gap_before_result: maxSilenceGap(clippedAudio, 0, finalMs),
    audio_segment_count_before_result: preResultAudioSegments.length,
    audio_output_count_before_result: events.filter((event) => event.type === "audio_output" && typeof event.event_ms === "number" && (typeof finalMs !== "number" || event.event_ms < finalMs)).length,
    audio_occupancy_ratio_before_result: windowMs ? Math.round((audioDuration / windowMs) * 10000) / 10000 : null,
    post_final_answer_latency_ms: summary.post_tool_final_latency_ms ?? null,
    waiting_speech_present: hasWaitingSpeech,
    waiting_task_relevance_score_when_spoke: relevance.whenSpoke,
    waiting_task_relevance_score_all: relevance.all,
    pre_result_hallucination: Boolean(summary.premature_final_answer) || hallucinatedBeforeFinal(preText),
    within_attempt_waiting_diversity_score: repetitionScore(preText) === null ? null : 1 - Number(repetitionScore(preText)),
    repetition_template_similarity: repetitionScore(preText),
    pending_tick_count: uniqueTickTimes.length,
    pending_tick_times_ms: uniqueTickTimes,
    boundary_detected_times_ms: [...new Set(boundaryDetected.map((time) => Math.round(time)))].sort((a, b) => a - b),
    boundary_tick_times_ms: [...new Set(boundaryTimes.map((time) => Math.round(time)))].sort((a, b) => a - b),
    has_output_after_boundary_tick: Boolean(summary.has_output_after_boundary_tick),
    pending_tick_skipped_final_ready: Number(summary.pending_tick_skipped_final_ready_count ?? 0) > 0,
    has_output_after_external_result: outputAfterFinal,
    final_core_answer_correct: Boolean(summary.post_tool_final_answer),
    close_1008: Boolean(summary.close_1008),
    close_1011: Boolean(summary.close_1011),
    send_error_count: Number(summary.client_send_error_count ?? 0),
    tool_call_cancellation_count: summary.cancelled_tool_call_ids?.length ?? 0,
  };
  writeJson(resolve(perAttemptDir, "pilot_attempt_record.json"), record);

  const copyIfExists = (from: string, to: string) => {
    if (existsSync(from)) cpSync(from, to);
  };
  copyIfExists(resolve(sourceAttemptDir, "raw_log.jsonl"), resolve(organizedDir, "raw_logs", `${runName}.raw_log.jsonl`));
  copyIfExists(resolve(sourceAttemptDir, "timeline", "events.jsonl"), resolve(organizedDir, "timelines", `${runName}.timeline.jsonl`));
  copyIfExists(resolve(sourceAttemptDir, "attempt_timeline.png"), resolve(organizedDir, "timelines", `${runName}.timeline.png`));
  for (const audioName of ["assistant_output.wav", "assistant_output_compressed.wav", "assistant_output_timeline.wav"]) {
    copyIfExists(resolve(sourceAttemptDir, "audio", audioName), resolve(organizedDir, "audio", `${runName}.${audioName}`));
  }
  return record;
}

function summarizeCell(condition: Condition, latencyMs: number, attempts: AttemptRecord[]): CellSummary {
  return {
    condition,
    latency_ms: latencyMs,
    attempt_count: attempts.length,
    valid_runs: attempts.filter((attempt) => attempt.valid).length,
    valid_run_rate: attempts.length ? Math.round((attempts.filter((attempt) => attempt.valid).length / attempts.length) * 1000) / 1000 : 0,
    retry_count: attempts.length - attempts.filter((attempt) => attempt.valid).length,
    "1008_error_count": attempts.filter((attempt) => attempt.close_1008).length,
    "1011_error_count": attempts.filter((attempt) => attempt.close_1011).length,
    send_error_count: attempts.reduce((sum, attempt) => sum + attempt.send_error_count, 0),
    tool_call_cancellation_count: attempts.reduce((sum, attempt) => sum + attempt.tool_call_cancellation_count, 0),
    pending_tick_count_avg: average(attempts.map((attempt) => attempt.pending_tick_count)),
    waiting_speech_present_rate: average(attempts.map((attempt) => (attempt.waiting_speech_present ? 1 : 0))),
    waiting_task_relevance_score_when_spoke_avg: average(attempts.map((attempt) => attempt.waiting_task_relevance_score_when_spoke)),
    waiting_task_relevance_score_all_avg: average(attempts.map((attempt) => attempt.waiting_task_relevance_score_all)),
    pre_result_hallucination_count: attempts.filter((attempt) => attempt.pre_result_hallucination).length,
    final_core_answer_correct_rate: average(attempts.map((attempt) => (attempt.final_core_answer_correct ? 1 : 0))),
    first_audio_time_ms_avg: average(attempts.map((attempt) => attempt.first_audio_time_ms)),
    max_silence_gap_before_result_avg: average(attempts.map((attempt) => attempt.max_silence_gap_before_result)),
    audio_segment_count_before_result_avg: average(attempts.map((attempt) => attempt.audio_segment_count_before_result)),
    audio_output_count_before_result_avg: average(attempts.map((attempt) => attempt.audio_output_count_before_result)),
    audio_occupancy_ratio_before_result_avg: average(attempts.map((attempt) => attempt.audio_occupancy_ratio_before_result)),
    post_final_answer_latency_ms_avg: average(attempts.map((attempt) => attempt.post_final_answer_latency_ms)),
    has_output_after_boundary_tick_count: attempts.filter((attempt) => attempt.has_output_after_boundary_tick).length,
    tick_skipped_final_ready_count: attempts.filter((attempt) => attempt.pending_tick_skipped_final_ready).length,
  };
}

async function runCell(params: {
  sourceRunsDir: string;
  organizedDir: string;
  condition: Condition;
  probeMode: ProbeMode;
  latencyMs: number;
  allAttempts: AttemptRecord[];
}): Promise<AttemptRecord[]> {
  const { sourceRunsDir, organizedDir, condition, probeMode, latencyMs, allAttempts } = params;
  const cellAttempts: AttemptRecord[] = [];
  let valid = 0;
  let launched = 0;
  let active = 0;
  let done = false;

  return await new Promise<AttemptRecord[]>((resolveCell, rejectCell) => {
    const maybeLaunch = () => {
      if (done) return;
      if (valid >= TARGET_VALID && active === 0) {
        done = true;
        resolveCell(cellAttempts);
        return;
      }
      while (
        active < CELL_CONCURRENCY &&
        launched < MAX_ATTEMPTS_PER_CELL &&
        valid + active < TARGET_VALID
      ) {
        launched += 1;
        active += 1;
        const retryIndex = launched;
        console.log(`[${condition} ${latencyMs}ms attempt ${retryIndex}] running active=${active}/${CELL_CONCURRENCY} valid=${valid}/${TARGET_VALID}`);
        Promise.resolve()
          .then(async () => {
            const sourceRunDir = await runProbeAttempt(sourceRunsDir, probeMode, latencyMs);
            const sourceAttemptDir = resolve(sourceRunDir, `condition_${probeMode}`, "attempt_0001");
            const runIndex = cellAttempts.length + 1;
            const record = await organizeAttempt({
              sourceRunDir,
              sourceAttemptDir,
              organizedDir,
              condition,
              latencyMs,
              runIndex,
              retryIndex,
            });
            cellAttempts.push(record);
            allAttempts.push(record);
            if (record.valid) valid += 1;
            console.log(`[${condition} ${latencyMs}ms attempt ${retryIndex}] ${record.valid ? "valid" : record.close_1008 ? "1008" : record.close_1011 ? "1011" : "not_valid"} valid=${valid}/${TARGET_VALID}`);
          })
          .then(() => {
            active -= 1;
            maybeLaunch();
          })
          .catch((error) => {
            done = true;
            rejectCell(error);
          });
      }
      if (active === 0 && (valid >= TARGET_VALID || launched >= MAX_ATTEMPTS_PER_CELL)) {
        done = true;
        resolveCell(cellAttempts);
      }
    };
    maybeLaunch();
  });
}

function writeReadme(rootDir: string, organizedDir: string, rows: CellSummary[], attempts: AttemptRecord[]): void {
  const total1008 = rows.reduce((sum, row) => sum + row["1008_error_count"], 0);
  const total1011 = rows.reduce((sum, row) => sum + row["1011_error_count"], 0);
  const totalSendErrors = rows.reduce((sum, row) => sum + row.send_error_count, 0);
  const notable = [...rows]
    .sort((a, b) => Number(b.max_silence_gap_before_result_avg ?? 0) - Number(a.max_silence_gap_before_result_avg ?? 0))
    .slice(0, 3)
    .map((row) => `- ${row.condition} ${row.latency_ms}ms: max silence avg ${row.max_silence_gap_before_result_avg ?? "NA"}ms, waiting speech rate ${row.waiting_speech_present_rate ?? "NA"}, post-final latency avg ${row.post_final_answer_latency_ms_avg ?? "NA"}ms`);
  const lines = [
    "# Boundary Tick Policy Pilot",
    "",
    "## Goal",
    "",
    "Compare fixed-time pending ticks with speech-boundary pending ticks for spoken waiting behavior in the TP1 Gemini Live native tool-wait benchmark.",
    "",
    "## Conditions",
    "",
    "- `periodic_tick_4s`: send external pending ticks every 4000 ms while the final result is not ready.",
    "- `tick_after_audio_idle_repeat_0s`: after each pre-final assistant audio-idle boundary, send a pending tick immediately if final result is not ready and the tick cooldown has elapsed.",
    "- `tick_after_audio_idle_repeat_1s`: after each pre-final assistant audio-idle boundary, wait 1000 ms, then send a pending tick if final result is not ready and the tick cooldown has elapsed.",
    "",
    "Audio-idle boundary detection uses the projected assistant audio playback cursor plus 300 ms without new assistant audio. Repeat boundary ticks use a cooldown to avoid dense tick spam.",
    "",
    "All pending ticks use `sendRealtimeInput(...)` and the same neutral message: `The lookup is still running. No final result is available yet.`",
    "",
    "## Run Shape",
    "",
    `- Latencies: ${LATENCIES_MS.join(", ")} ms`,
    `- Target valid runs per condition x latency: ${TARGET_VALID}`,
    `- Cell concurrency: ${CELL_CONCURRENCY}`,
    `- Max attempts per cell: ${MAX_ATTEMPTS_PER_CELL}`,
    "",
    "## Completed Counts",
    "",
    "| condition | latency_ms | attempts | valid | retries | 1008 | 1011 | send_error |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ...rows.map((row) => `| ${row.condition} | ${row.latency_ms} | ${row.attempt_count} | ${row.valid_runs} | ${row.retry_count} | ${row["1008_error_count"]} | ${row["1011_error_count"]} | ${row.send_error_count} |`),
    "",
    "## Important Files",
    "",
    "- `summary.csv` / `summary.json`: cell-level summary.",
    "- `metrics_by_condition_latency.csv`: same cell-level metrics in CSV form.",
    "- `per_attempt/`: full copied attempt artifacts plus pilot attempt records.",
    "- `raw_logs/`: flattened raw logs named by condition, latency, and run index.",
    "- `audio/`: flattened assistant audio files.",
    "- `timelines/`: flattened timeline JSON and per-attempt timeline PNGs.",
    "- `visualizations/`: pilot overlay timecharts.",
    "- `asr/` and `judge_outputs/`: reserved for later ASR / LLM judge passes; this pilot did not run ASR or LLM judge.",
    "",
    "## Stability",
    "",
    `- 1008: ${total1008}`,
    `- 1011: ${total1011}`,
    `- send errors: ${totalSendErrors}`,
    "",
    "## Initial Observations",
    "",
    ...notable,
    "",
    "These are first-pass timing observations from logs/audio events only. ASR and LLM judging were intentionally not run for this pilot.",
    "",
  ];
  writeFileSync(resolve(organizedDir, "README.md"), `${lines.join("\n")}\n`, "utf8");
  writeFileSync(resolve(rootDir, "README.md"), `${lines.join("\n")}\n`, "utf8");
}

async function main(): Promise<void> {
  const resultId = `${timestampForPath()}_boundary_tick_policy_pilot`;
  const rootDir = resolve(RESULT_DIR, resultId);
  const sourceRunsDir = resolve(rootDir, "source_runs");
  const organizedDir = resolve(rootDir, "organized");
  for (const dir of ["raw_logs", "audio", "asr", "timelines", "visualizations", "judge_outputs", "per_attempt"]) {
    mkdirSync(resolve(organizedDir, dir), { recursive: true });
  }
  mkdirSync(sourceRunsDir, { recursive: true });

  const command = `npm run boundary:tick-policy-pilot`;
  writeJson(resolve(rootDir, "config.json"), {
    result_id: resultId,
    command,
    conditions: CONDITIONS.map((condition) => condition.condition),
    latencies_ms: LATENCIES_MS,
    target_valid_per_cell: TARGET_VALID,
    max_attempts_per_cell: MAX_ATTEMPTS_PER_CELL,
    cell_concurrency: CELL_CONCURRENCY,
    source_runs_dir: relative(PROJECT_DIR, sourceRunsDir),
    organized_dir: relative(PROJECT_DIR, organizedDir),
  });

  const allAttempts: AttemptRecord[] = [];
  const rows: CellSummary[] = [];
  for (const { condition, probeMode } of CONDITIONS) {
    for (const latencyMs of LATENCIES_MS) {
      console.log(`\n=== ${condition} latency=${latencyMs} target_valid=${TARGET_VALID} ===`);
      const attempts = await runCell({ sourceRunsDir, organizedDir, condition, probeMode, latencyMs, allAttempts });
      const row = summarizeCell(condition, latencyMs, attempts);
      rows.push(row);
      if (row.valid_runs < TARGET_VALID) {
        console.warn(`WARNING: ${condition} ${latencyMs}ms ended with ${row.valid_runs}/${TARGET_VALID} valid after ${row.attempt_count} attempts.`);
      }
    }
  }

  writeJson(resolve(organizedDir, "summary.json"), { result_id: resultId, rows, attempts: allAttempts });
  writeJson(resolve(organizedDir, "corrected_summary.json"), { result_id: resultId, note: "No separate correction pass was run; this mirrors summary.json.", rows, attempts: allAttempts });
  writeCsv(resolve(organizedDir, "summary.csv"), rows as unknown as Array<Record<string, unknown>>);
  writeCsv(resolve(organizedDir, "corrected_summary.csv"), rows as unknown as Array<Record<string, unknown>>);
  writeCsv(resolve(organizedDir, "metrics_by_condition_latency.csv"), rows as unknown as Array<Record<string, unknown>>);
  writeCsv(resolve(organizedDir, "per_attempt_metrics.csv"), allAttempts as unknown as Array<Record<string, unknown>>);
  writeReadme(rootDir, organizedDir, rows, allAttempts);

  const plot = spawnSync("python3", [resolve(PROJECT_DIR, "scripts", "plot_boundary_tick_policy_pilot.py"), organizedDir], {
    cwd: PROJECT_DIR,
    encoding: "utf8",
    stdio: "inherit",
  });
  if (plot.status !== 0) console.warn(`Pilot plot script exited with status ${plot.status ?? "unknown"}.`);

  console.log(`\nBoundary tick policy pilot result: ${relative(PROJECT_DIR, rootDir)}`);
  console.log(`Organized: ${relative(PROJECT_DIR, organizedDir)}`);
  console.log(`Summary CSV: ${relative(PROJECT_DIR, resolve(organizedDir, "summary.csv"))}`);
  console.log(`Summary JSON: ${relative(PROJECT_DIR, resolve(organizedDir, "summary.json"))}`);
}

main().catch((error) => {
  console.error("boundary-tick-policy-pilot failed");
  console.error(error instanceof Error ? error.stack || error.message : String(error));
  process.exitCode = 1;
});
