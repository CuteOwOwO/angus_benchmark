import { spawnSync } from "node:child_process";
import {
  cpSync,
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { dirname, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

type Condition = "native_no_tick" | "external_single_tick";

type AttemptSummary = {
  session_valid?: boolean;
  close_1008?: boolean;
  close_1011?: boolean;
  client_send_error_count?: number;
  tool_call_success?: boolean;
  final_tool_response_sent?: boolean;
  waiting_audio_before_final_tool_result?: boolean;
  audio_after_tick?: boolean;
  post_tool_final_answer?: boolean;
  tool_call_time_ms?: number | null;
  first_audio_time_ms?: number | null;
  final_tool_response_sent_time_ms?: number | null;
  post_tool_final_latency_ms?: number | null;
  result_dir?: string;
};

type AttemptRecord = AttemptSummary & {
  condition: Condition;
  tick_mode: string;
  latency_ms: number;
  formal_attempt_index: number;
  source_result_dir: string;
  source_attempt_dir: string;
  relative_attempt_dir: string;
  pre_result_output_count: number;
  pre_result_output_frequency_per_sec: number | null;
  pre_result_text_transcription_event_count: number;
  pre_result_text_transcription_event_rate_per_sec: number | null;
  pre_result_window_ms: number | null;
  tick_status: "not_applicable" | "tick_skipped_final_before_tick" | "actual_tick_sent" | "tick_expected_but_not_sent";
  audio_occupancy_ratio_from_start_to_final: number | null;
  audio_occupancy_ratio_from_tool_call_to_final: number | null;
  assistant_audio_duration_from_start_to_final_ms: number | null;
  assistant_audio_duration_from_tool_call_to_final_ms: number | null;
  max_silent_gap_ms_from_start_to_final: number | null;
  post_final_answer_latency_ms: number | null;
};

type CellSummary = {
  condition: Condition;
  latency_ms: number;
  total_attempts: number;
  valid_attempts: number;
  "1008_count": number;
  "1011_count": number;
  retry_count: number;
  send_error_count: number;
  tool_call_success_count: number;
  final_tool_response_sent_count: number;
  waiting_audio_count: number;
  audio_after_tick_count: number;
  post_tool_final_answer_count: number;
  first_audio_time_ms_avg: number | null;
  first_audio_time_ms_median: number | null;
  pre_result_output_frequency_avg: number | null;
  pre_result_output_frequency_median: number | null;
  pre_result_text_transcription_event_count_avg: number | null;
  pre_result_text_transcription_event_rate_per_sec_avg: number | null;
  audio_occupancy_ratio_from_start_to_final_mean: number | null;
  audio_occupancy_ratio_from_start_to_final_median: number | null;
  audio_occupancy_ratio_from_tool_call_to_final_mean: number | null;
  audio_occupancy_ratio_from_tool_call_to_final_median: number | null;
  max_silent_gap_ms_from_start_to_final_mean: number | null;
  max_silent_gap_ms_from_start_to_final_median: number | null;
  post_final_answer_latency_ms_avg: number | null;
  post_final_answer_latency_ms_median: number | null;
};

const PROJECT_DIR = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const RESULT_DIR = resolve(PROJECT_DIR, "result");
const PROBE = resolve(PROJECT_DIR, "dist", "tau-live-tool-tick-factor-probe.js");
const CONDITIONS: Array<{ condition: Condition; tickMode: string }> = [
  { condition: "native_no_tick", tickMode: "native_no_tick" },
  { condition: "external_single_tick", tickMode: "client_status_tick_3000ms" },
];
const LATENCIES_MS = [3000, 5000, 8000, 12000];
const TARGET_VALID = 10;
const MAX_ATTEMPTS_PER_CELL = Number(process.env.FORMAL_MAX_ATTEMPTS_PER_CELL ?? 80);
const PCM_BYTES_PER_SECOND = 48_000;

function timestampForPath(date = new Date()): string {
  const [datePart, timePart] = date.toISOString().split("T");
  return `${datePart}_${timePart.replace("Z", "").replace(/\./g, "-").replace(/:/g, "-")}`;
}

function readJson<T>(path: string): T {
  return JSON.parse(readFileSync(path, "utf8")) as T;
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
  const text = String(value);
  if (!/[",\n]/.test(text)) return text;
  return `"${text.replace(/"/g, '""')}"`;
}

function writeCsv(path: string, rows: Array<Record<string, unknown>>): void {
  if (rows.length === 0) {
    writeFileSync(path, "", "utf8");
    return;
  }
  const headers = Object.keys(rows[0]);
  const lines = [headers.join(",")];
  for (const row of rows) lines.push(headers.map((header) => csvEscape(row[header])).join(","));
  writeFileSync(path, `${lines.join("\n")}\n`, "utf8");
}

function average(values: Array<number | null | undefined>): number | null {
  const nums = values.filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  if (!nums.length) return null;
  return Math.round(nums.reduce((sum, value) => sum + value, 0) / nums.length);
}

function averageFloat(values: Array<number | null | undefined>, digits = 3): number | null {
  const nums = values.filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  if (!nums.length) return null;
  const factor = 10 ** digits;
  return Math.round((nums.reduce((sum, value) => sum + value, 0) / nums.length) * factor) / factor;
}

function median(values: Array<number | null | undefined>): number | null {
  const nums = values.filter((value): value is number => typeof value === "number" && Number.isFinite(value)).sort((a, b) => a - b);
  if (!nums.length) return null;
  const mid = Math.floor(nums.length / 2);
  return nums.length % 2 ? nums[mid] : Math.round((nums[mid - 1] + nums[mid]) / 2);
}

function medianFloat(values: Array<number | null | undefined>, digits = 3): number | null {
  const nums = values.filter((value): value is number => typeof value === "number" && Number.isFinite(value)).sort((a, b) => a - b);
  if (!nums.length) return null;
  const mid = Math.floor(nums.length / 2);
  const raw = nums.length % 2 ? nums[mid] : (nums[mid - 1] + nums[mid]) / 2;
  const factor = 10 ** digits;
  return Math.round(raw * factor) / factor;
}

function findLatestProbeDir(before: Set<string>, stdout: string): string {
  const match = stdout.match(/Result directory:\s+(result\/[^\s]+)/);
  if (match) return resolve(PROJECT_DIR, match[1]);
  const after = readdirSync(RESULT_DIR)
    .filter((name) => name.endsWith("_tau_live_tool_tick_factor_probe"))
    .map((name) => resolve(RESULT_DIR, name))
    .filter((path) => !before.has(path));
  after.sort().reverse();
  if (!after[0]) throw new Error("Could not find probe result directory.");
  return after[0];
}

function listProbeDirs(): Set<string> {
  return new Set(
    readdirSync(RESULT_DIR)
      .filter((name) => name.endsWith("_tau_live_tool_tick_factor_probe"))
      .map((name) => resolve(RESULT_DIR, name)),
  );
}

function runProbeAttempt(tickMode: string, latencyMs: number): string {
  const before = listProbeDirs();
  const result = spawnSync(process.execPath, [
    PROBE,
    "--tick-modes",
    tickMode,
    "--attempts",
    "1",
    "--latency-ms",
    String(latencyMs),
    "--quiet-terminal-text",
  ], {
    cwd: PROJECT_DIR,
    encoding: "utf8",
    env: process.env,
    maxBuffer: 1024 * 1024 * 32,
  });
  if (result.stdout) {
    for (const line of result.stdout.split(/\n/).filter(Boolean)) {
      if (/Result directory|Attempts:|\] start|\] valid|\] not_valid|\] 1008|\] 1011|Summary:|Final comparison:/.test(line)) {
        console.log(`  ${line}`);
      }
    }
  }
  if (result.stderr) console.error(result.stderr);
  if (result.status !== 0) throw new Error(`Probe exited with status ${result.status ?? "unknown"}`);
  return findLatestProbeDir(before, result.stdout);
}

function firstEventMs(events: Array<Record<string, any>>, type: string): number | null {
  const event = events.find((item) => item.type === type && typeof item.event_ms === "number");
  return event ? event.event_ms : null;
}

function computePreResultTextTranscriptionActivity(
  events: Array<Record<string, any>>,
  finalMs: number | null,
): { count: number; rate: number | null; windowMs: number | null } {
  const userPromptMs = firstEventMs(events, "user_message_sent") ?? 0;
  const endMs = finalMs ?? firstEventMs(events, "session_closed") ?? Math.max(0, ...events.map((event) => Number(event.event_ms) || 0));
  const count = events.filter((event) => {
    if (event.type !== "text_output" && event.type !== "output_transcription") return false;
    return typeof event.event_ms === "number" && event.event_ms >= userPromptMs && event.event_ms < endMs;
  }).length;
  const windowMs = endMs > userPromptMs ? endMs - userPromptMs : null;
  return { count, rate: windowMs ? Math.round((count / (windowMs / 1000)) * 1000) / 1000 : null, windowMs };
}

function tickStatus(condition: Condition, latencyMs: number, summary: AttemptSummary): AttemptRecord["tick_status"] {
  if (condition === "native_no_tick") return "not_applicable";
  if (latencyMs <= 4000) return "tick_skipped_final_before_tick";
  return Number((summary as any).tick_send_success_count ?? 0) > 0 ? "actual_tick_sent" : "tick_expected_but_not_sent";
}

function computePostFinalAnswerLatency(events: Array<Record<string, any>>, finalMs: number | null): number | null {
  if (typeof finalMs !== "number") return null;
  const answer = events.find((event) => {
    if (event.type !== "text_output" && event.type !== "output_transcription") return false;
    if (typeof event.event_ms !== "number" || event.event_ms < finalMs) return false;
    const text = String(event.text ?? "").toLowerCase();
    return /shipped|ups|tracking|tomorrow/.test(text);
  });
  return answer ? answer.event_ms - finalMs : null;
}

function audioIntervals(events: Array<Record<string, any>>): Array<{ startMs: number; endMs: number }> {
  return events
    .filter((event) => event.type === "audio_output" && typeof event.event_ms === "number" && Number(event.bytes ?? 0) > 0)
    .map((event) => {
      const startMs = Number(event.event_ms);
      const durationMs = (Number(event.bytes) / PCM_BYTES_PER_SECOND) * 1000;
      return { startMs, endMs: startMs + durationMs };
    });
}

function clippedUnionDurationMs(intervals: Array<{ startMs: number; endMs: number }>, windowStartMs: number | null, windowEndMs: number | null): number | null {
  if (typeof windowStartMs !== "number" || typeof windowEndMs !== "number" || windowEndMs <= windowStartMs) return null;
  const clipped = intervals
    .map((interval) => ({
      startMs: Math.max(interval.startMs, windowStartMs),
      endMs: Math.min(interval.endMs, windowEndMs),
    }))
    .filter((interval) => interval.endMs > interval.startMs)
    .sort((a, b) => a.startMs - b.startMs);
  if (!clipped.length) return 0;
  const merged: Array<{ startMs: number; endMs: number }> = [];
  for (const interval of clipped) {
    const previous = merged.at(-1);
    if (!previous || interval.startMs > previous.endMs) {
      merged.push({ ...interval });
    } else {
      previous.endMs = Math.max(previous.endMs, interval.endMs);
    }
  }
  return Math.round(merged.reduce((sum, interval) => sum + (interval.endMs - interval.startMs), 0));
}

function mergedClippedIntervals(intervals: Array<{ startMs: number; endMs: number }>, windowStartMs: number | null, windowEndMs: number | null): Array<{ startMs: number; endMs: number }> {
  if (typeof windowStartMs !== "number" || typeof windowEndMs !== "number" || windowEndMs <= windowStartMs) return [];
  const clipped = intervals
    .map((interval) => ({
      startMs: Math.max(interval.startMs, windowStartMs),
      endMs: Math.min(interval.endMs, windowEndMs),
    }))
    .filter((interval) => interval.endMs > interval.startMs)
    .sort((a, b) => a.startMs - b.startMs);
  const merged: Array<{ startMs: number; endMs: number }> = [];
  for (const interval of clipped) {
    const previous = merged.at(-1);
    if (!previous || interval.startMs > previous.endMs) {
      merged.push({ ...interval });
    } else {
      previous.endMs = Math.max(previous.endMs, interval.endMs);
    }
  }
  return merged;
}

function maxSilentGapMs(intervals: Array<{ startMs: number; endMs: number }>, windowStartMs: number | null, windowEndMs: number | null): number | null {
  if (typeof windowStartMs !== "number" || typeof windowEndMs !== "number" || windowEndMs <= windowStartMs) return null;
  const merged = mergedClippedIntervals(intervals, windowStartMs, windowEndMs);
  let cursor = windowStartMs;
  let maxGap = 0;
  for (const interval of merged) {
    maxGap = Math.max(maxGap, interval.startMs - cursor);
    cursor = Math.max(cursor, interval.endMs);
  }
  return Math.round(Math.max(maxGap, windowEndMs - cursor));
}

function ratio(numeratorMs: number | null, denominatorMs: number | null): number | null {
  if (typeof numeratorMs !== "number" || typeof denominatorMs !== "number" || denominatorMs <= 0) return null;
  return Math.round((numeratorMs / denominatorMs) * 10_000) / 10_000;
}

function copyAttempt(sourceRunDir: string, tickMode: string, destCellDir: string, formalAttemptIndex: number): { attemptDir: string; summary: AttemptSummary; record: AttemptRecord } {
  const sourceAttemptDir = resolve(sourceRunDir, `condition_${tickMode}`, "attempt_0001");
  const destAttemptDir = resolve(destCellDir, `attempt_${String(formalAttemptIndex).padStart(4, "0")}`);
  if (existsSync(destAttemptDir)) rmSync(destAttemptDir, { recursive: true, force: true });
  cpSync(sourceAttemptDir, destAttemptDir, { recursive: true });
  const summary = readJson<AttemptSummary>(resolve(destAttemptDir, "summary.json"));
  const events = readJsonl(resolve(destAttemptDir, "timeline", "events.jsonl"));
  const finalMs = summary.final_tool_response_sent_time_ms ?? firstEventMs(events, "final_tool_response_sent");
  const preResult = computePreResultTextTranscriptionActivity(events, finalMs);
  const userPromptMs = firstEventMs(events, "user_message_sent") ?? 0;
  const toolCallMs = summary.tool_call_time_ms ?? firstEventMs(events, "tool_call_received");
  const audio = audioIntervals(events);
  const startToFinalAudioMs = clippedUnionDurationMs(audio, userPromptMs, finalMs);
  const toolCallToFinalAudioMs = clippedUnionDurationMs(audio, toolCallMs, finalMs);
  const startToFinalMaxSilentGapMs = maxSilentGapMs(audio, userPromptMs, finalMs);
  const record: AttemptRecord = {
    ...summary,
    condition: "native_no_tick",
    tick_mode: tickMode,
    latency_ms: 0,
    formal_attempt_index: formalAttemptIndex,
    source_result_dir: relative(PROJECT_DIR, sourceRunDir),
    source_attempt_dir: relative(PROJECT_DIR, sourceAttemptDir),
    relative_attempt_dir: relative(PROJECT_DIR, destAttemptDir),
    pre_result_output_count: preResult.count,
    pre_result_output_frequency_per_sec: preResult.rate,
    pre_result_text_transcription_event_count: preResult.count,
    pre_result_text_transcription_event_rate_per_sec: preResult.rate,
    pre_result_window_ms: preResult.windowMs,
    tick_status: tickStatus("native_no_tick", 0, summary),
    assistant_audio_duration_from_start_to_final_ms: startToFinalAudioMs,
    assistant_audio_duration_from_tool_call_to_final_ms: toolCallToFinalAudioMs,
    audio_occupancy_ratio_from_start_to_final: ratio(startToFinalAudioMs, preResult.windowMs),
    audio_occupancy_ratio_from_tool_call_to_final: ratio(
      toolCallToFinalAudioMs,
      typeof toolCallMs === "number" && typeof finalMs === "number" ? finalMs - toolCallMs : null,
    ),
    max_silent_gap_ms_from_start_to_final: startToFinalMaxSilentGapMs,
    post_final_answer_latency_ms: computePostFinalAnswerLatency(events, finalMs),
  };
  return { attemptDir: destAttemptDir, summary, record };
}

function summarizeCell(condition: Condition, latencyMs: number, attempts: AttemptRecord[]): CellSummary {
  return {
    condition,
    latency_ms: latencyMs,
    total_attempts: attempts.length,
    valid_attempts: attempts.filter((attempt) => attempt.session_valid).length,
    "1008_count": attempts.filter((attempt) => attempt.close_1008).length,
    "1011_count": attempts.filter((attempt) => attempt.close_1011).length,
    retry_count: attempts.length - attempts.filter((attempt) => attempt.session_valid).length,
    send_error_count: attempts.reduce((sum, attempt) => sum + Number(attempt.client_send_error_count ?? 0), 0),
    tool_call_success_count: attempts.filter((attempt) => attempt.tool_call_success).length,
    final_tool_response_sent_count: attempts.filter((attempt) => attempt.final_tool_response_sent).length,
    waiting_audio_count: attempts.filter((attempt) => attempt.waiting_audio_before_final_tool_result).length,
    audio_after_tick_count: attempts.filter((attempt) => attempt.audio_after_tick).length,
    post_tool_final_answer_count: attempts.filter((attempt) => attempt.post_tool_final_answer).length,
    first_audio_time_ms_avg: average(attempts.map((attempt) => attempt.first_audio_time_ms)),
    first_audio_time_ms_median: median(attempts.map((attempt) => attempt.first_audio_time_ms)),
    pre_result_output_frequency_avg: average(attempts.map((attempt) => attempt.pre_result_output_frequency_per_sec)),
    pre_result_output_frequency_median: median(attempts.map((attempt) => attempt.pre_result_output_frequency_per_sec)),
    pre_result_text_transcription_event_count_avg: average(attempts.map((attempt) => attempt.pre_result_text_transcription_event_count)),
    pre_result_text_transcription_event_rate_per_sec_avg: averageFloat(attempts.map((attempt) => attempt.pre_result_text_transcription_event_rate_per_sec)),
    audio_occupancy_ratio_from_start_to_final_mean: averageFloat(attempts.map((attempt) => attempt.audio_occupancy_ratio_from_start_to_final)),
    audio_occupancy_ratio_from_start_to_final_median: medianFloat(attempts.map((attempt) => attempt.audio_occupancy_ratio_from_start_to_final)),
    audio_occupancy_ratio_from_tool_call_to_final_mean: averageFloat(attempts.map((attempt) => attempt.audio_occupancy_ratio_from_tool_call_to_final)),
    audio_occupancy_ratio_from_tool_call_to_final_median: medianFloat(attempts.map((attempt) => attempt.audio_occupancy_ratio_from_tool_call_to_final)),
    max_silent_gap_ms_from_start_to_final_mean: averageFloat(attempts.map((attempt) => attempt.max_silent_gap_ms_from_start_to_final)),
    max_silent_gap_ms_from_start_to_final_median: medianFloat(attempts.map((attempt) => attempt.max_silent_gap_ms_from_start_to_final)),
    post_final_answer_latency_ms_avg: average(attempts.filter((attempt) => attempt.session_valid).map((attempt) => attempt.post_final_answer_latency_ms)),
    post_final_answer_latency_ms_median: median(attempts.filter((attempt) => attempt.session_valid).map((attempt) => attempt.post_final_answer_latency_ms)),
  };
}

function writeFinalMarkdown(resultDir: string, rows: CellSummary[]): void {
  const lines = [
    "# Tau Live Tool Formal Benchmark",
    "",
    "Frozen setting: native tool call, native `sendToolResponse(...)` final result, explicit `TOOL_RESULT` final payload, post-final observation window of 8000 ms, concurrency 1.",
    "",
    "Valid definition: an attempt is valid when the tool call succeeds, final tool response is sent, and post-final model output mentions the final order result (`shipped`, `UPS`, tracking, or `tomorrow`). The underlying probe also requires no 1008/1011 close for `session_valid`.",
    "",
    "| condition | latency_ms | total | valid | 1008 | 1011 | retries | tool_call | final_sent | waiting_audio | audio_after_tick | post_final | first_audio_avg | audio_occupancy_start_to_final | audio_occupancy_tool_to_final | max_silent_gap_ms | post_final_latency_avg |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ...rows.map((row) =>
      `| ${row.condition} | ${row.latency_ms} | ${row.total_attempts} | ${row.valid_attempts} | ${row["1008_count"]} | ${row["1011_count"]} | ${row.retry_count} | ${row.tool_call_success_count} | ${row.final_tool_response_sent_count} | ${row.waiting_audio_count} | ${row.audio_after_tick_count} | ${row.post_tool_final_answer_count} | ${row.first_audio_time_ms_avg ?? ""} | ${row.audio_occupancy_ratio_from_start_to_final_mean ?? ""} | ${row.audio_occupancy_ratio_from_tool_call_to_final_mean ?? ""} | ${row.max_silent_gap_ms_from_start_to_final_mean ?? ""} | ${row.post_final_answer_latency_ms_avg ?? ""} |`,
    ),
    "",
    "## Initial Interpretation",
    "",
    "- Compare `native_no_tick` against `external_single_tick` within each latency row. The tick condition sends exactly one natural external pending signal at 4000 ms when latency is greater than 4000 ms.",
    "- The 3000 ms latency cells are expected to have no external tick because the final result arrives before the fixed tick time.",
    "- The 5000 ms `external_single_tick` cell is the edge case: one tick arrives shortly before final.",
    "- Post-final answer latency is measured from final tool response to the first post-final text/transcription event containing final-answer keywords.",
    "- Stability should be read from total attempts, retries, and 1008/1011 counts rather than valid attempts alone.",
    "",
    "## Charts",
    "",
    "- `visualizations/first_audio_time.png`",
    "- `visualizations/pre_result_text_transcription_event_count.png`",
    "- `visualizations/pre_result_text_transcription_event_rate.png`",
    "- `visualizations/audio_occupancy_ratio_from_start_to_final.png`",
    "- `visualizations/audio_occupancy_ratio_from_tool_call_to_final.png`",
    "- `visualizations/max_silent_gap_ms_from_start_to_final.png`",
    "- `visualizations/post_final_answer_latency.png`",
    "- `visualizations/stability_retry_overview.png`",
    "",
  ];
  writeFileSync(resolve(resultDir, "final_comparison.md"), `${lines.join("\n")}\n`, "utf8");
}

async function main(): Promise<void> {
  const resultId = `${timestampForPath()}_tau_live_tool_formal_benchmark`;
  const resultDir = resolve(RESULT_DIR, resultId);
  mkdirSync(resultDir, { recursive: true });
  writeJson(resolve(resultDir, "config.json"), {
    result_id: resultId,
    conditions: CONDITIONS.map((condition) => condition.condition),
    latencies_ms: LATENCIES_MS,
    target_valid_per_cell: TARGET_VALID,
    max_attempts_per_cell: MAX_ATTEMPTS_PER_CELL,
    tick_rule: "external_single_tick sends one natural pending signal at 4000ms only when latency_ms > 4000",
    post_final_observation_ms: 8000,
    concurrency: 1,
  });

  const allAttempts: AttemptRecord[] = [];
  const rows: CellSummary[] = [];

  for (const { condition, tickMode } of CONDITIONS) {
    for (const latencyMs of LATENCIES_MS) {
      const cellDir = resolve(resultDir, `condition_${condition}`, `latency_${latencyMs}ms`);
      mkdirSync(cellDir, { recursive: true });
      const cellAttempts: AttemptRecord[] = [];
      let valid = 0;
      console.log(`\n=== ${condition} latency=${latencyMs}ms target_valid=${TARGET_VALID} ===`);
      while (valid < TARGET_VALID && cellAttempts.length < MAX_ATTEMPTS_PER_CELL) {
        const formalAttemptIndex = cellAttempts.length + 1;
        console.log(`[${condition} ${latencyMs}ms attempt ${formalAttemptIndex}] running`);
        const sourceRunDir = runProbeAttempt(tickMode, latencyMs);
        const copied = copyAttempt(sourceRunDir, tickMode, cellDir, formalAttemptIndex);
        copied.record.condition = condition;
        copied.record.latency_ms = latencyMs;
        copied.record.tick_mode = tickMode;
        copied.record.tick_status = tickStatus(condition, latencyMs, copied.summary);
        writeJson(resolve(copied.attemptDir, "formal_attempt_summary.json"), copied.record);
        cellAttempts.push(copied.record);
        allAttempts.push(copied.record);
        if (copied.record.session_valid) valid += 1;
        console.log(
          `[${condition} ${latencyMs}ms attempt ${formalAttemptIndex}] ${
            copied.record.session_valid ? "valid" : copied.record.close_1008 ? "1008" : copied.record.close_1011 ? "1011" : "not_valid"
          } valid=${valid}/${TARGET_VALID}`,
        );
      }
      const row = summarizeCell(condition, latencyMs, cellAttempts);
      rows.push(row);
      writeJson(resolve(cellDir, "summary.json"), { ...row, attempts: cellAttempts });
      writeCsv(resolve(cellDir, "summary.csv"), [row as unknown as Record<string, unknown>]);
      writeCsv(resolve(cellDir, "attempts.csv"), cellAttempts as unknown as Array<Record<string, unknown>>);
      if (row.valid_attempts < TARGET_VALID) {
        console.warn(`WARNING: ${condition} ${latencyMs}ms ended with ${row.valid_attempts}/${TARGET_VALID} valid after ${row.total_attempts} attempts.`);
      }
    }
  }

  writeJson(resolve(resultDir, "summary.json"), { result_id: resultId, rows, attempts: allAttempts });
  writeCsv(resolve(resultDir, "summary.csv"), rows as unknown as Array<Record<string, unknown>>);
  writeCsv(resolve(resultDir, "attempts.csv"), allAttempts as unknown as Array<Record<string, unknown>>);
  writeFinalMarkdown(resultDir, rows);

  const plot = spawnSync("python3", [resolve(PROJECT_DIR, "scripts", "plot_tau_live_tool_formal_benchmark.py"), resultDir], {
    cwd: PROJECT_DIR,
    encoding: "utf8",
    stdio: "inherit",
  });
  if (plot.status !== 0) console.warn(`Plot script exited with status ${plot.status ?? "unknown"}.`);

  console.log(`\nFormal benchmark result: ${relative(PROJECT_DIR, resultDir)}`);
  console.log(`Summary: ${relative(PROJECT_DIR, resolve(resultDir, "summary.csv"))}`);
  console.log(`Final comparison: ${relative(PROJECT_DIR, resolve(resultDir, "final_comparison.md"))}`);
}

main().catch((error) => {
  console.error("tau-live-tool-formal-benchmark failed");
  console.error(error instanceof Error ? error.stack || error.message : String(error));
  process.exitCode = 1;
});
