import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { config as loadEnv } from "dotenv";

const PROJECT_DIR = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const DEFAULT_ASR_DIR = resolve(
  PROJECT_DIR,
  "result",
  "2026-06-29_10-15-29-961_calendar_route_two_step_2step_tick_vs_no_tick_latency_sweep(2tool大跑)",
  "organized",
  "asr",
);
const DEFAULT_ENV_EXAMPLE = resolve(PROJECT_DIR, ".env.example");
const DEFAULT_MODEL = "gemma-3-27b-it";
const DEFAULT_REQUEST_TIMEOUT_MS = 45_000;
const OUTPUT_DIR_NAME = "calendar_route_phase_segmentation";

type Args = {
  asrDir: string;
  model: string;
  envFile?: string;
  keyVar?: string;
  force: boolean;
  limit?: number;
  retries: number;
  requestTimeoutMs: number;
  retryErrors: boolean;
  concurrency: number;
};

type AsrSegment = {
  segment_index: number;
  start_sec: number;
  end_sec: number;
  text: string;
};

type AsrRow = {
  condition: string;
  latency_ms: number;
  latency_s: number;
  attempt_index: number;
  attempt_id: string;
  audio_path: string;
  transcript: string;
  segments: AsrSegment[];
};

type PhaseJudgeParsed = {
  phase2_start_segment_index?: number | null;
  phase3_start_segment_index?: number | null;
  confidence?: number | null;
  phase1_label?: string;
  phase2_label?: string;
  phase3_label?: string;
  phase2_start_evidence?: string;
  phase3_start_evidence?: string;
  notes?: string;
};

type JudgeResult = {
  ok: boolean;
  parsed?: PhaseJudgeParsed;
  rawText: string;
  parseError?: string;
  attemptsUsed: number;
};

type AttemptOutputRow = {
  condition: string;
  latency_ms: string;
  latency_s: string;
  attempt_index: string;
  attempt_id: string;
  ok: string;
  confidence: string;
  phase2_start_segment_index: string;
  phase2_start_sec: string;
  phase2_start_evidence: string;
  phase3_start_segment_index: string;
  phase3_start_sec: string;
  phase3_start_evidence: string;
  phase1_text: string;
  phase2_text: string;
  phase3_text: string;
  phase1_token_count: string;
  phase2_token_count: string;
  phase3_token_count: string;
  parse_error: string;
  notes: string;
  audio_path: string;
};

function parseArgs(argv: string[]): Args {
  const args: Args = {
    asrDir: DEFAULT_ASR_DIR,
    model: process.env.GEMMA_MODEL || process.env.GEMINI_MODEL || DEFAULT_MODEL,
    force: false,
    retries: 2,
    requestTimeoutMs: DEFAULT_REQUEST_TIMEOUT_MS,
    retryErrors: false,
    concurrency: 1,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const item = argv[index];
    const next = argv[index + 1];
    if (item === "--asr-dir" && next) {
      args.asrDir = resolve(next);
      index += 1;
    } else if (item === "--model" && next) {
      args.model = next;
      index += 1;
    } else if (item === "--env-file" && next) {
      args.envFile = resolve(next);
      index += 1;
    } else if (item === "--key-var" && next) {
      args.keyVar = next;
      index += 1;
    } else if (item === "--force") {
      args.force = true;
    } else if (item === "--limit" && next) {
      args.limit = Number(next);
      index += 1;
    } else if (item === "--retries" && next) {
      args.retries = Number(next);
      index += 1;
    } else if (item === "--request-timeout-ms" && next) {
      args.requestTimeoutMs = Number(next);
      index += 1;
    } else if (item === "--retry-errors") {
      args.retryErrors = true;
    } else if (item === "--concurrency" && next) {
      args.concurrency = Math.max(1, Number(next));
      index += 1;
    } else if (item === "--help") {
      printHelp();
      process.exit(0);
    } else {
      throw new Error(`Unknown or incomplete argument: ${item}`);
    }
  }
  return args;
}

function printHelp(): void {
  console.log(
    [
      "Usage: npm run judge:calendar-route-phases -- [options]",
      "",
      "Options:",
      "  --asr-dir <path>     organized/asr folder with asr_summary.json",
      "  --model <name>       Gemma/Gemini model name, default gemma-3-27b-it",
      "  --env-file <path>    Optional env file",
      "  --key-var <name>     Use a specific API key env var, e.g. GEMINI_API_KEY",
      "  --force              Ignore cached raw responses",
      "  --limit <n>          Debug limit",
      "  --retries <n>        JSON/API retry count, default 2",
      "  --request-timeout-ms <n>  Per judge request timeout, default 45000",
      "  --retry-errors       Re-run cached judge responses whose previous result was not ok",
      "  --concurrency <n>    Concurrent judge requests, default 1",
    ].join("\n"),
  );
}

function loadEnvironment(args: Args): { keyVar: string; apiKey: string; model: string } {
  const envCandidates = [
    process.env.GEMINI_ENV_FILE,
    args.envFile,
    resolve(PROJECT_DIR, ".env"),
    DEFAULT_ENV_EXAMPLE,
  ].filter(Boolean) as string[];

  for (const path of envCandidates) {
    if (existsSync(path)) {
      loadEnv({ path, override: false });
      parseLooseEnvFile(path);
    }
  }

  const keyNames = args.keyVar ? [args.keyVar] : ["GEMMA_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"];
  for (const name of keyNames) {
    const value = process.env[name]?.trim();
    if (value && value !== "your_api_key_here") {
      return { keyVar: name, apiKey: value, model: process.env.GEMMA_MODEL || args.model };
    }
  }
  throw new Error(`Missing Gemma API key. Set one of ${keyNames.join(", ")}.`);
}

function parseLooseEnvFile(path: string): void {
  const text = readFileSync(path, "utf-8");
  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;
    const equalIndex = line.indexOf("=");
    if (equalIndex > 0) {
      const key = line.slice(0, equalIndex).trim();
      let value = line.slice(equalIndex + 1).trim();
      value = value.replace(/^["']|["']$/g, "");
      if (key && value && !process.env[key]) process.env[key] = value;
      continue;
    }
    if (/AIza[0-9A-Za-z_-]+/.test(line) && /gemma/i.test(line) && !process.env.GEMMA_API_KEY) {
      const match = line.match(/AIza[0-9A-Za-z_-]+/);
      if (match) process.env.GEMMA_API_KEY = match[0];
    }
  }
}

function readAsrRows(asrDir: string): AsrRow[] {
  const summaryPath = resolve(asrDir, "asr_summary.json");
  if (!existsSync(summaryPath)) throw new Error(`Missing ASR summary: ${summaryPath}`);
  const data = JSON.parse(readFileSync(summaryPath, "utf-8")) as { rows?: AsrRow[] };
  return (data.rows ?? [])
    .filter((row) => Array.isArray(row.segments) && row.segments.length > 0)
    .sort((left, right) => {
      const condition = left.condition.localeCompare(right.condition);
      if (condition !== 0) return condition;
      const latency = left.latency_ms - right.latency_ms;
      if (latency !== 0) return latency;
      return left.attempt_index - right.attempt_index;
    });
}

function phasePrompt(row: AsrRow): string {
  const segments = row.segments
    .map((segment) => {
      const start = Number(segment.start_sec).toFixed(2);
      const end = Number(segment.end_sec).toFixed(2);
      return `${segment.segment_index}. [${start}s-${end}s] ${segment.text}`;
    })
    .join("\n");

  return [
    "You are segmenting an ASR transcript from a spoken two-tool benchmark.",
    "",
    "Task context:",
    "- User asks when to leave for the next meeting this afternoon, wanting to arrive about 10 minutes early.",
    "- Tool 1 is calendar lookup: get the next meeting time/location.",
    "- Tool 2 is route lookup: get travel time/ETA to the meeting location.",
    "- Final answer is the departure-time recommendation, e.g. leave around/by 2:15 PM.",
    "",
    "Phase definitions:",
    "Phase 1: tool1/calendar lookup phase. Starts at the beginning and ends immediately before the assistant begins reporting the calendar result.",
    "Phase 2: tool2/route lookup phase. Starts when the assistant begins reporting the calendar result. It can include reporting the calendar result, saying it will check route/travel time, waiting for route, and reporting route/travel-time details. It ends immediately before the final departure-time answer begins.",
    "Phase 3: answer phase. Starts when the assistant begins the final answer/recommendation about what time the user should leave.",
    "",
    "Important boundary rules:",
    "- The Phase 2 start is NOT necessarily a tool-call timestamp. Use the spoken ASR content.",
    "- Phase 2 starts at the first segment where the assistant begins saying the calendar result, such as 'your next meeting is at 3 PM' or 'I found your next event'.",
    "- Phrases such as 'to figure out when you should leave, I need to check travel time' are Phase 2, NOT Phase 3.",
    "- Phase 3 starts only when the assistant begins giving the final departure recommendation, such as 'you should leave by/around 2:15 PM' or a sentence that directly leads into that recommendation.",
    "- If a final-answer lead-in and the actual recommendation are split across adjacent segments, set Phase 3 to the earliest lead-in segment that is part of that final answer.",
    "- ASR may mishear 'NTU EE' as 'NTUE'; treat that as the meeting location.",
    "",
    "Return segment indexes, not timestamps. Use null only if the boundary is genuinely not present.",
    "",
    "ASR segments:",
    segments,
    "",
    "Return ONLY valid JSON with exactly these fields:",
    JSON.stringify(
      {
        phase2_start_segment_index: 2,
        phase2_start_evidence: "short quoted evidence from that segment",
        phase3_start_segment_index: 5,
        phase3_start_evidence: "short quoted evidence from that segment",
        confidence: 0.0,
        notes: "brief explanation",
      },
      null,
      2,
    ),
  ].join("\n");
}

function extractJson(text: string): PhaseJudgeParsed {
  const trimmed = text.trim();
  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
  const candidate = fenced ? fenced[1].trim() : trimmed;
  try {
    return JSON.parse(candidate) as PhaseJudgeParsed;
  } catch {
    const objects = findJsonObjectCandidates(candidate);
    for (const objectText of objects.reverse()) {
      try {
        return JSON.parse(objectText) as PhaseJudgeParsed;
      } catch {
        // Keep searching.
      }
    }
    throw new Error("No parseable JSON object found.");
  }
}

function findJsonObjectCandidates(text: string): string[] {
  const candidates: string[] = [];
  for (let start = 0; start < text.length; start += 1) {
    if (text[start] !== "{") continue;
    let depth = 0;
    let inString = false;
    let escaped = false;
    for (let index = start; index < text.length; index += 1) {
      const char = text[index];
      if (inString) {
        if (escaped) escaped = false;
        else if (char === "\\") escaped = true;
        else if (char === '"') inString = false;
        continue;
      }
      if (char === '"') inString = true;
      else if (char === "{") depth += 1;
      else if (char === "}") {
        depth -= 1;
        if (depth === 0) {
          candidates.push(text.slice(start, index + 1));
          break;
        }
      }
    }
  }
  return candidates;
}

async function judgeJson(
  apiKey: string,
  model: string,
  prompt: string,
  cachePath: string,
  force: boolean,
  retries: number,
  timeoutMs: number,
  retryErrors: boolean,
): Promise<JudgeResult> {
  if (!force && existsSync(cachePath)) {
    const cached = JSON.parse(readFileSync(cachePath, "utf-8")) as JudgeResult;
    if (cached.ok || !retryErrors) return cached;
  }

  let rawText = "";
  let parseError = "";
  const attempts = Math.max(1, retries + 1);
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      rawText = await generateContentText(apiKey, model, prompt, timeoutMs);
      const parsed = extractJson(rawText);
      const result: JudgeResult = { ok: true, parsed, rawText, attemptsUsed: attempt };
      writeFileSync(cachePath, `${JSON.stringify(result, null, 2)}\n`, "utf-8");
      return result;
    } catch (error) {
      parseError = summarizeError(error);
      if (attempt < attempts) await sleep(500 * attempt);
    }
  }

  const result: JudgeResult = { ok: false, rawText, parseError, attemptsUsed: attempts };
  writeFileSync(cachePath, `${JSON.stringify(result, null, 2)}\n`, "utf-8");
  return result;
}

async function generateContentText(apiKey: string, model: string, prompt: string, timeoutMs: number): Promise<string> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const modelPath = model.startsWith("models/") ? model : `models/${model}`;
    const url = `https://generativelanguage.googleapis.com/v1beta/${modelPath}:generateContent`;
    const response = await fetch(`${url}?key=${encodeURIComponent(apiKey)}`, {
      method: "POST",
      signal: controller.signal,
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        contents: [{ role: "user", parts: [{ text: prompt }] }],
        generationConfig: {
          temperature: 0,
          responseMimeType: "application/json",
        },
      }),
    });
    const bodyText = await response.text();
    if (!response.ok) throw new Error(`generateContent HTTP ${response.status}: ${bodyText}`);
    const body = JSON.parse(bodyText) as {
      candidates?: Array<{ content?: { parts?: Array<{ text?: string }> } }>;
    };
    return body.candidates?.[0]?.content?.parts?.map((part) => part.text ?? "").join("") ?? "";
  } finally {
    clearTimeout(timeout);
  }
}

function buildAttemptOutput(row: AsrRow, result: JudgeResult): AttemptOutputRow {
  const parsed = result.parsed ?? {};
  const phase2Start = normalizeBoundary(parsed.phase2_start_segment_index, row.segments.length);
  const phase3Start = normalizeBoundary(parsed.phase3_start_segment_index, row.segments.length);
  const phase1Segments = phase2Start === null ? row.segments : row.segments.slice(0, phase2Start);
  const phase2Segments =
    phase2Start === null
      ? []
      : phase3Start === null
        ? row.segments.slice(phase2Start)
        : row.segments.slice(phase2Start, Math.max(phase2Start, phase3Start));
  const phase3Segments = phase3Start === null ? [] : row.segments.slice(phase3Start);
  return {
    condition: row.condition,
    latency_ms: String(row.latency_ms),
    latency_s: String(row.latency_s),
    attempt_index: String(row.attempt_index),
    attempt_id: row.attempt_id,
    ok: String(result.ok && phase2Start !== null && phase3Start !== null && phase2Start < phase3Start),
    confidence: asString(parsed.confidence),
    phase2_start_segment_index: phase2Start === null ? "" : String(phase2Start),
    phase2_start_sec: phase2Start === null ? "" : fmt(row.segments[phase2Start]?.start_sec),
    phase2_start_evidence: asString(parsed.phase2_start_evidence),
    phase3_start_segment_index: phase3Start === null ? "" : String(phase3Start),
    phase3_start_sec: phase3Start === null ? "" : fmt(row.segments[phase3Start]?.start_sec),
    phase3_start_evidence: asString(parsed.phase3_start_evidence),
    phase1_text: joinSegments(phase1Segments),
    phase2_text: joinSegments(phase2Segments),
    phase3_text: joinSegments(phase3Segments),
    phase1_token_count: String(tokenize(joinSegments(phase1Segments)).length),
    phase2_token_count: String(tokenize(joinSegments(phase2Segments)).length),
    phase3_token_count: String(tokenize(joinSegments(phase3Segments)).length),
    parse_error: result.ok ? "" : result.parseError || "parse_error",
    notes: asString(parsed.notes),
    audio_path: row.audio_path,
  };
}

function normalizeBoundary(value: unknown, length: number): number | null {
  if (value === null || value === undefined || value === "") return null;
  const numberValue = typeof value === "number" ? value : Number(value);
  if (!Number.isInteger(numberValue)) return null;
  if (numberValue < 0 || numberValue >= length) return null;
  return numberValue;
}

function joinSegments(segments: AsrSegment[]): string {
  return segments.map((segment) => segment.text.trim()).filter(Boolean).join(" ").replace(/\s+/g, " ").trim();
}

function tokenize(text: string): string[] {
  return text.toLowerCase().match(/[a-z0-9]+(?:'[a-z0-9]+)?/g) ?? [];
}

function writeCsv(path: string, rows: Record<string, unknown>[], headers: string[]): void {
  const lines = [headers.join(",")];
  for (const row of rows) {
    lines.push(headers.map((header) => csvEscape(row[header])).join(","));
  }
  writeFileSync(path, `${lines.join("\n")}\n`, "utf-8");
}

function csvEscape(value: unknown): string {
  const text = value === undefined || value === null ? "" : String(value);
  if (/[",\n\r]/.test(text)) return `"${text.replace(/"/g, '""')}"`;
  return text;
}

function summarize(rows: AttemptOutputRow[]): Record<string, unknown>[] {
  const groups = new Map<string, AttemptOutputRow[]>();
  for (const row of rows) {
    const key = `${row.condition}__${row.latency_ms}`;
    groups.set(key, [...(groups.get(key) ?? []), row]);
  }
  return [...groups.entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([key, items]) => {
      const [condition, latencyMs] = key.split("__");
      const okItems = items.filter((item) => item.ok === "true");
      const confidence = okItems.map((item) => Number(item.confidence)).filter(Number.isFinite);
      return {
        condition,
        latency_ms: latencyMs,
        attempts: items.length,
        segmented_ok_count: okItems.length,
        segmented_ok_rate: items.length ? fmt(okItems.length / items.length, 3) : "",
        mean_confidence: confidence.length ? fmt(confidence.reduce((sum, value) => sum + value, 0) / confidence.length, 3) : "",
        mean_phase1_tokens: meanTextNumber(okItems, "phase1_token_count"),
        mean_phase2_tokens: meanTextNumber(okItems, "phase2_token_count"),
        mean_phase3_tokens: meanTextNumber(okItems, "phase3_token_count"),
        parse_error_count: items.filter((item) => item.parse_error).length,
      };
    });
}

function meanTextNumber(rows: AttemptOutputRow[], key: keyof AttemptOutputRow): string {
  const values = rows.map((row) => Number(row[key])).filter(Number.isFinite);
  if (values.length === 0) return "";
  return fmt(values.reduce((sum, value) => sum + value, 0) / values.length, 1);
}

function asString(value: unknown): string {
  if (value === undefined || value === null) return "";
  return String(value);
}

function fmt(value: unknown, digits = 3): string {
  const numberValue = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(numberValue)) return "";
  return numberValue.toFixed(digits);
}

function cleanId(text: string): string {
  return text.replace(/[^0-9A-Za-z_-]+/g, "_");
}

function summarizeError(error: unknown): string {
  if (error instanceof Error) return `${error.name}: ${error.message}`;
  return String(error);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolveSleep) => setTimeout(resolveSleep, ms));
}

function rel(path: string): string {
  try {
    return relative(PROJECT_DIR, path);
  } catch {
    return path;
  }
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const env = loadEnvironment(args);
  const asrDir = resolve(args.asrDir);
  const outputDir = resolve(asrDir, "..", "judge_outputs", OUTPUT_DIR_NAME);
  const cacheDir = resolve(outputDir, "cache");
  const perAttemptDir = resolve(outputDir, "per_attempt");
  mkdirSync(cacheDir, { recursive: true });
  mkdirSync(perAttemptDir, { recursive: true });

  const allRows = readAsrRows(asrDir);
  const selectedRows = args.limit ? allRows.slice(0, args.limit) : allRows;
  console.log(`Phase segmentation inputs: ${selectedRows.length}`);
  console.log(`Model: ${env.model}`);
  console.log(`API key source: ${env.keyVar}`);
  console.log(`Output: ${rel(outputDir)}`);

  const outputs: AttemptOutputRow[] = [];
  const outputByKey = new Map<string, AttemptOutputRow>();
  let processed = 0;
  let cursor = 0;

  async function processRow(row: AsrRow): Promise<void> {
    const cachePath = resolve(
      cacheDir,
      `${cleanId(row.condition)}_${row.latency_ms}_${cleanId(row.attempt_id)}_phase_segmentation.json`,
    );
    const result = await judgeJson(
      env.apiKey,
      env.model,
      phasePrompt(row),
      cachePath,
      args.force,
      args.retries,
      args.requestTimeoutMs,
      args.retryErrors,
    );
    const outputRow = buildAttemptOutput(row, result);
    outputByKey.set(`${row.condition}__${row.latency_ms}__${row.attempt_index}`, outputRow);
    writeFileSync(
      resolve(perAttemptDir, `${cleanId(row.condition)}_${row.latency_ms}_${cleanId(row.attempt_id)}.phase.json`),
      `${JSON.stringify({ input: row, judge: result, derived: outputRow }, null, 2)}\n`,
      "utf-8",
    );
    processed += 1;
    if (processed % 10 === 0 || !result.ok || processed === selectedRows.length) {
      console.log(
        `Progress ${processed}/${selectedRows.length}; current=${row.condition}/${row.latency_ms}/${row.attempt_id}; ok=${outputRow.ok}; p2=${outputRow.phase2_start_segment_index}; p3=${outputRow.phase3_start_segment_index}`,
      );
    }
  }

  async function worker(): Promise<void> {
    for (;;) {
      const row = selectedRows[cursor];
      cursor += 1;
      if (!row) return;
      await processRow(row);
    }
  }

  const workerCount = Math.min(args.concurrency, selectedRows.length);
  await Promise.all(Array.from({ length: workerCount }, () => worker()));

  for (const row of selectedRows) {
    const output = outputByKey.get(`${row.condition}__${row.latency_ms}__${row.attempt_index}`);
    if (output) outputs.push(output);
  }

  const attemptHeaders = [
    "condition",
    "latency_ms",
    "latency_s",
    "attempt_index",
    "attempt_id",
    "ok",
    "confidence",
    "phase2_start_segment_index",
    "phase2_start_sec",
    "phase2_start_evidence",
    "phase3_start_segment_index",
    "phase3_start_sec",
    "phase3_start_evidence",
    "phase1_token_count",
    "phase2_token_count",
    "phase3_token_count",
    "parse_error",
    "notes",
    "phase1_text",
    "phase2_text",
    "phase3_text",
    "audio_path",
  ];
  writeCsv(resolve(outputDir, "phase_segmentation_attempts.csv"), outputs, attemptHeaders);
  const summaryRows = summarize(outputs);
  writeCsv(resolve(outputDir, "phase_segmentation_summary.csv"), summaryRows, [
    "condition",
    "latency_ms",
    "attempts",
    "segmented_ok_count",
    "segmented_ok_rate",
    "mean_confidence",
    "mean_phase1_tokens",
    "mean_phase2_tokens",
    "mean_phase3_tokens",
    "parse_error_count",
  ]);
  writeFileSync(
    resolve(outputDir, "README.md"),
    [
      "# Calendar Route Phase Segmentation",
      "",
      `ASR source: \`${rel(asrDir)}\``,
      `Model: \`${env.model}\``,
      `API key source: \`${env.keyVar}\``,
      "",
      "The LLM receives numbered ASR segments and returns two spoken-content boundaries:",
      "",
      "- `phase2_start_segment_index`: first segment where the assistant starts reporting the calendar result.",
      "- `phase3_start_segment_index`: first segment where the final departure-time answer begins.",
      "",
      "Derived phases:",
      "",
      "- Phase 1: beginning to before phase2 start.",
      "- Phase 2: phase2 start to before phase3 start.",
      "- Phase 3: phase3 start to end.",
      "",
      "This intentionally uses spoken content rather than tool-call timestamps.",
      "",
      "Generated files:",
      "",
      "- `phase_segmentation_attempts.csv`",
      "- `phase_segmentation_summary.csv`",
      "- `per_attempt/*.phase.json`",
      "- `cache/*_phase_segmentation.json`",
      "",
    ].join("\n"),
    "utf-8",
  );

  console.log("Phase segmentation outputs:");
  for (const path of ["phase_segmentation_attempts.csv", "phase_segmentation_summary.csv", "README.md"]) {
    console.log(`- ${rel(resolve(outputDir, path))}`);
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
