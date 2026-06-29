import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { config as loadEnv } from "dotenv";

const PROJECT_DIR = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const DEFAULT_PHASE_DIR = resolve(
  PROJECT_DIR,
  "result",
  "2026-06-29_10-15-29-961_calendar_route_two_step_2step_tick_vs_no_tick_latency_sweep(2tool大跑)",
  "organized",
  "judge_outputs",
  "calendar_route_phase_segmentation",
);
const DEFAULT_ENV_EXAMPLE = resolve(PROJECT_DIR, ".env.example");
const DEFAULT_MODEL = "gemma-4-31b-it";
const DEFAULT_REQUEST_TIMEOUT_MS = 45_000;
const OUTPUT_DIR_NAME = "calendar_route_phase_intent";

type Args = {
  phaseDir: string;
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

type PhaseRow = {
  condition: string;
  latency_ms: string;
  latency_s: string;
  attempt_index: string;
  attempt_id: string;
  ok: string;
  phase1_text: string;
  phase2_text: string;
  phase3_text: string;
  audio_path: string;
};

type IntentParsed = {
  phase1_calendar_intent_present?: number | null;
  phase1_intent_type?: string;
  phase1_evidence?: string;
  phase1_reason?: string;
  phase2_route_intent_present?: number | null;
  phase2_calendar_result_report_present?: number | null;
  phase2_intent_type?: string;
  phase2_route_evidence?: string;
  phase2_calendar_result_evidence?: string;
  phase2_reason?: string;
  generic_waiting_only?: number | null;
  notes?: string;
};

type JudgeResult = {
  ok: boolean;
  parsed?: IntentParsed;
  rawText: string;
  parseError?: string;
  attemptsUsed: number;
};

type OutputRow = {
  condition: string;
  latency_ms: string;
  latency_s: string;
  attempt_index: string;
  attempt_id: string;
  ok: string;
  phase1_calendar_intent_present: string;
  phase1_intent_type: string;
  phase1_evidence: string;
  phase1_reason: string;
  phase2_route_intent_present: string;
  phase2_calendar_result_report_present: string;
  phase2_intent_type: string;
  phase2_route_evidence: string;
  phase2_calendar_result_evidence: string;
  phase2_reason: string;
  generic_waiting_only: string;
  parse_error: string;
  notes: string;
  phase1_text: string;
  phase2_text: string;
  phase3_text: string;
  audio_path: string;
};

function parseArgs(argv: string[]): Args {
  const args: Args = {
    phaseDir: DEFAULT_PHASE_DIR,
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
    if (item === "--phase-dir" && next) {
      args.phaseDir = resolve(next);
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
      "Usage: npm run judge:calendar-route-phase-intent -- [options]",
      "",
      "Options:",
      "  --phase-dir <path>   calendar_route_phase_segmentation output folder",
      "  --model <name>       Gemma model name, default gemma-4-31b-it",
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

function parseCsv(text: string): Record<string, string>[] {
  const rows: string[][] = [];
  let current = "";
  let row: string[] = [];
  let inQuotes = false;
  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    const next = text[index + 1];
    if (inQuotes) {
      if (char === '"' && next === '"') {
        current += '"';
        index += 1;
      } else if (char === '"') {
        inQuotes = false;
      } else {
        current += char;
      }
      continue;
    }
    if (char === '"') inQuotes = true;
    else if (char === ",") {
      row.push(current);
      current = "";
    } else if (char === "\n") {
      row.push(current);
      rows.push(row);
      row = [];
      current = "";
    } else if (char !== "\r") {
      current += char;
    }
  }
  if (current.length > 0 || row.length > 0) {
    row.push(current);
    rows.push(row);
  }
  const [headers, ...body] = rows;
  if (!headers) return [];
  return body
    .filter((item) => item.some((cell) => cell.trim() !== ""))
    .map((cells) => Object.fromEntries(headers.map((header, index) => [header, cells[index] ?? ""])));
}

function readPhaseRows(phaseDir: string): PhaseRow[] {
  const path = resolve(phaseDir, "phase_segmentation_attempts.csv");
  if (!existsSync(path)) throw new Error(`Missing phase segmentation attempts CSV: ${path}`);
  return (parseCsv(readFileSync(path, "utf-8")) as unknown as PhaseRow[])
    .filter((row) => row.ok === "true")
    .sort((left, right) => {
      const condition = left.condition.localeCompare(right.condition);
      if (condition !== 0) return condition;
      const latency = Number(left.latency_ms) - Number(right.latency_ms);
      if (latency !== 0) return latency;
      return Number(left.attempt_index) - Number(right.attempt_index);
    });
}

function intentPrompt(row: PhaseRow): string {
  return [
    "You are judging spoken waiting behavior in a native two-tool benchmark.",
    "",
    "Task context:",
    "- User asks when to leave for the next meeting this afternoon, wanting to arrive about 10 minutes early.",
    "- Step/tool 1: calendar lookup for the next meeting time/location.",
    "- Step/tool 2: route/travel-time lookup to the meeting location.",
    "- Final answer: departure-time recommendation.",
    "",
    "You are given LLM-segmented spoken phases from ASR.",
    "",
    "Phase 1 should ideally mention what the assistant is doing for the first step: checking calendar / next event / next meeting / schedule.",
    "Phase 2 starts when the assistant reports the calendar result. It should ideally mention what the assistant is doing next: checking route / ETA / travel time / calculating when to leave.",
    "",
    "Judge only whether the assistant explicitly says the relevant next/current action somewhere in the phase text.",
    "Do not require exact wording.",
    "Do not give credit for generic waiting only, such as 'still waiting' or 'just a moment', unless it names the calendar/route/travel-time task.",
    "Do not judge final answer correctness.",
    "ASR may mishear NTU EE as NTUE.",
    "",
    "Phase 1 text:",
    row.phase1_text || "(empty)",
    "",
    "Phase 2 text:",
    row.phase2_text || "(empty)",
    "",
    "Return ONLY valid JSON with exactly these fields:",
    JSON.stringify(
      {
        phase1_calendar_intent_present: 1,
        phase1_intent_type: "calendar_lookup | combined_calendar_and_route | generic_waiting_only | none | other",
        phase1_evidence: "short quote or empty",
        phase1_reason: "brief reason",
        phase2_route_intent_present: 1,
        phase2_calendar_result_report_present: 1,
        phase2_intent_type: "route_lookup | travel_time_calculation | final_answer_only | generic_waiting_only | only_calendar_result | none | other",
        phase2_route_evidence: "short quote or empty",
        phase2_calendar_result_evidence: "short quote or empty",
        phase2_reason: "brief reason",
        generic_waiting_only: 0,
        notes: "brief notes",
      },
      null,
      2,
    ),
  ].join("\n");
}

function extractJson(text: string): IntentParsed {
  const trimmed = text.trim();
  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
  const candidate = fenced ? fenced[1].trim() : trimmed;
  try {
    return JSON.parse(candidate) as IntentParsed;
  } catch {
    const objects = findJsonObjectCandidates(candidate);
    for (const objectText of objects.reverse()) {
      try {
        return JSON.parse(objectText) as IntentParsed;
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

function buildOutput(row: PhaseRow, result: JudgeResult): OutputRow {
  const parsed = result.parsed ?? {};
  return {
    condition: row.condition,
    latency_ms: row.latency_ms,
    latency_s: row.latency_s,
    attempt_index: row.attempt_index,
    attempt_id: row.attempt_id,
    ok: String(result.ok),
    phase1_calendar_intent_present: numberField(parsed.phase1_calendar_intent_present),
    phase1_intent_type: stringField(parsed.phase1_intent_type),
    phase1_evidence: stringField(parsed.phase1_evidence),
    phase1_reason: stringField(parsed.phase1_reason),
    phase2_route_intent_present: numberField(parsed.phase2_route_intent_present),
    phase2_calendar_result_report_present: numberField(parsed.phase2_calendar_result_report_present),
    phase2_intent_type: stringField(parsed.phase2_intent_type),
    phase2_route_evidence: stringField(parsed.phase2_route_evidence),
    phase2_calendar_result_evidence: stringField(parsed.phase2_calendar_result_evidence),
    phase2_reason: stringField(parsed.phase2_reason),
    generic_waiting_only: numberField(parsed.generic_waiting_only),
    parse_error: result.ok ? "" : result.parseError || "parse_error",
    notes: stringField(parsed.notes),
    phase1_text: row.phase1_text,
    phase2_text: row.phase2_text,
    phase3_text: row.phase3_text,
    audio_path: row.audio_path,
  };
}

function summarize(rows: OutputRow[]): Record<string, unknown>[] {
  const groups = new Map<string, OutputRow[]>();
  for (const row of rows) {
    const key = `${row.condition}__${row.latency_ms}`;
    groups.set(key, [...(groups.get(key) ?? []), row]);
  }
  return [...groups.entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([key, items]) => {
      const [condition, latencyMs] = key.split("__");
      const okItems = items.filter((item) => item.ok === "true");
      return {
        condition,
        latency_ms: latencyMs,
        attempts: items.length,
        judged_ok_count: okItems.length,
        judged_ok_rate: items.length ? fmt(okItems.length / items.length) : "",
        phase1_calendar_intent_rate: meanBinary(okItems, "phase1_calendar_intent_present"),
        phase2_route_intent_rate: meanBinary(okItems, "phase2_route_intent_present"),
        phase2_calendar_result_report_rate: meanBinary(okItems, "phase2_calendar_result_report_present"),
        generic_waiting_only_rate: meanBinary(okItems, "generic_waiting_only"),
        parse_error_count: items.filter((item) => item.parse_error).length,
      };
    });
}

function meanBinary(rows: OutputRow[], key: keyof OutputRow): string {
  const values: number[] = rows.map((row) => Number(row[key])).filter((value) => value === 0 || value === 1);
  if (values.length === 0) return "";
  return fmt(values.reduce((sum, value) => sum + value, 0) / values.length);
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

function numberField(value: unknown): string {
  if (typeof value === "number" && Number.isFinite(value)) return String(value);
  if (typeof value === "string" && value.trim() !== "" && Number.isFinite(Number(value))) return String(Number(value));
  return "";
}

function stringField(value: unknown): string {
  if (value === undefined || value === null) return "";
  return String(value);
}

function fmt(value: number): string {
  return value.toFixed(3);
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
  const phaseDir = resolve(args.phaseDir);
  const outputDir = resolve(phaseDir, "..", OUTPUT_DIR_NAME);
  const cacheDir = resolve(outputDir, "cache");
  const perAttemptDir = resolve(outputDir, "per_attempt");
  mkdirSync(cacheDir, { recursive: true });
  mkdirSync(perAttemptDir, { recursive: true });

  const allRows = readPhaseRows(phaseDir);
  const selectedRows = args.limit ? allRows.slice(0, args.limit) : allRows;
  console.log(`Phase intent inputs: ${selectedRows.length}`);
  console.log(`Model: ${env.model}`);
  console.log(`API key source: ${env.keyVar}`);
  console.log(`Output: ${rel(outputDir)}`);

  const outputByKey = new Map<string, OutputRow>();
  let processed = 0;
  let cursor = 0;

  async function processRow(row: PhaseRow): Promise<void> {
    const cachePath = resolve(
      cacheDir,
      `${cleanId(row.condition)}_${row.latency_ms}_${cleanId(row.attempt_id)}_phase_intent.json`,
    );
    const result = await judgeJson(
      env.apiKey,
      env.model,
      intentPrompt(row),
      cachePath,
      args.force,
      args.retries,
      args.requestTimeoutMs,
      args.retryErrors,
    );
    const output = buildOutput(row, result);
    outputByKey.set(`${row.condition}__${row.latency_ms}__${row.attempt_index}`, output);
    writeFileSync(
      resolve(perAttemptDir, `${cleanId(row.condition)}_${row.latency_ms}_${cleanId(row.attempt_id)}.phase_intent.json`),
      `${JSON.stringify({ input: row, judge: result, derived: output }, null, 2)}\n`,
      "utf-8",
    );
    processed += 1;
    if (processed % 10 === 0 || !result.ok || processed === selectedRows.length) {
      console.log(
        `Progress ${processed}/${selectedRows.length}; current=${row.condition}/${row.latency_ms}/${row.attempt_id}; ok=${output.ok}; p1=${output.phase1_calendar_intent_present}; p2=${output.phase2_route_intent_present}`,
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

  const outputs: OutputRow[] = [];
  for (const row of selectedRows) {
    const output = outputByKey.get(`${row.condition}__${row.latency_ms}__${row.attempt_index}`);
    if (output) outputs.push(output);
  }

  writeCsv(resolve(outputDir, "phase_intent_attempts.csv"), outputs, [
    "condition",
    "latency_ms",
    "latency_s",
    "attempt_index",
    "attempt_id",
    "ok",
    "phase1_calendar_intent_present",
    "phase1_intent_type",
    "phase1_evidence",
    "phase1_reason",
    "phase2_route_intent_present",
    "phase2_calendar_result_report_present",
    "phase2_intent_type",
    "phase2_route_evidence",
    "phase2_calendar_result_evidence",
    "phase2_reason",
    "generic_waiting_only",
    "parse_error",
    "notes",
    "phase1_text",
    "phase2_text",
    "phase3_text",
    "audio_path",
  ]);

  const summaryRows = summarize(outputs);
  writeCsv(resolve(outputDir, "phase_intent_summary.csv"), summaryRows, [
    "condition",
    "latency_ms",
    "attempts",
    "judged_ok_count",
    "judged_ok_rate",
    "phase1_calendar_intent_rate",
    "phase2_route_intent_rate",
    "phase2_calendar_result_report_rate",
    "generic_waiting_only_rate",
    "parse_error_count",
  ]);

  writeFileSync(
    resolve(outputDir, "README.md"),
    [
      "# Calendar Route Phase Intent Judge",
      "",
      `Phase segmentation source: \`${rel(phaseDir)}\``,
      `Model: \`${env.model}\``,
      `API key source: \`${env.keyVar}\``,
      "",
      "This judge reads Phase 1 and Phase 2 text produced by `calendar_route_phase_segmentation`.",
      "",
      "It asks whether:",
      "",
      "- Phase 1 explicitly says the assistant is doing the first step: calendar / schedule / next meeting lookup.",
      "- Phase 2 explicitly says the assistant is doing the second step: route / ETA / travel-time / departure-time calculation.",
      "- Phase 2 reports the calendar result, which is useful bridge behavior but not sufficient by itself for route intent.",
      "",
      "Generic waiting alone does not count as task intent.",
      "",
      "Generated files:",
      "",
      "- `phase_intent_attempts.csv`",
      "- `phase_intent_summary.csv`",
      "- `per_attempt/*.phase_intent.json`",
      "- `cache/*_phase_intent.json`",
      "",
    ].join("\n"),
    "utf-8",
  );

  console.log("Phase intent outputs:");
  for (const path of ["phase_intent_attempts.csv", "phase_intent_summary.csv", "README.md"]) {
    console.log(`- ${rel(resolve(outputDir, path))}`);
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
