import { Behavior, GoogleGenAI, Modality } from "@google/genai";
import { config as loadEnv } from "dotenv";
import { appendFileSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const PROJECT_DIR = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const DEFAULT_ENV_FILE = resolve(PROJECT_DIR, "..", "angus_bench", ".env");
const RESULT_DIR = process.env.GEMINI_LIVE_CHECK_RESULT_DIR
  ? resolve(process.env.GEMINI_LIVE_CHECK_RESULT_DIR)
  : resolve(PROJECT_DIR, "result");

const USER_PROMPT =
  "I'm Anya Garcia, user id `anya_garcia_5901`. For reservation `JMO1MG`, how many total checked suitcases can I take? I think I'm a member, and I need the answer as a number with a brief explanation.";

const SYSTEM_INSTRUCTION = [
  "You are an airline customer support assistant.",
  "Use the provided tools when reservation or user profile details are needed.",
  "For this probe, the baggage rule is: economy + silver membership = 2 checked suitcases per passenger.",
  "Do not guess reservation details, membership, or the final suitcase total before the needed tool results are available.",
  "Answer briefly after you have enough tool results.",
].join("\n");

const RESERVATION_TOOL = "get_reservation_details";
const USER_TOOL = "get_user_details";
const DEFAULT_ATTEMPTS = 20;
const DEFAULT_TOOL_LATENCY_MS = 500;
const MAX_ATTEMPT_MS = 35_000;
const POST_FINAL_WAIT_MS = 6000;

const RESERVATION_RESULT = {
  reservation_id: "JMO1MG",
  cabin: "economy",
  passenger_count: 2,
  route: "DEN -> MIA",
  trip_type: "one_way",
  total_baggages: 1,
};

const USER_RESULT = {
  user_id: "anya_garcia_5901",
  name: "Anya Garcia",
  membership: "silver",
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
  };
  toolCall?: {
    functionCalls?: FunctionCall[];
  };
  toolCallCancellation?: {
    ids?: string[];
  };
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
  attempts: number;
  toolLatencyMs: number;
  quietTerminalText: boolean;
};

type ToolCallRecord = {
  event_ms: number | null;
  name?: string;
  args?: Record<string, unknown>;
};

type ToolResponseRecord = {
  event_ms: number | null;
  name?: string;
  response: Record<string, unknown>;
  send_error?: string;
};

type AttemptSummary = {
  attempt_index: number;
  session_valid: boolean;
  called_any_tool: boolean;
  called_get_reservation_details: boolean;
  get_reservation_details_args_correct: boolean;
  called_get_user_details: boolean;
  get_user_details_args_correct: boolean;
  both_tools_called: boolean;
  both_tool_args_correct: boolean;
  tool_order: string;
  final_answer_mentions_4: boolean;
  final_uses_reservation_result: boolean;
  final_uses_user_membership_result: boolean;
  premature_answer_before_tools: boolean;
  completed_two_tool_flow: boolean;
  close_1008: boolean;
  close_1011: boolean;
  close_1006: boolean;
  close_code: number | null;
  close_reason: string | null;
  client_send_error_count: number;
  raw_event_count: number;
  text_before_both_tools: string[];
  text_after_both_tools: string[];
  tool_calls: ToolCallRecord[];
  tool_responses: ToolResponseRecord[];
  send_errors: string[];
  errors: string[];
  result_dir: string;
};

type AggregateSummary = {
  attempts: number;
  any_tool_call_rate: number;
  two_tool_call_success_rate: number;
  reservation_tool_success_rate: number;
  user_tool_success_rate: number;
  both_tools_called_rate: number;
  both_tool_args_correct_rate: number;
  final_answer_correct_rate: number;
  completed_two_tool_flow_rate: number;
  premature_answer_rate: number;
  close_1008_count: number;
  close_1011_count: number;
  close_1006_count: number;
  send_error_count: number;
};

type AttemptState = {
  sessionOpenedAt?: number;
  setupCompleteAt?: number;
  promptSentAt?: number;
  bothToolResponsesAt?: number;
  sessionClosed: boolean;
  closeCode: number | null;
  closeReason: string | null;
  rawEventCount: number;
  toolCalls: ToolCallRecord[];
  toolResponses: ToolResponseRecord[];
  textBeforeBothTools: string[];
  textAfterBothTools: string[];
  cancelledToolCallIds: string[];
  sendErrors: string[];
  errors: string[];
};

loadEnv({ path: process.env.GEMINI_ENV_FILE || DEFAULT_ENV_FILE });

function usage(): string {
  return [
    "Usage: tau-two-tool-feasibility-probe [options]",
    "",
    "Options:",
    "  --attempts <n>          Number of attempts. Default: 20.",
    "  --tool-latency-ms <n>   Delay before each tool response. Default: 500.",
    "  --quiet-terminal-text   Suppress transcript text in terminal.",
    "  --help                  Show this help.",
  ].join("\n");
}

function parseArgs(argv: string[]): Args {
  const args: Args = {
    attempts: Number(process.env.TWO_TOOL_ATTEMPTS ?? DEFAULT_ATTEMPTS),
    toolLatencyMs: Number(process.env.TWO_TOOL_LATENCY_MS ?? DEFAULT_TOOL_LATENCY_MS),
    quietTerminalText: false,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") {
      console.log(usage());
      process.exit(0);
    }
    if (arg === "--quiet-terminal-text") {
      args.quietTerminalText = true;
      continue;
    }
    if (arg === "--attempts" || arg === "--tool-latency-ms") {
      const value = argv[index + 1];
      if (!value) throw new Error(`${arg} requires a value`);
      if (arg === "--attempts") args.attempts = Number(value);
      else args.toolLatencyMs = Number(value);
      index += 1;
      continue;
    }
    const inline = arg.match(/^--(attempts|tool-latency-ms)=(\d+)$/);
    if (inline) {
      if (inline[1] === "attempts") args.attempts = Number(inline[2]);
      else args.toolLatencyMs = Number(inline[2]);
      continue;
    }
    throw new Error(`Unknown argument: ${arg}\n${usage()}`);
  }
  if (!Number.isInteger(args.attempts) || args.attempts < 1) throw new Error(`Invalid --attempts: ${args.attempts}`);
  if (!Number.isInteger(args.toolLatencyMs) || args.toolLatencyMs < 0) {
    throw new Error(`Invalid --tool-latency-ms: ${args.toolLatencyMs}`);
  }
  return args;
}

function requireEnv(name: string): string {
  const value = process.env[name]?.trim();
  if (!value) throw new Error(`Missing ${name}. Set it in your shell or .env.`);
  return value;
}

function timestampForPath(date = new Date()): string {
  const [datePart, timePart] = date.toISOString().split("T");
  return `${datePart}_${timePart.replace("Z", "").replace(/\./g, "-").replace(/:/g, "-")}`;
}

function appendJsonl(path: string, value: unknown): void {
  appendFileSync(path, `${JSON.stringify(value)}\n`, "utf8");
}

function writeJson(path: string, value: unknown): void {
  writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

function csvEscape(value: unknown): string {
  if (value === null || value === undefined) return "";
  if (Array.isArray(value) || typeof value === "object") return csvEscape(JSON.stringify(value));
  const text = String(value);
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

function rate(count: number, total: number): number {
  return total ? Math.round((count / total) * 1000) / 1000 : 0;
}

function summarizeError(error: unknown): string {
  if (error instanceof Error) return `${error.name}: ${error.message}`;
  if (typeof error !== "object" || error === null) return String(error);
  const record = error as Record<string, unknown>;
  const fields = ["message", "type", "code", "reason", "name", "error"]
    .map((key) => (record[key] ? `${key}: ${String(record[key])}` : undefined))
    .filter(Boolean);
  return fields.length ? fields.join(", ") : Object.prototype.toString.call(error);
}

function eventTypes(message: LiveMessage): string[] {
  return [
    message.setupComplete ? "setupComplete" : undefined,
    message.serverContent ? "serverContent" : undefined,
    message.toolCall ? "toolCall" : undefined,
    message.toolCallCancellation ? "toolCallCancellation" : undefined,
  ].filter((value): value is string => Boolean(value));
}

function sanitizeMessage(message: LiveMessage): LiveMessage {
  return {
    ...message,
    serverContent: message.serverContent
      ? {
          ...message.serverContent,
          modelTurn: message.serverContent.modelTurn
            ? {
                parts: message.serverContent.modelTurn.parts?.map((part) =>
                  part.inlineData?.data
                    ? { ...part, inlineData: { ...part.inlineData, data: `<base64 ${part.inlineData.data.length} chars>` } }
                    : part,
                ),
              }
            : undefined,
        }
      : undefined,
  };
}

function makeTools(): unknown[] {
  return [
    {
      functionDeclarations: [
        {
          name: RESERVATION_TOOL,
          description: "Get the cabin, passenger count, route, trip type, and baggage information for an airline reservation.",
          behavior: Behavior.NON_BLOCKING,
          parametersJsonSchema: {
            type: "object",
            properties: {
              reservation_id: { type: "string", description: "Reservation id, such as JMO1MG." },
            },
            required: ["reservation_id"],
          },
        },
        {
          name: USER_TOOL,
          description: "Get the user's profile details, including airline membership tier.",
          behavior: Behavior.NON_BLOCKING,
          parametersJsonSchema: {
            type: "object",
            properties: {
              user_id: { type: "string", description: "User id, such as anya_garcia_5901." },
            },
            required: ["user_id"],
          },
        },
      ],
    },
  ];
}

function argsCorrect(call: ToolCallRecord): boolean {
  if (call.name === RESERVATION_TOOL) return call.args?.reservation_id === "JMO1MG";
  if (call.name === USER_TOOL) return call.args?.user_id === "anya_garcia_5901";
  return false;
}

function finalMentions4(text: string): boolean {
  return /(^|[^0-9])4([^0-9]|$)|four/i.test(text);
}

function usesReservationResult(text: string): boolean {
  return /economy|2 passenger|two passenger|passenger count|reservation|JMO1MG|DEN|MIA/i.test(text);
}

function usesMembershipResult(text: string): boolean {
  return /silver|member|membership/i.test(text);
}

function prematureAnswer(text: string): boolean {
  return finalMentions4(text) && /checked suitcase|checked bag|suitcase|bag|total/i.test(text);
}

function makeSummary(attemptIndex: number, attemptDir: string, state: AttemptState): AttemptSummary {
  const calls = state.toolCalls;
  const reservationCalls = calls.filter((call) => call.name === RESERVATION_TOOL);
  const userCalls = calls.filter((call) => call.name === USER_TOOL);
  const calledReservation = reservationCalls.length > 0;
  const calledUser = userCalls.length > 0;
  const reservationArgsOk = reservationCalls.some(argsCorrect);
  const userArgsOk = userCalls.some(argsCorrect);
  const bothToolsCalled = calledReservation && calledUser;
  const bothToolArgsCorrect = reservationArgsOk && userArgsOk;
  const afterText = state.textAfterBothTools.join(" ");
  const beforeText = state.textBeforeBothTools.join(" ");
  const finalAnswerCorrect = finalMentions4(afterText) && usesReservationResult(afterText) && usesMembershipResult(afterText);
  const close1008 = state.closeCode === 1008 || state.closeReason?.includes("1008") === true;
  const close1011 = state.closeCode === 1011 || state.closeReason?.includes("1011") === true;
  const close1006 = state.closeCode === 1006 || state.closeReason?.includes("1006") === true;
  return {
    attempt_index: attemptIndex,
    session_valid: !state.closeCode && state.errors.length === 0 && state.sendErrors.length === 0,
    called_any_tool: calls.length > 0,
    called_get_reservation_details: calledReservation,
    get_reservation_details_args_correct: reservationArgsOk,
    called_get_user_details: calledUser,
    get_user_details_args_correct: userArgsOk,
    both_tools_called: bothToolsCalled,
    both_tool_args_correct: bothToolArgsCorrect,
    tool_order: calls.map((call) => call.name || "unknown").join(" -> "),
    final_answer_mentions_4: finalMentions4(afterText),
    final_uses_reservation_result: usesReservationResult(afterText),
    final_uses_user_membership_result: usesMembershipResult(afterText),
    premature_answer_before_tools: prematureAnswer(beforeText),
    completed_two_tool_flow: bothToolsCalled,
    close_1008: close1008,
    close_1011: close1011,
    close_1006: close1006,
    close_code: state.closeCode,
    close_reason: state.closeReason,
    client_send_error_count: state.sendErrors.length,
    raw_event_count: state.rawEventCount,
    text_before_both_tools: state.textBeforeBothTools,
    text_after_both_tools: state.textAfterBothTools,
    tool_calls: state.toolCalls,
    tool_responses: state.toolResponses,
    send_errors: state.sendErrors,
    errors: state.errors,
    result_dir: attemptDir,
  };
}

function maybeMarkBothToolsComplete(state: AttemptState): void {
  const respondedReservation = state.toolResponses.some((response) => response.name === RESERVATION_TOOL && !response.send_error);
  const respondedUser = state.toolResponses.some((response) => response.name === USER_TOOL && !response.send_error);
  if (respondedReservation && respondedUser) state.bothToolResponsesAt ??= Date.now();
}

async function runOne(ai: GoogleGenAI, model: string, attemptIndex: number, attemptDir: string, args: Args): Promise<AttemptSummary> {
  mkdirSync(attemptDir, { recursive: true });
  const rawLogPath = resolve(attemptDir, "raw_log.jsonl");
  const timelinePath = resolve(attemptDir, "events.jsonl");
  writeJson(resolve(attemptDir, "config.json"), {
    attempt_index: attemptIndex,
    model,
    tool_latency_ms: args.toolLatencyMs,
    user_prompt: USER_PROMPT,
    system_instruction: SYSTEM_INSTRUCTION,
    tools: [RESERVATION_TOOL, USER_TOOL],
    expected_final_answer: "4 checked suitcases total",
  });

  const state: AttemptState = {
    sessionClosed: false,
    closeCode: null,
    closeReason: null,
    rawEventCount: 0,
    toolCalls: [],
    toolResponses: [],
    textBeforeBothTools: [],
    textAfterBothTools: [],
    cancelledToolCallIds: [],
    sendErrors: [],
    errors: [],
  };

  let session: Session | undefined;
  let done = false;
  let initialPromptSent = false;
  let finalObservationScheduled = false;
  const timers: ReturnType<typeof setTimeout>[] = [];

  const eventMs = () => (state.promptSentAt ? Date.now() - state.promptSentAt : null);
  const appendTimeline = (type: string, extra: Record<string, unknown> = {}) => appendJsonl(timelinePath, { type, event_ms: eventMs(), ...extra });

  const noteSendError = (type: string, error: unknown) => {
    const summary = summarizeError(error);
    state.sendErrors.push(`${type}: ${summary}`);
    appendTimeline("send_error", { send_type: type, error: summary });
    appendJsonl(rawLogPath, { type: "send_error", event_ms: eventMs(), send_type: type, error: summary });
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
    resolveRun();
  };

  const sendInitialUserPrompt = () => {
    if (initialPromptSent || done || state.sessionClosed || !session) return;
    initialPromptSent = true;
    session.sendClientContent({ turns: USER_PROMPT, turnComplete: true });
    state.promptSentAt = Date.now();
    appendTimeline("user_message_sent", { prompt: USER_PROMPT });
    appendJsonl(rawLogPath, { type: "user_message_sent", prompt: USER_PROMPT });
  };

  const scheduleFinalObservation = (resolveRun: () => void) => {
    if (finalObservationScheduled) return;
    finalObservationScheduled = true;
    appendTimeline("post_two_tool_observation_started", { wait_ms: POST_FINAL_WAIT_MS });
    timers.push(
      setTimeout(() => {
        appendTimeline("post_two_tool_observation_elapsed", { wait_ms: POST_FINAL_WAIT_MS });
        finish(resolveRun);
      }, POST_FINAL_WAIT_MS),
    );
  };

  const responseForCall = (call: FunctionCall): Record<string, unknown> => {
    if (call.name === RESERVATION_TOOL) return RESERVATION_RESULT;
    if (call.name === USER_TOOL) return USER_RESULT;
    return { error: `unknown tool ${call.name || "unknown"}` };
  };

  const sendToolResponse = (call: FunctionCall): boolean => {
    const response = responseForCall(call);
    const record: ToolResponseRecord = { event_ms: eventMs(), name: call.name, response };
    try {
      session?.sendToolResponse({
        functionResponses: [{ id: call.id, name: call.name, response }],
      });
      appendTimeline("tool_response_sent", { function_call_id: call.id, function_name: call.name, response });
      appendJsonl(rawLogPath, {
        type: "tool_response_sent",
        event_ms: eventMs(),
        function_call_id: call.id,
        function_name: call.name,
        response,
      });
    } catch (error) {
      record.send_error = noteSendError("tool_response", error);
    }
    state.toolResponses.push(record);
    maybeMarkBothToolsComplete(state);
    return !record.send_error;
  };

  await new Promise<void>(async (resolveRun) => {
    timers.push(
      setTimeout(() => {
        state.errors.push(`max attempt timeout ${MAX_ATTEMPT_MS}ms`);
        appendTimeline("max_attempt_timeout", { max_attempt_ms: MAX_ATTEMPT_MS });
        finish(resolveRun);
      }, MAX_ATTEMPT_MS),
    );

    try {
      session = (await ai.live.connect({
        model,
        config: {
          responseModalities: [Modality.AUDIO],
          outputAudioTranscription: {},
          systemInstruction: SYSTEM_INSTRUCTION,
          tools: makeTools() as any,
        },
        callbacks: {
          onopen: () => {
            state.sessionOpenedAt = Date.now();
            appendTimeline("session_opened", { model });
            appendJsonl(rawLogPath, { type: "session_opened", model });
          },
          onmessage: (message: LiveMessage) => {
            state.rawEventCount += 1;
            appendJsonl(rawLogPath, { type: "server_event", event_ms: eventMs(), event_types: eventTypes(message), message: sanitizeMessage(message) });
            appendTimeline("server_event", { event_types: eventTypes(message) });

            if (message.setupComplete) {
              state.setupCompleteAt ??= Date.now();
              appendTimeline("setup_complete");
              try {
                sendInitialUserPrompt();
              } catch (error) {
                noteSendError("initial_user_prompt", error);
                finish(resolveRun);
              }
              return;
            }

            const parts = message.serverContent?.modelTurn?.parts ?? [];
            const textPieces = [
              ...parts.map((part) => part.text).filter((text): text is string => Boolean(text)),
              message.serverContent?.outputTranscription?.text,
            ].filter((text): text is string => Boolean(text));

            for (const text of textPieces) {
              const phase = state.bothToolResponsesAt ? "after_both_tools" : "before_both_tools";
              if (!args.quietTerminalText) console.log(`[attempt ${attemptIndex} ${eventMs() ?? "?"}ms ${phase}] ${text.replace(/\s+/g, " ").trim()}`);
              if (state.bothToolResponsesAt) state.textAfterBothTools.push(text);
              else state.textBeforeBothTools.push(text);
              appendTimeline("text_output", { phase, text });
            }

            if (message.toolCall?.functionCalls?.length) {
              appendTimeline("tool_call_received", { function_calls: message.toolCall.functionCalls });
              appendJsonl(rawLogPath, { type: "tool_call_received", event_ms: eventMs(), function_calls: message.toolCall.functionCalls });
              for (const call of message.toolCall.functionCalls) {
                state.toolCalls.push({ event_ms: eventMs(), name: call.name, args: call.args });
                timers.push(
                  setTimeout(() => {
                    if (done || state.sessionClosed) return;
                    sendToolResponse(call);
                    if (state.bothToolResponsesAt) scheduleFinalObservation(resolveRun);
                  }, args.toolLatencyMs),
                );
              }
            }

            if (message.toolCallCancellation?.ids?.length) {
              for (const id of message.toolCallCancellation.ids) {
                if (!state.cancelledToolCallIds.includes(id)) state.cancelledToolCallIds.push(id);
              }
              appendTimeline("tool_call_cancellation_received", { ids: message.toolCallCancellation.ids });
              appendJsonl(rawLogPath, { type: "tool_call_cancellation_received", event_ms: eventMs(), ids: message.toolCallCancellation.ids });
            }

            if (message.serverContent?.turnComplete) appendTimeline("turn_complete");
          },
          onerror: (error) => {
            const summary = summarizeError(error);
            state.errors.push(summary);
            appendTimeline("socket_error", { error: summary });
            appendJsonl(rawLogPath, { type: "socket_error", event_ms: eventMs(), error: summary });
          },
          onclose: (event: { code?: number; reason?: string }) => {
            state.sessionClosed = true;
            state.closeCode = event.code ?? null;
            state.closeReason = event.reason || null;
            appendTimeline("session_closed", { code: event.code, reason: event.reason });
            appendJsonl(rawLogPath, { type: "session_closed", event_ms: eventMs(), code: event.code, reason: event.reason });
            finish(resolveRun);
          },
        },
      })) as Session;
    } catch (error) {
      state.errors.push(summarizeError(error));
      appendTimeline("connect_error", { error: summarizeError(error) });
      finish(resolveRun);
    }
  });

  const summary = makeSummary(attemptIndex, attemptDir, state);
  writeJson(resolve(attemptDir, "attempt_summary.json"), summary);
  return summary;
}

function aggregate(attempts: AttemptSummary[]): AggregateSummary {
  return {
    attempts: attempts.length,
    any_tool_call_rate: rate(attempts.filter((attempt) => attempt.called_any_tool).length, attempts.length),
    two_tool_call_success_rate: rate(attempts.filter((attempt) => attempt.both_tools_called).length, attempts.length),
    reservation_tool_success_rate: rate(attempts.filter((attempt) => attempt.get_reservation_details_args_correct).length, attempts.length),
    user_tool_success_rate: rate(attempts.filter((attempt) => attempt.get_user_details_args_correct).length, attempts.length),
    both_tools_called_rate: rate(attempts.filter((attempt) => attempt.both_tools_called).length, attempts.length),
    both_tool_args_correct_rate: rate(attempts.filter((attempt) => attempt.both_tool_args_correct).length, attempts.length),
    final_answer_correct_rate: rate(
      attempts.filter(
        (attempt) =>
          attempt.final_answer_mentions_4 &&
          attempt.final_uses_reservation_result &&
          attempt.final_uses_user_membership_result,
      ).length,
      attempts.length,
    ),
    completed_two_tool_flow_rate: rate(attempts.filter((attempt) => attempt.completed_two_tool_flow).length, attempts.length),
    premature_answer_rate: rate(attempts.filter((attempt) => attempt.premature_answer_before_tools).length, attempts.length),
    close_1008_count: attempts.filter((attempt) => attempt.close_1008).length,
    close_1011_count: attempts.filter((attempt) => attempt.close_1011).length,
    close_1006_count: attempts.filter((attempt) => attempt.close_1006).length,
    send_error_count: attempts.reduce((sum, attempt) => sum + attempt.client_send_error_count, 0),
  };
}

function writeReadme(path: string, resultDir: string, organizedDir: string, args: Args, aggregateSummary: AggregateSummary): void {
  const lines = [
    "# Native two-tool feasibility probe",
    "",
    "Task: Airline Suitcase Allowance Check",
    "",
    `Attempts: ${args.attempts}`,
    `Tool result latency: ${args.toolLatencyMs} ms`,
    `Result folder: ${resultDir}`,
    `Organized folder: ${organizedDir}`,
    "",
    "## Aggregate",
    "",
    "```json",
    JSON.stringify(aggregateSummary, null, 2),
    "```",
    "",
  ];
  writeFileSync(path, `${lines.join("\n")}\n`, "utf8");
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const apiKey = requireEnv("GEMINI_API_KEY");
  const model = requireEnv("GEMINI_LIVE_MODEL");
  const ai = new GoogleGenAI({ apiKey });

  const resultDir = resolve(RESULT_DIR, `${timestampForPath()}_two_tool_feasibility_probe`);
  const attemptsDir = resolve(resultDir, "attempts");
  const organizedDir = resolve(resultDir, "organized");
  mkdirSync(attemptsDir, { recursive: true });
  mkdirSync(organizedDir, { recursive: true });

  console.log(`Running native two-tool feasibility probe for model: ${model}`);
  console.log(`Attempts: ${args.attempts}`);
  console.log(`Tool latency: ${args.toolLatencyMs} ms`);
  console.log(`Result directory: ${relative(PROJECT_DIR, resultDir)}`);

  const attempts: AttemptSummary[] = [];
  for (let index = 1; index <= args.attempts; index += 1) {
    const attemptDir = resolve(attemptsDir, `attempt_${String(index).padStart(4, "0")}`);
    console.log(`[attempt ${index}/${args.attempts}] start`);
    const summary = await runOne(ai, model, index, attemptDir, args);
    attempts.push(summary);
    console.log(
      `[attempt ${index}/${args.attempts}] both_tools=${summary.both_tools_called} args_ok=${summary.both_tool_args_correct} final_4=${summary.final_answer_mentions_4} complete=${summary.completed_two_tool_flow} close=${summary.close_code ?? "none"}`,
    );
  }

  const aggregateSummary = aggregate(attempts);
  const summaryJson = {
    task: "airline_suitcase_allowance_check",
    user_prompt: USER_PROMPT,
    tool_latency_ms: args.toolLatencyMs,
    model,
    aggregate: aggregateSummary,
    attempts,
  };

  writeJson(resolve(resultDir, "summary.json"), summaryJson);
  writeCsv(resolve(resultDir, "summary.csv"), attempts as unknown as Array<Record<string, unknown>>);
  writeJson(resolve(organizedDir, "summary.json"), summaryJson);
  writeCsv(resolve(organizedDir, "summary.csv"), attempts as unknown as Array<Record<string, unknown>>);
  writeReadme(resolve(resultDir, "README.md"), resultDir, organizedDir, args, aggregateSummary);

  console.log("Summary:");
  console.log(JSON.stringify(aggregateSummary, null, 2));
  console.log(`Organized directory: ${relative(PROJECT_DIR, organizedDir)}`);
}

main().catch((error) => {
  console.error("tau-two-tool-feasibility-probe failed");
  console.error(summarizeError(error));
  process.exitCode = 1;
});
