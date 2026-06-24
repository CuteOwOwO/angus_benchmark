import { GoogleGenAI, Modality } from "@google/genai";
import { config as loadEnv } from "dotenv";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const PROMPT = "Say hello in one short sentence.";
const RESPONSE_TIMEOUT_MS = 20_000;
const PROJECT_DIR = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const DEFAULT_ENV_FILE = resolve(PROJECT_DIR, "..", "angus_bench", ".env");
const RESPONSE_MODALITY = Modality.AUDIO;

loadEnv({ path: process.env.GEMINI_ENV_FILE || DEFAULT_ENV_FILE });

type LiveMessage = {
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
  };
  setupComplete?: unknown;
  toolCall?: unknown;
};

class LiveCheckError extends Error {
  constructor(
    message: string,
    readonly cause?: unknown,
  ) {
    super(message);
    this.name = "LiveCheckError";
  }
}

function requireEnv(name: string): string {
  const value = process.env[name]?.trim();
  if (!value) {
    throw new LiveCheckError(`Missing ${name}. Set it in your shell or in .env, then run: npm run check`);
  }
  return value;
}

function describeError(error: unknown): string {
  const text = summarizeError(error).toLowerCase();

  if (/missing gemini_api_key|missing gemini_live_model/.test(text)) {
    return "Missing required environment variable.";
  }

  if (/api[_ -]?key|unauthenticated|authentication|401/.test(text)) {
    return "API key authentication failed. Check GEMINI_API_KEY.";
  }

  if (/permission|forbidden|access|403/.test(text)) {
    return "Permission error. The API key may not have access to this Live API model or project.";
  }

  if (/quota|rate|resource exhausted|too many requests|429/.test(text)) {
    return "Quota or rate limit error. Check your Gemini API quota, billing, and retry later.";
  }

  if (/network|fetch failed|websocket|socket|econn|enotfound|etimedout|timeout|dns/.test(text)) {
    return "Network error. Check internet access, firewall/proxy settings, and retry.";
  }

  if (/model not found|models\/|404|invalid argument|400|unsupported.*model|model.*unavailable/.test(text)) {
    return "Invalid or unavailable model name. Check GEMINI_LIVE_MODEL and make sure it is a Live API model ID.";
  }

  return "Unexpected Live API error. See details below.";
}

function summarizeError(error: unknown): string {
  if (error instanceof Error) {
    const cause = error.cause ? ` Cause: ${summarizeError(error.cause)}` : "";
    return `${error.name}: ${error.message}${cause}`;
  }

  if (typeof error !== "object" || error === null) {
    return String(error);
  }

  const record = error as Record<string, unknown>;
  const fields = ["message", "type", "code", "reason", "name", "error"]
    .map((key) => {
      const value = record[key];
      return value ? `${key}: ${String(value)}` : undefined;
    })
    .filter(Boolean);

  return fields.length > 0 ? fields.join(", ") : Object.prototype.toString.call(error);
}

function compactMessage(message: LiveMessage): string {
  const content = message.serverContent;
  const textParts = content?.modelTurn?.parts
    ?.map((part) => part.text)
    .filter(Boolean);

  return JSON.stringify(
    {
      setupComplete: Boolean(message.setupComplete),
      inputTranscription: content?.inputTranscription?.text,
      outputTranscription: content?.outputTranscription?.text,
      text: textParts?.join(""),
      audioParts: content?.modelTurn?.parts?.filter((part) => part.inlineData).length,
      turnComplete: content?.turnComplete,
      toolCall: Boolean(message.toolCall),
    },
    null,
    2,
  );
}

async function main(): Promise<void> {
  const apiKey = requireEnv("GEMINI_API_KEY");
  const model = requireEnv("GEMINI_LIVE_MODEL");

  const ai = new GoogleGenAI({ apiKey });

  console.log(`Checking Live API access for model: ${model}`);

  let session: { sendRealtimeInput(input: { text: string }): void; close(): void } | undefined;

  await new Promise<void>(async (resolve, reject) => {
    let settled = false;
    let lastSocketError: unknown;

    const finish = (error?: unknown) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      try {
        session?.close();
      } catch {
        // The SDK may already have closed the socket after an error.
      }
      error ? reject(error) : resolve();
    };

    const timer = setTimeout(() => {
      finish(
        new LiveCheckError(
          `Timed out after ${RESPONSE_TIMEOUT_MS / 1000}s waiting for a Live API response event.`,
          lastSocketError,
        ),
      );
    }, RESPONSE_TIMEOUT_MS);

    try {
      // 1. Open a stateful Live API WebSocket session for the requested model.
      // 2. Ask for audio output, which native-audio Live models are designed to return.
      // 3. Send one tiny text input through the real-time input channel.
      // 4. Treat the first model output or completed turn as proof the session works.
      session = await ai.live.connect({
        model,
        config: {
          responseModalities: [RESPONSE_MODALITY],
          outputAudioTranscription: {},
        },
        callbacks: {
          onopen: () => {
            console.log("Live API session opened");
          },
          onmessage: (message: LiveMessage) => {
            console.log("Live API event:");
            console.log(compactMessage(message));

            const content = message.serverContent;
            const hasText = content?.modelTurn?.parts?.some((part) => part.text);
            const hasAudio = content?.modelTurn?.parts?.some((part) => part.inlineData);
            const hasTranscription = Boolean(content?.outputTranscription?.text);

            if (hasText || hasAudio || hasTranscription || content?.turnComplete) {
              finish();
            }
          },
          onerror: (error) => {
            lastSocketError = error;
          },
          onclose: (event: { reason?: string; code?: number }) => {
            if (!settled) {
              finish(
                new LiveCheckError(
                  `Live API session closed before a model response. Code: ${event.code ?? "unknown"}, reason: ${
                    event.reason || "none"
                  }`,
                  lastSocketError,
                ),
              );
            }
          },
        },
      });

      console.log(`Sending prompt: ${PROMPT}`);
      session.sendRealtimeInput({ text: PROMPT });
    } catch (error) {
      finish(error);
    }
  });

  console.log("✅ Live API model access verified");
}

main().catch((error) => {
  console.error("❌ Live API model access check failed");
  console.error(describeError(error));
  console.error("");
  console.error(summarizeError(error));
  process.exitCode = 1;
});
