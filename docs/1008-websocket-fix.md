# Gemini Live 1008 WebSocket Fix Notes

This note documents the `fix/websocket-1008` branch: what was failing, what changed, why those changes are closer to the Gemini Live WebSocket protocol, and what evidence we have so far.

## Short Version

Earlier native tool-call benchmark runs could hit WebSocket close code `1008`, especially in native tool-call waiting experiments. The current branch did not change the benchmark's research question or final tool payload. Instead, it tightened the Live API session semantics:

- wait for `setupComplete` before sending the first user turn;
- send external pending ticks through `sendRealtimeInput(...)` instead of a new `sendClientContent(...)` user turn;
- record and respect `toolCallCancellation`;
- avoid sending `sendToolResponse(...)` for a function call that the server has cancelled;
- keep the final result on the native `sendToolResponse(...)` path;
- preserve the post-final observation window so the benchmark does not close early.

After these changes, a formal benchmark rerun showed `0` close-code `1008` and `0` close-code `1011` across all formal cells.

## Original Symptom

The project was testing whether Gemini Live could keep speaking naturally while a native tool call was pending. Early experiments saw instability in some native-tool runs:

- sessions sometimes closed with code `1008`;
- the failures were more visible around delayed tool responses and tick/pending-status experiments;
- some runs also failed behaviorally by not answering after the final tool result, but that is a separate model/turn-recovery issue and should not be confused with transport instability.

The working hypothesis for this branch was:

> Some 1008s may be caused by harness-level WebSocket/protocol details rather than by a fundamental model limitation.

This branch tests that hypothesis by aligning the harness more carefully with Live session semantics.

## Relevant Live API Concepts

The probe talks to Gemini Live over the SDK's WebSocket session abstraction.

The important event and send paths are:

- `setupComplete`: server event indicating the Live session is ready.
- `sendClientContent(...)`: sends complete client turns, used for the initial user request.
- `sendRealtimeInput(...)`: sends realtime input into the active session, used here for lightweight external pending ticks.
- `toolCall.functionCalls`: server asks the client/runtime to execute a declared native tool.
- `sendToolResponse(...)`: client/runtime resolves a native function call.
- `toolCallCancellation`: server cancels one or more pending tool calls.
- `serverContent.turnComplete`: model turn boundary. The benchmark observes this but does not use early turn completion as an immediate close condition.

## What Changed

### 1. Wait For `setupComplete` Before The First User Turn

Before this branch, some scripts sent the initial user prompt as soon as the socket opened. The fixed probe waits for `setupComplete` and only then sends:

```ts
session.sendClientContent({ turns: prompt.userPrompt, turnComplete: true });
```

Code: `src/tau-live-tool-tick-factor-probe.ts`

Relevant fields now logged per attempt:

- `setupComplete_time_ms`
- `setup_complete_before_prompt`
- `sent_after_setup_complete`

Why this matters:

The server has an explicit setup phase. Sending user content before setup completion may work sometimes, but it risks racing the session configuration and tool declarations.

### 2. External Pending Tick Uses `sendRealtimeInput(...)`

The successful single-tick condition still sends only one generic pending signal at `4000 ms` when the tool latency is greater than `4000 ms`:

```text
The lookup is still running. No final result is available yet.
```

The important implementation change is the channel:

```ts
session.sendRealtimeInput({ text: clientPendingMessage() });
```

The probe logs this as:

```json
{
  "type": "client_status_tick_sent",
  "send_method": "sendRealtimeInput"
}
```

Why this matters:

The tick is not meant to be a new completed user request. It is a small realtime status signal during the same interaction. Using `sendClientContent(... turnComplete: true ...)` for ticks may accidentally create extra complete user turns while a native function call is unresolved.

### 3. Native Final Result Still Uses `sendToolResponse(...)`

The fix did not move the answer into an external message. The final result remains a native tool response:

```ts
session.sendToolResponse({
  functionResponses: [{ id: call.id, name: call.name || MAIN_TOOL_NAME, response }],
});
```

The final response is intentionally explicit:

```json
{
  "event_type": "TOOL_RESULT",
  "phase": "final",
  "has_final_answer": true,
  "answer_now": true,
  "tool_name": "get_order_details",
  "order_id": "#A123",
  "status": "shipped",
  "carrier": "UPS",
  "tracking_number": "1Z999AA10123456784",
  "estimated_delivery": "tomorrow"
}
```

Why this matters:

The benchmark is specifically about native tool-call waiting behavior. External-only result injection is useful as a baseline, but it does not test whether the native tool-call path recovers after a pending tool result.

### 4. Respect `toolCallCancellation`

The probe now records server-side tool cancellation:

```ts
if (message.toolCallCancellation?.ids?.length) {
  for (const id of message.toolCallCancellation.ids) {
    if (!state.cancelledToolCallIds.includes(id)) state.cancelledToolCallIds.push(id);
  }
}
```

Before sending any tool response, it checks whether the function call was cancelled:

```ts
if (isCancelledCall(call)) {
  appendTimeline("tool_response_skipped_cancelled_call", ...);
  return false;
}
```

Why this matters:

Responding to a cancelled function call is a plausible protocol violation. The fixed harness treats cancellation as authoritative and logs the skipped response instead of trying to resolve a call the server no longer wants.

### 5. Do Not Close Early On Pre-final Or Early Post-final Turn Completion

The benchmark still uses a fixed post-final observation window:

```ts
const POST_FINAL_WAIT_MS = 8000;
```

After a successful final `sendToolResponse(...)`, it waits through that observation window before closing. This avoids closing the session just because the model produced an early `turnComplete` or because a waiting utterance finished before the final result arrived.

Why this matters:

The project's metric needs to know whether the model eventually answers after the final result. Closing immediately on the wrong turn boundary can erase that signal.

## Formal Rerun Evidence

### Sequential Rerun On The Fix Branch

Result folder:

```text
result/archived_2026-06-25_fix_websocket_api_runs/sequential_1137_formal_benchmark
```

Overall:

- total attempts: `123`
- valid attempts: `80`
- retries: `43`
- `1008`: `0`
- `1011`: `0`

This was the first strong signal that the WebSocket/protocol changes removed the reproduced 1008 instability in the formal matrix.

### Concurrent Rerun On The Fix Branch

Result folder:

```text
result/archived_2026-06-25_fix_websocket_api_runs/concurrent_1224_formal_benchmark
```

Overall:

- total attempts: `104`
- valid attempts: `80`
- retries: `24`
- `1008`: `0`
- `1011`: `0`
- elapsed time: about `4.91 min`

The previous sequential run took about `34.42 min`, so the concurrent runner was about `7.0x` faster for this formal benchmark.

The concurrent run also required one additional fix: per-attempt result directories now include process id plus a UUID fragment. Timestamp-only directories can collide under parallel child processes.

## Comparison To Older Behavior

An earlier formal run before this branch is:

```text
result/2026-06-22_11-15-58-195_tau_live_tool_formal_benchmark
```

In that run, native no-tick cells had visible `1008` counts:

| condition | latency_ms | old 1008 count |
| --- | ---: | ---: |
| native_no_tick | 3000 | 7 |
| native_no_tick | 5000 | 13 |
| native_no_tick | 8000 | 3 |
| native_no_tick | 12000 | 3 |

On the fixed branch's formal reruns, those formal cells showed `0` `1008`.

This does not prove Gemini Live can never produce `1008`. It does show that the previously observed formal-benchmark 1008s were reproducible enough to disappear after the harness was aligned more carefully with WebSocket/session semantics.

## Remaining Caveats

The 1008 fix is not the same as solving all benchmark behavior.

Remaining issues still tracked by the benchmark:

- some attempts are `not_valid` because the model does not produce the expected final answer after the native final result;
- long no-tick latency can still require more retries even when transport stability is good;
- waiting speech quality still needs ASR and LLM/self-BLEU analysis;
- server-side cancellation is now handled safely, but cancelled calls still count against final-response success for that attempt.

So the careful claim is:

> The fixed harness substantially reduced or eliminated the observed 1008 instability in the formal benchmark runs by aligning setup, tick injection, and tool-cancellation handling with Live WebSocket semantics.

The stronger claim would be unsafe:

> 1008 is permanently solved for all Gemini Live native tool-call workloads.

## How To Reproduce

Build:

```bash
npm run build
```

Run the formal benchmark sequentially:

```bash
npm run tau:live-tool-formal-benchmark
```

Run the formal benchmark with per-cell concurrency:

```bash
npm run tau:live-tool-formal-benchmark -- --cell-concurrency 50
```

Inspect stability:

```bash
cat result/<formal-result-folder>/summary.csv
```

Key columns:

- `total_attempts`
- `valid_attempts`
- `1008_count`
- `1011_count`
- `retry_count`
- `send_error_count`
- `tool_call_success_count`
- `final_tool_response_sent_count`
- `post_tool_final_answer_count`

## Code Pointers

Primary implementation:

- `src/tau-live-tool-tick-factor-probe.ts`

Formal benchmark runner:

- `src/tau-live-tool-formal-benchmark.ts`

Key areas in the probe:

- wait for `setupComplete` before `sendClientContent(...)`;
- external tick via `sendRealtimeInput(...)`;
- final native response via `sendToolResponse(...)`;
- cancellation handling via `toolCallCancellation`;
- post-final observation window;
- raw log and timeline fields for verifying event order.
