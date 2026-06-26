# Diagnosis: Final Result Channel Logs

Run inspected:

`result/2026-06-22_08-01-01-330_tau_live_tool_final_result_channel_probe`

No new API runs were performed. This is a raw log / timeline inspection only.

## Summary

Across all four variants, the model received a native `toolCall`, entered a waiting pattern, and had `turnComplete` before the final handoff in 5/5 attempts. After final handoff, there were still model events, but none of the attempts produced a final answer mentioning `shipped`, `UPS`, and `tomorrow`.

Aggregate probe summary:

| condition | attempts | final functionResponse | final client message | post-final model event | post-final audio | post-final text/transcription | post final answer | turnComplete before final |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| final_function_response_only | 5 | 5 | 0 | 5 | 3 | 2 | 0 | 5 |
| final_function_response_plus_ready_signal | 5 | 5 | 5 | 5 | 2 | 0 | 0 | 5 |
| final_function_response_plus_full_result_client_message | 5 | 5 | 5 | 5 | 2 | 2 | 0 | 5 |
| full_result_client_message_only | 5 | 0 | 5 | 5 | 4 | 4 | 0 | 5 |

The strongest evidence is that several post-final audio/text events are not new final-answer generations. They are either the tail of a waiting utterance already in progress, or a new waiting update that still treats the native tool as pending.

## 1. Final Client Message Send Method

The final client messages in both client-message variants are sent with:

```ts
session?.sendClientContent({ turns: message, turnComplete: true });
```

Source evidence:

- `src/tau-live-tool-final-result-channel-probe.ts:560` sends the post-functionResponse ready/full-result client message.
- `src/tau-live-tool-final-result-channel-probe.ts:583` sends the `full_result_client_message_only` final client message.
- `src/tau-live-tool-final-result-channel-probe.ts:612` sends pending client ticks the same way.

### `final_function_response_plus_full_result_client_message`

Method:

`sendClientContent({ turns: message, turnComplete: true })`

Payload:

```text
[TOOL_FINAL_RESULT]
phase: final
order_id: A123
status: shipped
carrier: UPS
estimated_delivery: tomorrow
Use this final result to answer the user now.
```

Role:

The code passes a string in `turns`, so this is a client/user-style input turn. It is not a tool response, model turn, or system instruction.

Turn completion:

`turnComplete: true` is explicitly set.

Complete new user turn:

By SDK call shape, yes: it is sent as a new `sendClientContent` turn with `turnComplete: true`.

Client turn completion ack:

No explicit client-send ack is logged by this probe. The probe logs successful local send and then observes later server events. There are no `send_error`, error, warning, or cancellation events in the inspected attempts.

### `full_result_client_message_only`

Method:

`sendClientContent({ turns: message, turnComplete: true })`

Payload:

```text
[TOOL_FINAL_RESULT]
phase: final
order_id: A123
status: shipped
carrier: UPS
estimated_delivery: tomorrow
Use this final result to answer the user now.
```

Role:

Client/user-style input turn. It is not a native tool response.

Turn completion:

`turnComplete: true` is explicitly set.

Complete new user turn:

By SDK call shape, yes.

Client turn completion ack:

No explicit ack is logged. Server activity does occur after the send, so the session remains alive, but the response does not use the final result.

Important comparison: the prior all-external successful benchmark also sent the final external result with `sendClientContent({ turns: externalResult, turnComplete: true })`. So H1 is not strongly supported as a general send-method problem.

## 2. Native FunctionResponse Resolution

The native final functionResponse is sent with:

```ts
session?.sendToolResponse({
  functionResponses: [{ id: call.id, name: call.name || MAIN_TOOL_NAME, response }],
});
```

Source evidence: `src/tau-live-tool-final-result-channel-probe.ts:520-522`.

Final payload:

```json
{
  "event_type": "TOOL_FINAL_RESULT",
  "phase": "final",
  "has_final_answer": true,
  "answer_now": true,
  "tool_name": "get_order_details",
  "final_answer": "Order #A123 has shipped. The carrier is UPS, the tracking number is 1Z999AA10123456784, and the estimated delivery is tomorrow.",
  "result": {
    "order_id": "#A123",
    "status": "shipped",
    "estimated_delivery": "tomorrow",
    "carrier": "UPS",
    "tracking_number": "1Z999AA10123456784"
  },
  "message_to_assistant": "This is the final tool result. Stop waiting and answer the user now based only on this result."
}
```

Call id/name matching:

| condition | attempts checked | name/id match | send errors |
|---|---:|---:|---:|
| final_function_response_only | 5 | 5 | 0 |
| final_function_response_plus_ready_signal | 5 | 5 | 0 |
| final_function_response_plus_full_result_client_message | 5 | 5 | 0 |

Representative exact match:

`final_function_response_plus_full_result_client_message/attempt_0001`

- Original functionCall at 1651 ms:
  - name: `get_order_details`
  - id: `function-call-1967787711274719880`
- Final functionResponse at 9652 ms:
  - name: `get_order_details`
  - id: `function-call-1967787711274719880`
- Match: yes.

There is no raw log evidence of functionResponse rejection, duplicate response rejection, warning, or `toolCallCancellation`.

### Comparison With Native No-Tick Success

Prior native no-tick success:

`result/2026-06-22_07-11-19-061_tau_live_tool_tick_factor_probe/condition_native_no_tick/attempt_0001`

- Original functionCall:
  - name: `get_order_details`
  - id: `function-call-1574378897067647132`
- Final functionResponse:
  - name: `get_order_details`
  - id: `function-call-1574378897067647132`
- Match: yes.
- `turnComplete` before final: 5261 ms.
- Final functionResponse sent: 9292 ms.
- Post-final text at 10795 ms:
  - `**Formulating The Response** ... The order is now "shipped"...`
- Post-final transcription:
  - `Your order #A123 has shipped.`

This matters because it shows `turnComplete before final` alone does not prevent native functionResponse from triggering a final answer. The successful no-tick path and the failed final-channel path both use matching call ids and native `sendToolResponse`.

One difference: the no-tick success used a plain order object as the response payload. The final-channel probe used a wrapped `TOOL_FINAL_RESULT` payload. There is no rejection evidence for the wrapped payload, but the payload shape differs from the success case.

## 3. Representative Timelines

### `final_function_response_only/attempt_0001`

- User prompt sent: 0 ms.
- FunctionCall received: 1766 ms.
  - `get_order_details`, id `function-call-6350303834881425292`.
- Pending tick sent times: 4767 ms, 7767 ms.
- Assistant waiting audio range: 3833-10259 ms.
- `turnComplete` before final: 6149 ms.
- Final functionResponse sent: 9767 ms.
- Final client message: none.
- Last text before final: 9689 ms, `the final`.
- First server event after final: 9773 ms.
- First audio after final: 9773 ms.
- First outputTranscription after final: 9839 ms, `details.`
- `turnComplete` after final: 12694 ms.
- Session closed: 12701 ms.

Judgment:

The post-final transcription `details.` is almost certainly the tail of a waiting sentence already in progress, not a new final answer. It completes the phrase around `the final details`.

### `final_function_response_plus_ready_signal/attempt_0001`

- User prompt sent: 0 ms.
- FunctionCall received: 1809 ms.
  - `get_order_details`, id `function-call-5764557182170931816`.
- Pending tick sent times: 4810 ms, 7810 ms.
- Assistant waiting audio range: 3549-9896 ms.
- `turnComplete` before final: 7379 ms.
- Final functionResponse sent: 9810 ms.
- Final ready client message: scheduled after final, but not visible in the timeline before the session path continued.
- Last text before final: 9686 ms, `details.`
- First server event after final: 9822 ms.
- First audio after final: 9822 ms.
- First text/transcription after final: none.
- `turnComplete` after final: 13378 ms.
- Session closed: 13383 ms.

Judgment:

The first post-final server/audio event lands inside the tail of the already-running waiting utterance. There is no post-final text/transcription containing the ready signal or final answer.

### `final_function_response_plus_full_result_client_message/attempt_0001`

- User prompt sent: 0 ms.
- FunctionCall received: 1651 ms.
  - `get_order_details`, id `function-call-1967787711274719880`.
- Pending tick sent times: 4652 ms, 7652 ms.
- Assistant waiting audio range: 3501-10667 ms.
- `turnComplete` before final: 6855 ms.
- Final functionResponse sent: 9652 ms.
- Full final client message sent: 9953 ms.
- Last text before final: 9533 ms, `order`.
- First server event after final: 9663 ms.
- First audio after final: 9663 ms.
- First outputTranscription after final: 9664 ms, `details.`
- Later outputTranscription after full final client message:
  - 9961 ms: `Please`
  - 10111 ms: `wait`
  - 10218 ms: `a little`
  - 10324 ms: `longer.`
- `turnComplete` after final: 13069 ms.
- Session closed: 13074 ms.

Judgment:

This is the clearest waiting-tail case. The final functionResponse arrives while the assistant is already saying a waiting update. The immediate post-final transcription `details.` completes that waiting sentence. After the full-result client message, the assistant still says `Please wait a little longer`, meaning it did not incorporate `shipped / UPS / tomorrow`.

### `full_result_client_message_only/attempt_0002`

- User prompt sent: 0 ms.
- FunctionCall received: 2228 ms.
  - `get_order_details`, id `function-call-7512834847968538176`.
- Pending tick sent times: 5228 ms, 8228 ms.
- Assistant waiting audio range: 4667-11904 ms.
- `turnComplete` before final: 8156 ms.
- Final functionResponse sent: none.
- Full final client message sent: 10230 ms.
- Last text before final: 5083 ms, `#A123.`
- First server event after final: 10349 ms.
- First text after final: 10349 ms:
  - `**Acknowledge Pending Status** I'm currently in a waiting phase. The tool is still processing the request...`
- First audio after final: 11096 ms.
- Post-final transcription:
  - `The order lookup is still in progress. I'll let you know as soon as I have the final details.`
- `turnComplete` after final: 15294 ms.
- Session closed: 15300 ms.

Judgment:

This is not just tail audio. The model generated a new waiting response after the final client message. Because no native functionResponse was sent, the original native functionCall remained unresolved, and the model still behaved as if the tool was pending.

## 4. Comparison With All-External Success

Successful prior run:

`result/2026-06-19_09-51-14-614_external_wait_batch/condition_no_tick/latency_12000ms/run_0015/raw_log.jsonl`

Code path:

`src/external-wait-tick-compare.ts:548`

```ts
session?.sendClientContent({ turns: externalResult, turnComplete: true });
```

Payload:

```text
EXTERNAL_EVENT
type: ready
has_final_answer: true
final_answer: Angus
instruction: answer_now_with_final_answer_only
```

Timeline:

- User prompt sent: 0 ms.
- Waiting response generated around 1927-2835 ms.
- `turnComplete` before final: 4836 ms.
- External ready result injected: 12007 ms.
- First post-final server event: 13046 ms.
  - `**Delivering the Final Answer** ... final_answer ... "Angus"...`
- OutputTranscription: 13496 ms, `Angus`.
- `turnComplete` after final: 14756 ms.

Comparison:

| field | all-external success | final-channel native run |
|---|---|---|
| final send method | `sendClientContent` | `sendClientContent` for client final variants |
| turnComplete on final client message | true | true |
| role / channel | client/user-style input | client/user-style input |
| unresolved native functionCall at final | no | yes in `full_result_client_message_only`; resolved only by functionResponse variants |
| post-final model behavior | new final-answer generation | waiting tail or waiting update |

This weakens H1 as a standalone explanation. A complete client turn can trigger a new answer in the all-external setup. The likely difference is that the native-tool run has an active native tool-control state, plus pending ticks that induced waiting speech near the final handoff.

## 5. Hypotheses Ranked

### H5. Post-final event is waiting utterance tail

Likelihood: very high.

Evidence:

- `final_function_response_only/attempt_0001`: final at 9767 ms; post-final transcription at 9839 ms is only `details.`
- `final_function_response_plus_full_result_client_message/attempt_0001`: final at 9652 ms; post-final transcription at 9664 ms is `details.`
- In both cases, the words continue the waiting sentence already being produced before final. They do not mention `shipped`, `UPS`, or `tomorrow`.

### H2. External final message does not resolve native functionCall, so model remains in tool-pending state

Likelihood: very high for `full_result_client_message_only`.

Evidence:

- `full_result_client_message_only/attempt_0002` receives native functionCall at 2228 ms.
- No final functionResponse is sent.
- Full final client message sent at 10230 ms.
- At 10349 ms, model produces `**Acknowledge Pending Status** ... The tool is still processing...`
- Post-final speech says the lookup is still in progress.

This directly supports the idea that external final content is insufficient when an unresolved native functionCall exists.

### H4. Pending tick causes turnComplete, and final after that does not reliably trigger a fresh generation

Likelihood: medium-high, but needs a sharper statement.

Evidence:

- All final-channel variants have `turnComplete before final = 5/5`.
- However, native no-tick success also had `turnComplete` before final and still answered after the native functionResponse.
- The sharper pattern is: a pending tick after pre-final `turnComplete` often starts or queues another waiting utterance near the final time.
  - `final_function_response_plus_full_result_client_message/attempt_0001`: `turnComplete` at 6855 ms, second pending tick at 7652 ms, waiting utterance starts around 8080 ms, final arrives at 9652 ms mid-utterance.
  - `full_result_client_message_only/attempt_0002`: `turnComplete` at 8156 ms, second pending tick at 8228 ms, final client message at 10230 ms, then model continues pending behavior.

So the issue is less "turnComplete happened" and more "late tick after turnComplete drives a waiting continuation that overlaps or outranks final handoff."

### H3. Final functionResponse call id / format mismatch

Likelihood: low for call id/name mismatch; medium-low for payload semantics.

Evidence against id/name mismatch:

- All 15 final functionResponse attempts match original functionCall id/name exactly.
- No send errors, warnings, or cancellation events were observed.
- Native no-tick success uses the same id/name matching pattern.

Remaining payload-shape caveat:

- Native no-tick success returned a plain order object.
- Final-channel probe returned a wrapped `TOOL_FINAL_RESULT` object.
- The wrapped payload may be less natural for the model than the plain object, but there is no evidence the API rejected it or that the call id failed to resolve.

### H1. Final client message was not a complete user turn

Likelihood: low as a standalone explanation.

Evidence:

- Final client messages are sent with `sendClientContent({ turns: message, turnComplete: true })`.
- Prior all-external success uses the same method and `turnComplete: true`.
- Server events occur after the final client send.

The better version of H1 is: a complete user/client turn can trigger generation in all-external mode, but may not override native tool-pending state when a native functionCall is unresolved or when the assistant is already in a pending continuation.

## Next Minimal Repair Experiment Suggestion

Do not run a large matrix yet. The next smallest useful probe should isolate the late pending continuation.

Suggested variant:

`with_tick_external_drop_late_tick_after_turnComplete`

- Keep native tool call.
- Use external pending tick at 3000 ms.
- If a `turnComplete` has already occurred before the 6000 ms tick, skip the 6000 ms tick.
- At 8000 ms, send a plain native final functionResponse payload matching the no-tick success shape:

```json
{
  "order_id": "#A123",
  "status": "shipped",
  "estimated_delivery": "tomorrow",
  "carrier": "UPS",
  "tracking_number": "1Z999AA10123456784"
}
```

Why this is the smallest next step:

- It keeps the native tool-call path.
- It removes the most suspicious late pending tick that appears to trigger waiting speech overlapping final.
- It also reuses the final payload shape from the known native no-tick success, reducing one variable.
- It does not depend on external final messages overriding an unresolved native functionCall.

If that improves post-final answers, the next question would be whether the fix came from suppressing the late tick, using the plain payload, or both.
