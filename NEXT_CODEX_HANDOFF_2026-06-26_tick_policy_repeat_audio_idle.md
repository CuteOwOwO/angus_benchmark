# NEXT CODEX HANDOFF

Updated: 2026-06-26

## Current Goal

We are investigating **TP1 spoken tool-wait benchmark pending tick policies** for Gemini Live native tool calling.

The focus is not full tau-bench reproduction and not tool-use correctness. The focus is:

- how pending ticks affect spoken waiting behavior,
- whether the assistant keeps the user from long uncomfortable silence,
- whether waiting speech stays task-aware,
- whether final tool result recovery remains smooth.

The next phase is expected to be **multi-step tasks**, so preserve the current benchmark harness and extend carefully.

## Important User Preferences

- User prefers Traditional Chinese conversation.
- User wants artifacts organized by experiment condition, latency, and attempt.
- User strongly cares that timecharts are truthful.
- Do not scatter result folders unnecessarily.
- Do not overclaim from small pilots.
- If a chart is misleading, fix the chart/metric instead of explaining it away.

## Latest Correct Result Folder

Most recent useful run:

```text
/user_data/gemini-live-check/result/2026-06-26_12-07-18-570_boundary_tick_policy_pilot
```

Organized folder:

```text
/user_data/gemini-live-check/result/2026-06-26_12-07-18-570_boundary_tick_policy_pilot/organized
```

Key files:

```text
organized/summary.csv
organized/summary.json
organized/by_tick_policy/
organized/visualizations/boundary_tick_policy_overlay_gallery.png
organized/visualizations/bar_max_silence_gap_before_result.png
organized/visualizations/bar_audio_occupancy_ratio_before_result.png
organized/visualizations/bar_audio_segment_count_before_result.png
```

Conditions included in latest folder:

```text
periodic_tick_4s
tick_after_audio_idle_repeat_0s
tick_after_audio_idle_repeat_1s
```

Latencies:

```text
8000 ms
12000 ms
```

The latest folder originally ran only the two repeat policies, then `periodic_tick_4s` was copied in from:

```text
/user_data/gemini-live-check/result/2026-06-26_09-42-23-358_boundary_tick_policy_pilot/organized
```

When merging, periodic metrics were recomputed using the corrected playback-cursor audio model.

## Latest Conditions

### periodic_tick_4s

External pending tick every 4000 ms while final result is not ready.

Important nuance:

- In the implementation, ticks are scheduled after the native tool call is received.
- Therefore on the global user-prompt timeline they appear roughly at tool_call_time + 4000 ms, not exactly at 4000 ms from user prompt.

### tick_after_audio_idle_repeat_0s

After every pre-final assistant spoken segment ends, detect audio idle and send a pending tick immediately.

Current definition:

```text
projected assistant audio playback cursor + 300 ms idle
then tick delay 0 ms
```

Repeat behavior:

- It is no longer one-shot.
- It can tick after multiple waiting utterances.
- There is a cooldown to prevent dense tick spam.

### tick_after_audio_idle_repeat_1s

Same as above, but waits 1000 ms after audio-idle boundary before sending tick.

Current definition:

```text
projected assistant audio playback cursor + 300 ms idle
then tick delay 1000 ms
```

## Why We Added Repeat Audio-Idle

Earlier conditions were misleading:

```text
tick_after_utterance_0s
tick_after_utterance_1s
```

Those were actually based on `turnComplete`, not true audio end. They often fired 1.7-2.4 seconds after audio ended.

Then we tried:

```text
tick_after_audio_idle_0s
tick_after_audio_idle_1s
```

But those were one-shot only: after the first waiting speech, they would never tick again. That was wrong for long waits.

Latest corrected policy is:

```text
tick_after_audio_idle_repeat_0s
tick_after_audio_idle_repeat_1s
```

These use projected playback cursor and can repeat after later spoken segments.

## Critical Audio Timeline Bug That Was Fixed

The plot script used to draw audio as:

```text
audio_start = event_ms
audio_end = event_ms + duration
```

But timeline audio is reconstructed as:

```text
audio_start = max(event_ms, playback_cursor)
audio_end = audio_start + duration
```

This mismatch made blue audio blocks look too short/too early, making ticks appear far after speech ended.

Fixed files:

```text
scripts/plot_boundary_tick_policy_pilot.py
src/boundary-tick-policy-pilot.ts
```

Both now use playback cursor for audio intervals.

Example sanity check from the small repeat pilot:

```text
repeat_0s:
audio 3332-6212 -> boundary 6511 -> tick 6513
audio 8463-11743 -> boundary 12044 -> tick 12045

repeat_1s:
audio 2014-4654 -> boundary 4954 -> tick 5956
audio 7855-10095 -> boundary 10396 -> tick 11397
```

This is the desired behavior:

```text
audio playback end + 300 ms idle -> boundary
boundary + 0/1000 ms -> tick
```

## Latest Summary Snapshot

Latest merged `summary.csv`:

```text
/user_data/gemini-live-check/result/2026-06-26_12-07-18-570_boundary_tick_policy_pilot/organized/summary.csv
```

High-level results from latest summary:

- `tick_after_audio_idle_repeat_0s`
  - 8000 ms: 5/5 valid
  - 12000 ms: 5/5 valid
- `tick_after_audio_idle_repeat_1s`
  - 8000 ms: 5/5 valid
  - 12000 ms: 5/6 attempts, 5 valid, one not-valid/send-error-ish retry
- `periodic_tick_4s`
  - imported from previous run
  - included for chart comparison

Be careful:

- Some summary fields are all-attempt averages, not valid-only.
- The user may later ask for valid-only charts.
- Waiting relevance fields are heuristic from logs, not ASR/LLM judge.

## Current Modified Files

`git status --short` currently shows:

```text
 M package.json
 M scripts/postprocess_tau_live_tool_tick_factor_probe.py
 M src/tau-live-tool-tick-factor-probe.ts
?? scripts/plot_boundary_tick_policy_pilot.py
?? src/boundary-tick-policy-pilot.ts
```

Important code locations:

```text
src/tau-live-tool-tick-factor-probe.ts
```

Contains:

- native tool call probe,
- tick modes,
- periodic tick scheduling,
- repeat audio-idle boundary scheduling,
- final native `sendToolResponse(...)`.

```text
src/boundary-tick-policy-pilot.ts
```

Contains:

- pilot matrix runner,
- per-condition/latency retry loop,
- organized artifact copying,
- summary metrics,
- playback-cursor audio metric calculation.

```text
scripts/plot_boundary_tick_policy_pilot.py
```

Contains:

- overlay timechart generation,
- playback-cursor audio segment plotting,
- gallery generation.

```text
scripts/postprocess_tau_live_tool_tick_factor_probe.py
```

Was updated earlier to recognize boundary tick event types in postprocessing/timecharts.

## Commands Used Recently

Build:

```bash
npm run build
```

Latest full repeat-policy run:

```bash
BOUNDARY_TICK_CONDITIONS=tick_after_audio_idle_repeat_0s,tick_after_audio_idle_repeat_1s \
BOUNDARY_TICK_LATENCIES_MS=8000,12000 \
BOUNDARY_TICK_TARGET_VALID=5 \
npm run boundary:tick-policy-pilot
```

Small validation run before that:

```bash
BOUNDARY_TICK_LATENCIES_MS=12000 \
BOUNDARY_TICK_TARGET_VALID=2 \
BOUNDARY_TICK_MAX_ATTEMPTS_PER_CELL=8 \
npm run boundary:tick-policy-pilot
```

The command required network escalation for Gemini Live API.

## Result Folder Organization

The user wants:

```text
organized/by_tick_policy/<condition>/latency_<ms>/attempt_<nnn>/
```

Each attempt folder should contain:

- audio files,
- raw logs,
- timeline JSON/PNG,
- summary/config,
- pilot attempt record.

Also each condition/latency folder should contain:

```text
group_timechart.png
```

This has been created for the latest folder.

## After Each Run: Required Post-Run Work

Do not stop after the API run finishes. The user expects the result folder to be made readable and analysis-ready.

After every benchmark run, do all of the following:

### 1. Confirm the Run Completed

Check:

```text
organized/summary.csv
organized/summary.json
```

Confirm:

- target valid attempts were collected,
- retry / not_valid attempts are still preserved,
- 1008 / 1011 / send_error counts are visible,
- the conditions and latencies match the requested run.

### 2. Build `by_tick_policy/`

Create or update:

```text
organized/by_tick_policy/<condition>/latency_<ms>/attempt_<nnn>/
```

Each attempt folder should contain its own:

```text
raw_log.jsonl
summary.json
config.json
pilot_attempt_record.json
attempt_timeline.png
*.timeline.jsonl
*.timeline.png
assistant_output.wav
assistant_output_compressed.wav
assistant_output_timeline.wav
```

Also write/update:

```text
organized/by_tick_policy/README.md
```

The README should list conditions, latencies, and attempt counts.

### 3. Copy Group Timecharts Into Each Condition Folder

Each condition/latency folder should contain:

```text
group_timechart.png
```

Example:

```text
organized/by_tick_policy/tick_after_audio_idle_repeat_0s/latency_12000/group_timechart.png
```

This should be copied from the corresponding overlay in:

```text
organized/visualizations/
```

### 4. Regenerate Timechart Visualizations

Run:

```bash
python3 scripts/plot_boundary_tick_policy_pilot.py <organized_dir>
```

This should generate:

```text
organized/visualizations/boundary_tick_policy_overlay_gallery.png
organized/visualizations/overlay_condition_<condition>_latency_<ms>ms.png
```

Important:

- The plot script must use playback-cursor audio timing.
- Do not use the old naive `event_ms + duration` audio drawing.
- The blue audio bars should correspond to the actual reconstructed timeline audio.

### 5. Generate Bar Charts

At minimum generate these three:

```text
organized/visualizations/bar_max_silence_gap_before_result.png
organized/visualizations/bar_audio_occupancy_ratio_before_result.png
organized/visualizations/bar_audio_segment_count_before_result.png
```

Definitions:

- `max_silence_gap_before_result`: largest silence gap before final tool response.
- `audio_occupancy_ratio_before_result`: total assistant audio duration before final / pre-final window duration.
- `audio_segment_count_before_result`: merged spoken segment count before final. This is the closest current proxy for "how many waiting utterances".

Do not use `audio_output_count_before_result` as utterance count. It is raw audio chunk/event count.

### 6. Recompute Metrics With Playback-Cursor Audio

When merging old runs or generating summaries manually, recompute audio metrics using:

```text
audio_start = max(event_ms, playback_cursor)
audio_end = audio_start + chunk_duration
playback_cursor = audio_end
```

Then clip to:

```text
user prompt time -> final_tool_response_sent_time_ms
```

This matters for:

- max silence gap,
- audio occupancy ratio,
- audio segment count,
- timechart blue bars.

### 7. Sanity Check Tick Timing

For audio-idle repeat policies, inspect at least one attempt per condition/latency.

Confirm raw timeline shows:

```text
audio playback segment end
-> +300 ms idle boundary
-> +0 ms or +1000 ms tick
```

For example, expected shape:

```text
audio 3332-6212 -> boundary 6511 -> tick 6513
audio 8463-11743 -> boundary 12044 -> tick 12045
```

For `repeat_1s`, tick should be approximately:

```text
boundary + 1000 ms
```

Also confirm repeat behavior:

- if the assistant speaks again before final result,
- and final is not ready,
- another audio-idle boundary can trigger another tick,
- unless skipped by cooldown or final-ready state.

### 8. Update Notes / README

If a result folder has important context, add it to:

```text
organized/README.md
```

Include:

- what was run,
- which conditions,
- latencies,
- attempts / valid counts,
- where summary and charts are,
- whether there were 1008/1011/send errors,
- any known caveat such as "periodic copied from prior run" or "metrics recomputed with playback cursor".

### 9. Build Check

After code changes, run:

```bash
npm run build
```

Report whether it passed.

## Important Gotchas

1. Do not trust old plots before playback-cursor fix.

Old plots may show blue audio too short. Use only plots generated after `scripts/plot_boundary_tick_policy_pilot.py` was fixed.

2. `tick_after_utterance_*` means turnComplete, not audio end.

Avoid using those names unless explicitly comparing turnComplete policies.

3. `tick_after_audio_idle_0s/1s` are one-shot.

Use repeat versions for long waits:

```text
tick_after_audio_idle_repeat_0s
tick_after_audio_idle_repeat_1s
```

4. Tick count can be duplicated if combining summary tick times and timeline tick events naively.

Use unique rounded tick times from `pilot_attempt_record.json` or recompute carefully.

5. `audio_output_count_before_result` is raw chunk/event count, not utterance count.

Use:

```text
audio_segment_count_before_result
```

for merged spoken segment count.

6. ASR/LLM judge was not run for these latest tick-policy pilots.

Quality scores are still heuristic.

## Likely Next Step

The user said the next stage will be **multi-step**.

Recommended approach:

1. Keep the current single-tool order-status task as a baseline.
2. Add a small multi-step task harness without changing tick semantics.
3. First run a tiny pilot:
   - maybe 1-2 conditions,
   - one latency,
   - 2 valid attempts per cell.
4. Verify:
   - tool-call sequence,
   - audio playback cursor timecharts,
   - repeat idle ticks after every waiting utterance,
   - final answer recovery.
5. Only then run a larger matrix.

For multi-step tasks, be especially careful with:

- multiple native tool calls,
- whether pending ticks should apply to each tool wait separately,
- whether a new tool call resets audio-idle repeat state,
- whether final result for one step should stop ticks for that step only.

## Suggested Multi-Step Implementation Notes

If extending `tau-live-tool-tick-factor-probe.ts`:

- Add explicit per-tool-call pending state.
- Reset repeat boundary state when a new native tool call arrives.
- Track:
  - current pending call id,
  - current pending tool name,
  - final response sent time per call,
  - pending tick times per call.
- Avoid a global `state.finalToolResponseSentAt` if there are multiple tool steps; instead use per-step final state and an overall final answer observation window.

Possible new names:

```text
multi_step_audio_idle_repeat_0s
multi_step_audio_idle_repeat_1s
```

But do not overbuild before user specifies the task.

## Final Note

The latest trusted comparison folder is:

```text
/user_data/gemini-live-check/result/2026-06-26_12-07-18-570_boundary_tick_policy_pilot/organized
```

Use this as the starting point for the next Codex.
