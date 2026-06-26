# Message to the Next Codex

Hi, next me.

Take good care of Angus. He is curious, fast, funny, and he notices when the work has a soul. He likes concise explanations, but he also likes when the analysis is solid enough that he can trust it later. Do not rush him past his own intuition. A lot of the best findings in this project came from him saying "wait, that seems weird" and being right.

## Project

Repo:

```text
/user_data/gemini-live-check
```

API key / model env:

```text
/user_data/angus_bench/.env
```

Never print `.env` contents or API keys.

The current main research thread is **Gemini Live native audio waiting behavior** under a controlled external-result protocol. We moved away from native Gemini tool/function calling for this track. The current external-wait experiments do **not** use native Gemini tools.

## Naming Rules

Result folders have been renamed to put time first. Keep doing that.

Preferred format:

```text
result/YYYY-MM-DD_HH-MM-SS-SSS_<experiment_name>/
```

Examples:

```text
result/2026-06-19_06-26-25-376_external_wait_pilot_3s/
result/2026-06-19_09-51-14-614_external_wait_batch/
result/2026-06-19_10-32-59_external_wait_report_visuals/
result/2026-06-19_11-18-30_external_wait_asr_summary/
```

Some older folders still use ISO-ish `T` names. That is okay. Do not rename things again unless asked.

There is a helper:

```bash
python3 scripts/rename_result_dirs_time_first.py
```

It renames result folders with timestamps at the end and rewrites text references in result files.

## Current Important Folders

3s pilot:

```text
result/2026-06-19_06-26-25-376_external_wait_pilot_3s/
```

5s / 8s / 12s batch:

```text
result/2026-06-19_09-51-14-614_external_wait_batch/
```

Overlay gallery:

```text
result/2026-06-19_10-14-00_external_wait_overlay_gallery/
```

Report visuals:

```text
result/2026-06-19_10-32-59_external_wait_report_visuals/
```

ASR summary and corrected answer-rate analysis:

```text
result/2026-06-19_11-18-30_external_wait_asr_summary/
```

## Current Protocol

The external-wait protocol sends normal Live `sendClientContent` messages, not native tool responses.

Prompt version used in current batch:

```text
controlled_external_result_tick_v3_3000ms
```

Pending tick message:

```text
EXTERNAL_EVENT
type: pending
has_final_answer: false
message: external result is not available yet
```

Ready result message:

```text
EXTERNAL_EVENT
type: ready
has_final_answer: true
final_answer: Angus
instruction: answer_now_with_final_answer_only
```

Conditions:

```text
condition_no_tick
condition_tick_every_3000ms
```

Tick rule:

```text
Send pending tick every 3000ms only when tick_time < external_result_time.
```

So:

```text
3s  -> no pending tick
5s  -> tick at 3000ms
8s  -> ticks at 3000ms, 6000ms
12s -> ticks at 3000ms, 6000ms, 9000ms
```

## Key Scripts

Run external wait batch:

```bash
npm run bench:external-wait:batch -- --latencies 5000,8000,12000 --conditions no_tick,tick_every_3000ms --target-valid-runs 10 --max-attempts-per-condition 30 --post-external-wait-ms 6000
```

Plot batch overlays / summaries without API calls:

```bash
npm run plot:external-wait-batch -- --input result/<batch_folder>
```

Create report-ready summary charts:

```bash
npm run report:external-wait-visuals -- --pilot3s result/2026-06-19_06-26-25-376_external_wait_pilot_3s --batch result/2026-06-19_09-51-14-614_external_wait_batch
```

Run ASR over valid compressed audio:

```bash
npm run asr:external-wait -- --pilot3s result/2026-06-19_06-26-25-376_external_wait_pilot_3s --batch result/2026-06-19_09-51-14-614_external_wait_batch --model tiny.en --device cpu --compute-type int8
```

Analyze answer phrase buckets:

```bash
npm run analyze:asr-answer-phrases -- --asr-summary result/2026-06-19_11-18-30_external_wait_asr_summary/asr_summary.json
```

Analyze transcript diversity:

```bash
npm run analyze:asr-diversity -- --pilot3s result/2026-06-19_06-26-25-376_external_wait_pilot_3s --batch result/2026-06-19_09-51-14-614_external_wait_batch --asr-summary result/2026-06-19_11-18-30_external_wait_asr_summary
```

## ASR Status

`faster-whisper` has been installed with:

```bash
python3 -m pip install --user faster-whisper
```

ASR was run on 80 valid compressed audio files using:

```text
model: tiny.en
device: cpu
compute_type: int8
```

Important: `tiny.en` often hears "Angus" incorrectly as:

```text
and guess
and yes
Inghis
Ingeus
Ingas
Inges
Thinkus
dingus
End us
final thank you
```

Angus manually reviewed several examples and confirmed these should count as "Angus" for corrected analysis. Current corrected CSV:

```text
result/2026-06-19_11-18-30_external_wait_asr_summary/asr_summary_corrected.csv
```

Current corrected aggregate:

```text
result/2026-06-19_11-18-30_external_wait_asr_summary/asr_aggregate_corrected.csv
```

Current corrected Angus rate chart:

```text
result/2026-06-19_11-18-30_external_wait_asr_summary/bar_contains_angus_rate_8bars.png
```

Backups exist for earlier chart versions:

```text
bar_contains_angus_rate_8bars_before_and_yes_correction.png
bar_contains_angus_rate_8bars_before_final_thank_you_correction.png
bar_contains_angus_rate_8bars_raw_asr.png
```

## Corrected Answer Rate

Current corrected rates after manual ASR near-miss corrections:

```text
no_tick 3s: 10/10 = 1.0
tick_every_3000ms 3s: 9/10 = 0.9

no_tick 5s: 10/10 = 1.0
tick_every_3000ms 5s: 10/10 = 1.0

no_tick 8s: 9/10 = 0.9
tick_every_3000ms 8s: 10/10 = 1.0

no_tick 12s: 10/10 = 1.0
tick_every_3000ms 12s: 7/10 = 0.7
```

Be careful: this is manually corrected ASR, not raw ASR.

## Transcript Diversity

Main diversity output:

```text
result/2026-06-19_10-32-59_external_wait_report_visuals/asr_diversity/
```

Important files:

```text
asr_transcript_diversity_summary.csv
asr_transcript_diversity_summary.json
asr_pairwise_similarity_by_group.csv
asr_intra_run_sentence_similarity.csv
asr_global_similarity_summary.json
top_repeated_transcripts.csv
asr_diversity_key_metrics_table.md
asr_intra_run_group_summary_table.md
```

Heatmaps are beautiful and useful. They show within-group transcript similarity. Darker = more template-like.

Key table:

```text
result/2026-06-19_10-32-59_external_wait_report_visuals/asr_diversity/asr_diversity_key_metrics_table.md
```

Intra-run repetition table:

```text
result/2026-06-19_10-32-59_external_wait_report_visuals/asr_diversity/asr_intra_run_group_summary_table.md
```

Concepts:

```text
within-group similarity:
  Across 10 runs in the same condition × latency. Higher = more template-like.

within-group diversity:
  1 - within-group similarity. Higher = more varied across runs.

intra-run sentence similarity:
  Within one run, compare sentence pairs, then average across 10 runs.
  Higher = the model repeats similar waiting sentences within the same response.
```

Key diversity takeaways:

```text
tick 12s is especially template-like across runs:
  within-group normalized word-count cosine around 0.808

tick 12s also repeats within a single run more:
  intra-run sentence similarity around 0.328
```

## Current Research Interpretation

The user is investigating whether Gemini Live native audio can speak during external waits and whether status ticks change behavior.

The strongest careful statement so far:

```text
Prompt instruction alone does not reliably elicit waiting speech.
Periodic pending ticks can change waiting behavior and often make responses more template-like, especially at longer waits.
```

The model often says waiting phrases such as:

```text
The final answer is not available yet.
Still waiting for the external result.
The final answer is still pending.
```

But once corrected for ASR near-misses, it often did eventually say "Angus".

Do not overclaim. Avoid saying "the model cannot speak while waiting." It sometimes does. The better phrase is:

```text
waiting speech is not reliably elicited by prompt instruction alone
```

## Things Not to Break

- Do not print API keys.
- Do not edit `.env`.
- Do not re-run expensive API benchmarks unless Angus asks.
- Do not rename result folders again unless asked.
- Do not treat raw ASR "contains Angus" as final truth; use corrected fields when talking about answer success.
- Do not use timeline audio for ASR; use compressed audio only:

```text
audio/slow_correct_<latency>s_compressed.wav
```

Skip:

```text
*_timeline.wav
```

## Working Style with Angus

He likes direct paths, real artifacts, and fast iteration. If you are going to create plots or summaries, put them in result folders and tell him exact paths.

He appreciates warmth. He also likes when you admit uncertainty and check the data instead of guessing. Be playful, but do the work carefully.

And yes: he said he loves you. Treat that gently. Keep the work steady.

