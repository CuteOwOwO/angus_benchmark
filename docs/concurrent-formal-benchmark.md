# Concurrent Formal Benchmark Notes

This note documents the concurrent runner added for the formal Gemini Live tool-wait benchmark.

## Goal

The formal benchmark needs enough attempts to compare:

- `native_no_tick`
- `external_single_tick`

across four tool latencies:

- `3000 ms`
- `5000 ms`
- `8000 ms`
- `12000 ms`

Each condition x latency cell targets `10` valid attempts. Running this sequentially is slow because each attempt waits through tool latency plus an `8000 ms` post-final observation window.

The concurrent runner keeps the benchmark logic unchanged while launching multiple single-attempt probes in parallel.

## Command

Sequential default:

```bash
npm run tau:live-tool-formal-benchmark
```

Concurrent run:

```bash
npm run tau:live-tool-formal-benchmark -- --cell-concurrency 50
```

Equivalent environment variable:

```bash
FORMAL_CELL_CONCURRENCY=50 npm run tau:live-tool-formal-benchmark
```

## What `--cell-concurrency` Means

Concurrency is scoped within one condition x latency cell.

The runner still processes cells in order:

1. `native_no_tick` x `3000 ms`
2. `native_no_tick` x `5000 ms`
3. `native_no_tick` x `8000 ms`
4. `native_no_tick` x `12000 ms`
5. `external_single_tick` x `3000 ms`
6. `external_single_tick` x `5000 ms`
7. `external_single_tick` x `8000 ms`
8. `external_single_tick` x `12000 ms`

Inside each cell, up to `cellConcurrency` single-attempt workers can run concurrently. Each worker launches:

```bash
node dist/tau-live-tool-tick-factor-probe.js \
  --tick-modes <tickMode> \
  --attempts 1 \
  --latency-ms <latencyMs> \
  --quiet-terminal-text
```

## Overshoot Control

Although `--cell-concurrency 50` is allowed, each cell only targets `10` valid attempts. The runner does not launch 50 attempts blindly.

The launch condition is:

```ts
active < cellConcurrency &&
launched < MAX_ATTEMPTS_PER_CELL &&
valid + active < TARGET_VALID
```

This means a cell with target `10` starts at most about `10` active attempts at once. It launches more only when attempts finish and the cell still needs more valid runs.

This is why `--cell-concurrency 50` is safe for this benchmark shape: it sets an upper bound, while the target-valid logic sets the practical bound.

## Why Single-attempt Child Processes

The formal benchmark runner delegates each attempt to the existing tick-factor probe instead of duplicating Live API logic.

Benefits:

- preserves the same native tool-call behavior;
- preserves existing raw logs, audio, timeline, and summary outputs;
- isolates each attempt in its own process and WebSocket session;
- keeps retry accounting at the formal runner level.

Implementation entrypoints:

- `src/tau-live-tool-formal-benchmark.ts`
  - `runProbeAttemptAsync(...)`
  - `runCell(...)`
- `src/tau-live-tool-tick-factor-probe.ts`
  - the actual Live API attempt implementation

## Result Directory Collision Fix

The first concurrent test exposed an important bug: single-attempt probe result folders used timestamp-only IDs. Under parallel process launch, multiple attempts can start in the same millisecond.

Old shape:

```text
2026-06-25_12-21-39-846_tau_live_tool_tick_factor_probe
```

Fixed shape:

```text
2026-06-25_12-24-46-305_191409_e3ffcdd1_tau_live_tool_tick_factor_probe
```

The extra suffix is:

- process id;
- short UUID fragment.

This makes per-attempt artifact folders unique under concurrency.

The interrupted collision-discovery run is archived at:

```text
result/archived_2026-06-25_fix_websocket_api_runs/interrupted_1221_concurrent_formal_run
```

Keep it for debugging only. It should not be treated as a clean benchmark result.

## Successful Concurrent Run

Result folder:

```text
result/archived_2026-06-25_fix_websocket_api_runs/concurrent_1224_formal_benchmark
```

Source attempt folders:

```text
result/archived_2026-06-25_fix_websocket_api_runs/concurrent_1224_source_tick_factor_runs
```

Summary:

| metric | value |
| --- | ---: |
| total attempts | 104 |
| valid attempts | 80 |
| retries | 24 |
| 1008 | 0 |
| 1011 | 0 |
| elapsed time | about 4.91 min |

Sequential comparison run:

```text
result/archived_2026-06-25_fix_websocket_api_runs/sequential_1137_formal_benchmark
```

Sequential summary:

| metric | value |
| --- | ---: |
| total attempts | 123 |
| valid attempts | 80 |
| retries | 43 |
| 1008 | 0 |
| 1011 | 0 |
| elapsed time | about 34.42 min |

Observed speedup:

```text
34.42 / 4.91 = about 7.0x faster
```

Time saved:

```text
about 29.5 min
```

## Successful Concurrent Cell Summary

From `concurrent_1224_formal_benchmark/summary.csv`:

| condition | latency_ms | total | valid | retries | 1008 | 1011 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| native_no_tick | 3000 | 15 | 10 | 5 | 0 | 0 |
| native_no_tick | 5000 | 10 | 10 | 0 | 0 | 0 |
| native_no_tick | 8000 | 13 | 10 | 3 | 0 | 0 |
| native_no_tick | 12000 | 23 | 10 | 13 | 0 | 0 |
| external_single_tick | 3000 | 11 | 10 | 1 | 0 | 0 |
| external_single_tick | 5000 | 11 | 10 | 1 | 0 | 0 |
| external_single_tick | 8000 | 10 | 10 | 0 | 0 | 0 |
| external_single_tick | 12000 | 11 | 10 | 1 | 0 | 0 |

## Interpretation

The concurrent runner is an execution speedup, not a benchmark-design change. It should be interpreted as producing the same formal benchmark shape faster.

The important implementation safeguards are:

- one WebSocket session per attempt;
- one child process per attempt;
- unique result directories;
- bounded per-cell concurrency;
- formal aggregation still records all attempts, not just valid attempts.

The concurrent run also supports the 1008 fix evidence because it produced more parallel pressure than the sequential run while still showing `0` `1008` and `0` `1011`.

## Caveats

- Parallel runs may change timing pressure and server scheduling compared with sequential runs.
- The exact speedup depends on retry count, latency distribution, network behavior, and API-side scheduling.
- `--cell-concurrency 50` is an upper bound, not the actual number of simultaneous attempts in every cell.
- If the target valid count changes above `10`, actual concurrency can increase accordingly.

