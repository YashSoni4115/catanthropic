# Baselines

`python -m catan_rl.eval.run --agents <lineup> --seeds 125 --workers 8 --keep-awake --out runs/baselines/<lineup>`

## Setup

- **Games:** seeds 0–124 × 4 rotations = 500 games per lineup. Every lineup uses the same seeds, so board, turn order, dev deck and dice are identical across lineups.
- **Provenance** (from each `metrics.json`):
  - fork `rng-streams` at 4732786, clean, base ecf9311.
  - Python 3.12.15; `PYTHONHASHSEED` unset.
  - Games no longer depend on `PYTHONHASHSEED`: fixed in the fork and tested.
- **Clustered CI:** a 95% percentile bootstrap (10,000 resamples) that resamples whole seeds, keeping each seed's 4 games together. Wilson is shown only as a reference.
- **Design effect (deff):** clustered bootstrap variance ÷ binomial variance p(1−p)/500. Effective n = 500 / deff. "n/a" means p is 0 or 1.
- **Peeking:** "(peeking)" bots read the true hidden state.
- **Timing:** 8 workers on macOS, run with `--keep-awake`. In-process time and wall-clock time agreed within 0.1 s for every lineup.

## Slot 0 (first agent in lineup)

| Lineup | n_seeds | Win rate | Clustered 95% CI | Wilson (ref) | deff | Mean VP | Win by position 0/1/2/3 | Mean turns | No-winner | games/s | Time |
|---|---|---|---|---|---|---|---|---|---|---|---|
| random ×4 | 125 | 0.248 | [0.244, 0.250] | [0.212, 0.288] | 0.01 | 5.88 | .312/.208/.224/.248 | 328.6 | 4 | 300 | 1.7 s |
| weighted_random vs 3 random | 125 | 0.452 | [0.414, 0.490] | [0.409, 0.496] | 0.76 | 7.20 | .392/.472/.456/.488 | 283.2 | 0 | 311 | 1.6 s |
| value_function (peeking) vs 3 random | 125 | 0.998 | [0.994, 1.000] | [0.989, 1.000] | 1.00 | 10.03 | 1/1/1/.992 | 89.5 | 0 | 34.8 | 14.4 s |
| alpha_beta (peeking) vs 3 random | 125 | 0.998 | [0.994, 1.000] | [0.989, 1.000] | 1.00 | 10.03 | 1/1/.992/1 | 85.6 | 0 | 1.67 | 299 s |
| alpha_beta (peeking) vs 3 value_function (peeking) | 125 | 0.330 | [0.292, 0.368] | [0.290, 0.372] | 0.83 | 7.28 | .432/.240/.336/.312 | 89.2 | 0 | 0.91 | 549 s |
| phi_greedy vs 3 random | 125 | 0.988 | [0.978, 0.996] | [0.974, 0.994] | 0.97 | 10.09 | .984/.992/.984/.992 | 137.4 | 0 | 5.09 | 98 s |
| phi_greedy vs 3 weighted_random | 125 | 0.934 | [0.910, 0.956] | [0.909, 0.953] | 1.17 | 9.93 | .928/.944/.952/.912 | 135.4 | 0 | 4.46 | 112 s |
| phi_greedy vs 3 value_function (peeking) | 125 | 0.094 | [0.072, 0.118] | [0.071, 0.123] | 0.85 | 5.95 | .096/.072/.120/.088 | 96.8 | 0 | 4.24 | 118 s |
| value_function (peeking) vs 3 phi_greedy | 125 | 0.548 | [0.502, 0.592] | [0.504, 0.591] | 1.09 | 8.65 | .552/.576/.592/.472 | 109.9 | 0 | 1.71 | 293 s |

phi_greedy = one-ply expectimax on Φ `phi_v1_pilot:hgb_A_nobelief:88ddfea18f7a` over one belief determinization per decision (non-peeking). These four runs were made without `--keep-awake` (under `caffeinate`).

**Milestone gates (phi_greedy):**
- M1 ≥ 90% vs random: **PASS** (0.988, CI [0.978, 0.996]).
- M2 ≥ 60% vs weighted_random: **PASS** (0.934, CI [0.910, 0.956]).
- M3 clustered CI lower bound > 25% vs value_function: **FAIL** (0.094, CI [0.072, 0.118]; pooled phi_greedy vs 1 value_function: 0.151, CI [0.136, 0.166]).

## Opponents, pooled per agent (clustered)

| Lineup | Agent | Win rate | Clustered 95% CI | deff |
|---|---|---|---|---|
| weighted_random vs 3 random | random | 0.183 | [0.170, 0.195] | 0.42 |
| value_function vs 3 random | random | 0.001 | [0.000, 0.002] | 0.99 |
| alpha_beta vs 3 random | random | 0.001 | [0.000, 0.002] | 1.00 |
| alpha_beta vs 3 value_function | value_function | 0.223 | [0.211, 0.236] | 0.35 |
| value_function vs 3 phi_greedy | phi_greedy | 0.151 | [0.136, 0.166] | 0.70 |

## Reading the design effect

- **deff is at or below 1 here, not above.** Rotation puts each slot in every turn-order position once per seed. This stratifies out the seat and board advantage, so per-seed win counts vary less than independent games would.
- **Pooled opponents have a much lower deff (0.35–0.42).** Within a seed, one opponent's win is another's loss, so their results are negatively correlated.
- **random ×4 is degenerate.** Identical agents replay the same game in all 4 rotations, so each slot wins exactly 1 of 4 per seed. The exception is one seed that hit the turn limit without a winner (4 no-winner games). This is why the clustered CI collapses to about 0.25 and deff ≈ 0. The Wilson interval [0.212, 0.288] is meaningless for this lineup.
- **Positive correlation would show up.** A test with fully correlated clusters (all 4 games of a seed share the outcome) gives deff ≈ 4.
- **Use the clustered CI.** Wilson is valid only when deff ≈ 1.

Paired lineup comparison: `python -m catan_rl.eval.run --compare DIR_A DIR_B [--slot 0]`. This computes the per-seed win-rate difference and resamples seeds jointly, keeping each seed's games from both lineups together.
