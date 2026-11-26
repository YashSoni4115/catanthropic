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
| bc_policy vs 3 random | 125 | 1.000 | [1.000, 1.000] | [0.992, 1.000] | n/a | 10.04 | 1/1/1/1 | 89.7 | 0 | 10.3 | 49 s |
| bc_policy vs 3 weighted_random | 125 | 0.974 | [0.958, 0.988] | [0.956, 0.985] | 1.10 | 9.96 | .976/.976/.952/.992 | 91.9 | 0 | 10.1 | 49 s |
| bc_policy vs 3 value_function (peeking) | 125 | 0.238 | [0.206, 0.270] | [0.203, 0.277] | 0.74 | 6.78 | .328/.216/.208/.200 | 89.2 | 0 | 4.25 | 118 s |
| value_function (peeking) vs 3 bc_policy | 125 | 0.242 | [0.214, 0.270] | [0.207, 0.281] | 0.56 | 6.94 | .232/.256/.264/.216 | 92.0 | 0 | 4.23 | 118 s |

bc_policy = `bc_v1` (argmax, non-peeking, 116k params), behavior-cloned from value_function on 3000 games (seeds 200000–202999). Run with `--keep-awake`. **Gate (bc_policy vs 3 value_function, CI containing or above 0.20): PASS** (0.238, CI [0.206, 0.270]).

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
| value_function vs 3 bc_policy | bc_policy | 0.253 | [0.243, 0.262] | 0.18 |

## PPO final evaluation (fresh seeds)

`python -m catan_rl.rl.final_eval --ckpts bc,runs/rl/A_noshape/final.pt,runs/rl/A_noshape/ckpt_0300.pt,runs/rl/B_shape/final.pt`

- **Mode:** argmax, non-peeking.
- **Seeds:** the same for every checkpoint and fresh: training-time evals used seeds 0–62.
  - vs value_function: 5000–5249 × 4 rotations = 1,000 games.
  - vs weighted_random and vs bc_policy: 5000–5124 × 4 = 500 games.
- **CIs:** seed-clustered 95%. "Diff vs bc" is the paired per-seed difference on the same seeds.
- **Runs:** A = `--shaping off`, B = `--shaping on` (Φ `phi_v1_pilot:hgb_A_nobelief:88ddfea18f7a`). Both: seed 0, 313 updates × 64 games, `--value-warmup 15`, lr 1e-4, code at 72636c6.

| Checkpoint | Opponents | Win rate | 95% CI | Diff vs bc | 95% CI |
|---|---|---|---|---|---|
| bc | 3 value_function | 0.255 | [0.233, 0.277] | — | — |
| A final (upd 313) | 3 value_function | 0.274 | [0.252, 0.296] | +0.019 | [−0.008, +0.045] |
| A @300 | 3 value_function | 0.284 | [0.261, 0.308] | +0.029 | [+0.002, +0.055] |
| B final (upd 313) | 3 value_function | **0.294** | **[0.271, 0.318]** | **+0.039** | [+0.012, +0.066] |
| bc | 3 weighted_random | 0.986 | [0.974, 0.996] | — | — |
| A final | 3 weighted_random | 0.984 | [0.972, 0.994] | −0.002 | [−0.014, +0.010] |
| A @300 | 3 weighted_random | 0.988 | [0.978, 0.996] | +0.002 | [−0.008, +0.012] |
| B final | 3 weighted_random | 0.978 | [0.964, 0.990] | −0.008 | [−0.022, +0.006] |
| bc | 3 bc_policy (mirror) | 0.250 | [0.250, 0.250] | — | — |
| A final | 3 bc_policy | 0.288 | [0.260, 0.316] | +0.038 | [+0.010, +0.066] |
| A @300 | 3 bc_policy | 0.302 | [0.274, 0.330] | +0.052 | [+0.024, +0.080] |
| B final | 3 bc_policy | 0.290 | [0.256, 0.324] | +0.040 | [+0.006, +0.074] |

**M3 (clustered CI lower bound > 25% vs 3 value_function): PASSED by B final** (0.294, CI lower bound 0.271, seeds 5000–5249).
- Caveat: B final is the best of three PPO checkpoints, and it was picked on these same seeds, so the estimate is slightly optimistic.
- A confirmation on fresh seeds (e.g. 6000–6249) is still pending.

B final is frozen as `checkpoints/ppo_B313.pt` (git-ignored), sha256 `5bc1f8c7d7c6de8c9af358830905b690a3fa26cc6a5fa71c691d00f56a967b7a`, identical to `runs/rl/B_shape/final.pt`.

B2 (continuation from B313, 600 updates, shaping on, Φ `phi_v1_pilot:hgb_A_nobelief:88ddfea18f7a`) is frozen as `checkpoints/ppo_B2_600.pt` (git-ignored, copy of `runs/rl/B2_shape/final.pt`).
- file sha256: `00386f881fadd5dcf311379d675c65f9e8ee66e9fce2645c2c8a1430bcd6dbab`
- parameter digest (printed by train.py as "final params sha256"): `312d250afebf53c7daf0df7cfb6fa0215da5c85e0ccd24ce86c2c99e96906426`. `ckpt_0600.pt` has the same parameter digest but a different file sha, because it stores different metadata.
- training eval (seeds 0–62, which are not fresh): 0.274 at update 0, 0.437 [0.373, 0.500] at update 600. Fresh-seed validation uses seeds 6000–6249.

### B2 fresh-seed validation (Spec V0)

Seeds 6000-6249 (vs_vf) and 6000-6124 (others), x4 rotations, argmax, non-peeking. Outputs: `runs/rl/B2_val/results.json`, `runs/rl/B2_val_ab/results.json`. alpha_beta and value_function are peeking.

| Checkpoint | Opponents | Win rate | 95% CI | Diff vs bc | 95% CI |
|---|---|---|---|---|---|
| bc | 3 value_function | 0.247 | [0.226, 0.269] | — | — |
| B313 | 3 value_function | 0.273 | [0.248, 0.298] | +0.026 | [−0.006, +0.058] |
| B2 | 3 value_function | **0.400** | **[0.368, 0.433]** | +0.153 | [+0.115, +0.192] |
| B313 | 3 bc_policy | 0.284 | [0.250, 0.320] | +0.034 | [+0.000, +0.070] |
| B2 | 3 bc_policy | **0.406** | **[0.362, 0.452]** | +0.156 | [+0.112, +0.202] |
| bc | 3 B313 | 0.220 | [0.190, 0.250] | — | — |
| B2 | 3 B313 | **0.384** | **[0.344, 0.424]** | +0.164 | [+0.114, +0.214] |
| bc | 3 weighted_random | 0.982 | [0.970, 0.992] | — | — |
| B313 | 3 weighted_random | 0.990 | [0.980, 0.998] | +0.008 | [−0.004, +0.020] |
| B2 | 3 weighted_random | 0.992 | [0.984, 0.998] | +0.010 | [−0.002, +0.024] |
| bc | 3 alpha_beta | 0.192 | [0.164, 0.220] | — | — |
| B2 | 3 alpha_beta | **0.366** | **[0.324, 0.408]** | +0.174 | [+0.124, +0.224] |

**V0 gate: REAL IMPROVEMENT, PASS.** B2's CI lower bound is above 0.25 vs alpha_beta (0.324), vs bc_policy (0.362) and vs B313 (0.344). Its value_function CI does not overlap B313's, and there is no regression vs weighted_random.
- These are fresh seeds, so there is no selection bias.
- B313's M3 number shrank on fresh seeds, from 0.294 to 0.273 (CI [0.248, 0.298]). Its lower bound is now below 0.25, so M3 for B313 is not confirmed. B2 now carries M3.
- The odd-profile check is done in Spec V1 below.

### Spec V1: behavior profile, KL to BC (seeds 7000-7099)

100 games each vs 3 value_function, argmax. Outputs: `runs/rl/V1/profile.json`, `runs/rl/V1/kl.json`. These are diagnostics only, so there are no CIs (n=100).

| Ckpt | Win | Mean VP | Turns | Turn-limit | Win-VP share: sett / city / LR / LA / VPcard | Dev buys/game | Dev plays/game | Robber moves/game |
|---|---|---|---|---|---|---|---|---|
| bc | 0.25 | 6.77 | 90.4 | 0.00 | .29 / .57 / .13 / .00 / .02 | 0.60 | 0.38 | 3.81 |
| B313 | 0.31 | 6.90 | 88.5 | 0.00 | .25 / .61 / .13 / .00 / .01 | 0.52 | 0.35 | 3.73 |
| B2 | 0.37 | 7.85 | 88.1 | 0.00 | .23 / .51 / .12 / .05 / .08 | 2.90 | 1.85 | 4.72 |

KL to BC: B2's argmax differs from BC's on 23.6% of its 5,335 decisions.
- dev_buy differs on 89%. Mean BC probability of B2's chosen dev_buy action is 0.11, so this is where B2 has drifted most.
- roll 41%, robber 33%, trade 31%, road 25%, discard 22%.
- settlement 6%, city 7%, play_dev 8%, end_turn 0.5%.

**V1 gate (odd profile = turn-limit > 2% or VP-source skew): CLEAR.**
- Turn-limit is 0% for all three checkpoints.
- B2's VP sources are more diverse than BC's, not skewed: it gained largest army and VP cards and kept cities as the main source.
- The main learned change is about 5x more dev-card buying (0.6 to 2.9 per game), which turns into largest army and VP cards.

### Reserved seed ranges (new)

- **B3 PPO training:** `--seed 2`, giving game seeds 3,000,000-3,999,999. Formula: train_seed_base + seed x 1e6 + n_games. A and B used seed 0, and B2 used seed 1 (2,000,000+).
- **B3 fresh validation:** seeds 8000-8249. Never used before.

### B3 (continuation from B2, 600 updates, seed 2) fresh validation, seeds 8000+

Training settings were the same as B2, `--init-from checkpoints/ppo_B2_600.pt`. Training eval on seeds 0-62 (not fresh): 0.437 at update 0, dropped to 0.333 at update 100, recovered to 0.440 by 400, ended at 0.460 at 600. The early dip followed the KL-to-BC pull restarting at full strength (kl_bc 3.4 to 1.7 within 3 updates).
- `runs/rl/B3_shape/final.pt` file sha256 `a31c55346d2d19c485a5dd9f9def3467340b48bdeae76a2e3dd60f4d22abcf9c`, params sha256 `f15b05701e7b2d495c8562559ffc34c594728112935f126b7ac8d373ccd2a2ab`. Not yet copied to `checkpoints/`.

| Ckpt | Opponents | Win | 95% CI | Diff vs bc | 95% CI |
|---|---|---|---|---|---|
| bc | 3 value_function | 0.229 | — | — | — |
| B2 | 3 value_function | 0.406 | [0.375, 0.436] | +0.177 | [+0.138, +0.217] |
| B3 | 3 value_function | **0.463** | **[0.432, 0.494]** | +0.234 | [+0.196, +0.272] |
| B2 | 3 bc_policy | 0.442 | [0.394, 0.490] | +0.192 | [+0.144, +0.240] |
| B3 | 3 bc_policy | **0.484** | **[0.436, 0.534]** | +0.234 | [+0.186, +0.284] |
| B2 | 3 B313 | 0.372 | [0.330, 0.414] | +0.158 | [+0.106, +0.212] |
| B3 | 3 B313 | 0.444 | [0.396, 0.494] | +0.230 | [+0.174, +0.288] |
| bc | 3 weighted_random | 0.976 | [0.958, 0.992] | — | — |
| B2 | 3 weighted_random | 0.992 | [0.982, 1.000] | +0.016 | [+0.002, +0.032] |
| B3 | 3 weighted_random | 0.994 | [0.986, 1.000] | +0.018 | [+0.006, +0.032] |
| bc | 3 alpha_beta | 0.186 | [0.154, 0.220] | — | — |
| B3 | 3 alpha_beta | **0.408** | **[0.366, 0.452]** | +0.222 | [+0.168, +0.276] |

bc vs 3 value_function is inferred as 0.229 = B2 0.406 minus B2's paired diff +0.177. The paired diff is exact; the 0.229 is derived, not read directly.

Paired B3 minus B2 vs 3 value_function (`eval.run --compare`, seeds 8000-8249, 1,000 games each, seed-clustered): **+0.057 [+0.020, +0.093]**. Re-running eval.run reproduced the final_eval numbers exactly (0.463 and 0.406), so the evals are deterministic. Outputs: `runs/rl/B3_val`, `runs/rl/B3_val_ab`, `runs/rl/B3_paired`.

**B3 gate: PASS.** The paired diff vs B2 has CI lower bound > 0; the vs_bc lower bound 0.436 and vs_ab lower bound 0.366 are both > 0.25; vs_wr 0.994 is >= 0.95.

## Reading the design effect

- **deff is at or below 1 here, not above.** Rotation puts each slot in every turn-order position once per seed. This stratifies out the seat and board advantage, so per-seed win counts vary less than independent games would.
- **Pooled opponents have a much lower deff (0.35–0.42).** Within a seed, one opponent's win is another's loss, so their results are negatively correlated.
- **random ×4 is degenerate.** Identical agents replay the same game in all 4 rotations, so each slot wins exactly 1 of 4 per seed. The exception is one seed that hit the turn limit without a winner (4 no-winner games). This is why the clustered CI collapses to about 0.25 and deff ≈ 0. The Wilson interval [0.212, 0.288] is meaningless for this lineup.
- **Positive correlation would show up.** A test with fully correlated clusters (all 4 games of a seed share the outcome) gives deff ≈ 4.
- **Use the clustered CI.** Wilson is valid only when deff ≈ 1.

Paired lineup comparison: `python -m catan_rl.eval.run --compare DIR_A DIR_B [--slot 0]`. This computes the per-seed win-rate difference and resamples seeds jointly, keeping each seed's games from both lineups together.
