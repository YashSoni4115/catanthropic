# Win probability v2 — pilot gate

Data: pilot (3000 games, 30,000 states, N=16 weighted_random rollouts, soft 4-way labels, viewer-first).
- Splits are by game: 2428 train / 292 val / 280 test games.
- No rollout ended without a winner (no-winner fraction 0.0), so no states were dropped.
- All CIs are 95% game-clustered bootstrap (resample test games).
- Models are selected on val; numbers below are on the 2800 test states.
- Excess = cross-entropy (CE) − entropy of the soft label (the label's own entropy is the floor).

Commands: `python -m catan_rl.winprob.train`, then `python -m catan_rl.winprob.eval`. Outputs go to `runs/winprob/{models,eval}` (git-ignored); the reliability plot is `runs/winprob/eval/reliability.png`.

## Model comparison (test, raw → after temperature scaling on val)

| Model | Features | CE [95% CI] | Excess | CE − VP-only | ECE raw → cal | Truth MAE raw → cal |
|---|---|---|---|---|---|---|
| uniform | – | 1.3863 | 0.877 | +0.463 | 0.110 | 0.313 |
| VP-only logistic | public VP + diffs | 0.9230 [0.886, 0.959] | 0.413 | 0 | 0.021 → 0.021 | 0.178 |
| value_function softmax (T=3.5e14, val-fit) | – | 0.9285 [0.893, 0.964] | 0.419 | +0.0055 [0.002, 0.009] | 0.042 → 0.041 | 0.174 |
| HGB | A without beliefs | 0.6611 [0.618, 0.703] | 0.151 | −0.262 [−0.287, −0.239] | 0.014 → 0.010 | 0.073 → 0.073 |
| HGB | A + independent beliefs | 0.6586 [0.616, 0.701] | 0.149 | −0.264 [−0.290, −0.240] | 0.015 → 0.010 | 0.071 → 0.072 |
| HGB | A + joint beliefs | **0.6583** [0.615, 0.700] | **0.149** | −0.265 [−0.290, −0.240] | 0.016 → 0.011 | 0.071 → 0.070 |
| HGB (selected on val) | A + joint beliefs + B | 0.6595 [0.617, 0.701] | 0.150 | −0.264 [−0.289, −0.239] | 0.018 → **0.012** | 0.070 → **0.0695** [0.058, 0.082] |
| MLP | A without beliefs | 0.6775 | 0.168 | −0.246 | 0.010 → 0.008 | 0.073 → 0.075 |
| MLP | A + independent beliefs | 0.6798 | 0.170 | −0.243 | 0.008 → 0.005 | 0.081 → 0.082 |
| MLP | A + joint beliefs | 0.6803 | 0.171 | −0.243 | 0.010 → 0.007 | 0.080 → 0.081 |
| MLP | A + joint beliefs + B | 0.7524 | 0.243 | −0.171 | 0.030 → 0.012 | 0.109 → 0.111 |

Fitted temperatures are 1.06–1.09 for HGB and 1.07–1.18 for MLP; both models are slightly overconfident.

### Feature-set differences (test CE, paired, game-clustered)

| Comparison | ΔCE [95% CI] |
|---|---|
| HGB joint − HGB no-belief | −0.0028 [−0.0054, −0.0003] |
| HGB indep − HGB no-belief | −0.0025 [−0.0053, +0.0001] |
| HGB indep − HGB joint | +0.0003 [−0.0015, +0.0020] |
| HGB joint+B − HGB joint | +0.0013 [−0.0018, +0.0044] |
| MLP joint+B − MLP joint | +0.072 [0.058, 0.087] (MLP overfits B: best epoch 6) |

Reading the differences:
- **Belief features:** they help only slightly. The gain is about 0.003 nats, and the CI excludes zero only for the joint version.
- **Independent vs joint marginals:** they are indistinguishable at this sample size.
- **Block B (`create_sample`):** it adds nothing to HGB and hurts the MLP.

### Selected model (`hgb_A_joint+B`, calibrated) by phase

| Phase | CE | Excess | CE − VP-only | ECE | Truth MAE |
|---|---|---|---|---|---|
| early | 0.925 | 0.179 | −0.368 [−0.405, −0.335] | 0.011 | 0.065 [0.049, 0.084] |
| mid | 0.627 | 0.150 | −0.275 [−0.311, −0.241] | 0.016 | 0.075 [0.059, 0.093] |
| late | 0.433 | 0.116 | −0.148 [−0.178, −0.119] | 0.018 | 0.066 [0.046, 0.093] |

## Sanity checks
- **Probabilities sum to 1:** this is asserted in `Model.predict` and in `eval`.
- **Only features A and B are used:** every model's column list is a subset of `names_a()` ∪ the `create_sample` names. This is asserted in the constructor, at eval time, and in the tests.
- **Monotone in VP lead:** mean predicted P(viewer wins) rises monotonically with the viewer's public-VP lead over the best opponent (test set).

| Lead | −7 | −6 | −5 | −4 | −3 | −2 | −1 | 0 | +1 | +2 | +3 | +4 | +5 | +6 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P̂ | .002 | .008 | .015 | .042 | .091 | .147 | .227 | .367 | .628 | .804 | .893 | .970 | .989 | .995 |

- **Reliability:** calibrated reliability bins have at most 0.05 gap between mean prediction and mean label; the largest gaps are in the 0.6–0.9 range, where the model is overconfident by about 0.04.

## Truth set and early-game check
- **Truth set:** 300 test states labelled with N=200 weighted_random rollouts.
  - Truth-label sampling error is 0.021 RMS, so even a perfect model would show a MAE of about 0.017.
  - The N=16 training labels themselves score a MAE of 0.047 [0.040, 0.054] against the truth. The model (0.0695) is worse than its own labels on these states. More data or more capacity should close part of that gap.
- **Early-game check:** 150 early truth-pool test states, each labelled with weighted_random (N=200) and value_function (N=64).
  - Model vs weighted_random truth: MAE 0.074 [0.061, 0.088].
  - Model vs value_function labels: MAE 0.134 [0.113, 0.156].
  - weighted_random truth vs value_function labels: MAE 0.116 [0.096, 0.137].
  - Mean P(viewer wins) is 0.287 under weighted_random and 0.305 under value_function.
  - The N=64 value_function labels alone contribute about 0.04 MAE of noise. The remaining roughly 0.08 is a real disagreement between the two policies in the early game.
  - The model matches its training policy (weighted_random) but not value_function play. If the target is strong play, early-game labels should come from a stronger rollout policy.

## Gate

| Criterion | Threshold | Actual | Result |
|---|---|---|---|
| Best model beats VP-only (CE difference CI excludes 0) | CI upper < 0 | −0.264 [−0.289, −0.239] | PASS |
| ECE after calibration | ≤ 0.03 | 0.012 [0.009, 0.020] | PASS |
| Truth-set MAE | ≤ 0.07 | 0.0695 [0.058, 0.082] | PASS, barely; the CI straddles the threshold |

**Overall: PASS** on point estimates.

How each threshold looks against the actual numbers:
- **ECE ≤ 0.03 is too loose.** Every HGB and MLP model passes, even uncalibrated. Something like 0.015 would actually separate models.
- **MAE ≤ 0.07 is borderline.**
  - The noise floor is about 0.017.
  - The model's own N=16 labels reach 0.047, so 0.05–0.06 is a realistic target after the full dataset.
  - The current 0.0695 rests on only 300 truth states, so its CI ([0.058, 0.082]) is wide.
  - Extending the truth set to 1000 states would tighten it.
- **"Beats VP-only" is easy** (0.26 nats). A stronger reference would be the no-belief HGB, which tests whether beliefs earn their cost.

## Next (not run)
- Full dataset, 12,000 games:
  `.venv-rl/bin/python -m catan_rl.winprob.build_dataset --games 12000 --n 16 --policy weighted_random --workers 8 --keep-awake --out runs/winprob/full`
- Extend the truth set to 1000 test states:
  `.venv-rl/bin/python -m catan_rl.winprob.label_study --data runs/winprob/pilot --truth 1000 --truth-n 200 --policy weighted_random --workers 8 --keep-awake --out runs/winprob/label_study`
  - The truth pool is drawn from the pilot test split; its first 300 states are the current truth set, and their labels are reproduced exactly.
