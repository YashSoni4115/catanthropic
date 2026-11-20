# robber_hotspot_v1

A standalone sub-model that predicts **P(tile T receives the robber on the
next placement)** as a distribution over the 19 board tiles. The trained
model's output feeds into the value-model / win-probability pipeline through
a fixed-length feature adapter.

## What it predicts

Given an ``observed_state`` (schema_version=1, no omniscient / hidden data),
the model outputs a length-19 probability distribution over tile indices in
canonical ``(x, y)`` order.

## Input features (424 dims)

Per tile (22 features × 19 tiles = 418), plus 6 global-context features.
See `feature_schema.json` for the full ordered list.

| Group | Features per tile |
|-------|-------------------|
| Static (9) | `pip_weight`, `is_desert`, `has_robber`, `resource.{BRICK,GRAIN,LUMBER,ORE,WOOL,DESERT}` |
| Structural (5) | `settlement_count`, `city_count`, `distinct_player_count`, `expected_output_per_turn`, `active_player_has_structure` |
| Opponent (4) | `vp_weighted_presence`, `belief_resource_wealth`, `leader_is_present`, `threat_player_present` |
| Historical (2) | `times_robbed_this_game`, `active_player_targeted_tile_before` |
| Strategic (2) | `would_block_bottleneck`, `steal_resolves_active_bottleneck` |

Global context (6): `turn_norm`, `active_vp`, `active_knight_count`,
`phase_early`, `phase_mid`, `phase_late`.

## Label extraction

`iter_robber_placements()` in `ai/train_robber_hotspot_v1.py` scans the
extracted self-play JSONL and detects **consecutive-row robber transitions**
per game. When `observed_state.robber` changes between rows, the *previous*
row's `observed_state` is the input and the *current* row's `(x, y)` is
resolved to a tile index (0-18). Filtered out: initial desert placement,
no-op moves, target tiles missing from the previous state.

## Output adapter (for downstream consumption)

```python
from ai.economy.robber_hotspot_features import get_robber_hotspot_features
features = get_robber_hotspot_features(observed_state, model=hgb_model)
```

Returns a fixed-length dict with 11 features (`ADAPTER_DIM`):

| Feature | Meaning |
|---------|---------|
| `robber.top1_tile_id` | Index of tile with highest predicted probability |
| `robber.top1_prob` | Probability of that tile |
| `robber.top3_entropy` | Entropy over the top-3 tiles (concentration signal) |
| `robber.vulnerability_score.opp{0,1,2}` | Per-opponent exposure — sum of P(robber → T) over tiles owner has structures on, weighted by city=2 / settlement=1. Padded to **MAX_OPPONENTS_SCHEMA_V1 = 3**. |
| `robber.expected_block_score.{BRICK,GRAIN,LUMBER,ORE,WOOL}` | Expected production lost to the robber per resource type |

Flatten to a length-11 vector with `flatten_adapter_features(features)`.

## Model

- **Primary**: `HistGradientBoostingClassifier` (multiclass, 19 classes)
- **Baseline (secondary)**: `LogisticRegression` (multinomial)
- **Uniform baseline**: 1/19 over all tiles — reported in `metrics.json`

Split-by-game (80/10/10) with a deterministic md5-hash on `game_id`. No
row-level leakage is possible.

## Artifacts

| File | Purpose |
|------|---------|
| `model_hgb.pkl` | Trained HGB classifier |
| `model_logreg.pkl` | Trained LogReg classifier |
| `split_summary.json` | Row/game counts per split, `vector_length` (424), `num_classes` (19), top-1 accuracy per split |
| `metrics.json` | Top-1 acc, top-3 acc, log-loss per model per split, plus uniform baseline |
| `feature_schema.json` | Ordered list of 424 feature names |
| `config.json` | Training config snapshot |

## Retraining

```
python -m ai.train_robber_hotspot_v1 \
    --input ai/selfplay/data/selfplay_guided_1000games_v3_extracted.jsonl \
    --output-dir ai/models/robber_hotspot_v1 \
    --split-ratios 80,10,10 \
    --seed 42
```

## Design notes

- **observed_state only** — never reads `omniscient_state` or hidden cards.
- **Fixed-length outputs** — belief-style opponent padding keeps the adapter
  vector length constant across 2/3/4-player games.
- **Non-invasive** — does not modify `schema_version=1` vectors or
  `economy_flat_v1.2.0`. The adapter is a separate appendable segment.
