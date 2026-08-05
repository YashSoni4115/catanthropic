# Vector Shape Report — schema_version=1

> Generated after `vec_v1.2.0` integration (adds `robber` + `strategic` segments).
> Vectorizer: `StateVectorizerV1` (VERSION `vec_v1.2.0`)
> Economy flattener: `economy_flat_v1.2.0`

---

## 1. Summary

| Metric | Value |
|--------|-------|
| Base vector length (no economy) | **2 614** |
| Economy section length | **91** |
| Robber section length | **11** |
| Strategic section length | **188** |
| Total vector length (with economy) | **2 904** |
| Stable across 2 / 3 / 4 players | ✅ Yes |
| Schema hash (economy-enabled) | `d50806f5e5b62cb3…` |

When `economy_features_enabled=True`, the vectorizer appends three fixed-width
segments in this order: **`economy` → `robber` → `strategic`**. All segments
remain fixed-width regardless of player count via padding to
`MAX_OPPONENTS_SCHEMA_V1 = 3` (→ `SLOTS = 4` in the strategic segment).

---

## 2. Group Layout

| # | Group | Start | End | Length |
|---|-------|------:|----:|-------:|
| 0 | `global` | 0 | 8 | **8** |
| 1 | `tile_slots` | 8 | 547 | **539** |
| 2 | `structure_slots` | 547 | 1 429 | **882** |
| 3 | `road_slots` | 1 429 | 2 458 | **1 029** |
| 4 | `players` | 2 458 | 2 582 | **124** |
| 5 | `engineered` | 2 582 | 2 614 | **32** |
| 6 | `economy` *(optional)* | 2 614 | 2 705 | **91** |
| 7 | `robber` *(optional)* | 2 705 | 2 716 | **11** |
| 8 | `strategic` *(optional)* | 2 716 | 2 904 | **188** |

Groups 0–5 are always present. Groups 6–8 are appended together when
`economy_features_enabled=True` (single toggle for backward compatibility).

---

## 3. Economy Feature Inventory (91 features)

Features are appended in **sorted key order** by the vectorizer.

### 3.1 `board.*` — 22 features

Board-level resource environment plus opening quality.

| Feature | Description |
|---------|-------------|
| `board.{R}.pip_total` | Sum of pips across all tiles of resource R |
| `board.{R}.scarcity_score` | Normalized scarcity (0 = abundant, 1 = very scarce) |
| `board.{R}.abundance_score` | Normalized abundance (0 = scarce, 1 = very abundant) |
| `board.{R}.concentration_score` | Spatial clustering of resource R |
| `board.opening.best_ev` | Expected-value score of the best opening candidate |
| `board.opening.best_balance` | Resource-balance score of the best opening candidate |

Where `{R}` ∈ {`BRICK`, `GRAIN`, `LUMBER`, `ORE`, `WOOL`} → 5 × 4 + 2 = **22**.

### 3.2 `self.*` — 54 features

Viewer-relative production, position, planning, expansion, pressure, and trade.

| Group | Features | Count |
|-------|----------|------:|
| Production per resource | `self.{R}.expected`, `.rank`, `.share`, `.leader_gap` | 20 |
| Production aggregates | `expected_total`, `total_rank`, `total_share`, `total_leader_gap`, `balance_score`, `diversity_score`, `blocked_total` | 7 |
| Port & robber | `port_leverage_score`, `robber_burden_score` | 2 |
| Expansion summary | `expansion_score` | 1 |
| Expansion detail | `expansion.reachable_site_count`, `.frontier_score`, `.top_site_score`, `.city_upside_score`, `.boxed_in_score` | 5 |
| Build plan readiness | `self.plan.{B}` for `road`, `settlement`, `city`, `dev` | 4 |
| Competitive pressure | `pressure.leader`, `.race`, `.block_threat`, `.contested_frontier`, `.resource_competition` | 5 |
| Trade need | `self.trade_need.{R}` | 5 |
| Trade surplus | `self.trade_surplus.{R}` | 5 |
| | | **54** |

### 3.3 `belief.*` — 15 features

Opponent resource-holding beliefs (padded to 3 opponent slots).

| Feature | Description |
|---------|-------------|
| `belief.opp{i}.{R}.p_at_least_one` | P(opponent *i* holds ≥ 1 of resource R) |

Where `{i}` ∈ {0, 1, 2} and `{R}` ∈ 5 resources → 3 × 5 = **15**.

When belief data is unavailable (e.g. `include_belief=False`), all values
default to `0.0`. The `belief_unavailable` flag in the economy state signals
downstream consumers, but is not included in the flat vector.

---

## 4. Cross-Player Stability

Probe results across 2-, 3-, and 4-player games:

```
  2-player: economy_len=91  total=2705  hash=3aee4da9b578324e...
  3-player: economy_len=91  total=2705  hash=3aee4da9b578324e...
  4-player: economy_len=91  total=2705  hash=3aee4da9b578324e...
```

Schema hash, economy length, and total vector length are **identical** across
all supported player counts.

---

## 5. Stale Constant Audit

| Location | Value | Status |
|----------|------:|--------|
| `ai/models/value_model_baseline_v1/split_summary.json` `vector_length` | 2 614 | ✅ Correct — baseline model (no economy) |
| `ai/models/value_model_economy_v1/split_summary.json` `vector_length` | 2 703 | ⚠️ **Stale** — old economy schema (89 features). New vectors are 2 705 (91 features). Model must be retrained. |
| Python code constants | — | ✅ No hardcoded vector-length constants found in `.py` files |

The `opening_value_evaluator_v1.py` reads `vector_length` from
`split_summary.json` at runtime and logs a mismatch warning when the produced
vector does not match. This is **correct** defensive behavior—the old model
artifact accurately records what the old model was trained on, and the warning
surfaces the need to retrain.

---

## 6. Delta from Previous Economy Schema

| Property | v1.1.x (old) | v1.2.0 (current) | Change |
|----------|-------------:|------------------:|--------|
| Economy feature count | 89 | 91 | **+2** |
| Total vector length | 2 703 | 2 705 | **+2** |
| Belief features | 10 (2 slots × 5) or variable | 15 (3 slots × 5, fixed) | Fixed-width padding |
| Opening features | conditional | always emitted | FIX 3 |
| Schema hash | varies | `3aee4da9…` (stable) | Deterministic |

The +2 comes from belief padding: the old schema emitted 2 opponent slots for a
3-player game (10 belief features), while v1.2.0 always emits 3 slots
(15 belief features), adding 5 new features but removing the variable-length
behavior.  The net effect is +2 because a 3-player game previously had 10 belief
features and now has 15, while the opening features that were conditionally
emitted are now always present (no count change for the common 3-player case,
but stabilizes 2-player and 4-player).

---

## 7. Test Coverage

All 74 tests pass after these changes:

```
ai/tests/ — 74 passed in 0.96s
```

Key shape-related test assertions:
- `test_vector_length_and_groups` — base length = 2 614
- `test_optional_economy_features_append_group` — economy group appended, `len > 2614`
- `test_vector_length_stable_with_and_without_opening` — opening features always emitted
- `test_three_opponent_vector_same_length_as_one_opponent` — belief padding stable
- `test_one_opponent_produces_max_slots` — all 3 belief slots present

---

*Report produced by `ai/tests/_vector_shape_probe.py`.*

---

## 8. `robber` segment (11 features) — new in vec_v1.2.0

Output of the `robber_hotspot_v1` adapter (`get_robber_hotspot_features`).
The vectorizer computes tile probabilities using a pip-weighted heuristic
(desert / current robber tile zeroed, remainder normalized) so no trained
model file is required at vectorization time. Callers may pass `model=…` to
substitute the trained sub-model.

| Idx | Feature |
|----:|---------|
| 0 | `robber.top1_tile_id` |
| 1 | `robber.top1_prob` |
| 2 | `robber.top3_entropy` |
| 3–5 | `robber.vulnerability_score.opp{0,1,2}` |
| 6–10 | `robber.expected_block_score.{BRICK,GRAIN,LUMBER,ORE,WOOL}` |

---

## 9. `strategic` segment (188 features) — new in vec_v1.2.0

Emitted by `ai/economy/strategic_features.py`. All per-player features use
**`SLOTS = 1 + MAX_OPPONENTS_SCHEMA_V1 = 4`** — active player at index 0,
opponents 1..3 sorted alphabetically by `player_id`, padded with `0.0`.

Flat order: **category alphabetical → field alphabetical → slot index**.
Names: `strategic.{category}.{field}.{i}`.

| Category | Dim | Content |
|----------|----:|---------|
| `board_control` | 20 | 5 fields × 4 slots: `expansion_space`, `blocked_expansion`, `city_upgrade_candidates`, `pip_ownership_share`, `board_centrality` |
| `dev_cards` | 28 | 7 fields × 4 slots: `total_bought`, `total_played`, `est_unplayed`, `p_road_building`, `p_year_of_plenty`, `p_monopoly`, `knight_to_dev_ratio` |
| `game_phase` | 13 | `turn_norm` (1) + `phase_one_hot` (3) + `settlements_remaining` (4) + `cities_remaining` (4) + `initial_placement_complete` (1) |
| `largest_army` | 21 | 3 per-slot fields × 4 + `held_one_hot` (5, incl. `unowned`) + `within_1_stealing` (4) |
| `longest_road` | 21 | 3 per-slot fields × 4 + `held_one_hot` (5) + `within_2_stealing` (4) |
| `robber_threat` | 20 | 5 fields × 4 slots: `aggression`, `victim_score`, `vulnerability`, `steal_ev`, `last_mover_one_hot` |
| `trade_ports` | 41 | 5 × `port_2to1_{R}` (4) + `port_3to1` (4) + `trade_activity_rate` (4) + `maritime_ratio` (4) + `surplus_score` (4) + `monopoly_risk` (5, per resource) |
| `vp_trajectory` | 24 | 6 fields × 4 slots: `vp_norm`, `hidden_vp`, `vp_gap_to_10`, `vp_rate_change`, `at_9_vp`, `city_conversion_rate` |
| **TOTAL** | **188** | |

`validate_strategic_schema(features)` enforces the declared dimensions.

### 9.1 History-dependent fields

Fields that require cross-turn history emit `0.0` when `history` isn't
supplied (safe zero-fill; no shape change):

- `robber_threat.aggression`, `robber_threat.victim_score`, `robber_threat.last_mover_one_hot`
- `vp_trajectory.vp_rate_change`
- `trade_ports.trade_activity_rate`, `trade_ports.maritime_ratio`

Downstream callers may pass `build_strategic_features(observed_state, history=…)`.

---

## 10. Delta vs. vec_v1.1.0

| Property | vec_v1.1.0 | vec_v1.2.0 | Δ |
|----------|-----------:|-----------:|---:|
| Baseline length | 2 614 | 2 614 | — |
| Economy length | 91 | 91 | — |
| Robber length | — | **11** | +11 |
| Strategic length | — | **188** | +188 |
| **Total (economy enabled)** | 2 705 | **2 904** | **+199** |

### Stale-artifact audit

- `ai/models/value_model_economy_v1/split_summary.json` — old
  `vector_length: 2705` is stale until retrained on vec_v1.2.0. Training
  pipeline auto-detects vector length from record shape and overwrites the
  file, so retraining alone is sufficient.
- `ai/models/value_model_baseline_v1/` — trained on base 2 614 (unchanged).

---

## 11. Test coverage after vec_v1.2.0

```
ai/tests/ — 100 passed in 1.06s
```

New coverage in `ai/tests/test_strategic_features_v1.py` (17 tests):
- `TestStrategicFeatures` (11): dim = 188, `SLOTS = 4`, all categories
  present, `validate_strategic_schema` pass/fail, flatten length matches,
  per-player padding, owner one-hots length = `SLOTS + 1`, stability across
  2/3/4 players, scaffold ≡ real name equality, no omniscient leakage on
  hidden dev cards, `at_9_vp` flag fires correctly.
- `TestVectorizerIntegration` (5): reads `vec.feature_groups`; asserts
  `robber` and `strategic` groups appear only when
  `economy_features_enabled=True`, with correct dims (11 / 188), and total
  length stable across player counts.
