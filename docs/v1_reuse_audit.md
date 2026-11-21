# V1 Reuse Audit

Scope: read-only audit of `v1/` (lowercase in the repo; the request says `V1/`). Nothing in `v1/` was changed.

Line numbers reference `v1/` at commit `b4cf156`.

**Bottom line:** The V1 Java engine is a *simplified, partly stubbed* Catan:
- Dev cards are never actually drawn.
- There are no steals, no player trades, no longest road and no dev-card plays.
- Discards are random.
- It is not reproducible from a seed.

It cannot serve as the RL environment without major work. Rewrite the engine and the bridge, and port selected Python pieces: the belief tracker, the observed-state redaction pattern, and the vectorizer slot layout.

`catan_rl/` does not exist in this workspace yet, so Part B4 can't be compared (see B4).

---

## Part A: Foundation

### A1. Bridge: `v1/ai/bridge/`

| File | Main symbols | What it does | V1 deps | Tests |
|---|---|---|---|---|
| `game_bridge_v1.py` | `GameBridge` (L158), `.play` (L192), `_handle_setup` (L259), `_handle_request` (L323), `action_to_response` (L55), `action_from_response` (L78), `observed_state_from_request` (L113), `build_default_scorer` (L400) | Spawns the JVM and runs the JSON-lines callback loop | `board_state_extractor`, `decision.*`, `setup_chooser`, value model pickle | `test_game_bridge_v1.py` (10) |
| `play_game_v1.py` | CLI `main` | Plays N games, decision layer vs Java bots | `game_bridge_v1` | none |
| `setup_chooser.py` | `choose_setup_settlement` (L34), `choose_setup_road` (L64) | Rule-based opening placement | `economy.board.openings`, `economy.constants` | indirect |
| Java side: `v1/src/game/SelfPlayGameRunner.java` | `ExternalPolicyClient` (L55-75), `requestDecision` (L859-881), `sendError` (L883-889), snapshot builder (L1817-2045) | Engine side of the protocol | — | none |

**Protocol**
- One JSON object per line over stdin/stdout.
- Java owns the game loop: it sends `decision_request` and blocks on `readLine()` until Python answers.

| Direction | Message | Fields |
|---|---|---|
| J→P | `decision_request` | `request_id, turn, phase ∈ {setup_settlement, setup_road, main, robber}, player, legal_action_kinds[], snapshot_payload{…}` |
| J→P | `error` | `message` (e.g. `illegal_action_kind:…`, `bad_setup_vertex`) |
| J→P | `game_over` | winner, turns, termination reason |
| J→P | (after game_over) | full per-turn trajectory rows |
| P→J | `decision_response` | `request_id, action_kind` + params (`vertex_*`, `edge_*`, `tile_*`; `trade_bank` give/get are **ignored** by Java) |

**Process and error handling**
- **Launch:** `java -cp src game.SelfPlayGameRunner --external-policy --ai-player N --max-turns M [--seed S]`, with cwd `v1/`.
- **No compile step in the bridge.** `javac` is run separately (by the selfplay scripts or `run.sh`).
- **Parsing:** Java parses responses by naive string scanning, not a JSON parser, and never checks `request_id`.
- **Bad responses:** a bad or illegal response gets an `error` line, then the engine **silently falls back to a random policy** for that decision. That hides agent bugs.
- **Timeouts:** `read_timeout_s` (L173) is stored but never used, so `readline()` can block forever.
- **stderr deadlock risk:** stderr is piped but never drained.
- **Per-turn cap:** the main phase is capped at 10 actions per turn (L800).

**Reset/step:** No reset. One JVM = one game, and there is no `step()` API (callback-driven). A gym-style env would need inversion of control.

**Seeding: broken.**
- `--seed` only seeds the runner's `Random` (L234).
- These sources are unseeded:
  - Board layout: `Board.java:46` `Collections.shuffle`
  - Dice: `Game.java:86` `Math.random()`
  - Dev deck: `Deck.java:32`
  - Seat order: `Game.java:35`
- **Measured:** two runs with `--seed 7` gave different output hashes.
- No dice-seed control.

**Throughput:**
- One flush per message.
- Every request carries a full board snapshot, built as a string.
- JVM startup cost for every game.
- No sleeps.

**Verdict: REWRITE.**
- Keep only the idea: a JSON-lines protocol with `legal_action_kinds` and phases.
- The new bridge needs:
  - a step/reset API
  - full seeding, including dice
  - parameterized actions that are actually honored
  - errors on illegal actions (no silent random fallback)
  - timeouts
  - per-seat agents
  - an event stream
  - a strict split between observed and full state on the wire

### A2. Java engine entry points: `v1/src/`

| Item | Finding |
|---|---|
| Headless entry | `game.SelfPlayGameRunner.main` (L231). Players P0-P3; snake setup (L518-581); main loop L278-399 |
| GUI entry | `game.GameRunner` → `gui.GameWindow` (used by `run.sh`) |
| Game creation | `new Game(players)`, which **shuffles the seat list** (`Game.java:35`). **Measured:** `--ai-player 0` played as `P3` |
| Seat swapping | Only **one** external seat. The other three are the built-in bot. There is no per-seat agent plug-in |
| Built-in bot | `SelfPlayGameRunner` L1038-1107: fixed priority city > settlement > road > bank trade, otherwise weighted random. Optional `--guided-policy`/`--opening-policy value_model` spawn Python subprocesses |
| Headless/fast | Yes: pure Java, no GUI. **Measured:** about 9 games/s single-threaded with all bots (5 games in 0.54 s, including 5 JVM starts) |
| Rules gaps | Dev buy takes resources but **assigns no card**. No steal, no longest road (`Game.java:362` TODO). Bank trade legality only checks "≥4 of a resource" and doesn't use ports. NPC trade validity is a TODO (`Game.java:272`). Discards are random |
| Size | About 7.8k LOC Java (`SelfPlayGameRunner` 2189, `Board` 872, `Game` 502, `Player` 417, GUI about 2.7k) |

**Verdict: REWRITE.**
- `Board.java` is the only piece worth mining: its hex/vertex/edge coordinate scheme and harbor placement. Use it as a reference, not a copy.
- **DROP** the GUI.

### A3. Action enumerator: `v1/ai/decision/legal_actions.py`, `actions.py`

| Action type | Python enum | Engine honors it? |
|---|---|---|
| Build road / settlement / city | Yes (L115 / L154 / L190, parameterized) | Yes |
| Initial placement | Separate (`setup_chooser.py`) | Yes |
| Bank 4:1 / port 3:1, 2:1 | Yes (L236) | **No.** Params ignored; Java runs its own heuristic trade |
| Player trade | No | No |
| Buy dev card | Yes | Takes resources only; no card received |
| Knight / Monopoly / YoP / Road Building | No | No |
| Robber move | Yes (L252) | Yes (tile only) |
| Steal target | Yes | **Ignored; no steal happens** |
| Discard | No | Engine-random |
| End turn | Yes (`pass`/`end_turn`) | Yes |
| Forced-move detection | None. A request is always sent, even when `pass` is the only option | — |

- **Known bug risks:**
  - Port list order differs between the two sides: Python assumes `[B,W,O,G,L,3:1]`, Java uses `[3:1,B,W,O,G,L]`.
  - The structure type comes from Java as an int but is compared to strings. Whether `_normalize_structure_type` (`board_state_extractor.py` ~L193) covers this in every path is UNKNOWN; it needs a targeted test.
- **Tests:** `test_decision_layer_v1.py` (14, shared with the decision layer).
- **Verdict: REWRITE.**
  - Coverage is about half of Catan.
  - The new enumerator must live next to the new engine, as the single source of truth.
  - `actions.py`'s `Action` dataclass shape can be REUSED WITH CHANGES: add dev-play, discard, steal and player-trade kinds, plus a canonical index for the policy head.

### A4. `observed_state` schema

**Built by:** `v1/ai/board_state_extractor.py`
- Snapshot entry point: `extract_board_state_from_snapshot(..., reveal_private=False)` (L1566)
- Live-game entry point: `extract_full_state_from_game` (L1485)
- Payload assembly: `_build_output_payload` (L1426)
- Opponent redaction: `_project_observed_players` (L1318-1360)

The output contains `omniscient_state` **and** `observed_state` in the same payload.

| Field group | Public/hidden | Leak? |
|---|---|---|
| tiles, structures, roads, robber, ports, dice roll, turn, phase | Public | — |
| per-player `victory_points_visible`, `knights_played_visible`, `largest_army_flag`, `ports`, `remaining_pieces`, `total_resource_cards` | Public | — |
| self `resource_cards{5}` | Private (self) | — |
| opponents' `resource_cards{5}` | Hidden; set to `None` with `hidden_from_viewer=True` | ⚠️ **Sent unredacted on the wire** in every `snapshot_payload` (`player_state[].resources` and `players[].resource_cards`); redaction happens Python-side only |
| opponents' dev-card subtypes | Hidden; set to `None` | Moot: dev cards don't exist in the engine |
| `victory_points` (total) | Public today | Becomes a leak once hidden VP cards exist; must split into public and hidden |
| `omniscient_state` | Full truth | ⚠️ Same payload as `observed_state`; any consumer can read it |
| `reveal_private=True` flag | — | ⚠️ Exposes everything (CLI `--reveal_private`) |
| `_aggregate_resource_totals` (L1366) | UNKNOWN | Need to check whether it is copied into `observed_state` |
| bank counts, deck size/order, dice RNG | Not exposed | — |

**Verdict: REUSE WITH CHANGES** (the pattern, not the code).
- Keep the explicit observed/omniscient split and the `hidden_from_viewer` mask convention.
- Change:
  - have the engine emit a per-viewer observation directly, so hidden fields never cross the wire
  - separate the public VP count from the hidden VP count
  - never ship `omniscient_state` alongside the observation, except in a separate eval/log channel
- The extractor (1752 LOC, **no direct tests**) is tied to the V1 snapshot format. → REWRITE.

### A5. Vectorizer: `v1/ai/vectorizer_v1.py` (553 LOC)

- **Main symbols:** `StateVectorizerV1` (L44), `VERSION="vec_v1.2.0"`, `make_training_row` (L528).
- **Safety guard:** `vectorize(view=...)` refuses anything other than `observed_state`.
- **Dims:** 2614 base, or 2904 with economy (+91), robber (+11) and strategic (+188) features. Fixed for 2-4 players (`MAX_OPPONENTS_SCHEMA_V1=3`).
- **Layout (`v1/ai/docs/vector_shape_report_v1.md`):**

| Block | Index range |
|---|---|
| global | 0-8 |
| tile slots | 8-547 |
| structure slots | 547-1429 |
| road slots | 1429-2458 |
| players | 2458-2582 |
| engineered | 2582-2614 |

- It is already mostly per-tile, per-vertex and per-edge one-hots, plus per-player blocks.
- **Hidden info:** opponent counts are −1 plus a mask bit when hidden. No leakage when fed `observed_state`.
- **Tests:** `test_vectorizer_v1.py` (6), `test_strategic_features_v1.py` (17, partially).
- **Deps:** `economy/*` (builders, `ml/flatten_v1`, `robber_hotspot_features`, `strategic_features`).

**Verdict: REUSE WITH CHANGES as a reference / baseline encoder only.**
- Its input is the V1 snapshot schema, and the economy blocks pull in about 70 files of `economy/`.
- For a small net, prefer a new per-tile/per-vertex/per-edge tensor encoder (graph or structured) built on the new engine's observation.
- Keep the masking convention (−1 + mask).
- DROP the economy/strategic hand-crafted blocks unless ablations show value.

### A6. State approximator and rule helpers

| File | Symbols | Exact? | Verdict |
|---|---|---|---|
| `decision/legal_actions.py` L37-40 | `COST_ROAD/SETTLEMENT/CITY/DEV_CARD` | Correct standard costs | REUSE (copy constants) |
| `decision/simulator.py` | `apply_action` (L215) + `_apply_*` | **Approximate**: no production, no longest-road update, no port gain (its own docstring L10-20 says so) | DROP; the new engine does exact transitions |
| `decision/heuristics.py` | `rule_secure_longest_road` (L180), `rule_secure_largest_army` (L240), `apply_heuristics` (L258) | Heuristic. **No exact longest-road path calculation exists anywhere in V1**, Python or Java | DROP (maybe reuse ideas for a scripted bot) |
| `economy/constants.py` | pip totals, thresholds | Static numbers | REUSE WITH CHANGES (only pip/probability tables) |
| `economy/board/ports.py` | `PORT_VERTEX_MAP`, `PORT_RESOURCE_MAP` | Tied to V1 vertex indexing | REWRITE alongside the new board |
| VP counting | Reads engine `victory_points` | No independent recount | REWRITE in the engine |

### A7. Tests: `v1/ai/tests/`

**131 passed in 3.81 s** (124 `def test_` functions plus parametrization).

| File | # | Tests | Depends on reusable code? |
|---|---:|---|---|
| test_resource_belief_tracker_v1.py | 8 | belief tracker | **Yes**: port with the tracker |
| test_economy_belief_v1.py | 1 | belief adapter + tracker | Partial (adapter is dropped) |
| test_vectorizer_v1.py | 6 | vectorizer | Only if the vectorizer is kept as a baseline |
| test_strategic_features_v1.py | 17 | strategic/robber features, vectorizer | No (dropped) |
| test_robber_hotspot_v1.py | 9 | robber hotspot features | No |
| test_economy_*.py (board 5, flatten 1, integration 26, player 2, trade 2) | 36 | economy package | No |
| test_decision_layer_v1.py | 14 | decision layer / simulator | No |
| test_game_bridge_v1.py | 10 | V1 bridge | No (bridge is rewritten); keep as protocol-case ideas |
| test_dataset_builder_v1.py | 15 | dataset builder | No |
| test_trajectory_dataset_builder_v1.py | 5 | trajectory builder | No |
| test_split_manifest_v1.py | 3 | split manifest | Maybe (deterministic game-level splits) |

Directly reusable: about 8-11 tests (belief tracker and split manifest). The rest test dropped components.

Untested: `board_state_extractor.py`, `opening_value_evaluator_v1.py`, `setup_chooser.py`, all training scripts, and the entire Java engine.

---

## Part B: Belief tracker

### B1. Event stream

**There is no event stream.** The bridge sends full snapshots only. "Events" are limited to:
- `action_taken` / `recent_actions`: kind labels such as `build_road+trade_bank`, with no parameters
- the dice roll
- whatever can be inferred from snapshot diffs

| Event | Emitted? | Exact for tracking? |
|---|---|---|
| Dice production | Roll only; production must be recomputed from the board | Recomputable (if robber position is known) |
| Builds | Kind label + board diff | Yes (cost is known) |
| Bank/port trades | Kind label only; **no give/get types** | ❌ Lossy |
| Player trades | Not implemented | N/A |
| Discards | Not emitted; only the hand-size change is visible | ❌ Types not visible |
| Robber move | Robber position in snapshot | Yes |
| Steals | **Do not occur in the engine** | N/A |
| Steal card type visible to non-participants? | N/A (no steals) | — |
| Dev card buy | Kind label | Yes (cost) |
| Dev card identity hidden? | Moot: no card is drawn | — |
| Dev card plays | Not implemented | N/A |

**Implication:** exact public-information tracking against V1 is impossible because of lossy trades and discards. The new engine must emit typed public events per viewer, for example:
- `steal{thief, victim, card: hidden|type}`
- `discard{player, cards}`
- `trade{give, get}`
- `dev_buy{player}`
- `dev_play{type, …}`
- `production{player: {res: n}}`

### B2. Full-information logs

- **Yes.** Every selfplay row has `player_state[i].resources` with exact per-player hands.
- Extracted rows also carry `omniscient_state`.
- Rows are logged **per turn, not per action** (L345-361), so they are too coarse for step-level tracker validation.
- They are also not reproducible (unseeded).

Usable as a weak sanity check only. The new engine needs per-event ground-truth hands in a separate log channel.

### B3. Existing V1 opponent-hand estimation

`v1/ai/resource_belief_tracker_v1.py`:
- **Symbols:** `ResourceBeliefTrackerV1` (L33), `PlayerBelief` (L25), `BeliefContradictionError` (L20), `_enumerate_compositions` (L342), `_prob_can_afford` (L295).
- **Method:** exact distribution over hand compositions per player.
- **Events (`apply_event`, L67-81):**
  - `known_resource_gain`
  - `known_spend`
  - `random_steal` (uniform-random card, Bayesian update, L183-240)
  - `public_total_reconcile`
  - Anything else raises `ValueError`.
- **Options:** strict mode raises on contradictions; prune ε = 1e-15.
- **Not handled natively:** discard, monopoly, YoP, player trades. These must be encoded by the caller as gain/spend (monopoly needs a "lose all of type X" event).
- **No producer:** nothing in V1 emits these events from real games; the tracker was only exercised in tests.
- **Tests:** 8 + 1 (adapter).
- **Exactness:** exact for the events it supports (not heuristic).

**Verdict: REUSE WITH CHANGES.**
- Add `discard_known`, `discard_hidden`, `monopoly_loss`, `yop_gain` and `trade` handlers.
- Watch combinatorial blow-up of the composition support on large hands.
- `economy/belief/adapter.py` (`build_belief_economy_v1`, L44) → DROP.

### B4. Compare with `catan_rl/belief/`

**`catan_rl/` does not exist in this workspace**, so this comparison can't be done yet. To do it, I need the path to the new repo or tracker module.

Checklist for when it exists:
1. Composition enumeration vs particle/count representation.
2. `random_steal` Bayesian update.
3. `public_total_reconcile` consistency checks.
4. Strict contradiction mode.
5. Event types ignored by either side: monopoly, YoP, discard, trades.

---

## Part C: Other assets

### C1. Trained models (all scikit-learn pickles; no torch)

| Model | Path | Test metrics | Verdict |
|---|---|---|---|
| value_model_baseline_v1 | `v1/ai/models/value_model_baseline_v1/` (hgb 1.1 MB, logreg 73 KB) | acc 0.75, **ROC AUC 0.5** | DROP |
| value_model_economy_v1 | `v1/ai/models/value_model_economy_v1/` | identical to baseline (suspicious) | DROP |
| robber_hotspot_v1 | `v1/ai/models/robber_hotspot_v1/` (hgb 3.3 MB) | logloss ≈ 2.91-3.03 ≈ ln 19 (uniform) | DROP |
| win_model_v1 / _stageaware | `v1/ai/selfplay/models/` | acc 0.75, AUC 0.5 | DROP |
| catan_winner_model | `v1/ai/selfplay/models/catan_winner_model/` | acc 0.75, AUC 0.5, logloss 9.0 | DROP |

Accuracy 0.75 = the 4-player base rate. **None of these beats chance**, so none is usable as a baseline, feature or opponent. They are also tied to V1 vectors.

### C2. Logged games and datasets

| Location | Size | Rows | Format | Ground truth? |
|---|---|---|---|---|
| `v1/data/*.jsonl` (12 files) | 424 MB | 0-47k each | JSONL, per turn | Yes (`player_state.resources`) |
| `v1/ai/selfplay/data/` | **41 GB** | ≈3.2M | JSONL raw + "extracted" (`observed_state` + `omniscient_state`) | Yes |

These datasets are of limited value:
- They come from the stubbed engine (no dev cards, steals or longest road).
- They are unseeded and logged per turn.
- Play is bot vs bot (random/guided).

**Verdict: DROP**, and don't copy them into the new repo. Optionally keep one small file as a format fixture.

### C3. Heuristic agents

| Agent | Where | Usable? |
|---|---|---|
| Java built-in bot | `SelfPlayGameRunner` L1038-1107 | Only inside the V1 engine. Port the priority logic (city > settlement > road > trade) as a scripted baseline in the new env |
| Decision layer | `ai/decision/decision_layer_v1.py` | No: it depends on an AUC-0.5 value model. Phase-2 gate: 0/10 wins at weight 0.06 (other weights UNKNOWN; see `v1/data/phase2_weight_sweep/sweep_aggregate.json`). In this audit's measurement, 0/2 wins and 15 fallbacks/game |
| `setup_chooser.py` | `ai/bridge/` | Rule-based pip/diversity opening. REUSE WITH CHANGES as a scripted opening baseline (re-index to the new board) |
| `opening_scorer_v1.py` / `opening_value_evaluator_v1.py` | `ai/selfplay/`, `ai/decision/` | DROP (relies on the value model) |

---

## Copy list (priority order, into `catan_rl/`)

1. `v1/ai/resource_belief_tracker_v1.py` → `catan_rl/belief/` (then extend its event types)
2. `v1/ai/tests/test_resource_belief_tracker_v1.py` → `catan_rl/tests/`
3. `v1/ai/decision/actions.py` (the `Action` dataclass only, as a starting point for the action schema)
4. `v1/ai/bridge/setup_chooser.py` (scripted opening baseline; re-index)
5. Cost constants from `v1/ai/decision/legal_actions.py` L37-40 (copy the constants, not the file)
6. `v1/ai/split_manifest_v1.py` + `test_split_manifest_v1.py` (deterministic game-level splits for eval)
7. Reference only, do not copy: `v1/ai/vectorizer_v1.py` + `v1/ai/docs/vector_shape_report_v1.md`, `v1/src/board/Board.java` (coordinate scheme), `_project_observed_players` in `board_state_extractor.py` (redaction pattern)

## Drop list

- **Engine and bridge:** the whole Java engine `v1/src/**` (GUI included), `v1/run.sh`, and `v1/ai/bridge/game_bridge_v1.py` / `play_game_v1.py`
- **Decision stack:** `v1/ai/decision/*` except the copied bits (simulator, heuristics, scorer, decision layer, opening evaluator, `_diagnose_build_delta.py`)
- **Features and extraction:** `v1/ai/economy/**` (≈70 files), `v1/ai/board_state_extractor.py`
- **Dataset and training pipeline:** `dataset_builder_v1.py`, `trajectory_dataset_builder_v1.py`, `train_win_model_v1.py`, `train_robber_hotspot_v1.py`, and the notebook
- **Selfplay scripts:** everything in `v1/ai/selfplay/*.py`
- **Artifacts:** all models (`v1/ai/models/**`, `v1/ai/selfplay/models/**`) and all datasets (`v1/data/**`, `v1/ai/selfplay/data/**`, 41 GB)
- **Tests:** every test except the belief tracker and split manifest tests

## Top 5 risks / gaps

1. **Incomplete rules.**
   - Missing: dev cards (no draws), dev-card plays, steals, player trades, longest road, port-aware trade legality, discard choice.
   - Anything trained on V1 learns the wrong game.
2. **Hidden-info leakage.** Opponents' exact hands are on the wire in every request, and `omniscient_state` ships next to `observed_state`. Redaction is opt-in on the Python side. The new env must redact at the source.
3. **No event stream.** Trades and discards are untyped, so exact belief tracking is impossible on V1. The new engine must emit typed per-viewer events plus ground-truth hands in a separate channel.
4. **Not reproducible.**
   - Dice, board, deck and seat order use unseeded RNGs (`Math.random`, `Collections.shuffle` without an RNG).
   - Seat assignment is shuffled.
   - Verified: the same seed gave different outputs.
   - This blocks deterministic evaluation and debugging.
5. **Silent fallbacks and throughput.**
   - Illegal or malformed actions silently become random moves (15 fallbacks/game measured).
   - There's no read timeout, and stderr is never drained (deadlock risk).
   - There's one JVM per game and no reset.
   - The full snapshot is resent every decision.

## Measured throughput

Measured on a 10-core Mac, cwd `v1/`, with existing `.class` files. The temp script was in `/tmp`; nothing in the repo was written.

| Setup | Games/s | Decisions/s | Notes |
|---|---:|---:|---|
| Pure Java, all built-in bots, 1 process | ≈9.2 | n/a | 5 games / 0.54 s, includes JVM start |
| Bridge, trivial Python policy (`unavailable` → engine fallback), 1 worker | 12.3 | ≈1,050 | 10 games, 858 decisions, 0.81 s |
| Same, 8 workers (process pool) | 30.0 | ≈2,700 | 40 games, 3,601 decisions, 1.33 s. Scales poorly (~2.5× from 8 workers). The bottleneck is likely JVM startup and JSON |
| Bridge + V1 decision layer (value model + vectorizer), 1 game process | ≈0.02 | ≈2–4 | 2 games in 105 s, all truncated at 400 turns, 0 wins (the last game logged 224 decisions; per-game counts for the other game weren't printed) |

The trivial-policy games are short because `unavailable` hands control to the bot. The fallback-random behavior means these numbers reflect the protocol cost per decision, not realistic game length.

UNKNOWN: throughput with a persistent process and in-process reset. That doesn't exist in V1, so it can't be measured.
