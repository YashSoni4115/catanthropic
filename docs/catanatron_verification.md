# Catanatron verification

- **Checkout:** `~/Desktop/catanatron_eval/catanatron` (outside the repo).
- **Venv:** `~/Desktop/catanatron_eval/.venv`, Python 3.12.15, installed with `pip install -e ".[gym]"`.
- **Scratch script:** `~/Desktop/catanatron_eval/verify.py`.
- Catanatron source and `v1/` were not modified.
- **Branch:** the repo's default branch is **`main`**; there is no `master`. `main` was used.

## 1. Version and license

| Item | Value |
|---|---|
| Commit | `ecf931181b9a65bb4116a2153fb78c16f1438e00` (2026-09-08) |
| Package version | 3.3.0 (`pyproject.toml`) |
| Python | `requires-python = ">=3.11"` (system 3.9 is too old; used Homebrew 3.12) |
| Deps | core: networkx, click, rich; gym extra: `gymnasium<=0.29.1`, numpy, pandas, fastparquet, pygame |
| License | **GPL-3.0** (`LICENSE`: "GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007"; `pyproject.toml`: `GPL-3.0-or-later`). **Not MIT.** |

**License implication:** importing or modifying Catanatron in a distributed work makes that work GPL-3.0. Running it privately as a training env is unaffected. Model weights trained on it are not obviously derivative, but that is not legal advice. Decide this before vendoring or patching its source.

## 2. Rules coverage

All paths are under `catanatron/catanatron/catanatron/`.

| Rule | Status | Code path / notes |
|---|---|---|
| Robber move | IMPLEMENTED | `apply_action.py:354-370`; targets in `models/actions.py:208-249`. The current tile is excluded (`:225`). Optional `friendly_robber` |
| 7-roll discard | IMPLEMENTED | `apply_action.py:278-300` sets `discard_counts = n//2` when n > `discard_limit` (default 7); `apply_discard` `:316-351` |
| Steal | IMPLEMENTED | `state_functions.py:321-323` (uniform card via `state.random`); only victims with ≥1 card (`actions.py:236`) |
| Knight | IMPLEMENTED | `apply_action.py:373-381` |
| Monopoly | IMPLEMENTED | `apply_action.py:399-417` |
| Year of Plenty | IMPLEMENTED | `apply_action.py:384-396`; one card allowed if the bank is short (`actions.py:124-144`) |
| Road Building | IMPLEMENTED | `apply_action.py:420-430`, free roads `:197-211` |
| VP card | IMPLEMENTED | Counted in `ACTUAL_VICTORY_POINTS` on buy (`state_functions.py:266-267`) |
| Same-turn dev restriction; one dev card per turn | IMPLEMENTED, with an edge case | `player_can_play_dev` (`state_functions.py:231-237`) uses `{CARD}_OWNED_AT_START` + `HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN`. The check is per card type: holding a knight from earlier and buying a second knight still lets you play "a knight", which is correct under the rules. Dev cards are playable before the roll (`actions.py:60`) |
| Longest road | IMPLEMENTED | `models/board.py:192-229` (road), `:86-139` (a settlement that cuts a road recomputes); `maintain_longest_road` (`state_functions.py:25-43`) |
| Largest army | IMPLEMENTED | `maintain_largest_army` (`state_functions.py:46-68`), ≥3 knights, strictly more than the holder |
| Ports 2:1 / 3:1, bank 4:1 | IMPLEMENTED | `actions.py:300-340`; `apply_action.py:433-448` |
| Domestic player trades | **PARTIAL** | Offer/accept/reject/confirm/cancel are applied (`apply_action.py:451-533`), but **`generate_playable_actions` never emits `OFFER_TRADE`** (`actions.py:56-89`), so bots and the gym can't initiate trades. Domestic trade types are also absent from the gym action array |
| Setup phase | IMPLEMENTED | Snake order `apply_action.py:178-196`; second settlement pays out `:142-149` |
| Win at 10 VP | IMPLEMENTED | `Game(vps_to_win=10)` (`game.py:77,103`), checked on `ACTUAL_VICTORY_POINTS` (`:192-207`). `TURNS_LIMIT=1000` (`game.py:20`) |
| Bank limit 19 per resource | IMPLEMENTED | `models/decks.py:32-34`; a resource the bank can't fully pay goes to no one (`apply_action.py:581-595`) |
| Dev deck 25 (14/2/2/2/5) | IMPLEMENTED | `models/decks.py:68-75` |

## 3. Reproducibility

- **Seeding:** `Game(players, seed=S)` creates a per-game `random.Random(S)` (`game.py:99-100`) and passes it to `State`. That RNG drives:
  - seat order: `random.sample` (`state.py:100`)
  - map: `CatanMap.from_template(..., rng)` (`state.py:104-105`, `models/map.py:336-344`)
  - dev deck shuffle (`state.py:122`)
  - dice (`apply_action.py:273,537-544`)
  - steals (`state_functions.py:323`)
- **Separate dice seed:** none. Dice, map, deck, seats and bots all share **one stream**.
- **Bots share the stream:** bots draw from `game.state.random` (`models/player.py:127`, `minimax.py:61`, `value.py:171`, …), so changing one player's policy changes later dice and steals.
- **`copy()` shares the RNG object** (`game.py:218`, `state.py:167`). A search bot simulating on a copy advances the real game's stream. The result is still deterministic, but not independent of the agent.
- **Global-RNG fallbacks:**
  - `models/map.py:336` and `apply_action.py:543` fall back to the global `random` when no rng is given.
  - The gym env calls `random.seed(seed)` on the global module before `build_map` (`gym/envs/catanatron_env.py:172-176`).
  - `players/stdio.py:181,200` uses the global `random`.

**Test** (same seed, same player classes, played twice, full `action_records` compared as strings, seat order included):

| Players | Seed | Actions | Identical? |
|---|---|---:|---|
| 4× RandomPlayer | 1234 | 900 | **Yes** |
| AlphaBeta + 3× Random | 7 | 207 | **Yes** |

## 4. Event stream

`state.action_records: List[ActionRecord(action=Action(color, action_type, value), result)]` holds one record per action (`models/enums.py:113,126`; appended at `apply_action.py:123`).

Observer hook: `GameObserver.before/step/after` via `Game.play(accumulators=[...])` (`observer.py:13-43`, `game.py:121-147`). `step` receives the action before it is applied; read `action_records[-1]` afterwards for the result.

| Event | Recorded? | How | Non-participant in real Catan sees |
|---|---|---|---|
| Dice | Exact | ROLL `value = result = (d1, d2)` (verified) | everything |
| Resources produced | **Not recorded** | Recompute from roll + board + robber, including bank shortage (`yield_resources`, `apply_action.py:547-595`) | everything (public) |
| Steal | Exact | MOVE_ROBBER `value=(coord, victim\|None)`, `result=<resource>` (verified: `((2,-1,-1), RED) -> ORE`) | victim + count only, **not the card type** |
| Discard | Exact | One `DISCARD_RESOURCE` record per card, `value = result = resource` | **card types** (discards are face up) |
| Bank/port trade | Exact | MARITIME_TRADE `value` 5-tuple: give = non-None of the first 4, get = last (verified) | everything |
| Monopoly | **Partial** | Resource only; **amounts per victim not recorded** (`apply_action.py:399-417`). Recover them by diffing hands before and after | in practice the total, usually per victim too (announced) |
| Dev buy | Exact | `value` rewritten to the drawn card, `result` = card (verified: `-> VICTORY_POINT`) | buyer only, **not the identity** |
| Dev plays | Exact | KNIGHT/ROAD_BUILDING `value=None` (followed by robber/road actions); YoP `value=(r1[, r2])`; MONOPOLY `value=resource` | everything (VP cards are never played; revealed at win) |
| Domestic trade | Exact if it happens | OFFER/ACCEPT/REJECT 10/11-tuples, CONFIRM 11-tuple incl. counterparty | everything |

**Engine-side redaction:** `serialization.client_view(doc, perspective)` (`serialization.py:349-400`) hides other players' steal results and dev-buy identities (`SECRET_VALUE`/`SECRET_RESULT`) and collapses opponents' hands to counts. It works on the JSON form only.

## 5. Discards

- **Mechanism:** chosen by the player's `decide()`, **one card per action**, as `DISCARD_RESOURCE(value=resource)`. One option is generated per resource type held (`actions.py:281-289`), and the player repeats until `discard_counts[i] == 0`. Merged in `41ba0db deterministic discards (#362)`.
  - Branch `origin/bryan/deterministic-discard` (31 files, +763/−211) is the engine change plus a frontend batch-discard dialog.
  - The old engine-random `DISCARD` (`random.sample`, value `None`) is gone on `main`.
- **Recording:** the card types are recorded (one record per card).
- **Tracker implication:**
  - Discards are public and typed, so use `DiscardEvent(cards=...)` (exact, cheap).
  - The hidden-count discard path and its discard model are not needed for Catanatron, which also avoids the expensive case.
  - A `RandomPlayer` discards by picking a resource *type* uniformly at each step. That is neither uniform over sub-multisets nor hypergeometric, which matters only if we ever model opponents' discards as hidden.

## 6. State

- **Flat `state.player_state` dict, keyed `P{seat}_{FIELD}`** (`state.py:22-43`; seat via `player_key(state, color)`):
  - `{WOOD,BRICK,SHEEP,WHEAT,ORE}_IN_HAND`
  - `{KNIGHT,YEAR_OF_PLENTY,MONOPOLY,ROAD_BUILDING,VICTORY_POINT}_IN_HAND`, `PLAYED_*`, `*_OWNED_AT_START`
  - `VICTORY_POINTS` (public), `ACTUAL_VICTORY_POINTS`
  - pieces available, `HAS_ROAD`, `HAS_ARMY`, `HAS_ROLLED`, `HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN`, `LONGEST_ROAD_LENGTH`
- **Other state fields:** `board` (buildings, `robber_coordinate`, `map`), `resource_freqdeck` (bank), `development_listdeck` (ordered; `pop()` from the end), `colors`, `current_*`, `discard_counts`, `current_trade`, `random`.
- **Ground truth per action:** yes. Read `get_player_freqdeck(state, color)` (`state_functions.py:149`) in an observer's `step`/`after`, or after each `game.play_tick()`. Every hand is available after every action.
- **A clean per-viewer `observed_state` must hide, for each opponent:**
  - resource `*_IN_HAND` (expose the total only)
  - dev `*_IN_HAND` and `*_OWNED_AT_START` (expose the dev-card count only)
  - `ACTUAL_VICTORY_POINTS` (expose `VICTORY_POINTS`)
- **And globally:**
  - `development_listdeck` order (expose the count)
  - `state.random`, `game.seed`
  - other players' steal `result` and dev-buy `value`/`result` in `action_records`
- `serialization.client_view` already does most of this on the JSON form and can serve as a reference. Monopoly per-victim amounts should be *added* (public in real play).

## 7. Action space

- **Enumeration:** `generate_playable_actions(state)` (`models/actions.py:56-89`) → `game.playable_actions`. There are 18 `ActionType` values (`enums.py:68-106`), parameterized through `value`.
- **Fixed index** for the policy head: `gym/envs/action_space.py:11-74` (`get_action_array`), which excludes domestic trades.
  - Measured size: **370 for 4 players**, 332 for 2 players. The docstring's `[0, 327]` (`catanatron_env.py:251`) is stale.
- **Forced moves:** detectable as `len(playable_actions) == 1` (already used by Greedy/MCTS). Nothing is auto-played: ROLL, END_TURN and each discard card are explicit decisions.
- **Measured over 50 random 4-player games:** **55.4% of 52,284 decisions were forced**. Auto-playing these would roughly halve the decisions sent to a model.

## 8. Throughput (this Mac, 10 cores)

- Every engine action is one `decide()` call, so decisions = actions.
- 8 workers = `ProcessPoolExecutor`.

| Setup | Workers | Games/s | Decisions/s | Avg decisions/game |
|---|---:|---:|---:|---:|
| 4× RandomPlayer | 1 | 73.5 | 81.6k | 1112 |
| 4× RandomPlayer | 8 | 354.9 | 381k | 1075 |
| AlphaBeta (default depth 2) + 3× Random | 1 | 0.95 | 208 | 220 |
| AlphaBeta + 3× Random | 8 | 3.22 | 810 | 252 |
| Gym env, 4p (learner vs 3 Random), random valid actions | 1 | — | **7,108 env steps/s** | (72 episodes in 20k steps) |

- Measured 4× AlphaBeta: UNKNOWN (expected about 4× slower than 1× AB; not measured).
- 3 of the 800 random games hit the turn limit with no winner.
- AB at 8 workers scales only 3.4×. The likely cause is the small batch of 40 games plus process startup; this was not profiled.

## 9. Gym env (`gym/envs/catanatron_env.py`)

- **4 players with custom opponents:** yes. `config={"enemies": [<any Player>, …]}` (3 enemies = 4 players). Other options: `map_type`, `vps_to_win`, `representation` (`vector` or `mixed`), `reward_function`, `invalid_action_reward`.
  - The learner is always `BLUE`.
  - Seat order is still randomized by the seed.
  - An episode is truncated after more than 10 invalid actions.
- **Observation:** `features.create_sample(game, BLUE)` (`features.py:517`), measured at **1002 dims** for 4 players (vector form).
  - Hidden info is exposed only for the learner itself: own hand, own dev cards, own `ACTUAL_VPS`.
  - Opponents get public VPS, army/road flags, pieces, `LONGEST_ROAD_LENGTH`, played-dev counts, and `NUM_RESOURCES_IN_HAND` / `NUM_DEVS_IN_HAND` (counts only).
  - **No opponent hidden info leaks.** Steal and dev-buy history is not in the features at all.
- **Seeding caveat:** `reset(seed)` uses the **global** `random` for the map (`:172-173`). It's reproducible per process, but not isolated.
- **Turn-taking:** the env is single-agent. Opponents act inside `step()`, so self-play or league training needs a multi-agent wrapper (UNKNOWN whether one exists; none found in `gym/`).

### Bot leakage (matters for opponents and eval)

- `AlphaBetaPlayer`, `ValueFunctionPlayer`, `GreedyPlayoutsPlayer` and `MCTSPlayer` all search on `game.copy()`, which holds the true hands and deck.
- `tree_search_utils.execute_spectrum` reads the victim's real hand and the real deck composition, including opponents' actual unplayed dev cards (`:56-60`, `:90-115`).
- Playouts use the real deck order.
- **They are cheating bots:** fine as strong league opponents, but not a fair hidden-information baseline.

---

## Verdict: **PASS WITH FIXES**

Full rules including longest road, all dev cards, ports and bank limits; exact typed event records; deterministic seeding; a fast pure-Python engine (≈80k decisions/s single core); a working 4-player gym env without opponent leakage. The fixes below are wrappers or small engine patches. Domestic trading is the only real rules gap.

| # | Fix | Where | Effort |
|---|---|---|---|
| 1 | **Domestic trade generation is missing**: `OFFER_TRADE` is never generated, and trade actions are absent from the gym action array. Needed if we want negotiation | `models/actions.py`, `gym/envs/action_space.py` | **Large** (action-space design and response protocol), or **drop** domestic trades for v1 of the RL work |
| 2 | Monopoly per-victim amounts are not recorded | Wrapper: diff hands around the action (no source change) | Small |
| 3 | Production per roll is not recorded | Wrapper: diff hands, or recompute via `yield_resources` | Small |
| 4 | Per-viewer `observed_state` and event redaction (steal card, dev-buy identity, deck order, `ACTUAL_VPS`) for the live `State` | Our wrapper, modeled on `serialization.client_view` | Small–Medium |
| 5 | Separate RNG streams (dice, board, deck, steals) independent of agents and search copies; gym map uses the global RNG | Wrapper: pass a pre-built `catan_map` and our own RNGs; full separation needs an engine patch to `State`/`roll_dice` | Medium |
| 6 | Auto-play forced moves (55% of decisions) | Wrapper loop | Small |
| 7 | Multi-agent / self-play API (the gym is single-agent with built-in opponents) | Our own env on top of `Game.play_tick()` | Medium |
| 8 | Fair (non-peeking) search baselines: AB, MCTS and Playouts search on true hidden state | Determinize from belief samples before `copy()` | Medium |
| 9 | Stale action-space docstring (`[0,327]` vs 370 for 4p) | Ignore (cosmetic) | Small |
| — | GPL-3.0 license | Decide on vendoring vs. dependency before shipping code | Policy, not code |
