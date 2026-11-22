# Engine: Catanatron fork with separated RNG streams

| | |
|---|---|
| Fork location | `~/Desktop/catanatron-fork` (local git repo, branch `rng-streams`) |
| Upstream | https://github.com/bcollazo/catanatron |
| Base commit | `ecf931181b9a65bb4116a2153fb78c16f1438e00` (`main`, 2026-09-08, v3.3.0) |
| Pinned fork commit | `4732786` (`rng-streams` HEAD; working tree clean). All `catan_rl` results (baselines, winprob, Φ, phi_greedy) were produced on this commit |
| License | **GPL-3.0-or-later**. Upstream `LICENSE` is kept unchanged in the fork. Code that imports or modifies the fork is subject to GPL-3.0 when distributed |
| Python | ≥ 3.11. Installed editable into `catanthropic/.venv-rl` (Python 3.12): `pip install -e "~/Desktop/catanatron-fork[dev,gym,web]"`. The gym and web extras are only needed to run upstream's test suite; we don't use `catanatron_gym` |

## Patch summary

`Game(players, seed=S)` now derives five independent streams from `S` with `random.Random(f"{S}:{name}")`. String seeds are hashed with SHA-512 by CPython, so the streams don't overlap.

| Stream | Used for | Code |
|---|---|---|
| `map` | Board tiles, numbers, ports | `state.py` `State.__init__` → `CatanMap.from_template(..., rng=streams["map"])` |
| `seats` | Seat order | `State.__init__` |
| `deck` | Dev deck shuffle | `State.__init__` |
| `dice` → `state.dice_rng` | Every roll | `apply_action.apply_roll` → `roll_dice(state.dice_rng)` |
| `steal` → `state.steal_rng` | Robber steal card | `state_functions.player_deck_random_select` |
| `state.random` (unchanged) | Bots and other uses | still `random.Random(S)` |

### Guarantees

- **Policy independence:** the dice sequence and steal draws depend only on `S`, never on which players are seated or what they decide. Board, seat order and deck depend only on `S`.
- **Copy semantics:**
  - `State.copy()` / `Game.copy()` **clone** `dice_rng` and `steal_rng` (an exact, independent copy), so simulating on a copy never advances the real game's streams.
  - `state.random` remains shared, as upstream does.
- **No future-dice leakage to search bots:**
  - A cloned dice stream would let rollouts replay the real upcoming rolls.
  - `State.reseed_rule_streams(rng)` gives a copy fresh dice and steal streams.
  - `players/playouts.py` (`run_playout`, `_run_seeded_playout`, used by GreedyPlayouts and MCTS) calls it on every rollout copy.
  - AlphaBeta uses `execute_spectrum` (it enumerates outcomes) and never rolls.

### Global RNG

- `roll_dice(rng=None)` now raises instead of falling back to the global `random` module.
- `models/map.py:initialize_tiles` keeps its global fallback, marked with a comment. The engine path (`State`) always passes `streams["map"]`, so `Game(seed=S)` never touches it. The fallback is only reachable from external callers that seed the global module themselves: web, cli, gym, and upstream's tests. Removing it outright broke 6 upstream tests (`tests/models/test_map.py`, `test_spiral.py`) that call `initialize_tiles()` directly.
- A test checks that `Game(seed=S)` produces the same map, seats and deck regardless of global `random` state.

### Serialization

- `state_to_json` saves `dice_random_state` and `steal_random_state`; `state_from_json` restores them.
- Older documents without these keys derive them deterministically.
- `client_view` strips both keys.

### Behavior change vs upstream

The same seed no longer gives the same game as upstream: the board, seats, deck and dice all differ. Replays of upstream-recorded games still work, because replay uses recorded results.

## Files changed (fork, relative to `ecf9311`)

| File | Change |
|---|---|
| `catanatron/catanatron/state.py` | `derive_streams`, `clone_rng`, `streams=` arg, `dice_rng`/`steal_rng`, `reseed_rule_streams`, cloned on `copy()` |
| `catanatron/catanatron/game.py` | Passes `streams=derive_streams(seed)` |
| `catanatron/catanatron/apply_action.py` | Roll uses `state.dice_rng`; `roll_dice` requires an rng |
| `catanatron/catanatron/state_functions.py` | Steal uses `state.steal_rng` |
| `catanatron/catanatron/serialization.py` | Persist, restore and redact the rule streams |
| `catanatron/catanatron/models/map.py` | Comment only (global fallback scope) |
| `catanatron/catanatron/players/playouts.py` | Reseed rule streams on rollout copies |
| `tests/test_replay.py` | Replay helper nulls and restores the new streams (as upstream does for `state.random`) |
| `tests/test_rng_streams.py` | New: policy-independent dice, identical records, search leaves the real dice untouched, copy clones, global-RNG independence, playout isolation |
