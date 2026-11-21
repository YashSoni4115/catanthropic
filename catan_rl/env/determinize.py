"""Determinization: a playable Game whose hidden state is re-sampled from one viewer's beliefs.

Read from the real game: public state only (board, buildings, roads, robber, bank, turn, phase,
prompts, public VP, played cards, card counts) and the viewer's own hand/dev cards.
Overwritten: every opponent's resource hand, dev hand, owned-at-start flags and actual VP; the dev
deck; hidden results in action_records; bot RNG and dice/steal streams. The output is a function of
(public state, viewer's own cards, trackers, seed) only.

Resource hands: product of tracker marginals conditioned on the per-resource conservation sums
(19 - bank - viewer's own). Rejection sampler: all opponents but one from their marginals, the last
derived from the sums, accepted w.p. P_last(h) / max P_last; after ``max_tries`` falls back to
exact enumeration (or raises DeterminizeError if the joint support is too large).
"""
from __future__ import annotations

import itertools
import random
from collections import Counter
from typing import Dict, List, Sequence, Tuple

from catanatron.game import Game
from catanatron.models.actions import generate_playable_actions
from catanatron.models.enums import RESOURCES as CAT_RES, ActionRecord, ActionType, Action
from catanatron.state_functions import player_key

from catan_rl.belief.devcards import DEV_DECK, PLAYABLE, DevCardBelief
from catan_rl.belief.tracker import RESOURCES as TRK_RES, BeliefTracker, Hand
from catan_rl.env.event_adapter import RES

PER_RESOURCE = 19
CAT_OF_TRK = [CAT_RES.index(next(c for c, t in RES.items() if t == r)) for r in TRK_RES]  # tracker idx -> catanatron idx
EXACT_LIMIT = 2_000_000


class DeterminizeError(RuntimeError):
    pass


def conservation_sums(bank_freqdeck: Sequence[int], own: Hand) -> Hand:
    """Per resource (tracker order): total held by all opponents = 19 - bank - viewer's own."""
    return tuple(PER_RESOURCE - bank_freqdeck[CAT_OF_TRK[i]] - own[i] for i in range(5))


def own_hand(tracker: BeliefTracker, viewer: str) -> Hand:
    pmf = tracker.players[viewer].pmf
    if len(pmf) != 1:
        raise DeterminizeError(f"viewer {viewer}'s own hand is not known exactly: {len(pmf)} candidates")
    return next(iter(pmf))


def _sub(S, hs):
    return tuple(S[i] - sum(h[i] for h in hs) for i in range(5))


def joint_support(pmfs: Dict[str, Dict[Hand, float]], S: Hand):
    """Exact constrained joint: list of (assignment dict, weight) and normalizer Z (independent prob of S)."""
    order = sorted(pmfs, key=lambda p: (len(pmfs[p]), p))
    others, last = order[:-1], order[-1]
    n = 1
    for p in others:
        n *= len(pmfs[p])
    if n > EXACT_LIMIT:
        raise DeterminizeError(f"joint support too large for exact enumeration ({n} > {EXACT_LIMIT})")
    out, Z = [], 0.0
    for combo in itertools.product(*(sorted(pmfs[p].items()) for p in others)):
        h_last = _sub(S, [h for h, _ in combo])
        w = pmfs[last].get(h_last, 0.0)
        for _, q in combo:
            w *= q
        if w > 0:
            out.append(({**{p: h for p, (h, _) in zip(others, combo)}, last: h_last}, w))
            Z += w
    return out, Z


def sample_hands(pmfs: Dict[str, Dict[Hand, float]], S: Hand, rng: random.Random, max_tries: int = 2000):
    order = sorted(pmfs)
    last = max(order, key=lambda p: (len(pmfs[p]), p))  # derive the most diffuse hand
    others = [p for p in order if p != last]
    tables = {}
    for p in others:
        keys = sorted(pmfs[p])
        cum, acc = [], 0.0
        for k in keys:
            acc += pmfs[p][k]
            cum.append(acc)
        tables[p] = (keys, cum)
    P_last = pmfs[last]
    m = max(P_last.values())
    for t in range(1, max_tries + 1):
        hs = {p: rng.choices(tables[p][0], cum_weights=tables[p][1])[0] for p in others}
        h = _sub(S, hs.values())
        w = P_last.get(h, 0.0)
        if w > 0 and rng.random() * m < w:
            hs[last] = h
            return hs, {"tries": t, "method": "rejection"}
    support, Z = joint_support(pmfs, S)
    if not support:
        raise DeterminizeError("no hand assignment satisfies the conservation sums (beliefs inconsistent)")
    assignment = rng.choices([a for a, _ in support], weights=[w for _, w in support])[0]
    return assignment, {"tries": max_tries, "method": "exact_fallback"}


def deal_dev(db: DevCardBelief, public_vp: Dict[str, int], vps_to_win: int, rng: random.Random, max_tries: int = 1000):
    """Uniform deal of the unseen multiset to opponents' (old, new) slots and the deck (random order).
    Rejects deals that would give an opponent >= vps_to_win actual VP (public fact: game not over)."""
    base = sorted(db.unseen.elements())
    for t in range(1, max_tries + 1):
        cards = base[:]
        rng.shuffle(cards)
        out, i = {}, 0
        for p in db.opponents():
            o, n = db.old[p], db.new[p]
            out[p] = (Counter(cards[i:i + o]), Counter(cards[i + o:i + o + n]))
            i += o + n
        if all(public_vp[p] + out[p][0]["VICTORY_POINT"] + out[p][1]["VICTORY_POINT"] < vps_to_win for p in out):
            return out, cards[i:], t
    raise DeterminizeError("could not deal dev cards without ending the game")


def _mask(r: ActionRecord, viewer: str) -> ActionRecord:
    a = r.action
    if a.action_type == ActionType.BUY_DEVELOPMENT_CARD and a.color.value != viewer:
        return ActionRecord(action=Action(a.color, a.action_type, None), result=None)
    if a.action_type == ActionType.MOVE_ROBBER and a.value[1] is not None and viewer not in (a.color.value, a.value[1].value):
        return ActionRecord(action=a, result=None)
    return r


def determinize(game: Game, viewer, tracker: BeliefTracker, db: DevCardBelief, seed, max_tries: int = 2000) -> Game:
    v = getattr(viewer, "value", viewer)
    rng = random.Random(f"{seed}:determinize:{v}")
    g = game.copy()
    st = g.state
    opps = [c for c in st.colors if c.value != v]
    own = own_hand(tracker, v)
    S = conservation_sums(st.resource_freqdeck, own)
    hands, info = sample_hands({c.value: tracker.players[c.value].pmf for c in opps}, S, rng, max_tries)
    pvp = {c.value: st.player_state[f"{player_key(st, c)}_VICTORY_POINTS"] for c in opps}
    if db.deck_size != len(st.development_listdeck):
        raise DeterminizeError(f"dev belief deck size {db.deck_size} != public deck size {len(st.development_listdeck)}")
    dev, deck, info["dev_deals"] = deal_dev(db, pvp, g.vps_to_win, rng)
    for c in opps:
        k, h = player_key(st, c), hands[c.value]
        for i in range(5):
            st.player_state[f"{k}_{CAT_RES[CAT_OF_TRK[i]]}_IN_HAND"] = h[i]
        old, new = dev[c.value]
        for card in DEV_DECK:
            st.player_state[f"{k}_{card}_IN_HAND"] = old[card] + new[card]
        for card in PLAYABLE:
            st.player_state[f"{k}_{card}_OWNED_AT_START"] = old[card] > 0
        st.player_state[f"{k}_ACTUAL_VICTORY_POINTS"] = pvp[c.value] + old["VICTORY_POINT"] + new["VICTORY_POINT"]
    st.development_listdeck = deck
    st.action_records = [_mask(r, v) for r in st.action_records]
    g.random = st.random = random.Random(rng.getrandbits(64))
    st.reseed_rule_streams(rng)
    g.playable_actions = generate_playable_actions(st)
    g.determinization = info
    return g


def phase(state) -> str:
    if state.is_initial_build_phase:
        return "setup"
    vp = max(state.player_state[f"{player_key(state, c)}_VICTORY_POINTS"] for c in state.colors)
    return "early" if vp <= 4 else "mid" if vp <= 7 else "late"
