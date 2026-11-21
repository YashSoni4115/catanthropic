"""Dev-card belief for one viewer.

Unseen multiset U = 25-card deck - viewer's own cards (held or played) - cards others played publicly.
U is dealt exchangeably to every opponent's hidden slots and to the deck remainder, so opponent p's
hidden hand (n_p = public count) is a multivariate hypergeometric draw of size n_p from U.
Not conditioned on anything beyond public counts (e.g. not on "the game is not over yet").

Owned-at-start: cards bought before p's last TurnEndEvent are ``old`` (playable), later ones ``new``.
Plays always come from old cards (engine rule).
"""
from __future__ import annotations

from collections import Counter
from math import comb
from typing import Dict, Iterable, List, Optional

from catan_rl.belief import events as ev
from catan_rl.belief.tracker import BeliefContradictionError

DEV_DECK: Dict[str, int] = {"KNIGHT": 14, "VICTORY_POINT": 5, "ROAD_BUILDING": 2, "YEAR_OF_PLENTY": 2, "MONOPOLY": 2}
PLAYABLE = ("KNIGHT", "YEAR_OF_PLENTY", "MONOPOLY", "ROAD_BUILDING")


def hypergeom_pmf(N: int, K: int, n: int) -> List[float]:
    """P(k of the n drawn are special), N cards of which K special, k = 0..n."""
    return [comb(K, k) * comb(N - K, n - k) / comb(N, n) for k in range(n + 1)]


class DevCardBelief:
    def __init__(self, viewer: str, players: Iterable[str], deck: Dict[str, int] = DEV_DECK):
        self.viewer, self.players = viewer, list(players)
        self.unseen = Counter(deck)
        self.own: Counter = Counter()  # viewer's held cards
        self.old = dict.fromkeys(self.players, 0)
        self.new = dict.fromkeys(self.players, 0)
        self.played = {p: Counter() for p in self.players}

    # ---------------------------------------------------------------- updates
    def apply(self, e) -> None:
        if isinstance(e, ev.DevBuyEvent):
            self.new[e.player] += 1
            if e.player == self.viewer:
                if e.card is None:
                    raise ValueError("viewer's own DevBuyEvent must reveal the card")
                self._take(e.card)
                self.own[e.card] += 1
        elif isinstance(e, ev.DevPlayEvent):
            p = e.player
            if self.old[p] <= 0:
                raise BeliefContradictionError(f"{p} played {e.card} with no playable (old) cards")
            self.old[p] -= 1
            self.played[p][e.card] += 1
            if p == self.viewer:
                self.own[e.card] -= 1
            else:
                self._take(e.card)
        elif isinstance(e, ev.TurnEndEvent):
            self.old[e.player] += self.new[e.player]
            self.new[e.player] = 0

    def _take(self, card: str) -> None:
        if self.unseen[card] <= 0:
            raise BeliefContradictionError(f"no unseen {card} left")
        self.unseen[card] -= 1

    # ---------------------------------------------------------------- queries
    def opponents(self) -> List[str]:
        return [p for p in self.players if p != self.viewer]

    def hidden(self, p: str) -> int:
        return self.old[p] + self.new[p]

    @property
    def n_unseen(self) -> int:
        return sum(self.unseen.values())

    @property
    def deck_size(self) -> int:
        return self.n_unseen - sum(self.hidden(o) for o in self.opponents())

    def count_pmf(self, p: str, card: str, old_only: bool = False) -> List[float]:
        """P(p holds k cards of type ``card``) among all hidden (or only old/playable) cards."""
        n = self.old[p] if old_only else self.hidden(p)
        return hypergeom_pmf(self.n_unseen, self.unseen[card], n)

    def p_at_least_one(self, p: str, card: str, old_only: bool = False) -> float:
        return 1.0 - self.count_pmf(p, card, old_only)[0]

    def expected_count(self, p: str, card: str) -> float:
        return self.hidden(p) * self.unseen[card] / self.n_unseen if self.n_unseen else 0.0

    def summary(self, p: str, public_vp: Optional[int] = None) -> dict:
        e_vp = self.expected_count(p, "VICTORY_POINT")
        out = {
            "hidden_cards": self.hidden(p),
            "p_vp": self.p_at_least_one(p, "VICTORY_POINT"),
            "p_knight": self.p_at_least_one(p, "KNIGHT"),
            "p_playable_knight": self.p_at_least_one(p, "KNIGHT", old_only=True),
            "expected_vp_cards": e_vp,  # = expected hidden VP (each VP card is worth 1)
        }
        if public_vp is not None:
            out["expected_total_vp"] = public_vp + e_vp
        return out
