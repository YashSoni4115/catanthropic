"""Convert catanatron action records into per-viewer catan_rl.belief events.

True hands are read only to compute *public* quantities (production, setup payouts,
monopoly per-victim amounts, YoP when the bank is short, public totals). The only hidden
quantity, the stolen card, is revealed to thief and victim only.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence, Tuple

from catanatron.models.enums import RESOURCES, ActionRecord, ActionType
from catanatron.state import State
from catanatron.state_functions import get_player_freqdeck

from catan_rl.belief import events as ev

# catanatron resource name -> tracker resource name
RES = {"WOOD": "LUMBER", "BRICK": "BRICK", "SHEEP": "WOOL", "WHEAT": "GRAIN", "ORE": "ORE"}
DEV_PLAYS = {ActionType.PLAY_KNIGHT_CARD: "KNIGHT", ActionType.PLAY_YEAR_OF_PLENTY: "YEAR_OF_PLENTY",
             ActionType.PLAY_MONOPOLY: "MONOPOLY", ActionType.PLAY_ROAD_BUILDING: "ROAD_BUILDING"}
Hands = Dict[str, Tuple[int, ...]]  # color name -> freqdeck in catanatron RESOURCES order


def hands(state: State) -> Hands:
    return {c.value: tuple(get_player_freqdeck(state, c)) for c in state.colors}


def cards(vec: Sequence[int]) -> Dict[str, int]:
    return {RES[r]: n for r, n in zip(RESOURCES, vec) if n}


def _split(before, after):
    gain = [max(a - b, 0) for b, a in zip(before, after)]
    loss = [max(b - a, 0) for b, a in zip(before, after)]
    return gain, loss


def public_events(record: ActionRecord, before: Hands, after: Hands) -> List[Tuple[object, frozenset]]:
    """Returns (event, private_to) pairs. ``private_to`` empty = identical for all viewers;
    otherwise the event is the *revealed* version for those viewers (StealEvent only)."""
    a, t, actor = record.action, record.action.action_type, record.action.color.value
    out: List[Tuple[object, frozenset]] = []
    pub = frozenset()
    if t in DEV_PLAYS:
        out.append((ev.DevPlayEvent(actor, DEV_PLAYS[t]), pub))
    if t == ActionType.MARITIME_TRADE:
        give = Counter(RES[r] for r in a.value[:4] if r is not None)
        out.append((ev.BankTradeEvent(actor, dict(give), {RES[a.value[4]]: 1}), pub))
    elif t == ActionType.BUY_DEVELOPMENT_CARD:
        out.append((ev.DevBuyEvent(actor, record.result), frozenset((actor,))))
        out.append((ev.DevBuyEvent(actor), frozenset(("*",))))
    elif t == ActionType.DISCARD_RESOURCE:
        out.append((ev.DiscardEvent(actor, {RES[record.result]: 1}, 1), pub))
    elif t == ActionType.PLAY_YEAR_OF_PLENTY:
        gain, _ = _split(before[actor], after[actor])
        out.append((ev.YearOfPlentyEvent(actor, cards(gain)), pub))
    elif t == ActionType.PLAY_MONOPOLY:
        i = RESOURCES.index(a.value)
        per = {c: before[c][i] - after[c][i] for c in before if c != actor}
        out.append((ev.MonopolyEvent(actor, RES[a.value], sum(per.values()), per), pub))
    elif t == ActionType.MOVE_ROBBER:
        victim = a.value[1]
        if victim is not None:
            v, card = victim.value, RES[record.result]
            out.append((ev.StealEvent(actor, v, card), frozenset((actor, v))))
            out.append((ev.StealEvent(actor, v, None), frozenset(("*",))))  # everyone else
    else:  # ROLL, builds (incl. free setup builds + 2nd-settlement payout), knight, road building, end turn
        for c in before:
            gain, loss = _split(before[c], after[c])
            if any(loss):
                out.append((ev.SpendEvent(c, cards(loss)), pub))
            if any(gain):
                out.append((ev.ProductionEvent(c, cards(gain)), pub))
    if t == ActionType.END_TURN:
        out.append((ev.TurnEndEvent(actor), pub))
    return out


def events_for(viewer: str, pairs, after: Hands) -> List[object]:
    """Select the viewer's version of each event, then reconcile every opponent's public total."""
    out = []
    for e, private_to in pairs:
        if not private_to or viewer in private_to or ("*" in private_to and not _is_party(e, viewer)):
            out.append(e)
    out += [ev.TotalReconcileEvent(c, sum(h)) for c, h in after.items() if c != viewer]
    return out


def _is_party(e, viewer) -> bool:
    if isinstance(e, ev.StealEvent):
        return viewer in (e.thief, e.victim)
    return isinstance(e, ev.DevBuyEvent) and viewer == e.player
