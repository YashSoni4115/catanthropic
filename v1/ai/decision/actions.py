"""Action representation for the decision layer.

An :class:`Action` is a lightweight, immutable, hashable description of a
single move that a player may take.  It is deliberately decoupled from the
Java engine's string kinds so we can parameterise moves (vertex / edge /
tile / trade offer) while still round-tripping to the runner's
``build_road`` / ``build_settlement`` / ... vocabulary via
:attr:`Action.engine_kind`.

Only these action types are supported by decision_layer_v1 because the
Java runner's ``legalActionKinds`` only exposes them and the value model
has zero training signal for the rest:

    BUILD_ROAD, BUILD_SETTLEMENT, BUILD_CITY,
    BUY_DEV_CARD, TRADE_BANK, MOVE_ROBBER, END_TURN

TODO: add PLAY_KNIGHT, PLAY_ROAD_BUILDING, PLAY_YEAR_OF_PLENTY,
PLAY_MONOPOLY, PROPOSE_TRADE / ACCEPT_TRADE / REJECT_TRADE once the
Java engine loop supports them AND self-play data contains examples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple


class ActionType(str, Enum):
    BUILD_ROAD = "build_road"
    BUILD_SETTLEMENT = "build_settlement"
    BUILD_CITY = "build_city"
    BUY_DEV_CARD = "buy_dev_card"
    TRADE_BANK = "trade_bank"
    MOVE_ROBBER = "move_robber"
    END_TURN = "end_turn"

    # Not enumerated by legal_actions() in v1 -- see module docstring.
    PLAY_KNIGHT = "play_knight"
    PLAY_ROAD_BUILDING = "play_road_building"
    PLAY_YEAR_OF_PLENTY = "play_year_of_plenty"
    PLAY_MONOPOLY = "play_monopoly"
    PROPOSE_TRADE = "propose_trade"
    ACCEPT_TRADE = "accept_trade"
    REJECT_TRADE = "reject_trade"


SUPPORTED_ACTION_TYPES: Tuple[ActionType, ...] = (
    ActionType.BUILD_ROAD,
    ActionType.BUILD_SETTLEMENT,
    ActionType.BUILD_CITY,
    ActionType.BUY_DEV_CARD,
    ActionType.TRADE_BANK,
    ActionType.MOVE_ROBBER,
    ActionType.END_TURN,
)


@dataclass(frozen=True)
class Action:
    """A parameterised, hashable action.

    Only the fields relevant to :attr:`type` are populated; the rest stay
    ``None``.  ``__repr__`` is compact for logging.
    """

    type: ActionType

    # Board-slot parameters
    vertex: Optional[Tuple[int, int, int]] = None   # (x, y, orientation) for settlement/city
    edge: Optional[Tuple[int, int, int]] = None     # (x, y, orientation) for road
    tile: Optional[Tuple[int, int]] = None          # (x, y) for robber

    # Robber steal target: opponent player_id or None if no valid victim
    steal_target: Optional[str] = None

    # Bank / port trade
    give_resource: Optional[str] = None
    take_resource: Optional[str] = None
    give_qty: Optional[int] = None                  # 4 / 3 / 2 depending on port

    # Origin metadata (owner attaches after construction if needed)
    tags: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def engine_kind(self) -> str:
        """String matching Java runner's action-kind vocabulary."""
        if self.type == ActionType.END_TURN:
            return "pass"
        return self.type.value

    def short(self) -> str:
        t = self.type.value
        if self.vertex is not None:
            return f"{t}@v{self.vertex}"
        if self.edge is not None:
            return f"{t}@e{self.edge}"
        if self.tile is not None:
            tgt = f"->{self.steal_target}" if self.steal_target else ""
            return f"{t}@t{self.tile}{tgt}"
        if self.type == ActionType.TRADE_BANK:
            return f"trade_{self.give_qty}:{self.give_resource}->{self.take_resource}"
        return t
