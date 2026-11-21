"""Typed, engine-agnostic public events consumed by the belief tracker.

Card multisets are ``Dict[str, int]`` keyed by resource name
(BRICK, WOOL, ORE, GRAIN, LUMBER). ``None`` means "not publicly visible".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Union

Cards = Dict[str, int]


@dataclass(frozen=True)
class ProductionEvent:
    player: str
    cards: Cards


@dataclass(frozen=True)
class SpendEvent:
    player: str
    cards: Cards


@dataclass(frozen=True)
class BankTradeEvent:
    player: str
    give: Cards
    get: Cards


@dataclass(frozen=True)
class PlayerTradeEvent:
    """Player ``a`` gives ``give`` to ``b``; ``b`` gives ``get`` to ``a``."""
    a: str
    b: str
    give: Cards
    get: Cards


@dataclass(frozen=True)
class DiscardEvent:
    player: str
    cards: Optional[Cards]
    count: int


@dataclass(frozen=True)
class StealEvent:
    thief: str
    victim: str
    card: Optional[str] = None


@dataclass(frozen=True)
class MonopolyEvent:
    player: str
    resource: str
    total: int
    per_victim: Optional[Dict[str, int]] = None


@dataclass(frozen=True)
class YearOfPlentyEvent:
    player: str
    resources: Cards


@dataclass(frozen=True)
class DevBuyEvent:
    player: str


@dataclass(frozen=True)
class TotalReconcileEvent:
    player: str
    total: int


Event = Union[
    ProductionEvent, SpendEvent, BankTradeEvent, PlayerTradeEvent, DiscardEvent,
    StealEvent, MonopolyEvent, YearOfPlentyEvent, DevBuyEvent, TotalReconcileEvent,
]
