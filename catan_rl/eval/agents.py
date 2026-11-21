"""Named seat agents for evaluation.

A factory takes a Color and returns a CatanEnv seat agent: a catanatron Player, or a
callback ``(env, color) -> action_index``. "Peeking" agents read hidden information
(catanatron's search/value bots use the full game state) and are labeled in all outputs.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable, Dict

from catanatron.models.player import RandomPlayer
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer


@dataclass(frozen=True)
class AgentSpec:
    name: str
    factory: Callable  # color -> seat agent
    peeking: bool = False


REGISTRY: Dict[str, AgentSpec] = {}


def register(name: str, factory: Callable, peeking: bool = False) -> None:
    """Register our own agent. ``factory(color)`` returns a callback ``(env, color) -> index``
    (or a catanatron Player). With workers > 1, register at import time of a module the
    workers also import, or pass the agent as "package.module:factory"."""
    REGISTRY[name] = AgentSpec(name, factory, peeking)


register("random", RandomPlayer)
register("weighted_random", WeightedRandomPlayer)
register("value_function", ValueFunctionPlayer, peeking=True)
register("alpha_beta", AlphaBetaPlayer, peeking=True)


def get(name: str) -> AgentSpec:
    if name not in REGISTRY and ":" in name:
        mod, attr = name.split(":", 1)
        register(name, getattr(importlib.import_module(mod), attr))
    if name not in REGISTRY:
        raise KeyError(f"unknown agent {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name]


def label(name: str) -> str:
    return f"{name} (peeking)" if get(name).peeking else name
