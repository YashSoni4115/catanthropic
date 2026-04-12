from __future__ import annotations

from typing import Dict, List, Literal, Optional, TypedDict


ResourceName = Literal["BRICK", "WOOL", "ORE", "GRAIN", "LUMBER"]
PortName = Literal[
    "three_to_one",
    "brick_two_to_one",
    "wool_two_to_one",
    "ore_two_to_one",
    "grain_two_to_one",
    "lumber_two_to_one",
]
PlanArchetype = Literal["road", "settlement", "city", "dev"]
SourceView = Literal["board_only", "observed", "omniscient"]


class ResourceStatsV1(TypedDict):
    resource: ResourceName
    tile_count: int
    pip_total: float
    pip_share: float
    baseline_pip_total: float
    scarcity_score: float
    abundance_score: float
    mean_pips_per_tile: float
    strong_number_share: float
    concentration_hhi: float
    concentration_score: float
    token_quality_score: float
    number_token_counts: Dict[str, int]
    tags: List[str]


class OpeningCandidateV1(TypedDict):
    vertex_id: str
    x: int
    y: int
    orientation: int
    adjacent_tiles: List[Dict[str, int]]
    total_pips: float
    pip_by_resource: Dict[str, float]
    diversity_score: float
    evenness_score: float
    recipe_coverage_score: float
    balance_score: float
    expansion_frontier_count: int
    expansion_frontier_score: float
    upgrade_quality_score: float
    port: Optional[str]
    port_alignment_score: float
    ev_score: float
    ev_score_norm: float
    archetype_scores: Dict[str, float]
    tags: List[str]


class PlayerEconomyV1(TypedDict):
    player_id: Optional[str]
    player_name: Optional[str]
    hidden_from_viewer: bool
    victory_points_visible: Optional[int]
    total_resource_cards: Optional[int]
    expected_production_total: float
    expected_production_by_resource: Dict[str, float]
    blocked_production_total: float
    blocked_production_by_resource: Dict[str, float]
    diversity_score: float
    balance_score: float
    orientation_scores: Dict[str, float]
    favored_plan: str
    bottlenecks: List[Dict[str, float | str]]
    port_leverage_score: float
    expansion_potential: Dict[str, float | int | List[str]]
    upgrade_potential: Dict[str, float | int]
    robber_burden_score: float
    relative_table_metrics: Dict[str, float | int]
    pressure: Dict[str, float | int | List[str]]
    ports: Dict[str, bool]
    tags: List[str]


class BeliefPlayerV1(TypedDict):
    player_id: str
    known_total: Optional[int]
    support_size: int
    confidence_score: float
    expected_hidden_resource_counts: Dict[str, float]
    p_at_least_one: Dict[str, float]
    p_can_afford: Dict[str, float]
    tags: List[str]


class TradePlayerProfileV1(TypedDict):
    player_id: Optional[str]
    player_name: Optional[str]
    need_scores: Dict[str, float]
    surplus_scores: Dict[str, float]
    need_top2: List[str]
    offer_top2: List[str]
    strongest_resource: Optional[str]
    weakest_resource: Optional[str]
    affordability: Dict[str, float | bool | None]
    relative_table_metrics: Dict[str, float | int]
    tags: List[str]


class EconomyContextV1(TypedDict):
    extractor_schema_version: Optional[str]
    source_view: SourceView
    state_id: Optional[str]
    position_id: Optional[str]
    turn_index: Optional[int]
    current_player_id: Optional[str]
    board_fingerprint: str


class EconomyStateV1(TypedDict):
    schema_version: str
    context: EconomyContextV1
    board_economy: Dict[str, object]
    player_economy: Dict[str, object]
    belief_economy: Dict[str, object]
    trade_profile: Dict[str, object]
    summary_tags: List[str]
