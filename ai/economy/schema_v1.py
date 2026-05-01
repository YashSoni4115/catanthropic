from __future__ import annotations

from typing import Dict, List, Literal, Optional, TypedDict


BRICK = "BRICK"
WOOL = "WOOL"
ORE = "ORE"
GRAIN = "GRAIN"
LUMBER = "LUMBER"

BOARD_ONLY = "board_only"
OBSERVED = "observed"
OMNISCIENT = "omniscient"

RESOURCE_NAMES = (BRICK, WOOL, ORE, GRAIN, LUMBER)
SOURCE_VIEWS = (BOARD_ONLY, OBSERVED, OMNISCIENT)

SYMBOLIC_BOARD_TAGS = (
    "balanced_board",
    "polarized_board",
    "brick_scarce",
    "wool_scarce",
    "ore_scarce",
    "grain_scarce",
    "lumber_scarce",
    "brick_abundant",
    "wool_abundant",
    "ore_abundant",
    "grain_abundant",
    "lumber_abundant",
    "brick_concentrated",
    "wool_concentrated",
    "ore_concentrated",
    "grain_concentrated",
    "lumber_concentrated",
    "robber_fragile_brick",
    "robber_fragile_wool",
    "robber_fragile_ore",
    "robber_fragile_grain",
    "robber_fragile_lumber",
)

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
    baseline_tile_count: int
    baseline_ratio: float
    scarcity_deviation: float
    abundance_deviation: float
    scarcity_score: float
    abundance_score: float
    mean_pips_per_tile: float
    strong_number_share: float
    concentration_hhi: float
    concentration_score: float
    token_quality_score: float
    number_token_counts: Dict[str, int]
    tags: List[str]


class ResourceEnvironmentV1(TypedDict):
    by_resource: Dict[str, ResourceStatsV1]
    strongest_resources: List[str]
    weakest_resources: List[str]
    total_productive_pips: float
    baseline_total_pips: float
    expected_average_tile_pips: float
    resource_ratio_spread: float
    board_balance_score: float
    board_polarization_score: float
    board_tags: List[str]


class DesertRobberV1(TypedDict):
    desert_location: Optional[Dict[str, int]]
    robber_location: Optional[Dict[str, int]]
    robber_on_desert: bool
    desert_neighbor_count: int
    desert_neighbor_total_pips: float
    desert_neighbor_resource_pips: Dict[str, float]
    desert_centrality_score: float
    robber_fragility_by_resource: Dict[str, float]
    strong_concentrated_resources: List[str]
    robber_fragile_resources: List[str]
    tags: List[str]


class PortOpportunityV1(TypedDict):
    available: bool
    ports: Dict[str, object]
    notes: List[str]


class BoardEconomyV1(TypedDict):
    resource_environment: ResourceEnvironmentV1
    desert_robber: DesertRobberV1
    port_opportunity: PortOpportunityV1


class OpeningCandidateV1(TypedDict):
    vertex_id: str
    x: int
    y: int
    orientation: int
    adjacent_tiles: List[Dict[str, object]]
    ev_score: float
    ev_score_norm: float
    total_pips: float
    resource_mix: Dict[str, float]
    pip_by_resource: Dict[str, float]
    diversity_score: float
    evenness_score: float
    recipe_coverage_by_recipe: Dict[str, float]
    recipe_coverage_score: float
    scarcity_capture_score: float
    synergy_score: float
    robber_fragility_score: float
    key_tile_pips: float
    key_tile_loss_share: float
    expansion_frontier_count: int
    expansion_frontier_score: float
    port: Optional[str]
    port_alignment_score: float
    opening_score: float
    tags: List[str]


class OpeningPairV1(TypedDict):
    pair_id: str
    vertex_ids: List[str]
    total_pips: float
    total_pips_norm: float
    resource_mix: Dict[str, float]
    pip_by_resource: Dict[str, float]
    diversity_score: float
    evenness_score: float
    recipe_coverage_by_recipe: Dict[str, float]
    recipe_coverage_score: float
    scarcity_capture_score: float
    synergy_score: float
    robber_fragility_score: float
    key_tile_pips: float
    key_tile_loss_share: float
    expansion_frontier_score: float
    port_alignment_score: float
    opening_pair_score: float
    tags: List[str]


class OpeningEconomyV1(TypedDict):
    schema_version: str
    opening_candidates: List[OpeningCandidateV1]
    opening_pairs: List[OpeningPairV1]
    opening_rankings: Dict[str, object]
    scoring_notes: Dict[str, str]


class PlayerEconomyV1(TypedDict):
    player_id: Optional[str]
    player_name: Optional[str]
    hidden_from_viewer: bool
    structure_count: int
    settlement_count: int
    city_count: int
    expected_income: Dict[str, float]
    expected_production_by_resource: Dict[str, float]
    total_expected_income: float
    expected_production_total: float
    production_counts: Dict[str, int]
    settlement_weighted_income: Dict[str, float]
    city_weighted_income: Dict[str, float]
    blocked_income: Dict[str, float]
    blocked_production_by_resource: Dict[str, float]
    total_blocked_income: float
    blocked_production_total: float
    blocked_counts: Dict[str, int]
    contributing_structures: List[Dict[str, object]]
    diversity_score: float
    evenness_score: float
    concentration_score: float
    bottlenecks: Dict[str, Dict[str, float]]
    scarcity_capture_score: float
    strong_resources: List[str]
    weak_resources: List[str]
    orientation_scores: Dict[str, float]
    top_orientations: List[str]
    favored_plan: Optional[str]
    expansion: Dict[str, object]
    expansion_potential: Dict[str, object]
    upgrade_potential: Dict[str, object]
    robber_burden: Dict[str, object]
    robber_burden_score: float
    resource_ranks: Dict[str, object]
    total_income_rank: Optional[int]
    leader_gap: Dict[str, float]
    relative_table_metrics: Dict[str, object]
    pressure: PlayerPressureV1
    tags: List[str]


class PlayerEconomyBlockV1(TypedDict):
    schema_version: str
    by_player: List[PlayerEconomyV1]
    table_leaders: Dict[str, object]
    table_pressure: Dict[str, object]


class PlayerExpansionV1(TypedDict):
    schema_version: str
    reachable_site_count: int
    top_reachable_sites: List[Dict[str, object]]
    top_reachable_site_score: float
    top_reachable_site_ev: float
    average_top_reachable_site_score: float
    frontier_score: float
    boxed_in_score: float
    appears_boxed_in: bool
    appears_open: bool
    frontier_resource_diversity_score: float
    frontier_recipe_coverage_score: float
    plan_continuity_score: float
    upgradeable_settlement_count: int
    top_upgradeable_settlements: List[Dict[str, object]]
    best_upgradeable_settlement_score: float
    average_upgradeable_settlement_score: float
    city_upside_score: float
    ore_grain_upgrade_path_score: float
    tags: List[str]


class PlayerPressureV1(TypedDict):
    schema_version: str
    contested_frontier_score: float
    race_pressure_score: float
    block_threat_score: float
    resource_competition_score: float
    leader_pressure_score: float
    top_rival_player_id: Optional[str]
    contested_site_count: int
    contested_top_site_fraction: float
    top_contested_sites: List[Dict[str, object]]
    opponent_equal_or_faster_site_count: int
    player_ahead_site_count: int
    strongest_rival_by_expansion_race: Optional[str]
    vulnerable_frontier_count: int
    appears_cuttable: bool
    strongest_blocker_rival: Optional[str]
    resource_overlap_by_rival: Dict[str, float]
    bottleneck_resource_pressure_score: float
    scarce_capture_pressure_score: float
    strong_resource_line_contested: bool
    strongest_resource_competition_rival: Optional[str]
    is_public_economy_leader: bool
    leader_gap_pressure: float
    opponents_stronger_combined_frontier: bool
    strong_now_fragile_later: bool
    weak_now_latent_expansion: bool
    strongest_frontier_rival: Optional[str]
    tags: List[str]


class PlayerPressureBlockV1(TypedDict):
    schema_version: str
    by_player: Dict[str, PlayerPressureV1]
    table_pressure: Dict[str, object]


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
    board_economy: BoardEconomyV1
    player_economy: PlayerEconomyBlockV1
    belief_economy: Dict[str, object]
    trade_profile: Dict[str, object]
    summary_tags: List[str]


economy_state_v1 = EconomyStateV1
board_economy = BoardEconomyV1
resource_environment = ResourceEnvironmentV1
desert_robber = DesertRobberV1
port_opportunity = PortOpportunityV1
opening_economy = OpeningEconomyV1
player_economy = PlayerEconomyBlockV1
player_expansion = PlayerExpansionV1
player_pressure = PlayerPressureBlockV1
