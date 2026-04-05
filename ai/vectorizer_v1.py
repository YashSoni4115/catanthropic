from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


RESOURCE_ORDER = ["BRICK", "WOOL", "ORE", "GRAIN", "LUMBER", "DESERT", "UNKNOWN"]
PHASE_ORDER = ["setup", "roll", "main", "trade", "build", "unknown"]
OWNER_REL_ORDER = ["self", "op1", "op2", "op3", "none"]
TRACKED_RESOURCES = ["BRICK", "WOOL", "ORE", "GRAIN", "LUMBER"]

PROBABILITY_WEIGHTS = {
    2: 1.0,
    3: 2.0,
    4: 3.0,
    5: 4.0,
    6: 5.0,
    8: 5.0,
    9: 4.0,
    10: 3.0,
    11: 2.0,
    12: 1.0,
}


@dataclass(frozen=True)
class FeatureGroup:
    name: str
    start: int
    end: int


@dataclass
class VectorizedState:
    x: List[float]
    feature_names: List[str]
    feature_groups: List[FeatureGroup]
    key_indices: Dict[str, int]
    version: str
    metadata: Dict[str, Any]


class StateVectorizerV1:
    VERSION = "vec_v1.1.0"

    def __init__(self, adjacency_features_enabled: bool = True):
        self.adjacency_features_enabled = adjacency_features_enabled

    def feature_spec(self, state: Dict[str, Any]) -> List[FeatureGroup]:
        x, _, feature_groups, _, _ = self._vectorize_internal(state)
        _ = x
        return feature_groups

    def vectorize(self, state: Dict[str, Any], view: str = "observed_state") -> VectorizedState:
        if view != "observed_state":
            raise ValueError("vec_v1 is gameplay-safe and supports only observed_state")

        x, feature_names, feature_groups, key_indices, meta = self._vectorize_internal(state)
        return VectorizedState(
            x=x,
            feature_names=feature_names,
            feature_groups=feature_groups,
            key_indices=key_indices,
            version=self.VERSION,
            metadata=meta,
        )

    def _vectorize_internal(self, state: Dict[str, Any]) -> Tuple[List[float], List[str], List[FeatureGroup], Dict[str, int], Dict[str, Any]]:
        observed = state.get("observed_state", state)
        encoding = state.get("encoding_metadata", {})
        slot_ordering = encoding.get("slot_ordering", {})

        tile_slots = slot_ordering.get("tile_slots", [])
        structure_slots = slot_ordering.get("structure_slots", [])
        road_slots = slot_ordering.get("road_slots", [])

        tile_valid_mask = slot_ordering.get("tile_slot_valid", [True] * len(tile_slots))
        structure_valid_mask = slot_ordering.get("structure_slot_valid", [True] * len(structure_slots))
        road_valid_mask = slot_ordering.get("road_slot_valid", [True] * len(road_slots))

        turn_metadata = observed.get("turn_metadata", {})
        transition_context = observed.get("transition_context", {})
        players_block = observed.get("players", {})
        ordered_players = players_block.get("all_players", [])
        current_player_id = turn_metadata.get("current_player_id")

        tile_map = {(t.get("x"), t.get("y")): t for t in observed.get("tiles", []) if isinstance(t, dict)}
        structure_map = {
            (s.get("x"), s.get("y"), s.get("orientation")): s
            for s in observed.get("structures", [])
            if isinstance(s, dict)
        }
        road_map = {
            (r.get("x"), r.get("y"), r.get("orientation")): r
            for r in observed.get("roads", [])
            if isinstance(r, dict)
        }

        x: List[float] = []
        feature_names: List[str] = []
        feature_groups: List[FeatureGroup] = []
        key_indices: Dict[str, int] = {}

        def add_group(name: str, values: List[float], subfeature_names: List[str]) -> None:
            if len(values) != len(subfeature_names):
                raise ValueError(f"Group '{name}' values/name length mismatch")
            start = len(x)
            for value, subname in zip(values, subfeature_names):
                index = len(x)
                x.append(float(value))
                full_name = f"{name}.{subname}"
                feature_names.append(full_name)
                key_indices[full_name] = index
            feature_groups.append(FeatureGroup(name=name, start=start, end=len(x)))

        global_features: List[float] = []
        global_names: List[str] = []
        turn_index = turn_metadata.get("turn_index")
        global_features.append(float(turn_index) if isinstance(turn_index, int) else -1.0)
        global_names.append("turn_index")

        phase = str(turn_metadata.get("current_phase") or "unknown").lower()
        phase_onehot = [1.0 if phase == p else 0.0 for p in PHASE_ORDER]
        if sum(phase_onehot) == 0.0:
            phase_onehot[-1] = 1.0
        global_features.extend(phase_onehot)
        global_names.extend([f"phase.{p}" for p in PHASE_ORDER])

        dice_roll = transition_context.get("latest_dice_roll")
        global_features.append(float(dice_roll) if isinstance(dice_roll, int) else -1.0)
        global_names.append("latest_dice_roll")

        add_group("global", global_features, global_names)

        robber = observed.get("robber") or {}
        robber_xy = (robber.get("x"), robber.get("y"))

        tile_values: List[float] = []
        tile_names: List[str] = []
        for idx, slot in enumerate(tile_slots):
            sx, sy = slot.get("x"), slot.get("y")
            tile = tile_map.get((sx, sy))
            tile_values.append(1.0 if bool(tile_valid_mask[idx]) else 0.0)
            tile_names.append(f"slot{idx}.valid")
            tile_values.append(1.0 if tile is not None else 0.0)
            tile_names.append(f"slot{idx}.present")

            tile_type = str((tile or {}).get("type") or "UNKNOWN").upper()
            if tile_type not in RESOURCE_ORDER:
                tile_type = "UNKNOWN"
            tile_values.extend([1.0 if tile_type == resource else 0.0 for resource in RESOURCE_ORDER])
            tile_names.extend([f"slot{idx}.type.{resource}" for resource in RESOURCE_ORDER])

            number = (tile or {}).get("number")
            tile_values.append(float(number) if isinstance(number, int) else -1.0)
            tile_names.append(f"slot{idx}.number")
            tile_values.append(1.0 if bool((tile or {}).get("robber", False)) else 0.0)
            tile_names.append(f"slot{idx}.robber")
        add_group("tile_slots", tile_values, tile_names)

        player_ids = [p.get("player_id") for p in ordered_players if isinstance(p, dict)]
        rel_map: Dict[Optional[str], str] = {None: "none"}
        if current_player_id is not None:
            rel_map[current_player_id] = "self"
            opponents = [pid for pid in player_ids if pid is not None and pid != current_player_id]
        else:
            opponents = [pid for pid in player_ids if pid is not None]
            if opponents:
                rel_map[opponents[0]] = "self"
                opponents = opponents[1:]
        for i, pid in enumerate(opponents[:3], start=1):
            rel_map[pid] = f"op{i}"

        structure_values: List[float] = []
        structure_names: List[str] = []
        for idx, slot in enumerate(structure_slots):
            key = (slot.get("x"), slot.get("y"), slot.get("orientation"))
            structure = structure_map.get(key)
            structure_values.append(1.0 if bool(structure_valid_mask[idx]) else 0.0)
            structure_names.append(f"slot{idx}.valid")
            structure_values.append(1.0 if structure is not None else 0.0)
            structure_names.append(f"slot{idx}.present")

            structure_type = str((structure or {}).get("type") or "")
            structure_values.append(1.0 if structure_type == "settlement" else 0.0)
            structure_names.append(f"slot{idx}.is_settlement")
            structure_values.append(1.0 if structure_type == "city" else 0.0)
            structure_names.append(f"slot{idx}.is_city")

            owner_rel = rel_map.get((structure or {}).get("owner_id"), "none")
            structure_values.extend([1.0 if owner_rel == rel else 0.0 for rel in OWNER_REL_ORDER])
            structure_names.extend([f"slot{idx}.owner.{rel}" for rel in OWNER_REL_ORDER])
        add_group("structure_slots", structure_values, structure_names)

        road_values: List[float] = []
        road_names: List[str] = []
        for idx, slot in enumerate(road_slots):
            key = (slot.get("x"), slot.get("y"), slot.get("orientation"))
            road = road_map.get(key)
            road_values.append(1.0 if bool(road_valid_mask[idx]) else 0.0)
            road_names.append(f"slot{idx}.valid")
            road_values.append(1.0 if road is not None else 0.0)
            road_names.append(f"slot{idx}.present")

            owner_rel = rel_map.get((road or {}).get("owner_id"), "none")
            road_values.extend([1.0 if owner_rel == rel else 0.0 for rel in OWNER_REL_ORDER])
            road_names.extend([f"slot{idx}.owner.{rel}" for rel in OWNER_REL_ORDER])
        add_group("road_slots", road_values, road_names)

        player_values: List[float] = []
        player_names: List[str] = []

        def player_block(i: int, player: Dict[str, Any]) -> Tuple[List[float], List[str]]:
            is_hidden = bool(player.get("hidden_from_viewer", False))
            prefix = f"p{i}"
            values: List[float] = []
            names: List[str] = []

            def append_metric(metric_name: str, value: float) -> None:
                values.append(float(value))
                names.append(f"{prefix}.{metric_name}")

            append_metric(
                "victory_points_visible",
                player.get("victory_points_visible", -1) if player.get("victory_points_visible") is not None else -1,
            )
            append_metric(
                "knights_played_visible",
                player.get("knights_played_visible", -1) if player.get("knights_played_visible") is not None else -1,
            )
            append_metric(
                "total_resource_cards",
                player.get("total_resource_cards", -1) if player.get("total_resource_cards") is not None else -1,
            )

            dev_cards = player.get("development_cards", {}) if isinstance(player.get("development_cards"), dict) else {}
            append_metric(
                "dev_total_cards",
                dev_cards.get("total_cards", -1) if dev_cards.get("total_cards") is not None else -1,
            )
            append_metric("largest_army_flag", 1.0 if bool(player.get("largest_army_flag", False)) else 0.0)

            ports = player.get("ports", {}) if isinstance(player.get("ports"), dict) else {}
            for key in [
                "three_to_one",
                "brick_two_to_one",
                "wool_two_to_one",
                "ore_two_to_one",
                "grain_two_to_one",
                "lumber_two_to_one",
            ]:
                append_metric(f"port.{key}", 1.0 if bool(ports.get(key, False)) else 0.0)

            remaining = player.get("remaining_pieces", {}) if isinstance(player.get("remaining_pieces"), dict) else {}
            append_metric(
                "pieces.roads_remaining",
                remaining.get("roads_remaining", -1) if remaining.get("roads_remaining") is not None else -1,
            )
            append_metric(
                "pieces.settlements_remaining",
                remaining.get("settlements_remaining", -1) if remaining.get("settlements_remaining") is not None else -1,
            )
            append_metric(
                "pieces.cities_remaining",
                remaining.get("cities_remaining", -1) if remaining.get("cities_remaining") is not None else -1,
            )

            resource_cards = player.get("resource_cards", {}) if isinstance(player.get("resource_cards"), dict) else {}
            for resource in TRACKED_RESOURCES:
                if is_hidden:
                    append_metric(f"resource.{resource}", -1.0)
                else:
                    value = resource_cards.get(resource)
                    append_metric(f"resource.{resource}", float(value) if isinstance(value, int) else -1.0)

            grouped = dev_cards.get("grouped_counts", {}) if isinstance(dev_cards.get("grouped_counts"), dict) else {}
            for key in ["knight", "progress", "victory_point"]:
                if is_hidden:
                    append_metric(f"dev_grouped.{key}", -1.0)
                else:
                    value = grouped.get(key)
                    append_metric(f"dev_grouped.{key}", float(value) if isinstance(value, int) else -1.0)

            append_metric("hidden_from_viewer", 1.0 if is_hidden else 0.0)

            for resource in TRACKED_RESOURCES:
                append_metric(f"mask.resource.{resource}", 1.0 if is_hidden else 0.0)
            for key in ["knight", "progress", "victory_point"]:
                append_metric(f"mask.dev_grouped.{key}", 1.0 if is_hidden else 0.0)

            return values, names

        for i in range(4):
            player = ordered_players[i] if i < len(ordered_players) else {}
            block_values, block_names = player_block(i, player)
            player_values.extend(block_values)
            player_names.extend(block_names)

        add_group("players", player_values, player_names)
        key_indices["player0_resource_brick"] = key_indices["players.p0.resource.BRICK"]
        key_indices["player1_resource_brick"] = key_indices["players.p1.resource.BRICK"]

        self_expected = 0.0
        opp_expected = 0.0
        self_blocked = 0.0
        opp_blocked = 0.0
        self_expected_by_res: Dict[str, float] = {resource: 0.0 for resource in TRACKED_RESOURCES}
        opp_expected_by_res: Dict[str, float] = {resource: 0.0 for resource in TRACKED_RESOURCES}
        self_blocked_by_res: Dict[str, float] = {resource: 0.0 for resource in TRACKED_RESOURCES}
        opp_blocked_by_res: Dict[str, float] = {resource: 0.0 for resource in TRACKED_RESOURCES}

        self_settlement_local_ev: List[float] = []
        self_city_local_ev: List[float] = []
        opp_settlement_local_ev: List[float] = []
        opp_city_local_ev: List[float] = []

        weight_by_tile = {
            (tile.get("x"), tile.get("y")): PROBABILITY_WEIGHTS.get(tile.get("number"), 0.0)
            for tile in observed.get("tiles", [])
            if isinstance(tile, dict)
        }
        resource_by_tile = {
            (tile.get("x"), tile.get("y")): str(tile.get("type") or "UNKNOWN").upper()
            for tile in observed.get("tiles", [])
            if isinstance(tile, dict)
        }
        robber_xy = (robber.get("x"), robber.get("y"))

        for structure in observed.get("structures", []):
            if not isinstance(structure, dict):
                continue
            owner_rel = rel_map.get(structure.get("owner_id"), "none")
            multiplier = 2.0 if structure.get("type") == "city" else 1.0
            adjacency = structure.get("adjacent_tiles", []) if isinstance(structure.get("adjacent_tiles"), list) else []
            local_sum = 0.0
            for adj in adjacency:
                if not isinstance(adj, dict):
                    continue
                coord = (adj.get("x"), adj.get("y"))
                weight = weight_by_tile.get(coord, 0.0) * multiplier
                local_sum += weight
                tile_resource = resource_by_tile.get(coord, "UNKNOWN")

                if not self.adjacency_features_enabled:
                    continue

                if owner_rel == "self":
                    self_expected += weight
                    if tile_resource in TRACKED_RESOURCES:
                        self_expected_by_res[tile_resource] += weight
                    if coord == robber_xy:
                        self_blocked += weight
                        if tile_resource in TRACKED_RESOURCES:
                            self_blocked_by_res[tile_resource] += weight
                elif owner_rel in {"op1", "op2", "op3"}:
                    opp_expected += weight
                    if tile_resource in TRACKED_RESOURCES:
                        opp_expected_by_res[tile_resource] += weight
                    if coord == robber_xy:
                        opp_blocked += weight
                        if tile_resource in TRACKED_RESOURCES:
                            opp_blocked_by_res[tile_resource] += weight

            if not self.adjacency_features_enabled:
                continue

            if owner_rel == "self":
                if structure.get("type") == "city":
                    self_city_local_ev.append(local_sum)
                elif structure.get("type") == "settlement":
                    self_settlement_local_ev.append(local_sum)
            elif owner_rel in {"op1", "op2", "op3"}:
                if structure.get("type") == "city":
                    opp_city_local_ev.append(local_sum)
                elif structure.get("type") == "settlement":
                    opp_settlement_local_ev.append(local_sum)

        def mean_and_max(values: List[float]) -> Tuple[float, float]:
            if not values:
                return 0.0, 0.0
            return (sum(values) / len(values), max(values))

        self_settlement_mean, self_settlement_max = mean_and_max(self_settlement_local_ev)
        self_city_mean, self_city_max = mean_and_max(self_city_local_ev)
        opp_settlement_mean, opp_settlement_max = mean_and_max(opp_settlement_local_ev)
        opp_city_mean, opp_city_max = mean_and_max(opp_city_local_ev)

        engineered_values = [
            self_expected,
            opp_expected,
            self_blocked,
            opp_blocked,
            *[self_expected_by_res[r] for r in TRACKED_RESOURCES],
            *[opp_expected_by_res[r] for r in TRACKED_RESOURCES],
            *[self_blocked_by_res[r] for r in TRACKED_RESOURCES],
            *[opp_blocked_by_res[r] for r in TRACKED_RESOURCES],
            self_settlement_mean,
            self_settlement_max,
            self_city_mean,
            self_city_max,
            opp_settlement_mean,
            opp_settlement_max,
            opp_city_mean,
            opp_city_max,
        ]
        engineered_names = [
            "self_expected_production_total",
            "opp_expected_production_total",
            "self_blocked_production_total",
            "opp_blocked_production_total",
            *[f"self_expected_production.{r}" for r in TRACKED_RESOURCES],
            *[f"opp_expected_production.{r}" for r in TRACKED_RESOURCES],
            *[f"self_blocked_production.{r}" for r in TRACKED_RESOURCES],
            *[f"opp_blocked_production.{r}" for r in TRACKED_RESOURCES],
            "self_settlement_local_ev_mean",
            "self_settlement_local_ev_max",
            "self_city_local_ev_mean",
            "self_city_local_ev_max",
            "opp_settlement_local_ev_mean",
            "opp_settlement_local_ev_max",
            "opp_city_local_ev_mean",
            "opp_city_local_ev_max",
        ]
        add_group("engineered", engineered_values, engineered_names)

        key_indices["engineered_self_expected_production"] = key_indices["engineered.self_expected_production_total"]
        key_indices["engineered_opp_expected_production"] = key_indices["engineered.opp_expected_production_total"]
        key_indices["engineered_self_blocked_production"] = key_indices["engineered.self_blocked_production_total"]
        key_indices["engineered_opp_blocked_production"] = key_indices["engineered.opp_blocked_production_total"]

        schema_hash = hashlib.sha256("|".join(feature_names).encode("utf-8")).hexdigest()

        meta = {
            "state_id": (turn_metadata.get("state_id") if isinstance(turn_metadata, dict) else None),
            "position_id": (turn_metadata.get("position_id") if isinstance(turn_metadata, dict) else None),
            "turn_index": turn_metadata.get("turn_index"),
            "current_player_id": current_player_id,
            "vector_length": len(x),
            "resource_order": RESOURCE_ORDER,
            "tracked_resources": TRACKED_RESOURCES,
            "feature_schema_hash": schema_hash,
            "adjacency_features_enabled": self.adjacency_features_enabled,
        }
        return x, feature_names, feature_groups, key_indices, meta


def make_training_row(
    state_payload: Dict[str, Any],
    vectorized: VectorizedState,
    label_win: Optional[int] = None,
    game_id: Optional[str] = None,
    split: Optional[str] = None,
) -> Dict[str, Any]:
    turn_meta = (state_payload.get("observed_state", {}).get("turn_metadata", {})
                 if isinstance(state_payload.get("observed_state"), dict)
                 else {})

    return {
        "game_id": game_id,
        "state_id": vectorized.metadata.get("state_id"),
        "position_id": vectorized.metadata.get("position_id"),
        "turn_index": vectorized.metadata.get("turn_index"),
        "current_player_id": vectorized.metadata.get("current_player_id"),
        "vector_version": vectorized.version,
        "feature_schema_hash": vectorized.metadata.get("feature_schema_hash"),
        "extractor_schema_version": state_payload.get("schema_version"),
        "split": split,
        "y_win": label_win,
        "x": vectorized.x,
        "sample_weight": 1.0,
        "last_action": turn_meta.get("last_action") if isinstance(turn_meta, dict) else None,
    }
