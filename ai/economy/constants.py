from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Set, Tuple


BOARD_SIZE = 7
STRUCTURE_ORIENTATION_COUNT = 2
ROAD_ORIENTATION_COUNT = 3

TRACKED_RESOURCES: Tuple[str, ...] = ("BRICK", "WOOL", "ORE", "GRAIN", "LUMBER")
RESOURCE_SHORT = {
    "BRICK": "B",
    "WOOL": "W",
    "ORE": "O",
    "GRAIN": "G",
    "LUMBER": "L",
}

PORT_NAMES: Tuple[str, ...] = (
    "three_to_one",
    "brick_two_to_one",
    "wool_two_to_one",
    "ore_two_to_one",
    "grain_two_to_one",
    "lumber_two_to_one",
)

PIP_WEIGHTS: Dict[int, float] = {
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

VALID_TILE_COORDINATES: Set[Tuple[int, int]] = {
    (1, 1),
    (2, 1),
    (3, 1),
    (1, 2),
    (2, 2),
    (3, 2),
    (4, 2),
    (1, 3),
    (2, 3),
    (3, 3),
    (4, 3),
    (5, 3),
    (2, 4),
    (3, 4),
    (4, 4),
    (5, 4),
    (3, 5),
    (4, 5),
    (5, 5),
}

STANDARD_RESOURCE_TILE_COUNTS: Dict[str, int] = {
    "BRICK": 3,
    "WOOL": 4,
    "ORE": 3,
    "GRAIN": 4,
    "LUMBER": 4,
}

TOTAL_PRODUCTIVE_BOARD_PIPS = 58.0
AVERAGE_TILE_PIPS = TOTAL_PRODUCTIVE_BOARD_PIPS / 18.0
STANDARD_RESOURCE_BASELINE_PIPS: Dict[str, float] = {
    resource: count * AVERAGE_TILE_PIPS
    for resource, count in STANDARD_RESOURCE_TILE_COUNTS.items()
}

BUILD_RECIPES: Dict[str, Dict[str, int]] = {
    "road": {"BRICK": 1, "LUMBER": 1},
    "settlement": {"BRICK": 1, "LUMBER": 1, "WOOL": 1, "GRAIN": 1},
    "city": {"ORE": 3, "GRAIN": 2},
    "dev": {"ORE": 1, "GRAIN": 1, "WOOL": 1},
}

OPENING_BALANCE_WEIGHTS: Dict[str, float] = {
    "diversity": 0.4,
    "evenness": 0.4,
    "recipe_coverage": 0.2,
}

OPENING_ARCHETYPE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "road": {
        "resource_component": 0.45,
        "expansion_frontier": 0.35,
        "port_alignment": 0.20,
    },
    "settlement": {
        "resource_component": 0.45,
        "expansion_frontier": 0.35,
        "ev": 0.20,
    },
    "city": {
        "resource_component": 0.55,
        "ev": 0.25,
        "upgrade_quality": 0.20,
    },
    "dev": {
        "resource_component": 0.50,
        "ev": 0.25,
        "port_fallback": 0.25,
    },
}

MAX_SINGLE_VERTEX_PIPS = 15.0
MAX_PAIR_PIPS = 30.0
EXPECTED_PRODUCTION_CAP = 18.0

PORT_VERTEX_MAP: Dict[str, Tuple[Tuple[int, int, int], ...]] = {
    "brick_two_to_one": ((4, 1, 0), (4, 2, 1)),
    "wool_two_to_one": ((4, 5, 0), (5, 6, 1)),
    "ore_two_to_one": ((1, 3, 0), (2, 5, 1)),
    "grain_two_to_one": ((0, 1, 0), (1, 3, 1)),
    "lumber_two_to_one": ((2, 0, 0), (2, 1, 1)),
    "three_to_one": (
        (0, 0, 0),
        (1, 1, 1),
        (5, 2, 0),
        (6, 4, 1),
        (5, 4, 0),
        (6, 5, 1),
        (3, 5, 0),
        (3, 6, 1),
    ),
}

PORT_RESOURCE_MAP: Dict[str, str | None] = {
    "brick_two_to_one": "BRICK",
    "wool_two_to_one": "WOOL",
    "ore_two_to_one": "ORE",
    "grain_two_to_one": "GRAIN",
    "lumber_two_to_one": "LUMBER",
    "three_to_one": None,
}


def clamp_board_coord(x_coord: int, y_coord: int) -> bool:
    return 0 <= x_coord < BOARD_SIZE and 0 <= y_coord < BOARD_SIZE


def vertex_id(x_coord: int, y_coord: int, orientation: int) -> str:
    return f"vertex:{x_coord}:{y_coord}:{orientation}"


def road_id(x_coord: int, y_coord: int, orientation: int) -> str:
    return f"road:{x_coord}:{y_coord}:{orientation}"


def vertex_adjacent_tiles(x_coord: int, y_coord: int, orientation: int) -> List[Dict[str, int]]:
    if orientation == 0:
        candidates = [
            (x_coord, y_coord),
            (x_coord, y_coord + 1),
            (x_coord + 1, y_coord + 1),
        ]
    else:
        candidates = [
            (x_coord, y_coord),
            (x_coord, y_coord - 1),
            (x_coord - 1, y_coord - 1),
        ]
    return [{"x": tx, "y": ty} for tx, ty in candidates if clamp_board_coord(tx, ty)]


def vertex_incident_roads(x_coord: int, y_coord: int, orientation: int) -> List[Dict[str, int]]:
    if orientation == 0:
        candidates = [
            (x_coord, y_coord, 0),
            (x_coord, y_coord, 1),
            (x_coord, y_coord + 1, 2),
        ]
    else:
        candidates = [
            (x_coord, y_coord - 1, 0),
            (x_coord - 1, y_coord - 1, 1),
            (x_coord - 1, y_coord - 1, 2),
        ]
    return [
        {"x": rx, "y": ry, "orientation": ro}
        for rx, ry, ro in candidates
        if clamp_board_coord(rx, ry) and 0 <= ro < ROAD_ORIENTATION_COUNT
    ]


def road_endpoint_vertices(x_coord: int, y_coord: int, orientation: int) -> List[Dict[str, int]]:
    if orientation == 0:
        candidates = [(x_coord, y_coord, 0), (x_coord, y_coord + 1, 1)]
    elif orientation == 1:
        candidates = [(x_coord, y_coord, 0), (x_coord + 1, y_coord + 1, 1)]
    else:
        candidates = [(x_coord, y_coord - 1, 0), (x_coord + 1, y_coord + 1, 1)]
    return [
        {"x": vx, "y": vy, "orientation": vo}
        for vx, vy, vo in candidates
        if clamp_board_coord(vx, vy) and 0 <= vo < STRUCTURE_ORIENTATION_COUNT
    ]


def adjacent_vertices_for_vertex(x_coord: int, y_coord: int, orientation: int) -> List[Dict[str, int]]:
    neighbors: List[Dict[str, int]] = []
    seen: Set[Tuple[int, int, int]] = set()
    for road in vertex_incident_roads(x_coord, y_coord, orientation):
        endpoints = road_endpoint_vertices(road["x"], road["y"], road["orientation"])
        for endpoint in endpoints:
            key = (endpoint["x"], endpoint["y"], endpoint["orientation"])
            if key == (x_coord, y_coord, orientation) or key in seen:
                continue
            seen.add(key)
            neighbors.append(endpoint)
    return neighbors


def build_valid_structure_slots() -> Tuple[Tuple[int, int, int], ...]:
    valid: Set[Tuple[int, int, int]] = set()
    for x_coord, y_coord in VALID_TILE_COORDINATES:
        for vertex in (
            (x_coord, y_coord, 0),
            (x_coord, y_coord, 1),
            (x_coord + 1, y_coord + 1, 1),
            (x_coord - 1, y_coord - 1, 0),
            (x_coord, y_coord + 1, 1),
            (x_coord, y_coord - 1, 0),
        ):
            vx, vy, vo = vertex
            if clamp_board_coord(vx, vy) and 0 <= vo < STRUCTURE_ORIENTATION_COUNT:
                valid.add(vertex)
    return tuple(sorted(valid))


VALID_STRUCTURE_SLOTS: Tuple[Tuple[int, int, int], ...] = build_valid_structure_slots()
VALID_STRUCTURE_SLOT_SET: Set[Tuple[int, int, int]] = set(VALID_STRUCTURE_SLOTS)


def build_valid_road_slots(structure_slots: Iterable[Tuple[int, int, int]]) -> Tuple[Tuple[int, int, int], ...]:
    structure_slot_set = set(structure_slots)
    valid: Set[Tuple[int, int, int]] = set()
    for x_coord in range(BOARD_SIZE):
        for y_coord in range(BOARD_SIZE):
            for orientation in range(ROAD_ORIENTATION_COUNT):
                endpoints = road_endpoint_vertices(x_coord, y_coord, orientation)
                if len(endpoints) != 2:
                    continue
                t0 = (endpoints[0]["x"], endpoints[0]["y"], endpoints[0]["orientation"])
                t1 = (endpoints[1]["x"], endpoints[1]["y"], endpoints[1]["orientation"])
                if t0 in structure_slot_set and t1 in structure_slot_set:
                    valid.add((x_coord, y_coord, orientation))
    return tuple(sorted(valid))


VALID_ROAD_SLOTS: Tuple[Tuple[int, int, int], ...] = build_valid_road_slots(VALID_STRUCTURE_SLOTS)


def default_structure_slots() -> List[Dict[str, int]]:
    return [
        {"x": x_coord, "y": y_coord, "orientation": orientation}
        for x_coord, y_coord, orientation in VALID_STRUCTURE_SLOTS
    ]


def default_road_slots() -> List[Dict[str, int]]:
    return [
        {"x": x_coord, "y": y_coord, "orientation": orientation}
        for x_coord, y_coord, orientation in VALID_ROAD_SLOTS
    ]


def recipe_resources(recipe_name: str) -> Tuple[str, ...]:
    recipe = BUILD_RECIPES.get(recipe_name, {})
    return tuple(sorted(recipe.keys()))


def compact_resource_list(resources: Sequence[str]) -> str:
    return "".join(RESOURCE_SHORT.get(resource, resource[:1]) for resource in resources)
