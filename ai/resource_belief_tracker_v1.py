from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple


RESOURCES: Tuple[str, ...] = ("BRICK", "WOOL", "ORE", "GRAIN", "LUMBER")
RESOURCE_INDEX: Dict[str, int] = {resource: idx for idx, resource in enumerate(RESOURCES)}

SPEND_RECIPES: Dict[str, Tuple[int, int, int, int, int]] = {
    "road": (1, 0, 0, 0, 1),
    "settlement": (1, 1, 0, 1, 1),
    "city": (0, 0, 3, 2, 0),
    "dev_card": (0, 1, 1, 1, 0),
}

Hand = Tuple[int, int, int, int, int]


class BeliefContradictionError(RuntimeError):
    pass


@dataclass
class PlayerBelief:
    player_id: str
    pmf: Dict[Hand, float]
    known_total: Optional[int] = None
    last_event: Optional[str] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class ResourceBeliefTrackerV1:
    def __init__(self, strict: bool = True, prune_epsilon: float = 1e-15) -> None:
        self.strict = strict
        self.prune_epsilon = prune_epsilon
        self.players: Dict[str, PlayerBelief] = {}

    def initialize_player(
        self,
        player_id: str,
        observed_total_cards: int,
        known_counts: Optional[Dict[str, int]] = None,
    ) -> None:
        if observed_total_cards < 0:
            raise ValueError("observed_total_cards must be >= 0")
        fixed = self._normalize_known_counts(known_counts or {})
        fixed_total = sum(fixed.values())
        if fixed_total > observed_total_cards:
            raise ValueError("known_counts exceed observed_total_cards")

        free_total = observed_total_cards - fixed_total
        free_resources = [resource for resource in RESOURCES if resource not in fixed]

        pmf: Dict[Hand, float] = {}
        for allocation in self._enumerate_compositions(total=free_total, dimensions=len(free_resources)):
            counts = [0, 0, 0, 0, 0]
            for resource, value in fixed.items():
                counts[RESOURCE_INDEX[resource]] = value
            for resource, value in zip(free_resources, allocation):
                counts[RESOURCE_INDEX[resource]] = value
            hand = tuple(counts)
            pmf[hand] = pmf.get(hand, 0.0) + 1.0

        self.players[player_id] = PlayerBelief(player_id=player_id, pmf=self._normalize(pmf), known_total=observed_total_cards)

    def apply_event(self, event: Dict[str, Any]) -> None:
        event_type = str(event.get("type") or "").strip().lower()
        if event_type == "known_resource_gain":
            self._apply_known_gain(event)
            return
        if event_type == "known_spend":
            self._apply_known_spend(event)
            return
        if event_type == "random_steal":
            self._apply_random_steal(event)
            return
        if event_type == "public_total_reconcile":
            self._apply_public_total_reconcile(event)
            return
        raise ValueError(f"Unsupported event type: {event_type}")

    def apply_events(self, events: Iterable[Dict[str, Any]]) -> None:
        for event in events:
            self.apply_event(event)

    def reconcile_public_totals(self, player_id: str, total_cards: int) -> None:
        self.apply_event({"type": "public_total_reconcile", "player_id": player_id, "total_cards": total_cards})

    def get_player_summary(self, player_id: str) -> Dict[str, Any]:
        belief = self._get_player(player_id)
        pmf = belief.pmf

        expected_counts: Dict[str, float] = {}
        p_at_least_one: Dict[str, float] = {}
        for resource in RESOURCES:
            idx = RESOURCE_INDEX[resource]
            expected_counts[resource] = sum(prob * hand[idx] for hand, prob in pmf.items())
            p_at_least_one[resource] = sum(prob for hand, prob in pmf.items() if hand[idx] >= 1)

        affordability = {
            spend_name: self._prob_can_afford(pmf, recipe)
            for spend_name, recipe in SPEND_RECIPES.items()
        }

        return {
            "player_id": player_id,
            "known_total": belief.known_total,
            "support_size": len(pmf),
            "expected_counts": expected_counts,
            "p_at_least_one": p_at_least_one,
            "p_can_afford": affordability,
            "last_event": belief.last_event,
            "diagnostics": dict(belief.diagnostics),
        }

    def snapshot(self) -> Dict[str, Any]:
        output: Dict[str, Any] = {}
        for player_id in sorted(self.players.keys()):
            output[player_id] = self.get_player_summary(player_id)
        return output

    def _apply_known_gain(self, event: Dict[str, Any]) -> None:
        player_id = self._require_player_id(event)
        resource = self._normalize_resource(event.get("resource"))
        amount = int(event.get("amount", 1))
        if amount < 0:
            raise ValueError("known_resource_gain amount must be >= 0")

        belief = self._get_player(player_id)
        idx = RESOURCE_INDEX[resource]

        updated: Dict[Hand, float] = {}
        for hand, prob in belief.pmf.items():
            shifted = list(hand)
            shifted[idx] += amount
            key = tuple(shifted)
            updated[key] = updated.get(key, 0.0) + prob

        belief.pmf = self._normalize(updated)
        if belief.known_total is not None:
            belief.known_total += amount
        belief.last_event = "known_resource_gain"
        belief.diagnostics = {"resource": resource, "amount": amount}
        self._update_known_total_from_support(belief)

    def _apply_known_spend(self, event: Dict[str, Any]) -> None:
        player_id = self._require_player_id(event)
        spend_vector = self._parse_spend_vector(event)
        belief = self._get_player(player_id)

        updated: Dict[Hand, float] = {}
        valid_mass = 0.0
        for hand, prob in belief.pmf.items():
            if any(hand[idx] < spend_vector[idx] for idx in range(len(RESOURCES))):
                continue
            post = tuple(hand[idx] - spend_vector[idx] for idx in range(len(RESOURCES)))
            updated[post] = updated.get(post, 0.0) + prob
            valid_mass += prob

        if valid_mass <= 0.0:
            self._handle_contradiction(
                belief,
                reason="known_spend_infeasible",
                details={"spend_vector": spend_vector},
            )
            return

        belief.pmf = self._normalize(updated)
        if belief.known_total is not None:
            belief.known_total -= sum(spend_vector)
            if belief.known_total < 0:
                self._handle_contradiction(
                    belief,
                    reason="known_total_became_negative_after_spend",
                    details={"spend_vector": spend_vector},
                )
                return
        belief.last_event = "known_spend"
        belief.diagnostics = {"spend_vector": spend_vector, "dropped_mass": 1.0 - valid_mass}
        self._update_known_total_from_support(belief)

    def _apply_random_steal(self, event: Dict[str, Any]) -> None:
        thief_id = self._require_id(event, "thief_id")
        victim_id = self._require_id(event, "victim_id")
        amount = int(event.get("amount", 1))
        if amount < 1:
            raise ValueError("random_steal amount must be >= 1")

        for _ in range(amount):
            self._apply_single_random_steal(thief_id=thief_id, victim_id=victim_id)

    def _apply_single_random_steal(self, thief_id: str, victim_id: str) -> None:
        thief = self._get_player(thief_id)
        victim = self._get_player(victim_id)

        resource_type_distribution = {resource: 0.0 for resource in RESOURCES}
        victim_post: Dict[Hand, float] = {}

        for hand, prob in victim.pmf.items():
            total_cards = sum(hand)
            if total_cards <= 0:
                continue
            for idx, resource in enumerate(RESOURCES):
                count = hand[idx]
                if count <= 0:
                    continue
                transition_prob = prob * (count / total_cards)
                resource_type_distribution[resource] += transition_prob
                post = list(hand)
                post[idx] -= 1
                key = tuple(post)
                victim_post[key] = victim_post.get(key, 0.0) + transition_prob

        total_type_mass = sum(resource_type_distribution.values())
        if total_type_mass <= 0.0:
            self._handle_contradiction(
                victim,
                reason="random_steal_from_empty_support",
                details={"thief_id": thief_id},
            )
            return

        victim.pmf = self._normalize(victim_post)

        thief_post: Dict[Hand, float] = {}
        for hand, prob in thief.pmf.items():
            for idx, resource in enumerate(RESOURCES):
                p_resource = resource_type_distribution[resource]
                if p_resource <= 0.0:
                    continue
                post = list(hand)
                post[idx] += 1
                key = tuple(post)
                thief_post[key] = thief_post.get(key, 0.0) + prob * p_resource

        thief.pmf = self._normalize(thief_post)

        victim.last_event = "random_steal_victim"
        thief.last_event = "random_steal_thief"
        victim.diagnostics = {"stolen_type_distribution": resource_type_distribution}
        thief.diagnostics = {"gained_type_distribution": resource_type_distribution}
        self._update_known_total_from_support(victim)
        self._update_known_total_from_support(thief)

    def _apply_public_total_reconcile(self, event: Dict[str, Any]) -> None:
        player_id = self._require_player_id(event)
        total_cards = int(event.get("total_cards"))
        if total_cards < 0:
            raise ValueError("public_total_reconcile total_cards must be >= 0")

        belief = self._get_player(player_id)
        updated = {hand: prob for hand, prob in belief.pmf.items() if sum(hand) == total_cards}
        if not updated:
            self._handle_contradiction(
                belief,
                reason="public_total_reconcile_empty_support",
                details={"required_total": total_cards},
            )
            return

        belief.pmf = self._normalize(updated)
        belief.known_total = total_cards
        belief.last_event = "public_total_reconcile"
        belief.diagnostics = {"required_total": total_cards}
        self._update_known_total_from_support(belief)

    def _parse_spend_vector(self, event: Dict[str, Any]) -> Hand:
        spend_type = event.get("spend_type")
        if isinstance(spend_type, str):
            key = spend_type.strip().lower()
            if key not in SPEND_RECIPES:
                raise ValueError(f"Unknown spend_type: {spend_type}")
            return SPEND_RECIPES[key]

        spend = event.get("spend")
        if not isinstance(spend, dict):
            raise ValueError("known_spend requires spend_type or spend dict")

        vec = [0, 0, 0, 0, 0]
        for resource in RESOURCES:
            value = int(spend.get(resource, 0))
            if value < 0:
                raise ValueError("spend resource values must be >= 0")
            vec[RESOURCE_INDEX[resource]] = value
        return tuple(vec)

    def _normalize(self, pmf: Dict[Hand, float]) -> Dict[Hand, float]:
        clean = {hand: prob for hand, prob in pmf.items() if prob > self.prune_epsilon}
        total = sum(clean.values())
        if total <= 0.0:
            raise BeliefContradictionError("PMF normalization failed: zero total mass")
        return {hand: prob / total for hand, prob in clean.items()}

    def _prob_can_afford(self, pmf: Dict[Hand, float], recipe: Hand) -> float:
        return sum(
            prob
            for hand, prob in pmf.items()
            if all(hand[idx] >= recipe[idx] for idx in range(len(RESOURCES)))
        )

    def _normalize_known_counts(self, known_counts: Dict[str, int]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for resource, value in known_counts.items():
            norm_resource = self._normalize_resource(resource)
            int_value = int(value)
            if int_value < 0:
                raise ValueError("known_counts values must be >= 0")
            out[norm_resource] = int_value
        return out

    def _normalize_resource(self, resource: Any) -> str:
        value = str(resource or "").strip().upper()
        if value not in RESOURCE_INDEX:
            raise ValueError(f"Unknown resource: {resource}")
        return value

    def _handle_contradiction(self, belief: PlayerBelief, reason: str, details: Dict[str, Any]) -> None:
        belief.last_event = "contradiction"
        belief.diagnostics = {"reason": reason, **details}
        if self.strict:
            raise BeliefContradictionError(f"Belief contradiction for player={belief.player_id}: {reason} details={details}")

    def _require_player_id(self, event: Dict[str, Any]) -> str:
        return self._require_id(event, "player_id")

    def _require_id(self, event: Dict[str, Any], key: str) -> str:
        value = str(event.get(key) or "").strip()
        if not value:
            raise ValueError(f"Missing required event field: {key}")
        return value

    def _get_player(self, player_id: str) -> PlayerBelief:
        if player_id not in self.players:
            raise ValueError(f"Unknown player_id: {player_id}")
        return self.players[player_id]

    def _update_known_total_from_support(self, belief: PlayerBelief) -> None:
        totals = {sum(hand) for hand in belief.pmf.keys()}
        belief.known_total = next(iter(totals)) if len(totals) == 1 else None

    def _enumerate_compositions(self, total: int, dimensions: int) -> Iterable[Tuple[int, ...]]:
        if dimensions <= 0:
            return
        if dimensions == 1:
            yield (total,)
            return
        for first in range(total + 1):
            for rest in self._enumerate_compositions(total - first, dimensions - 1):
                yield (first,) + rest
