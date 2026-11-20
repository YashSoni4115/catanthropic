from __future__ import annotations

from typing import Dict, List


def summarize_trade_context_v1(trade_context: Dict[str, object]) -> List[str]:
    leaders = trade_context.get("leaders", {}) if isinstance(trade_context.get("leaders"), dict) else {}
    counterparties = trade_context.get("counterparties", []) if isinstance(trade_context.get("counterparties"), list) else []
    lines: List[str] = []
    for resource_short, player_id in sorted(leaders.items()):
        lines.append(f"{player_id} strongest {resource_short.lower()} economy.")
    for counterparty in counterparties[:3]:
        player_id = counterparty.get("player_id")
        needs = counterparty.get("need_top2", [])
        offers = counterparty.get("offer_top2", [])
        if needs:
            lines.append(f"{player_id} likely wants {'/'.join(needs)}.")
        if offers:
            lines.append(f"{player_id} most likely offers {'/'.join(offers)}.")
    return lines
