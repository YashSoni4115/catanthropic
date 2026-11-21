"""CLI.

    python -m catan_rl.eval.run --agents A,B,C,D --seeds 125 --workers 8 --out DIR [--seed-base 0] [--keep-awake]
    python -m catan_rl.eval.run --compare DIR_A DIR_B [--slot 0]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

from catan_rl.eval.match import load, metrics, paired_comparison, provenance, run_match, save


def keep_awake() -> bool:
    """macOS: prevent sleep while this process lives (caffeinate -w PID). No-op elsewhere."""
    if sys.platform != "darwin":
        return False
    subprocess.Popen(["caffeinate", "-dims", "-w", str(os.getpid())])
    return True


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--agents")
    p.add_argument("--seeds", type=int, default=125)
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out")
    p.add_argument("--keep-awake", action="store_true")
    p.add_argument("--compare", nargs=2, metavar=("DIR_A", "DIR_B"))
    p.add_argument("--slot", type=int, default=0)
    a = p.parse_args(argv)

    if a.compare:
        c = paired_comparison(load(a.compare[0]), load(a.compare[1]), slot=a.slot)
        print(json.dumps(c, indent=2))
        return c

    lineup = a.agents.split(",")
    caffeinated = keep_awake() if a.keep_awake else False
    wall0, proc0 = time.time(), time.perf_counter()  # perf_counter (mach clock) stops during sleep on macOS
    results = run_match(lineup, a.seeds, a.seed_base, a.workers)
    wall, proc = time.time() - wall0, time.perf_counter() - proc0
    m = metrics(results, lineup)
    m["provenance"] = provenance(lineup, range(a.seed_base, a.seed_base + a.seeds))
    m["timing"] = {"wall_clock_s": wall, "in_process_s": proc, "games_per_s": len(results) / proc,
                   "workers": a.workers, "keep_awake": caffeinated}
    save(a.out, results, m)
    for s in m["per_slot"]:
        w, d = s["win_rate"], s["win_rate"]["design_effect"]
        print(f"slot {s['slot']} {s['agent']:28} win {w['p']:.3f} clustered [{w['ci95'][0]:.3f}, {w['ci95'][1]:.3f}] "
              f"wilson [{w['wilson_ci95_reference'][0]:.3f}, {w['wilson_ci95_reference'][1]:.3f}] "
              f"deff {'n/a' if d is None else f'{d:.2f}'}  VP {s['mean_final_vp']:.2f}")
    t = m["timing"]
    print(f"games {m['games']}  seeds {a.seeds}  no-winner {m['no_winner_games']}  turns {m['mean_game_length_turns']:.1f}  "
          f"{t['games_per_s']:.2f} games/s  in-process {t['in_process_s']:.1f}s  wall {t['wall_clock_s']:.1f}s")
    return m


if __name__ == "__main__":
    main()
