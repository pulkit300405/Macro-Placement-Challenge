"""
Baseline Evaluator
==================
Calculates placement cost FROM SCRATCH every time (no incremental tricks).
This is the SLOW reference implementation.

Metrics:
  - HPWL  : Half Perimeter Wire Length (lower = better routing)
  - Overlap: Total overlapping area between macros (must be 0 to be valid)
  - Cost   : weighted combination used as proxy score

Used for:
  1. Correctness reference (incremental must match this)
  2. Speed benchmark baseline (target: incremental is 300x faster)

Author: Team ZeroLatency (Pulkit + Lakshita)
"""

from typing import Dict, Tuple
from parser import Netlist, Macro


# ─── Cost Weights (tune later) ────────────────────────────────────────────────

HPWL_WEIGHT    = 1.0
OVERLAP_WEIGHT = 10.0   # heavy penalty — overlaps disqualify submission


# ─── Baseline Evaluator ───────────────────────────────────────────────────────

class BaselineEvaluator:
    """
    Full from-scratch cost evaluator.
    Call evaluate(placement) to get cost dict.
    """

    def __init__(self, netlist: Netlist):
        self.netlist = netlist

    def evaluate(self, placement: Dict[str, Tuple[float, float]]) -> dict:
        """
        Compute full placement cost.

        Args:
            placement: {macro_name: (x, y)} — positions of ALL movable macros

        Returns:
            dict with keys: hpwl, overlap, cost, valid
        """
        hpwl    = self._compute_hpwl(placement)
        overlap = self._compute_overlap(placement)

        cost = (HPWL_WEIGHT * hpwl) + (OVERLAP_WEIGHT * overlap)

        return {
            "hpwl"    : hpwl,
            "overlap" : overlap,
            "cost"    : cost,
            "valid"   : overlap == 0.0
        }

    # ── HPWL ─────────────────────────────────────────────────────────────────

    def _compute_hpwl(self, placement: Dict[str, Tuple[float, float]]) -> float:
        """
        Half Perimeter Wire Length across all nets.
        For each net: HPWL = (max_x - min_x) + (max_y - min_y)
        """
        total_hpwl = 0.0
        macros = self.netlist.macros

        for net in self.netlist.nets:
            if len(net.pins) < 2:
                continue

            xs = []
            ys = []

            for (macro_name, ox, oy) in net.pins:
                if macro_name not in macros:
                    continue

                macro = macros[macro_name]

                if macro_name in placement:
                    px, py = placement[macro_name]
                else:
                    px, py = macro.x, macro.y

                pin_x = px + macro.width / 2 + ox
                pin_y = py + macro.height / 2 + oy

                xs.append(pin_x)
                ys.append(pin_y)

            if len(xs) < 2:
                continue

            total_hpwl += (max(xs) - min(xs)) + (max(ys) - min(ys))

        return total_hpwl

    # ── Overlap ───────────────────────────────────────────────────────────────

    def _compute_overlap(self, placement: Dict[str, Tuple[float, float]]) -> float:
        """
        Total overlapping area between all pairs of movable macros.
        O(n^2) naive — correct but slow. Incremental will fix this.
        """
        macros = self.netlist.macros
        names  = list(placement.keys())
        total_overlap = 0.0

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a = macros[names[i]]
                b = macros[names[j]]
                ax, ay = placement[names[i]]
                bx, by = placement[names[j]]

                ox = max(0.0, min(ax + a.width,  bx + b.width)  - max(ax, bx))
                oy = max(0.0, min(ay + a.height, by + b.height) - max(ay, by))
                total_overlap += ox * oy

        return total_overlap

    # ── Boundary Check ────────────────────────────────────────────────────────

    def check_bounds(self, placement: Dict[str, Tuple[float, float]]) -> dict:
        """check if all macros are within core area"""
        macros = self.netlist.macros
        cx, cy = self.netlist.core_x, self.netlist.core_y
        cw, ch = self.netlist.core_width, self.netlist.core_height

        results = {}
        for name, (x, y) in placement.items():
            m = macros[name]
            results[name] = (
                x >= cx and y >= cy and
                x + m.width  <= cx + cw and
                y + m.height <= cy + ch
            )
        return results

    def all_in_bounds(self, placement) -> bool:
        return all(self.check_bounds(placement).values())


# ─── Helper ───────────────────────────────────────────────────────────────────

def placement_from_netlist(netlist: Netlist) -> Dict[str, Tuple[float, float]]:
    """extract current x,y of all movable macros"""
    return {name: (m.x, m.y) for name, m in netlist.movable_macros().items()}


# ─── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, time
    from parser import ICCAD04Parser

    if len(sys.argv) < 2:
        print("Usage: python evaluator_baseline.py path/to/benchmark.aux")
        sys.exit(1)

    nl = ICCAD04Parser().parse(sys.argv[1])
    nl.summary()

    placement = placement_from_netlist(nl)
    ev = BaselineEvaluator(nl)

    runs = 100
    t0 = time.perf_counter()
    for _ in range(runs):
        result = ev.evaluate(placement)
    t1 = time.perf_counter()

    avg_ms = (t1 - t0) / runs * 1000

    print(f"\n[Baseline Eval Results]")
    print(f"  HPWL    : {result['hpwl']:.2f}")
    print(f"  Overlap : {result['overlap']:.2f}")
    print(f"  Cost    : {result['cost']:.4f}")
    print(f"  Valid   : {result['valid']}")
    print(f"\n[Speed]")
    print(f"  Avg eval time : {avg_ms:.2f} ms  ({runs} runs)")
    print(f"  Evals/sec     : {1000/avg_ms:.0f}")
    print(f"\n  >> Save this number. Incremental target: 300x faster.")
