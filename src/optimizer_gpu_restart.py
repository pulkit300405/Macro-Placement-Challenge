from typing import Dict, Tuple, List
from evaluator_baseline import BaselineEvaluator, placement_from_netlist
from parser import Netlist
import numpy as np
import concurrent.futures
import copy


def _single_restart(args):
    netlist, base_placement, seed, noise_scale = args
    rng = np.random.default_rng(seed)
    movable = list(netlist.movable_macros().keys())
    cx, cy = netlist.core_x, netlist.core_y
    cw, ch = netlist.core_width, netlist.core_height

    placement = dict(base_placement)
    for name in movable:
        m = netlist.macros[name]
        x, y = placement[name]
        nx = float(np.clip(x + rng.normal(0, noise_scale), cx, cx + cw - m.width))
        ny = float(np.clip(y + rng.normal(0, noise_scale), cy, cy + ch - m.height))
        placement[name] = (nx, ny)

    ev = BaselineEvaluator(netlist)
    result = ev.evaluate(placement)
    return placement, result["cost"]


class ParallelRestarter:

    def __init__(self, netlist: Netlist, n_restarts: int = 8, noise_scale: float = 500.0):
        self.netlist = netlist
        self.n_restarts = n_restarts
        self.noise_scale = noise_scale

    def run(self, base_placement: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
        print(f"  [Restart] Running {self.n_restarts} parallel restarts...")

        args = [
            (self.netlist, base_placement, i * 42, self.noise_scale)
            for i in range(self.n_restarts)
        ]

        best_placement = base_placement
        best_cost = float("inf")

        with concurrent.futures.ProcessPoolExecutor(max_workers=min(self.n_restarts, 4)) as ex:
            futures = [ex.submit(_single_restart, a) for a in args]
            for i, f in enumerate(concurrent.futures.as_completed(futures)):
                placement, cost = f.result()
                print(f"    restart {i:2d} | cost {cost:.4f}")
                if cost < best_cost:
                    best_cost = cost
                    best_placement = placement

        print(f"  [Restart] Best restart cost: {best_cost:.4f}")
        return best_placement
