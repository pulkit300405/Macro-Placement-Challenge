from typing import Dict, Tuple, List
from evaluator_incremental import IncrementalEvaluator
from parser import Netlist
import numpy as np


class PairwiseSwap:

    def __init__(self, netlist: Netlist, evaluator: IncrementalEvaluator):
        self.netlist = netlist
        self.evaluator = evaluator
        self.max_iters = 100
        self.candidates_per_macro = 10

    def optimize(self, placement: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
        self.evaluator.initialize(placement)
        current = dict(placement)
        best = dict(placement)
        best_cost = self.evaluator.evaluate()["cost"]

        movable = list(self.netlist.movable_macros().keys())

        for it in range(self.max_iters):
            improved = False
            np.random.shuffle(movable)

            for i, a in enumerate(movable):
                candidates = self._get_candidates(a, movable, current)

                for b in candidates:
                    ax, ay = current[a]
                    bx, by = current[b]

                    ma = self.netlist.macros[a]
                    mb = self.netlist.macros[b]

                    if not self._fits_bounds(mb, ax, ay) or not self._fits_bounds(ma, bx, by):
                        continue

                    before = self.evaluator.evaluate()["cost"]

                    self.evaluator.move(a, bx, by)
                    self.evaluator.move(b, ax, ay)
                    after = self.evaluator.evaluate()["cost"]

                    if after < before:
                        current[a] = (bx, by)
                        current[b] = (ax, ay)
                        if after < best_cost:
                            best_cost = after
                            best = dict(current)
                            improved = True
                    else:
                        self.evaluator.move(a, ax, ay)
                        self.evaluator.move(b, bx, by)

            if it % 20 == 0:
                print(f"  [Swap] iter {it:3d} | cost {best_cost:.4f}")

            if not improved:
                break

        return best

    def _get_candidates(self, name: str, movable: List[str], placement: Dict) -> List[str]:
        x, y = placement[name]
        distances = []
        for other in movable:
            if other == name:
                continue
            ox, oy = placement[other]
            dist = abs(x - ox) + abs(y - oy)
            distances.append((dist, other))
        distances.sort()
        return [d[1] for d in distances[:self.candidates_per_macro]]

    def _fits_bounds(self, macro, x, y) -> bool:
        cx = self.netlist.core_x
        cy = self.netlist.core_y
        cw = self.netlist.core_width
        ch = self.netlist.core_height
        return (x >= cx and y >= cy and
                x + macro.width <= cx + cw and
                y + macro.height <= cy + ch)
