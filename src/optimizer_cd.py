from typing import Dict, Tuple
from evaluator_incremental import IncrementalEvaluator
from parser import Netlist
import numpy as np


class CoordinateDescent:

    def __init__(self, netlist: Netlist, evaluator: IncrementalEvaluator):
        self.netlist = netlist
        self.evaluator = evaluator
        self.step_size = 100.0
        self.min_step = 1.0
        self.decay = 0.85
        self.max_iters = 200

    def optimize(self, placement: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
        self.evaluator.initialize(placement)
        current = dict(placement)
        best = dict(placement)
        best_cost = self.evaluator.evaluate()["cost"]

        movable = list(self.netlist.movable_macros().keys())
        step = self.step_size
        cx = self.netlist.core_x
        cy = self.netlist.core_y
        cw = self.netlist.core_width
        ch = self.netlist.core_height

        for it in range(self.max_iters):
            improved = False
            np.random.shuffle(movable)

            for name in movable:
                m = self.netlist.macros[name]
                x, y = current[name]

                best_local_cost = self.evaluator.evaluate()["cost"]
                best_dx, best_dy = 0.0, 0.0

                for dx, dy in [(step,0),(-step,0),(0,step),(0,-step)]:
                    nx = max(cx, min(cx + cw - m.width,  x + dx))
                    ny = max(cy, min(cy + ch - m.height, y + dy))
                    res = self.evaluator.move(name, nx, ny)
                    if res["cost"] < best_local_cost:
                        best_local_cost = res["cost"]
                        best_dx, best_dy = nx - x, ny - y
                    self.evaluator.move(name, x, y)

                if best_dx != 0 or best_dy != 0:
                    nx = x + best_dx
                    ny = y + best_dy
                    self.evaluator.move(name, nx, ny)
                    current[name] = (nx, ny)
                    if best_local_cost < best_cost:
                        best_cost = best_local_cost
                        best = dict(current)
                        improved = True

            step = max(self.min_step, step * self.decay)

            if it % 20 == 0:
                print(f"  [CD] iter {it:3d} | cost {best_cost:.4f} | step {step:.1f}")

            if step <= self.min_step and not improved:
                break

        return best
