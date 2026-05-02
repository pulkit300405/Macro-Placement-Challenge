from typing import Dict, Tuple
from parser import Netlist, Macro


class ConstraintChecker:

    def __init__(self, netlist: Netlist):
        self.netlist = netlist
        self.cx = netlist.core_x
        self.cy = netlist.core_y
        self.cw = netlist.core_width
        self.ch = netlist.core_height

    def clamp_to_bounds(self, macro: Macro, x: float, y: float) -> Tuple[float, float]:
        nx = max(self.cx, min(self.cx + self.cw - macro.width,  x))
        ny = max(self.cy, min(self.cy + self.ch - macro.height, y))
        return nx, ny

    def is_in_bounds(self, macro: Macro, x: float, y: float) -> bool:
        return (x >= self.cx and y >= self.cy and
                x + macro.width  <= self.cx + self.cw and
                y + macro.height <= self.cy + self.ch)

    def has_overlap(self, placement: Dict[str, Tuple[float, float]]) -> bool:
        macros = self.netlist.macros
        names = list(placement.keys())
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                a, b = macros[names[i]], macros[names[j]]
                ax, ay = placement[names[i]]
                bx, by = placement[names[j]]
                ox = max(0.0, min(ax+a.width, bx+b.width) - max(ax, bx))
                oy = max(0.0, min(ay+a.height, by+b.height) - max(ay, by))
                if ox * oy > 0:
                    return True
        return False

    def total_overlap(self, placement: Dict[str, Tuple[float, float]]) -> float:
        macros = self.netlist.macros
        names = list(placement.keys())
        total = 0.0
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                a, b = macros[names[i]], macros[names[j]]
                ax, ay = placement[names[i]]
                bx, by = placement[names[j]]
                ox = max(0.0, min(ax+a.width, bx+b.width) - max(ax, bx))
                oy = max(0.0, min(ay+a.height, by+b.height) - max(ay, by))
                total += ox * oy
        return total

    def validate(self, placement: Dict[str, Tuple[float, float]]) -> dict:
        macros = self.netlist.macros
        out_of_bounds = [n for n, (x,y) in placement.items()
                         if not self.is_in_bounds(macros[n], x, y)]
        overlap = self.total_overlap(placement)
        return {
            "valid": len(out_of_bounds) == 0 and overlap == 0.0,
            "out_of_bounds": out_of_bounds,
            "total_overlap": overlap
        }
