"""
ICCAD04 Benchmark Parser
========================
Parses the IBM ICCAD04 benchmark suite files:
  - .aux   -> lists all other files
  - .nodes -> macro/cell dimensions
  - .nets  -> net connectivity
  - .pl    -> initial placement (x, y, orientation)
  - .scl   -> row/site structure (core area definition)

Author: Team ZeroLatency (Pulkit + Lakshita)
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


# ─── Data Classes ────────────────────────────────────────────────────────────

@dataclass
class Macro:
    name: str
    width: float
    height: float
    x: float = 0.0
    y: float = 0.0
    orientation: str = "N"
    is_fixed: bool = False
    is_terminal: bool = False   # i/o pins, fixed always

    @property
    def area(self):
        return self.width * self.height

    @property
    def cx(self):
        return self.x + self.width / 2

    @property
    def cy(self):
        return self.y + self.height / 2


@dataclass
class Net:
    name: str
    pins: List[Tuple[str, float, float]] = field(default_factory=list)
    # pins -> list of (macro_name, pin_offset_x, pin_offset_y)


@dataclass
class Row:
    origin_x: float
    origin_y: float
    site_width: float
    site_height: float
    num_sites: int
    orientation: str = "N"

    @property
    def width(self):
        return self.num_sites * self.site_width

    @property
    def height(self):
        return self.site_height


@dataclass
class Netlist:
    macros: Dict[str, Macro] = field(default_factory=dict)
    nets: List[Net] = field(default_factory=list)
    rows: List[Row] = field(default_factory=list)

    # derived from rows
    core_x: float = 0.0
    core_y: float = 0.0
    core_width: float = 0.0
    core_height: float = 0.0

    def movable_macros(self):
        """return only macros that can be moved (not terminals/fixed)"""
        return {n: m for n, m in self.macros.items() if not m.is_fixed and not m.is_terminal}

    def fixed_macros(self):
        return {n: m for n, m in self.macros.items() if m.is_fixed or m.is_terminal}

    def summary(self):
        total = len(self.macros)
        movable = len(self.movable_macros())
        fixed = len(self.fixed_macros())
        print(f"[Netlist Summary]")
        print(f"  Macros  : {total} total | {movable} movable | {fixed} fixed")
        print(f"  Nets    : {len(self.nets)}")
        print(f"  Rows    : {len(self.rows)}")
        print(f"  Core    : ({self.core_x:.1f}, {self.core_y:.1f}) "
              f"-> {self.core_width:.1f} x {self.core_height:.1f}")


# ─── Parser ──────────────────────────────────────────────────────────────────

class ICCAD04Parser:
    """
    Parses full ICCAD04 benchmark given path to .aux file.

    Usage:
        parser = ICCAD04Parser()
        netlist = parser.parse("data/iccad04/ibm01/ibm01.aux")
        netlist.summary()
    """

    def __init__(self):
        self.netlist = Netlist()

    def parse(self, aux_path: str) -> Netlist:
        """entry point — reads .aux, then delegates to sub-parsers"""
        aux_path = os.path.abspath(aux_path)
        base_dir = os.path.dirname(aux_path)

        files = self._parse_aux(aux_path, base_dir)

        if "nodes" in files:
            self._parse_nodes(files["nodes"])
        if "nets" in files:
            self._parse_nets(files["nets"])
        if "pl" in files:
            self._parse_pl(files["pl"])
        if "scl" in files:
            self._parse_scl(files["scl"])

        self._compute_core_bounds()
        return self.netlist

    # ── .aux ─────────────────────────────────────────────────────────────────

    def _parse_aux(self, aux_path, base_dir) -> dict:
        """reads aux file, returns dict of {type: full_path}"""
        files = {}
        with open(aux_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # format: RowBasedPlacement : ibm01.nodes ibm01.nets ...
                if ":" in line:
                    parts = line.split(":")
                    filenames = parts[1].strip().split()
                    for fname in filenames:
                        ext = fname.split(".")[-1].lower()
                        files[ext] = os.path.join(base_dir, fname)
        return files

    # ── .nodes ───────────────────────────────────────────────────────────────

    def _parse_nodes(self, path: str):
        """parse macro dimensions"""
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("UCLA"):
                    continue
                if line.startswith("NumNodes") or line.startswith("NumTerminals"):
                    continue

                parts = line.split()
                if len(parts) < 3:
                    continue

                name = parts[0]
                w = float(parts[1])
                h = float(parts[2])
                is_terminal = "terminal" in line.lower()

                self.netlist.macros[name] = Macro(
                    name=name,
                    width=w,
                    height=h,
                    is_terminal=is_terminal,
                    is_fixed=is_terminal
                )

    # ── .nets ─────────────────────────────────────────────────────────────────

    def _parse_nets(self, path: str):
        """parse net connectivity"""
        current_net = None

        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("UCLA"):
                    continue
                if line.startswith("NumNets") or line.startswith("NumPins"):
                    continue

                if line.startswith("NetDegree"):
                    # e.g. "NetDegree : 4  net_name"
                    parts = line.split()
                    net_name = parts[-1] if len(parts) > 2 else f"net_{len(self.netlist.nets)}"
                    current_net = Net(name=net_name)
                    self.netlist.nets.append(current_net)

                elif current_net is not None:
                    # pin line: "macro_name  I  : offset_x  offset_y"
                    parts = line.split()
                    if len(parts) >= 2:
                        macro_name = parts[0]
                        ox = float(parts[3]) if len(parts) > 3 else 0.0
                        oy = float(parts[4]) if len(parts) > 4 else 0.0
                        current_net.pins.append((macro_name, ox, oy))

    # ── .pl ──────────────────────────────────────────────────────────────────

    def _parse_pl(self, path: str):
        """parse initial placement positions"""
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("UCLA"):
                    continue

                parts = line.split()
                if len(parts) < 3:
                    continue

                name = parts[0]
                x = float(parts[1])
                y = float(parts[2])
                orient = parts[3] if len(parts) > 3 else "N"
                fixed = "/FIXED" in line or "/FIXED_NI" in line

                if name in self.netlist.macros:
                    self.netlist.macros[name].x = x
                    self.netlist.macros[name].y = y
                    self.netlist.macros[name].orientation = orient.replace("/FIXED", "").replace("/FIXED_NI", "").strip()
                    if fixed:
                        self.netlist.macros[name].is_fixed = True

    # ── .scl ─────────────────────────────────────────────────────────────────

    def _parse_scl(self, path: str):
        """parse row structure (core area)"""
        current_row = {}

        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("UCLA"):
                    continue

                if line.startswith("CoreRow"):
                    current_row = {}

                elif line.startswith("End"):
                    if current_row:
                        row = Row(
                            origin_x=current_row.get("SubrowOrigin", 0.0),
                            origin_y=current_row.get("Coordinate", 0.0),
                            site_width=current_row.get("SiteWidth", 1.0),
                            site_height=current_row.get("Height", 1.0),
                            num_sites=current_row.get("NumSites", 0),
                            orientation=current_row.get("SiteOrient", "N")
                        )
                        self.netlist.rows.append(row)
                    current_row = {}

                else:
                    # key : value pairs
                    parts = line.replace(":", " ").split()
                    for i in range(0, len(parts) - 1, 2):
                        key = parts[i]
                        try:
                            val = float(parts[i + 1])
                            current_row[key] = val
                        except:
                            current_row[key] = parts[i + 1]

    # ── core bounds ──────────────────────────────────────────────────────────

    def _compute_core_bounds(self):
        """derive core area from row definitions"""
        if not self.netlist.rows:
            return

        min_x = min(r.origin_x for r in self.netlist.rows)
        min_y = min(r.origin_y for r in self.netlist.rows)
        max_x = max(r.origin_x + r.width for r in self.netlist.rows)
        max_y = max(r.origin_y + r.height for r in self.netlist.rows)

        self.netlist.core_x = min_x
        self.netlist.core_y = min_y
        self.netlist.core_width = max_x - min_x
        self.netlist.core_height = max_y - min_y


# ─── Quick Test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python parser.py path/to/benchmark.aux")
        sys.exit(1)

    p = ICCAD04Parser()
    nl = p.parse(sys.argv[1])
    nl.summary()

    # print first 5 movable macros
    movable = list(nl.movable_macros().values())[:5]
    print("\nSample movable macros:")
    for m in movable:
        print(f"  {m.name}: ({m.width:.1f} x {m.height:.1f}) @ ({m.x:.1f}, {m.y:.1f})")

    # print first 3 nets
    print("\nSample nets:")
    for net in nl.nets[:3]:
        print(f"  {net.name}: {len(net.pins)} pins")
