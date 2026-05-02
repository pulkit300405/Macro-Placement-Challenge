import os
import subprocess
import tempfile
from typing import Dict, Tuple
from parser import Netlist


class OpenROADRunner:

    def __init__(self, openroad_bin: str = "openroad", design_dir: str = ""):
        self.openroad_bin = openroad_bin
        self.design_dir = design_dir

    def write_placement(self, netlist: Netlist,
                        placement: Dict[str, Tuple[float, float]],
                        out_path: str):
        with open(out_path, "w") as f:
            f.write("UCLA pl 1.0\n")
            for name, macro in netlist.macros.items():
                if name in placement:
                    x, y = placement[name]
                else:
                    x, y = macro.x, macro.y
                fixed = "/FIXED" if macro.is_fixed else ""
                f.write(f"{name}\t{x:.6f}\t{y:.6f}\t: {macro.orientation}{fixed}\n")

    def run_pnr(self, pl_path: str, tcl_template: str) -> dict:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tcl", delete=False) as f:
            tcl_path = f.name
            f.write(tcl_template.replace("{{PL_FILE}}", pl_path))

        try:
            result = subprocess.run(
                [self.openroad_bin, tcl_path],
                capture_output=True, text=True, timeout=600
            )
            metrics = self._parse_output(result.stdout + result.stderr)
            metrics["returncode"] = result.returncode
            return metrics
        except subprocess.TimeoutExpired:
            return {"error": "timeout", "returncode": -1}
        except FileNotFoundError:
            return {"error": "openroad_not_found", "returncode": -1}
        finally:
            os.unlink(tcl_path)

    def _parse_output(self, output: str) -> dict:
        metrics = {"wns": None, "tns": None, "area": None}
        for line in output.splitlines():
            line = line.strip().lower()
            if "wns" in line:
                parts = line.split()
                for i, p in enumerate(parts):
                    if "wns" in p and i+1 < len(parts):
                        try:
                            metrics["wns"] = float(parts[i+1])
                        except:
                            pass
            if "tns" in line:
                parts = line.split()
                for i, p in enumerate(parts):
                    if "tns" in p and i+1 < len(parts):
                        try:
                            metrics["tns"] = float(parts[i+1])
                        except:
                            pass
            if "area" in line:
                parts = line.split()
                for i, p in enumerate(parts):
                    if "area" in p and i+1 < len(parts):
                        try:
                            metrics["area"] = float(parts[i+1])
                        except:
                            pass
        return metrics

    def evaluate_placement(self, netlist: Netlist,
                           placement: Dict[str, Tuple[float, float]],
                           tcl_template: str) -> dict:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pl", delete=False) as f:
            pl_path = f.name

        try:
            self.write_placement(netlist, placement, pl_path)
            return self.run_pnr(pl_path, tcl_template)
        finally:
            if os.path.exists(pl_path):
                os.unlink(pl_path)
