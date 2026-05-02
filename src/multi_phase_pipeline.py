import time
from typing import Dict, Tuple
from parser import ICCAD04Parser, Netlist
from evaluator_baseline import BaselineEvaluator, placement_from_netlist
from evaluator_incremental import IncrementalEvaluator
from optimizer_cd import CoordinateDescent
from optimizer_swap import PairwiseSwap
from optimizer_gpu_restart import ParallelRestarter
from macro_constraints import ConstraintChecker
from clustering import MacroClusterer


class MacroPlacementPipeline:

    def __init__(self, aux_path: str, n_restarts: int = 8):
        self.netlist = ICCAD04Parser().parse(aux_path)
        self.netlist.summary()

        self.ev_inc = IncrementalEvaluator(self.netlist)
        self.ev_base = BaselineEvaluator(self.netlist)
        self.constraints = ConstraintChecker(self.netlist)
        self.clusterer = MacroClusterer(self.netlist)
        self.n_restarts = n_restarts

    def run(self) -> Dict[str, Tuple[float, float]]:
        t0 = time.perf_counter()

        placement = placement_from_netlist(self.netlist)
        init_cost = self.ev_base.evaluate(placement)["cost"]
        print(f"\n[Pipeline] Initial cost: {init_cost:.4f}")

        print("\n[Phase 1] Coordinate Descent...")
        cd = CoordinateDescent(self.netlist, self.ev_inc)
        placement = cd.optimize(placement)
        cost = self.ev_base.evaluate(placement)["cost"]
        print(f"[Phase 1] Done. Cost: {cost:.4f}")

        print("\n[Phase 2] Pairwise Swaps...")
        sw = PairwiseSwap(self.netlist, self.ev_inc)
        placement = sw.optimize(placement)
        cost = self.ev_base.evaluate(placement)["cost"]
        print(f"[Phase 2] Done. Cost: {cost:.4f}")

        print("\n[Phase 3] Parallel Restarts...")
        restarter = ParallelRestarter(self.netlist, n_restarts=self.n_restarts)
        best_restart = restarter.run(placement)
        restart_cost = self.ev_base.evaluate(best_restart)["cost"]

        if restart_cost < cost:
            placement = best_restart
            cost = restart_cost
            print(f"[Phase 3] Restart improved! Cost: {cost:.4f}")

            print("\n[Phase 4] Final CD pass on best restart...")
            cd2 = CoordinateDescent(self.netlist, self.ev_inc)
            cd2.max_iters = 100
            placement = cd2.optimize(placement)
            cost = self.ev_base.evaluate(placement)["cost"]
            print(f"[Phase 4] Done. Cost: {cost:.4f}")
        else:
            print(f"[Phase 3] No improvement from restarts.")

        elapsed = time.perf_counter() - t0
        val = self.constraints.validate(placement)

        print(f"\n{'='*50}")
        print(f"FINAL COST    : {cost:.4f}")
        print(f"VALID         : {val['valid']}")
        print(f"OVERLAP       : {val['total_overlap']:.2f}")
        print(f"OUT OF BOUNDS : {len(val['out_of_bounds'])}")
        print(f"TIME          : {elapsed:.1f}s")
        print(f"{'='*50}")

        return placement

    def save_placement(self, placement: Dict[str, Tuple[float, float]], out_path: str):
        with open(out_path, "w") as f:
            f.write("UCLA pl 1.0\n")
            for name, macro in self.netlist.macros.items():
                if name in placement:
                    x, y = placement[name]
                else:
                    x, y = macro.x, macro.y
                fixed = "/FIXED" if macro.is_fixed else ""
                f.write(f"{name}\t{x:.6f}\t{y:.6f}\t: {macro.orientation}{fixed}\n")
        print(f"[Pipeline] Placement saved to: {out_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python multi_phase_pipeline.py path/to/benchmark.aux [output.pl]")
        sys.exit(1)

    aux = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "results/output.pl"

    pipeline = MacroPlacementPipeline(aux, n_restarts=8)
    final_placement = pipeline.run()
    pipeline.save_placement(final_placement, out)
