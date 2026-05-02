import sys
import time
sys.path.insert(0, "../src")

from parser import ICCAD04Parser
from evaluator_baseline import BaselineEvaluator, placement_from_netlist
from evaluator_incremental import IncrementalEvaluator


def bench_baseline(netlist, placement, runs=50):
    ev = BaselineEvaluator(netlist)
    t0 = time.perf_counter()
    for _ in range(runs):
        ev.evaluate(placement)
    t1 = time.perf_counter()
    return (t1 - t0) / runs * 1000


def bench_incremental(netlist, placement, runs=500):
    ev = IncrementalEvaluator(netlist)
    ev.initialize(placement)
    movable = list(netlist.movable_macros().keys())
    import numpy as np
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    for _ in range(runs):
        name = movable[rng.integers(len(movable))]
        x, y = placement[name]
        ev.move(name, x + rng.uniform(-10, 10), y + rng.uniform(-10, 10))
    t1 = time.perf_counter()
    return (t1 - t0) / runs * 1000


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python eval_speed.py path/to/benchmark.aux")
        sys.exit(1)

    nl = ICCAD04Parser().parse(sys.argv[1])
    pl = placement_from_netlist(nl)

    print("Benchmarking...")
    base_ms = bench_baseline(nl, pl)
    incr_ms = bench_incremental(nl, pl)
    speedup = base_ms / incr_ms if incr_ms > 0 else 0

    print(f"\n[Benchmark Results]")
    print(f"  Baseline eval   : {base_ms:.2f} ms/eval")
    print(f"  Incremental eval: {incr_ms:.4f} ms/eval")
    print(f"  Speedup         : {speedup:.0f}x")
    print(f"  Target          : 300x")
    print(f"  Status          : {'ACHIEVED' if speedup >= 300 else 'NOT YET - tune more'}")
