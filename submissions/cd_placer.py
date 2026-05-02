import torch
import numpy as np
from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost


class CDPlacer:

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        from macro_place.loader import load_benchmark_from_dir
        _, plc = load_benchmark_from_dir(
            f"external/MacroPlacement/Testcases/ICCAD04/{benchmark.name}"
        )
        placement = benchmark.macro_positions.clone()
        movable = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
        indices = torch.where(movable)[0].tolist()
        sizes = benchmark.macro_sizes
        cw = benchmark.canvas_width
        ch = benchmark.canvas_height
        net_nodes = benchmark.net_nodes
        num_hard = benchmark.num_hard_macros

        placement = self._greedy_init(placement, indices, sizes, cw, ch)
        print(f"\n    greedy proxy={compute_proxy_cost(placement, benchmark, plc)['proxy_cost']:.4f}")
        placement = self._cd(placement, indices, sizes, cw, ch, net_nodes, num_hard)
        print(f"    cd done proxy={compute_proxy_cost(placement, benchmark, plc)['proxy_cost']:.4f}")
        return placement

    def _greedy_init(self, placement, indices, sizes, cw, ch):
        indices = sorted(indices, key=lambda i: -sizes[i, 1].item())
        gap = 0.001
        cx, cy, rh = 0.0, 0.0, 0.0
        for idx in indices:
            w, h = sizes[idx, 0].item(), sizes[idx, 1].item()
            if cx + w > cw:
                cx = 0.0
                cy += rh + gap
                rh = 0.0
            placement[idx, 0] = cx + w / 2
            placement[idx, 1] = cy + h / 2
            cx += w + gap
            rh = max(rh, h)
        return placement

    def _hpwl(self, pos, net_nodes, num_hard):
        total = 0.0
        for net in net_nodes:
            valid = net[net < num_hard]
            if len(valid) < 2:
                continue
            pts = pos[valid]
            total += (pts[:, 0].max() - pts[:, 0].min()).item()
            total += (pts[:, 1].max() - pts[:, 1].min()).item()
        return total

    def _no_overlap(self, pos, idx, sizes_np, all_indices):
        w1, h1 = sizes_np[idx]
        x1, y1 = pos[idx]
        for j in all_indices:
            if j == idx:
                continue
            w2, h2 = sizes_np[j]
            x2, y2 = pos[j]
            if abs(x1-x2) < (w1+w2)/2 and abs(y1-y2) < (h1+h2)/2:
                return False
        return True

    def _cd(self, placement, indices, sizes, cw, ch, net_nodes, num_hard):
        pos = placement[:num_hard].clone()
        sizes_np = sizes[:num_hard].numpy()

        step = min(cw, ch) * 0.04
        min_step = min(cw, ch) * 0.001
        decay = 0.75
        dirs = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]

        best_cost = self._hpwl(pos, net_nodes, num_hard)
        best_pos = pos.clone()
        iters = 0

        while step > min_step and iters < 80:
            iters += 1
            improved = False
            perm = np.random.permutation(len(indices))

            for pi in perm:
                idx = indices[pi]
                w, h = sizes_np[idx]
                ox, oy = pos[idx, 0].item(), pos[idx, 1].item()
                best_local = best_cost
                best_nx, best_ny = ox, oy

                for dx, dy in dirs:
                    nx = float(np.clip(ox + dx*step, w/2, cw - w/2))
                    ny = float(np.clip(oy + dy*step, h/2, ch - h/2))
                    pos[idx, 0] = nx
                    pos[idx, 1] = ny

                    if not self._no_overlap(pos.numpy(), idx, sizes_np, indices):
                        pos[idx, 0] = ox
                        pos[idx, 1] = oy
                        continue

                    c = self._hpwl(pos, net_nodes, num_hard)
                    if c < best_local:
                        best_local = c
                        best_nx, best_ny = nx, ny

                    pos[idx, 0] = ox
                    pos[idx, 1] = oy

                if best_local < best_cost:
                    pos[idx, 0] = best_nx
                    pos[idx, 1] = best_ny
                    best_cost = best_local
                    best_pos = pos.clone()
                    improved = True

            if iters % 10 == 0:
                print(f"    iter {iters:3d} | hpwl {best_cost:.2f} | step {step:.2f}")

            if not improved:
                step *= decay

        placement[:num_hard] = best_pos
        return placement