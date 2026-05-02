from typing import Dict, Tuple, List
from parser import Netlist
import numpy as np


class MacroClusterer:

    def __init__(self, netlist: Netlist, n_clusters: int = 8):
        self.netlist = netlist
        self.n_clusters = n_clusters

    def cluster(self, placement: Dict[str, Tuple[float, float]]) -> Dict[str, int]:
        movable = list(self.netlist.movable_macros().keys())
        if not movable:
            return {}

        coords = np.array([placement[n] for n in movable], dtype=float)
        labels = self._kmeans(coords, self.n_clusters)
        return {name: int(labels[i]) for i, name in enumerate(movable)}

    def _kmeans(self, coords: np.ndarray, k: int, max_iter: int = 50) -> np.ndarray:
        n = len(coords)
        k = min(k, n)
        rng = np.random.default_rng(42)
        centers = coords[rng.choice(n, k, replace=False)]

        labels = np.zeros(n, dtype=int)
        for _ in range(max_iter):
            dists = np.linalg.norm(coords[:, None] - centers[None, :], axis=2)
            new_labels = np.argmin(dists, axis=1)
            if np.all(new_labels == labels):
                break
            labels = new_labels
            for c in range(k):
                mask = labels == c
                if mask.any():
                    centers[c] = coords[mask].mean(axis=0)

        return labels

    def cluster_hpwl_contribution(self, cluster_id: int, cluster_map: Dict[str, int],
                                   placement: Dict[str, Tuple[float, float]]) -> float:
        macros_in = [n for n, c in cluster_map.items() if c == cluster_id]
        total = 0.0
        for net in self.netlist.nets:
            net_macros = [p[0] for p in net.pins]
            if any(m in macros_in for m in net_macros):
                xs, ys = [], []
                for mname, ox, oy in net.pins:
                    if mname not in self.netlist.macros:
                        continue
                    m = self.netlist.macros[mname]
                    px, py = placement.get(mname, (m.x, m.y))
                    xs.append(px + m.width/2 + ox)
                    ys.append(py + m.height/2 + oy)
                if len(xs) >= 2:
                    total += (max(xs)-min(xs)) + (max(ys)-min(ys))
        return total
