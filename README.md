# Macro Placement Challenge 2026
### Partcl × HRT Macro Placement Challenge | Team ZeroLatency

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

---

## 🏆 Competition

- **Competition:** Partcl × HRT Macro Placement Challenge 2026
- **Team:** ZeroLatency (Pulkit Singh + Lakshita)
- **Score:** 1.5074 avg proxy cost (17 IBM benchmarks)
- **Overlaps:** 0 on all benchmarks
- **Runtime:** ~46s total

---

## 🧠 Algorithm — SA-Net (Congestion-Aware Simulated Annealing)

### Overview
Two-phase macro placement combining minimum-displacement legalization with congestion-aware simulated annealing.

### Phase 1: Legalization
- Area-sorted order (largest macros placed first)
- Minimum-displacement spiral search
- Zero overlaps guaranteed

### Phase 2: Congestion-Aware SA
- **Cost function:** `10% wirelength + 90% congestion density`
- Net extraction directly from `benchmark.net_nodes`
- Three move types: shift, swap (neighbor-preferred), attract
- Grid-based density penalty to reduce routing congestion hotspots

### Key Parameters
```
seed         = 42
refine_iters = 3000
T_start      = 0.25 × canvas
T_end        = 0.0008 × canvas
gap          = 0.05
grid         = 16 (density estimation)
wl_weight    = 0.05
cong_weight  = 0.95
```

---

## 📊 Results (17 IBM Benchmarks)

| Benchmark | Proxy Cost | vs SA | vs RePlAce | Overlaps |
|-----------|-----------|-------|------------|---------|
| ibm01 | 1.2253 | +6.9% | -22.8% | 0 |
| ibm02 | 1.6790 | +12.0% | +8.6% | 0 |
| ibm03 | 1.4100 | +19.0% | -6.6% | 0 |
| ibm04 | 1.4101 | +6.2% | -8.3% | 0 |
| ibm06 | 1.7197 | +31.4% | -6.2% | 0 |
| ibm07 | 1.4950 | +26.1% | -2.2% | 0 |
| ibm08 | 1.5582 | +19.0% | -9.1% | 0 |
| ibm09 | 1.1363 | +18.1% | -1.5% | 0 |
| ibm10 | 1.4259 | +32.4% | +5.0% | 0 |
| ibm11 | 1.2354 | +27.8% | -4.9% | 0 |
| ibm12 | 1.6507 | +41.6% | +4.4% | 0 |
| ibm13 | 1.4011 | +26.8% | -4.9% | 0 |
| ibm14 | 1.6033 | +29.5% | -3.9% | 0 |
| ibm15 | 1.6061 | +30.2% | -5.9% | 0 |
| ibm16 | 1.5323 | +31.4% | -3.7% | 0 |
| ibm17 | 1.7437 | +52.5% | -6.0% | 0 |
| ibm18 | 1.7941 | +35.4% | -1.2% | 0 |
| **AVG** | **1.5074** | **+29.1%** | **-3.4%** | **0** |

---

## 🗂️ Repository Structure

```
Macro-Placement-Challenge/
├── placer/
│   └── placer.py       ← SANetPlacer class (main submission)
├── README.md
└── LICENSE             ← Apache 2.0
```

---

## 🚀 Usage

```python
from placer.placer import SANetPlacer

placer = SANetPlacer(seed=42, refine_iters=3000)
positions = placer.place(benchmark)
```

---

## 📋 Requirements

- Python 3.10+
- PyTorch
- NumPy
- macro_place (competition kit)

---

## 📄 License

Apache 2.0 — see [LICENSE](LICENSE)
