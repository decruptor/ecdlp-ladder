# Classical solutions to the "Brace for impact" ECDLP ladder (arXiv:2508.14011)

**Status: DRAFT — not published anywhere. Local working notes for the team.**

We solved the low rungs of the secp256k1-shaped ECDLP challenge ladder from
*"Brace for impact: ECDLP challenges for quantum cryptanalysis"*
(Dallaire-Demers, Doyle, Foo — arXiv:2508.14011v2, 25 Mar 2026) **classically**,
before any quantum team demonstrates them.

## The challenge (as specified by the paper)

For each bit-length k ∈ {6..256} the paper publishes a "secpkk1" instance:
a k-bit prime p, the (prime) group order n = |E(F_p)| of E: y² = x³ + 7 over F_p,
and two deterministic NUMS points derived from the SHA-256 labels `"Quantum"` (P)
and `"Challenge"` (Q). The task on every card is:

    find κ such that  P = [κ]·Q  over E(F_p).

**Important convention note (verified):** contrary to what one might assume from
interval-puzzle practice (e.g. the 2019-2025 "Bitcoin puzzle" tradition), the
secret κ is **not** interval-restricted. The construction is fully deterministic
(Algorithm 3 of the paper) and κ is an arbitrary element of [1, n−1]. The right
classical algorithm is therefore **Pollard's rho with distinguished points**
(expected ≈ √(πn/2) ≈ 1.25·√n group operations), *not* a kangaroo. The paper's own
Section 3 calibration matches this.

All 20 published rungs were independently verified before solving:

* every published P, Q is on its curve and has full order n (we checked [n]Q = O,
  [n+1]Q = Q, n prime by Miller-Rabin, and independently recomputed n with SEA);
* every P, Q was **recomputed from the labels** (SHA-256("Quantum"/"Challenge"),
  x = h+δ search, canonical y) and matches the paper's cards exactly;
* no rung has a small embedding degree (r = ord_n(p) is large everywhere) and no
  rung is anomalous — no MOV/Smart shortcut exists (we checked).

**One erratum found in the paper (v2):** the 144-bit card prints `n` equal to `p`
(22300745198530623141535718272648361505978547 twice). The true group order, by SEA,
is `22300745198530623141528449602726112266830289` (also prime). The printed P and Q
are consistent with the true order. (Worth reporting upstream.)

## Results

| rung | κ (hex) | method | hardware | wall time | verified |
|------|---------|--------|----------|-----------|----------|
| 6  | `0x17` | BSGS | 1 core | <0.01 s | ✅ |
| 8  | `0x18` | BSGS | 1 core | <0.01 s | ✅ |
| 12 | `0x7ea` | BSGS | 1 core | <0.01 s | ✅ |
| 16 | `0x2304` | BSGS | 1 core | <0.01 s | ✅ |
| 24 | `0x687a6b` | BSGS | 1 core | 0.01 s | ✅ |
| 32 | `0xc6a8b85f` | BSGS | 1 core | 0.6 s | ✅ |
| 48 | `0x142b32783630` | parallel Pollard rho (DP) | M4 (10 threads) | 0.2 s (10.1e6 ops, 88 M/s) | ✅ |
| 64 | `0xfc14e401202418fd` | parallel Pollard rho (DP, 128-bit limbs) | M4 (10 threads) | 395 s (7.5e9 ops, 19 M/s) | ✅ |

Every κ above was verified by an **independent** pure-Python point-arithmetic check
`[κ]·Q == P` (see `verify_all.py`), not just by the solver that produced it.

## Solver notes

* `rho.c` — single-file C solver: Montgomery arithmetic (1 or 2×64-bit limbs),
  r-adding walks (32 jumps, M_j = [s_j]Q + [t_j]P), OpenMP-parallel with W
  independent walks per thread in lockstep and Montgomery's trick batch inversion
  once per round; distinguished points shared through a sharded 64-slot hash table.
  Throughput measured: ~50-90 M group ops/s on an Apple M4 (10 cores, shared with
  other desktop load). Roughly 17 modmuls per step.
* Small rungs (≤32 bits) solved by baby-step/giant-step in pure Python.

## What remains (and what it costs)

Rho cost ≈ 1.25·√n group ops. Measured M4 throughput: ~22 M ops/s with the
2-limb solver (10 threads, machine shared with other desktop load);
GPU estimates use RCKangaroo-class throughput (≈14.5 G jump/s per RTX 4090,
see GPU_PREP.md):

| rung | expected group ops | M4 CPU (measured rate) | 1× RTX 4090 |
|------|--------------------|------------------------|-------------|
| 80   | 1.4e12 | **SOLVED 2026-10-24** — 1862 s @ 877 Mops/s, 1 RTX PRO 6000 Blackwell (κ verified ×2; result_80.json) | ~1.5 min |
| 96   | 3.6e14 | ~6 months | ~6 h |
| 112  | 8.9e16 | centuries | ~66 days |
| 128  | 2.2e19 | — | ~1.5e4 years (record territory) |

Notes:
* The 256-bit rung is the real secp256k1 prime, so stock tools (RCKangaroo,
  JeanLucPons/Kangaroo) apply to it directly — but those tools are hard-coded to
  secp256k1, so the intermediate rungs (80-240) need a small patch (swap p, n and
  the NUMS points) or a generalized GPU kangaroo. The rung primes are mostly
  special-form (p = 2^k − c), which GPU Montgomery/Barrett reduction likes.
* Full-range DLP on GPU costs ≈ 2√n jump operations with a kangaroo over [0, n);
  RCKangaroo-class throughput is ~1-2 G jumps/s per high-end GPU.

## Reproducibility

* `challenges.json` — all 20 rungs (p, n, P, Q in SEC1 compressed form), with the
  144-bit erratum corrected and flagged.
* `verify_points.py` — recomputes every card from the labels and checks orders.
* `sea_all.py` logic — SEA point-counting cross-check (PARI/GP).
* `solve_small.py`, `run_rho.py`, `rho.c` — solvers.
* `verify_all.py` — independent verification of every reported κ.
* All solutions verified: P = [κ]Q holds exactly.
