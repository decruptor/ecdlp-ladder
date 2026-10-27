# Classically solving the "Brace for impact" ECDLP ladder (rungs 6–80)

**Author:** Decruptor (pseudonymous)
**License:** GPLv3 (see [`LICENSE`](LICENSE))
**Related:** kangaroo walk constants paper — [kangaroo-constants](https://github.com/decruptor/kangaroo-constants)

This repo contains the solvers, the verified challenge data and the results for
the low rungs of the secp256k1-shaped ECDLP challenge ladder from
*"Brace for impact: ECDLP challenges for quantum cryptanalysis"*
(Dallaire-Demers, Doyle, Foo — arXiv:2508.14011v2, 25 Mar 2026), solved
**classically**, before any quantum team demonstrates them.

## The challenge — and one correction to how it is usually read

For each bit-length k the paper publishes a "secpkk1" instance: a k-bit prime p,
the group order n of E: y² = x³ + 7 over F_p, and two deterministic NUMS points
derived from SHA-256 labels `"Quantum"` (P) and `"Challenge"` (Q). The task on
every card is: find κ such that P = [κ]·Q over E(F_p).

**Important convention (verified):** contrary to what interval-puzzle practice
suggests, the secret κ is **not interval-restricted**. The construction is fully
deterministic (Algorithm 3 of the paper) and κ is an arbitrary element of
[1, n−1]. The right classical algorithm is therefore **Pollard's rho with
distinguished points** (expected ≈ √(πn/2) ≈ 1.25·√n group operations), **not a
kangaroo**. The paper's own Section 3 calibration matches this. Any rung
"attempts" using kangaroo tools over a sub-interval are solving a different
problem than the card poses.

All 20 published rungs were verified before solving:

* every published P, Q is on its curve and has full order n ([n]Q = O,
  [n+1]Q = Q, n prime by Miller-Rabin, n independently recomputed with SEA);
* every P, Q was **recomputed from the labels** (SHA-256("Quantum"/"Challenge"),
  x = h+δ search, canonical y) and matches the paper's cards exactly;
* no rung has a small embedding degree and no rung is anomalous — no MOV/Smart
  shortcut exists.

## Erratum found in the paper (v2)

The 144-bit card prints `n` equal to `p`
(22300745198530623141535718272648361505978547 twice). The true group order, by
SEA, is **22300745198530623141528449602726112266830289** (also prime). The
printed P and Q are consistent with the true order. Reported upstream;
[`challenges.json`](challenges.json) carries the corrected value, flagged.

## Results: rungs 6–80

Every κ below was verified by an **independent** pure-Python point-arithmetic
check `[κ]·Q == P` (`verify_all.py`), not just by the solver that produced it.

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
| 80 | `0x27f35e126a10e6e7d3d4` | parallel Pollard rho (DP, CUDA) | 1× RTX PRO 6000 Blackwell (sm_120) | 1862 s (1.63e12 ops, 877 M/s) | ✅ |

(The published ladder skips some bit-lengths; rungs 6, 8, 12, 16, 24, 32, 48,
64, 80 are the cards solved so far, per `challenges.json`.)

## Solver notes

* [`rho.c`](rho.c) — single-file C solver: Montgomery arithmetic (1 or 2×64-bit
  limbs), r-adding walks (32 jumps, M_j = [s_j]Q + [t_j]P), OpenMP-parallel with
  W independent walks per thread in lockstep and Montgomery's trick batch
  inversion once per round; distinguished points shared through a sharded
  64-slot hash table. Measured throughput ~50–90 M group ops/s on an Apple M4
  (10 cores, shared with other desktop load); roughly 17 modmuls per step.
* Small rungs (≤32 bits) solved by baby-step/giant-step in pure Python
  (`solve_small.py`).
* [`rho80.cu`](rho80.cu) — the **CUDA port** of `rho.c` (see [`GPU80.md`](GPU80.md)):
  one thread = one r-adding walk, warp-wide batch inversion, sharded DP table in
  device memory, host-side collision resolution and `[κ]Q == P` verification.
  Specialized per rung at compile time with `-DRUNG=<bits>`; a `CPU_REF` build
  compiles the identical walk for bit-for-bit cross-checking against `rho.c`
  (validated: identical κ and identical step counts on rung 48; first-24-DP
  walk cross-checks on rungs 64 and 80; field-arithmetic fuzz vs Python,
  312500 vectors × 3 rungs, 0 mismatches).
  `rho80.cu` is preserved **unmodified** (tag `pristine`): it predates the
  undefined-behavior investigation in §6 of the paper — the ternary-select-over-shuffle
  at lines 826–827 is the faulty construct, kept for reference.
* [`rho80_fixed.cu`](rho80_fixed.cu) — the **rung-80 production source**: the post-fix
  build that passed the paper's four validation gates and produced the rung-80 result
  (three workers, one per RTX PRO 6000, full 385,024-walk grid). Same lineage as
  `rho80.cu` plus the §6.5 fix (shuffle into a temporary, then an explicit edge-lane
  overwrite — lines 877–880), `RHO_SM_FRAC` launch scaling, checkpointing, and a
  RUNG = 96 block (the pre-v6 rung-96 workers were built from it).
* The **rung-96 (v6, endomorphism) production sources** — the exact files the twelve
  fleet binaries were compiled from (md5 prefixes in brackets):
  * [`rho80v6.cu`](rho80v6.cu) [493a20a6] — the canonical v6 source: GLV endomorphism
    (RHO_ENDO = 1 by default), affine walk with division-step inversion, the §6.5 fix
    in source, `RHO_SM_FRAC`.
  * [`rho80v6b.cu`](rho80v6b.cu) [a781158e] — the RTX 4090 workers' source: `rho80v6.cu`
    with a two-line launch-shape change (`__launch_bounds__(128,4)`, 128 threads/block),
    generated by [`rho80v6b_gen.sh`](rho80v6b_gen.sh) `rho80v6.cu 128 4` (the script
    writes to `/root/`, as run on the box; its output reproduces byte-for-byte).
    Built `nvcc -O3 -DRUNG=96 -arch=sm_89`.
  * [`rho80v6sw.cu`](rho80v6sw.cu) [a5d8b82e] — `rho80v6.cu` with the launch shape as
    compile-time knobs; the RTX PRO 6000 workers were built
    `-DRUNG=96 -arch=sm_120 -DRHO_TPB=256 -DRHO_LBMIN=4`, the A100
    `-DRUNG=96 -arch=sm_80 -DRHO_TPB=512 -DRHO_LBMIN=2`.

  The production sources build for the host exactly like `rho80.cu` (note `-x c++`;
  clang otherwise treats `.cu` as CUDA):

      clang++ -x c++ -O3 -DCPU_REF -DRUNG=48 -o ref48  rho80_fixed.cu && ./ref48  selftest 3
      clang++ -x c++ -O3 -DCPU_REF -DRUNG=48 -o ref48e rho80v6.cu     && ./ref48e selftest 3

  All four pass (3/3, same κ values; the endomorphism builds take different walk paths
  and fewer steps, and the three v6 files walk identically on the host — they differ
  only in GPU launch shape). Two in-source comments are shipped as they ran and are
  superseded by the paper: the fix comment calls the fault an nvcc "miscompile" (the
  paper's §6 shows it is source-level undefined behavior), and the header's arch table
  lists the RTX 6000D as sm_89 (it is Blackwell, compute capability 12.0, sm_120).
* [`dbg2.cu`](dbg2.cu), [`dbg2-sm89.sass`](dbg2-sm89.sass) — the §6.3 minimal
  reproducer (30-line kernel) and its sm_89 disassembly (Table 6.3).

## GPU prep for rungs 80–144

Rho cost ≈ 1.25·√n group ops. Measured M4 CPU throughput ~22 M ops/s (2-limb
solver, 10 threads, machine shared with other desktop load); GPU estimates use
RCKangaroo-class throughput (~14.5 G jump/s per RTX 4090; see GPU80.md's
0.5–2 Gops/s conservative planning number for this kernel):

| rung | expected group ops | M4 CPU (measured rate) | 1× RTX 4090-class |
|------|--------------------|------------------------|-------------------|
| 80   | 1.4e12 | **SOLVED 2026-10-24** — 1862 s @ 877 Mops/s, 1× RTX PRO 6000 Blackwell (κ verified ×2; `result_80.json`) | ~12–47 min (rho-kernel estimate; see GPU80.md) |
| 96   | 3.6e14 | **SOLVED 2026-10-26** — 13,280 s by the winning worker (RTX 4090, endomorphism build, 3.56 G walk steps/s) of a 12-worker fleet; κ verified ×2 (`result_96.json`) | ~6 h |
| 112  | 8.9e16 | centuries | ~66 days |
| 128  | 2.2e19 | — | ~1.5e4 years (record territory) |

Notes:

* The 256-bit rung is the real secp256k1 prime, so stock tools (RCKangaroo,
  JeanLucPons/Kangaroo) apply to it directly — but those tools are hard-coded to
  secp256k1, so the intermediate rungs (80–240) need a small patch (swap p, n
  and the NUMS points) or a generalized GPU kangaroo. The rung primes are mostly
  special-form (p = 2^k − c), which GPU Montgomery/Barrett reduction likes.
* `rho80.cu` covers rung 80 (p = 2^80 − 8259, 4×32-bit limbs, Montgomery
  domain R = 2^128 matching `rho.c -DNW=2`); the `rho80v6*.cu` sources cover rung 96
  (p = 2^96 − 7245, GLV endomorphism). Rungs 112/144 need a new rung
  block in the rung table (NW64 rule: smallest NW64 with 64·NW64 ≥ bitlen(p)+2 —
  112/128: NW64 = 3; 144: NW64 = 3).
* **Verified on real hardware.** CUDA execution (kernel launch, shuffles,
  atomics, DP table traffic) is no longer hypothetical: the rung-80 solve ran
  on a real GPU (RTX PRO 6000 Blackwell, sm_120, CUDA 13.0) and was doubly
  verified — the kernel's own `[κ]Q == P` check plus an independent
  pure-Python EC replay (`verify_all.py`, all rungs PASS). See
  [`result_80.json`](result_80.json) and [`GPU80.md`](GPU80.md) for the
  validation record, and [`test_transcript.txt`](test_transcript.txt) for the
  local (no-GPU) validation transcript. To reproduce on your own card: build
  RUNG=48, run `./rho48 selftest 3` (seconds), then `./rho80 smoke`, then the
  real solve; compare `smoke`'s lane-0 DP tap against `rho_ref80`'s to confirm
  the GPU walk matches the CPU walk bit-for-bit.

## Reproducibility

* [`challenges.json`](challenges.json) — all 20 rungs (p, n, P, Q in SEC1
  compressed form), with the 144-bit erratum corrected and flagged.
* [`verify_points.py`](verify_points.py) — recomputes every card from the labels
  and checks orders.
* [`solve_small.py`](solve_small.py), [`run_rho.py`](run_rho.py),
  [`rho.c`](rho.c) — solvers.
* [`verify_all.py`](verify_all.py) — independent verification of every reported κ
  (all solutions verified: P = [κ]Q holds exactly).
* [`GPU80.md`](GPU80.md), [`GPU_PREP.md`](GPU_PREP.md) — GPU port documentation
  and prep notes; [`test_transcript.txt`](test_transcript.txt) — the local
  validation transcript (regenerate with `make_transcript.py`).
