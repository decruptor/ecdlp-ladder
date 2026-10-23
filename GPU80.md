# GPU80 — CUDA rho for rung 80 (and the other ladder rungs)

`rho80.cu` is a CUDA port of `rho.c` (parallel Pollard rho with distinguished
points) specialized per rung at compile time. Rung 80 is the current target:

    secp80k1:  p = 2^80 - 8259  (80-bit prime, NOT 256-bit)
               n = 1208925819612459780337609 (~2^80, prime)
               P = 030D08430A6C00FAE05651   Q = 02CF45D656507F3699875E
    task: full-range DLP  P = [kappa]Q,  kappa in [1, n-1]  (no interval)

Field size note: the "80" refers to the group order n; the field prime p is
also 80 bits here (p = 2^80 - 8259), so the arithmetic needs 3 x 32-bit limbs.
We use **4 x 32-bit limbs** on purpose: the Montgomery domain is
R = 2^128 = 2^(64*NW64), exactly the domain rho.c uses with its 64-bit limbs
(`rho.c -DNW=2`). That makes every walk reproducible bit-for-bit against the
CPU solver for the same seed, and keeps the schoolbook+REDC carry bound
(p < 0.61*R) with lots of margin.

## Build

CPU reference / cross-check build (no GPU needed — this is what was validated
locally on the Mac):

    clang++ -x c++ -O3 -DCPU_REF -DRUNG=80 -o rho_ref80 rho80.cu
    clang++ -x c++ -O3 -DCPU_REF -DRUNG=48 -o rho_ref48 rho80.cu
    # or: make ref80 ref48   /   ./build.sh ref 48 80

CUDA build (CUDA >= 11.4):

    nvcc -O3 -DRUNG=80 -arch=sm_89 -o rho80 rho80.cu
    # or: make cuda RUNG=80 ARCH=sm_89

Arch table for the fleet:

| GPU                          | -arch   |
|------------------------------|---------|
| RTX 6000 Ada / RTX 6000D     | sm_89   |
| RTX 4090 / L40S              | sm_89   |
| RTX PRO 6000 (Blackwell)     | sm_120 (nvcc >= 12.8; else sm_90) |
| A100                         | sm_80   |
| H100                         | sm_90   |

`-arch=native` also works on the GPU box. `-DRUNG=<bits>` recompiles for
another rung (32/48/64/80 have constant blocks in the rung table at the top of
rho80.cu; add more by adding a block with p, n, Px, Py, Qx, Qy from
challenges.json — points decompressed from the SEC1 form — plus NW64, DPBITS,
TBLBITS defaults).

## Design (what maps where)

* One CUDA thread = one r-adding walk (32 jumps, M_j = [s_j]Q + [t_j]P, jump
  index from bits 8..12 of the Montgomery-form x — identical to rho.c).
* Per step: one mixed Jacobian addition (~11 modmuls), then the point is made
  affine by a **warp-wide batch inversion** (Montgomery's trick over the 32
  lanes via two shfl scans + one Fermat inversion per warp per step,
  amortized ~4 modmuls/step).  Per-thread state stays in registers; no queues,
  no per-thread local-memory traffic.
* Distinguished points: low `dpbits` bits of the Montgomery-form affine x are
  zero.  Events go into a sharded open-addressing table in device global
  memory (64-slot shards, atomic claim -> payload -> publish), same probing
  order as rho.c.
* Collisions are written to a small candidate ring; the **host** resolves them
  exactly like rho.c (same-point: kappa = (a1-a2)*(b2-b1)^-1; negation:
  kappa = -(a1+a2)*(b1+b2)^-1) and verifies [kappa]Q == P before declaring
  SOLVED.
* Reseed-on-merge / infinity handled in-kernel (rare, divergent — fine).
* The kernel stops when the host writes the found flag; the host polls a
  device-side candidate ring and per-thread step counters every ~1.5 s.

The `CPU_REF` build compiles the identical walk/DP/collision code for the
host: N "lanes" (default 32, like a warp) run in lockstep and are batch
inverted serially — which produces the *same* zi values as the warp scan, so
lane 0 of the CPU build is the same walk as lane 0 of the GPU kernel for the
same seed.

## Run

    ./rho80                    # solve rung 80 (P,Q compiled in), ~1.4e12 group ops
    ./rho80 smoke              # ~20 s of walking, prints rate + lane-0 DP tap
    ./rho80 selftest 3         # 3 random-kappa DLPs on the same curve, then check
    ./rho80 solve DPBITS TBLBITS LANES SEED DBG

    ./rho_ref80 solve 22 22 32 12345 32    # CPU: same walk logic, dbg tap on
    ./rho_ref48 solve                      # CPU: solve rung 48 (seconds)

Arguments: `dpbits` (default 22 for rung 80; expected DPs ~ 2^40.3 / 2^dpbits),
`tblbits` (default 22 -> 4M slots, ~235 MB device memory), `lanes` (CPU build
only), `seed` (default 0x123456789ABCDEF, same default as rho.c), `dbg`
(record the first DBG distinguished points of lane 0 with their step numbers —
use this to cross-check the GPU run against rho.c/rho_ref bit-for-bit).

### Selftest expectations

* Pipeline validation on a small rung (do this first on the GPU box):
  `nvcc -DRUNG=48 ... && ./rho48 selftest 3` — solves three random-kappa DLPs
  on secp48k1 (~2.6e7 group ops each, i.e. seconds) and prints PASS/FAIL per
  instance plus `SELFTEST PASSED`.
* `./rho80 selftest 3` solves three random 80-bit DLPs — that costs 3 x
  1.4e12 group ops, i.e. 3x the real run; it is *not* part of the normal
  rung-80 workflow.  For rung 80 use `smoke` (quick rate + DP sanity) and rely
  on the RUNG=48 selftest for pipeline correctness.
* The real solve prints `SOLVED kappa=0x... verified=1` only after an
  independent host-side scalar multiplication [kappa]Q == P.

## Expected runtime math (rung 80)

* Expected group ops: 1.25*sqrt(n) = 1.25 * 2^40 ≈ **1.4e12** (rho.c measured
  25.4e6 ops for the 2^24-sqrt rung 48 = 1.51*sqrt(n), so 1.25-1.5 is right).
* Per-step cost: ~11 modmuls (mixed addition) + ~10 (two shfl scans for the
  warp batch inversion) + ~6 (affine conversion) + ~4 (amortized Fermat
  inversion) ≈ **31 modmuls/step**, each a 4x4-limb 32-bit Montgomery
  multiply (~16-24 32-bit mad/hwi ops).
* Throughput model: instruction-issue bound. An Ada-class GPU (RTX 6000 Ada:
  142 SMs x 4 schedulers x 1 warp-instruction/cycle @ ~2.5 GHz) issues
  ~1.4e12 lane-instructions/s; at ~2500-3500 lane-instructions/step that is
  **0.4-1.1 G group-ops/s**.  Conservative planning number: **0.5-2 Gops/s**,
  i.e. rung 80 in **12-47 minutes** on one card (A100/Blackwell similar or
  better; the kernel is integer-issue bound, so newer archs scale directly).
* The CPU run currently in the background (rho2, ~19-22 Mops/s on the M4)
  needs ~18-26 h for the same work; the GPU should beat it by ~3 orders of
  magnitude per card.
* If the achieved rate is much lower, check (a) `-arch` mismatch (no fast
  integer mad), (b) register spills from jac_madd locals (`-Xptxas -v`),
  (c) occupancy (`__launch_bounds__(256)`), (d) DP table traffic (raise
  dpbits).

## Validation performed locally (no GPU on the Mac)

All of this is recorded in `test_transcript.txt` (regenerate with
`python3 make_transcript.py`).

1. **Field arithmetic fuzz vs Python** (`fuzz_fe.py`): mont_mul, fe_add,
   fe_sub and fe_inv of the 32-bit-limb Montgomery code were checked against
   Python long-int arithmetic — 312500 vectors each for rungs 48, 64 and 80,
   0 mismatches.  Rungs 32/48 additionally carry a built-in 3e6-case
   `unsigned __int128` self-check (see `INTERNAL fuzz` line, 0 mismatches).
2. **CUDA-only code paths syntax check** (`cuda_stubs.h`): the kernel,
   shuffles, atomics and launch/poll code are compiled with a stub header
   under clang++ (no nvcc on this machine) — catches typos/type errors in
   exactly the parts the CPU_REF build does not compile.
3. **Jump table cross-check** vs rho.c (mode-98 dump, seed 98): all 32
   (s_j, t_j, M_j) and Q identical, bit-for-bit, for rung 48 (rho1, NW=1)
   AND rung 80 (rho2, NW=2 = the CUDA build's domain).
4. **Full rung-48 solve cross-check**: rho.c with 1 thread / 1 walk, seed
   12345, dpbits 6 vs `rho_ref48 solve 6 21 1 12345` — identical kappa and
   identical step count (25,420,821), i.e. a bit-for-bit walk.
5. **Walk cross-check on the big-limb domain**: first 24 distinguished
   points of lane 0 agree bit-for-bit with rho2 for rung 64 and rung 80
   (rho.c RHODEBUG=1, 1 thread/1 walk, dpbits 12 vs
   `rho_refXX smoke 12 22 1 81985529216486895 25`).
6. **End-to-end selftest on CPU_REF**: random-kappa DLPs solved and verified
   — rung 32: 3/3 PASS; rung 48: 3/3 PASS (steps 39.5M/42.8M/13.0M,
   consistent with the ~2.1e7 mean).  This selftest caught a real bug during
   development: the first version of the warp batch inversion used
   zi = total^-1 x prefix (missing the suffix product) — correct only for
   1 lane, which is why the single-walk cross-checks had passed while the
   32-lane selftest stalled.  Both the kernel and the CPU lane loop now use
   zi = prefix x suffix x total^-1.
7. NOT verified locally (needs the GPU box): actual CUDA execution — kernel
   launch, real __shfl semantics, atomicCAS DP insertion, candidate ring,
   memcpy polling.  These are kept standard/simple on purpose.  First things
   to run on the box: `./rho48 selftest 3` (RUNG=48 CUDA build), then
   `./rho80 smoke`, then the real solve; compare `smoke`'s lane-0 DP tap
   against `rho_ref80`'s to confirm the GPU walk matches the CPU walk.

## Recompiling for another rung

Add a block to the rung table in rho80.cu:

    #elif RUNG == 96
    #define RUNG_BITS   96
    #define NW64        2      /* R = 2^128; keep p < 0.61*R */
    #define DPBITS_DEF  25     /* ~ log2(1.25*sqrt(n)) - 18 */
    #define TBLBITS_DEF 22
    #define P_HEX ...  N_HEX ...  PX/PY/QX/QY_HEX ...

NW64 rule: choose the smallest NW64 with 64*NW64 >= bitlen(p) + 2
(the Montgomery domain is then R = 2^(64*NW64), matching rho.c's -DNW=NW64).
Rungs up to 96: NW64=2.  Rungs 112/128: NW64=3.  Rung 256: NW64=5.
DPBITS rule of thumb: dpbits = ceil(log2(1.25*sqrt(n))) - 18 (rung 80: 22),
tblbits = ceil(log2(expected DPs)) + 3, and keep the DP table under ~1 GB.
