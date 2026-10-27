/*
 * rho80.cu — CUDA port of rho.c (parallel Pollard rho with distinguished points)
 * for the "Brace for impact" ECDLP ladder (arXiv:2508.14011), specialized per rung.
 *
 *   E/F_p : y^2 = x^3 + 7,  full-range DLP  P = [kappa]*Q,  kappa in [1, n-1]
 *
 * Field arithmetic: Montgomery, 32-bit limbs (GPU friendly).  The Montgomery
 * domain is R = 2^(64*NW64) — the SAME domain rho.c uses with its 64-bit limbs
 * (rho.c NW == NW64 here) — so with the same seed the walk/DP sequence is
 * bit-for-bit identical to rho.c.  NW64 is chosen per rung so that
 * p < 0.61*R (required by the schoolbook+REDC carry bounds; see mont_mul note).
 *
 * Walk: 32-jump r-adding walk (identical structure to rho.c), one walk per CUDA
 * thread.  Affine coordinates are recovered once per step by a warp-wide batch
 * inversion (Montgomery's trick over the 32 lanes via shfl scans; one Fermat
 * inversion per warp per step, amortized 32x).  The CPU_REF build runs the exact
 * same lane logic on the host (batch inversion emulated serially) so the walk,
 * DP and collision logic is testable without a GPU.
 *
 * Distinguished points: low dpbits of the Montgomery-form affine x are zero.
 * Shared hash table in device global memory (open addressing, 64-slot shards,
 * atomic insert).  Collisions are resolved on the HOST exactly like rho.c
 * (same-point and negation cases; kappa verified by [k]Q == P before declaring
 * success).
 *
 * ============================ BUILD =============================
 * CPU reference / fuzz / cross-check build (no GPU needed, clang or gcc):
 *   clang++ -O3 -DCPU_REF -DRUNG=48 -o rho_ref48 rho80.cu
 *   clang++ -O3 -DCPU_REF -DRUNG=80 -o rho_ref80 rho80.cu
 *   clang++ -O3 -DCPU_REF -DRUNG=64 -o rho_ref64 rho80.cu
 *   clang++ -O3 -DCPU_REF -DRUNG=32 -o rho_ref32 rho80.cu
 *
 * CUDA build (CUDA >= 11.4):
 *   Arch table (fleet): RTX 6000 Ada / RTX 6000D      -> sm_89
 *                       RTX PRO 6000 (Blackwell)      -> sm_120 (or sm_90 fallback)
 *                       A100                          -> sm_80
 *                       RTX 4090 / L40                -> sm_89
 *   Default is sm_89; override with ARCH=<sm>:
 *   make cuda ARCH=sm_90          # nvcc -O3 -DRUNG=80 -arch=sm_90 -o rho80 rho80.cu
 *   make cuda ARCH=sm_80 RUNG=80  # A100
 *   (or: nvcc -O3 -DRUNG=80 -arch=native -o rho80 rho80.cu on the GPU box)
 *
 * ============================ USAGE =============================
 *   ./rho80                 # solve the compiled-in rung's real target (P,Q)
 *   ./rho80 selftest [n]    # solve n (default 3) random-kappa DLPs on the same
 *                           # curve and check the answers (costs n * full solve!)
 *   ./rho80 smoke           # ~a few seconds of walking + on-curve spot checks
 *   ./rho_ref48 fuzz [n]    # field-arithmetic fuzz vectors for fuzz_fe.py
 *   ./rho_ref48 jumptest    # dump jump table in rho.c "mode 98" format
 *   ./rho_ref48 solve [dpbits] [tblbits] [lanes] [seed] [dbg]
 *
 * ====================== SASS-PORT PATCH (lane 2, 2026-10-25) ======================
 * Techniques extracted from RCKangaroo v4 (RetiredC) turbo-SASS kernel and applied
 * to this walker (source of truth: ~/Devs/RCKangaroo, RCGpuCore.cu + RCGpuUtils.h
 * + mod_mul/mod_inv/main.asm).  RC's SASS is secp256k1-specialized (0x3D1 sparse-P
 * constants baked into IMAD chains), so no cubin can drop into our per-rung primes
 * 2^b - c; what ports is the ALGORITHM and its scheduling discipline:
 *
 *   1. Division-step modular inversion (TCHES 2020 "fast invariant-free" GCD, the
 *      algorithm RC implements in InvModP / mod_inv.asm) replaces the Fermat
 *      exponentiation in fe_inv.  RC's own SASS comment prices InvMod256 at
 *      "about 56 MulMod256"; our Fermat fe_inv costs ~1.5*bitlen(p) mont_muls
 *      (144 at rung 96) EVERY warp-round — the single dominant kernel cost.
 *      RHO_FASTINV=1 (default).  The result is the canonical inverse, so the walk
 *      sequence is bit-identical either way (verified by fuzz diff + DP-tap diff).
 *   2. Affine r-adding walk (RHO_AFFINE=1): the jump add is done directly on
 *      affine coordinates, lambda = (jy-y)/(jx-x) with (jx-x) inverted by the
 *      SAME warp-wide batch inversion the Jacobian path already pays for.  This
 *      replaces jac_madd (10 mont_muls) + pending_to_affine (4 mont_muls) per
 *      step with 3 mont_muls, matching RC's 4-mul affine point-add.  Degenerate
 *      lanes (dx==0: point==jump -> double via lambda=3x^2/(2y); point==-jump ->
 *      infinity -> reseed) contribute 2y / R1 to the batch product so the warp
 *      product never vanishes.  Walk/DP/scalar stream stays bit-identical.
 *   3. Jump table in __shared__ (RHO_SM_JT=1): 1 KB copied once per block; the
 *      32-lane gather (each lane a different jump) hits shared instead of
 *      L2-cached global — RC keeps jump tables in LDS for the same reason.
 *
 *   Not portable here: RC's hand-scheduled IMAD.WIDE.U32.X mul chains (.reuse /
 *   scoreboard-annotated SASS, mod_mul.asm) — that is the V5_MUL_IMPL=3 seam,
 *   needs p-specific SASS per rung.  No shfl-vs-constant ternaries introduced
 *   (sm_120 CUDA 13.0 miscompile class; edge lanes are overridden AFTER shfl).
 *
 *   Toggles (all default on): -DRHO_FASTINV=0 / -DRHO_AFFINE=0 / -DRHO_SM_JT=0
 *   restore the pristine path piecewise for A/B.  rho_multi.cu compatibility:
 *   walk_step / pending_to_affine / jac_* are untouched and still compiled;
 *   fe_inv keeps its signature (fast path behind the same call).
 *
 * ====================== V6 PATCH (Decruptor rung-96 campaign, 2026-10-25) ======================
 * Staged on top of the SASS-PORT kernel (md5 d12addeb26824882cc8a53294de154d1).
 * Compile-time switches (defaults in parentheses):
 *
 *   RHO_CAND_HANDSHAKE (1)  STAGE 0a: kernel-side candidate handshake.  When any
 *      lane publishes a DP_CAND record the whole grid PARKS (spins on
 *      g_cand_pend/g_found_flag) instead of walking on for up to the host's
 *      1.5 s poll interval.  Host clears g_cand_pend to release the park and
 *      writes g_found_flag = 1 only to stop (candidate genuine / budget stop).
 *      STAGE 2b fix: the flag is now 0/1 only — the old 2-on-every-poll
 *      tri-state made the no-pend threads exit at the first park boundary
 *      after the first poll, killing the grid ~1.5 s into every
 *      single-worker run.  Fixes the step metrology (GPU selftest "steps"
 *      were poll_interval x rate, i.e. VOID as difficulty metrics) and
 *      stops the ~1.5 s post-candidate walk overrun.  Walk stream UNCHANGED.
 *   RHO_CURVE_GUARD (1)     STAGE 0b: on-curve check at every DP event before
 *      insertion; an off-curve lane (the ~0.13% whole-warp init corruption) is
 *      reseeded instead of poisoning the table.  Recovers ~0.16% throughput.
 *      CPU_REF runs the identical check (never fires there: 0 in 188k events).
 *   RHO_SP_MUL (1)          STAGE 1a: special-form Montgomery multiplication
 *      for p = HI*2^(32*PH) - C (true for every rung prime here).  Requires
 *      canonical (< p) inputs, which every call site provides.  BIT-IDENTICAL
 *      results to the generic schoolbook+REDC (same T, same m_i, same tail);
 *      verified by 200k-vector fuzz diff special-vs-generic + __int128
 *      cross-check on the NL==2 rungs.  ckpt/compat preserved.
 *   RHO_ENDO (1)            STAGE 2: order-6 endomorphism symmetry walk
 *      (Wiener-Zuccherato).  See the RHO_ENDO block below.  CHANGES THE WALK:
 *      checkpoints written by a non-endo build are REFUSED (ckpt version 2).
 *   RHO_FASTINV/RHO_AFFINE/RHO_SM_JT (1/1/1) inherited from the SASS port.
 *
 * RHO_ENDO details: the automorphism group G = {+1,-1,+zeta,-zeta,+zeta^2,
 * -zeta^2} (zeta: (x,y)->(beta*x,y), beta^3=1 in F_p; zeta acts on E[n] as
 * [lambda], lambda^2+lambda+1 == 0 mod n) is quotiented out: every visited
 * point is canonicalized to a fixed orbit representative (numerically smallest
 * of {x, beta*x, beta^2*x}, y picked so its Montgomery form is odd), and the
 * scalars are transformed by the matching mu in {+-1, +-lambda, +-lambda^2}.
 * Walks on the quotient set of size n/6 need sqrt(6) fewer steps
 * (K: 1.2533 -> 0.5117).  Scalars live in a Montgomery-scalar domain (a*Rn mod
 * n, Rn = 2^(32*NL)) so the mu-multiplication is one fast REDC mul; DP records
 * and candidates carry mont-domain scalars and resolve_candidate converts back.
 * Fruitless-cycle guard: a history window of the last RHO_CYCWIN canonical-x
 * hashes; a state repeating ANY window entry -> reseed.  This is required,
 * not optional: consecutive steps with the same jump index close structurally
 * (the canon multipliers telescope to 1+lambda+lambda^2 == 0 mod n for
 * runs of 3 with all-mu(lambda|lambda^2), and to the double -1 flip for
 * runs of 2), so 2- and 3-cycles occur every ~1e3 steps per lane and would
 * otherwise trap the walk permanently.  Measured closure rates for same-jump
 * runs: m=2 1/36 of mu-pairs, m=3 2/216, m>=4 negligible; RHO_CYCWIN=8
 * covers every material length.  Plus the existing DP self-merge reseed.
 * Deferral rule: a cycle round whose state is itself a DP defers the reseed
 * for one round so dp_check can still emit a genuine cross-lane candidate,
 * but the deferral is BOUNDED (RHO_CYC_DEF rounds) — a cycle whose states are
 * ALL distinguished points would otherwise defer forever (observed: exact
 * 2-cycles with both states DP-marked, dp_ev=1 every round, lane frozen for
 * 1.5e5 rounds).  After the bound the lane reseeds unconditionally.
 * beta/lambda are DERIVED at init (not hardcoded) and verified numerically
 * against the curve points before the walk starts; "endotest" mode unit-checks
 * the canonicalization on device vs host.
 *
 * Time-sharing / restartability (see GPU80.md):
 *   RHO_SM_FRAC=0.4 ./rho80 ...   # CUDA only: launch grid = 0.4 x SMs x 8
 *                                 # blocks (share the card with a serving lane)
 *   ./rho80 ... --ckpt /path/f    # + RHO_CKPT_EVERY_S (default 300 s):
 *                                 # periodic DP-table checkpoint; rerun the
 *                                 # SAME command to resume after a kill
 *
 * Expected cost (rho with distinguished points): ~1.25*sqrt(n) group ops.
 * Rung 80: n ~ 2^80 -> ~1.4e12 group ops; at 0.5-2 Gops/s on one Ada-class GPU
 * that is roughly 12-50 minutes (see GPU80.md for the throughput math).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>

#if defined(__CUDACC__) && !defined(CPU_STUB)
  #include <cuda_runtime.h>
  #define HD  __host__ __device__
  #define DI  __device__
  #define DC  __constant__
  #define FI  __forceinline__
#else
  #define HD
  #define DI
  #define DC
  #define FI  inline
#endif

#ifndef RUNG
#define RUNG 80
#endif

/* ------------------------------------------------------------------ */
/* Rung table: compile-time curve constants.  Add a rung by adding a   */
/* block here and rebuilding with -DRUNG=<bits>.                       */
/* ------------------------------------------------------------------ */
#if RUNG == 32
#define RUNG_BITS   32
#define NW64        1      /* Montgomery domain R = 2^64  (rho.c -DNW=1) */
#define DPBITS_DEF  8
#define TBLBITS_DEF 16
#define P_HEX   "fffffba1"
#define N_HEX   "fffdfbe5"
#define PX_HEX  "1e0ef3eb"
#define PY_HEX  "4555b1d6"
#define QX_HEX  "ef01bff4"
#define QY_HEX  "56ff81a4"
#elif RUNG == 48
#define RUNG_BITS   48
#define NW64        1      /* R = 2^64 (rho.c -DNW=1) */
#define DPBITS_DEF  6
#define TBLBITS_DEF 22
#define P_HEX   "fffffffff58f"
#define N_HEX   "ffffff1b5011"
#define PX_HEX  "769ec7f35857"
#define PY_HEX  "b4a02702bdb"
#define QX_HEX  "210644f01a41"
#define QY_HEX  "553d08cef988"
#elif RUNG == 64
#define RUNG_BITS   64
#define NW64        2      /* R = 2^128 (rho.c -DNW=2; n is 65 bits) */
#define DPBITS_DEF  14
#define TBLBITS_DEF 22
#define P_HEX   "ffffffffffffed79"
#define N_HEX   "10000000153aa1c27"
#define PX_HEX  "7f3ebcb70e7ff266"
#define PY_HEX  "c8a99d561dd12d1"
#define QX_HEX  "989c2061ef0e994e"
#define QY_HEX  "47cad8f3cdc61403"
#elif RUNG == 80
#define RUNG_BITS   80
#define NW64        2      /* R = 2^128 (rho.c -DNW=2) */
#define DPBITS_DEF  22
#define TBLBITS_DEF 22
#define P_HEX   "ffffffffffffffffdfbd"     /* p = 2^80 - 8259 */
#define N_HEX   "fffffffffe06e604efc9"
#define PX_HEX  "d08430a6c00fae05651"
#define PY_HEX  "51f122611becfa454627"
#define QX_HEX  "cf45d656507f3699875e"
#define QY_HEX  "3bffc8e5352fdbabb520"
#elif RUNG == 96
#define RUNG_BITS   96
#define NW64        2      /* R = 2^128: 128 >= 96+2, REDC carry bound holds */
#define DPBITS_DEF  26
#define TBLBITS_DEF 23
#define P_HEX   "ffffffffffffffffffffe3b3"     /* p = 2^96 - 7245 (20 f's) */
#define N_HEX   "1000000000001f6d164e69747"
#define PX_HEX  "9e1f276c17bce3d8df9d22bb"
#define PY_HEX  "613eda6a9e313ff01e53a6d4"
#define QX_HEX  "ccf37e1e144a24433c0582a6"
#define QY_HEX  "2c1f6651fc6dd7f61981852d"
#else
#error "No constants for this RUNG; add a block to the rung table."
#endif

/* 32-bit limbs; NL = 2*NW64 so R = 2^(64*NW64) = 2^(32*NL), matching rho.c. */
#define NL      (2*NW64)
#define NJUMPS  32

/* ---- v6 STAGE 1a: special-form prime shape --------------------------------
 * Every rung prime has the exact shape  p = HI*2^(32*SP_PH) - C  with
 * HI = 1 (SP_HI16=0) or HI = 2^16 (SP_HI16=1) and C < 2^32:
 *   rung 32: 2^32 - 1119    rung 48: 2^48 - 2673   rung 64: 2^64 - 4743
 *   rung 80: 2^80 - 8259    rung 96: 2^96 - 7245
 * SP_PL = number of nonzero 32-bit limbs of p.  mont_mul inputs are canonical
 * (< p), so limbs SP_PL..NL-1 of every input are zero.  mont_init derives C
 * at runtime and refuses to start if the shape does not hold exactly. */
#if RUNG == 32
#define SP_PL 1
#define SP_PH 1
#define SP_HI16 0
#elif RUNG == 48
#define SP_PL 2
#define SP_PH 1
#define SP_HI16 1
#elif RUNG == 64
#define SP_PL 2
#define SP_PH 2
#define SP_HI16 0
#elif RUNG == 80
#define SP_PL 3
#define SP_PH 2
#define SP_HI16 1
#elif RUNG == 96
#define SP_PL 3
#define SP_PH 3
#define SP_HI16 0
#endif

#ifndef DPBITS
#define DPBITS  DPBITS_DEF
#endif
#ifndef TBLBITS
#define TBLBITS TBLBITS_DEF
#endif

/* ---- v6 toggles (defaults; overridable via -D) ---------------------------- */
#ifndef RHO_CAND_HANDSHAKE
#define RHO_CAND_HANDSHAKE 1    /* STAGE 0a: park-on-candidate (CUDA) */
#endif
#ifndef RHO_CURVE_GUARD
#define RHO_CURVE_GUARD 1       /* STAGE 0b: on-curve check at DP events */
#endif
#ifndef RHO_SP_MUL
#define RHO_SP_MUL 1            /* STAGE 1a: special-form Montgomery mul */
#endif
#ifndef RHO_ENDO
#define RHO_ENDO 1              /* STAGE 2: order-6 symmetry walk */
#endif
#ifndef RHO_CYCWIN
#define RHO_CYCWIN 8            /* STAGE 2: fruitless-cycle history window */
#endif
#ifndef RHO_CYC_DEF
#define RHO_CYC_DEF 3           /* STAGE 2: max deferred reseed rounds */
#endif
#ifndef RHO_CYC_ESC
#define RHO_CYC_ESC 4           /* STAGE 2b: escape steps before giving up */
#endif

typedef struct { uint32_t w[NL]; } fe;    /* field element, Montgomery form */
typedef struct { uint32_t w[NL]; } sc;    /* scalar mod n                   */
typedef struct { fe X, Y, Z;      } jac;  /* Jacobian point (Montgomery)    */

/* Host/device constant pairs: device code reads the __constant__/__device__
 * copies, host code (and the CPU_REF build) reads plain globals. */
typedef struct {
    fe P_mod, P_neg, N_mod, N_neg, R1_mont, R2_mont;
    fe P_montX, P_montY, Q_montX, Q_montY;
    uint32_t g_inv;              /* -p^-1 mod 2^32 */
    uint32_t pinv30;             /* -p^-1 mod 2^30 (fe_inv_gcd reduction constant) */
    uint32_t sp_c;               /* v6 STAGE 1a: C in p = HI*2^(32*SP_PH) - C */
    /* v6 STAGE 2: endomorphism constants (RHO_ENDO) */
    uint32_t ninv32;             /* -n^-1 mod 2^32 (sc_mulred REDC constant) */
    fe beta_m;                   /* mont(beta), beta^3 = 1, beta != 1 in F_p */
    sc lam_m, nlam_m, lam2_m, nlam2_m;  /* +-lambda, +-lambda^2 as mont scalars */
    sc scR1, scR2;               /* Rn mod n, Rn^2 mod n (scalar Montgomery) */
} FeConst;

#if defined(__CUDACC__)
DC  FeConst cK;
DI  fe c_jx[NJUMPS], c_jy[NJUMPS];
DI  sc c_js[NJUMPS], c_jt[NJUMPS];
#if RHO_ENDO
DI  sc c_jsm[NJUMPS], c_jtm[NJUMPS];   /* mont-scalar copies (s*Rn mod n) */
#endif
#endif
static FeConst hK;
static fe h_jx[NJUMPS], h_jy[NJUMPS];
static sc h_js[NJUMPS], h_jt[NJUMPS];
#if RHO_ENDO
static sc h_jsm[NJUMPS], h_jtm[NJUMPS];
#endif

#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
#define KC   (&cK)
#define JX   (c_jx)
#define JY   (c_jy)
#if RHO_ENDO
#define JS   (c_jsm)      /* STAGE 2: walk scalars live in the mont-scalar domain */
#define JT   (c_jtm)
#else
#define JS   (c_js)
#define JT   (c_jt)
#endif
#else
#define KC   (&hK)
#define JX   (h_jx)
#define JY   (h_jy)
#if RHO_ENDO
#define JS   (h_jsm)
#define JT   (h_jtm)
#else
#define JS   (h_js)
#define JT   (h_jt)
#endif
#endif

/* One distinguished-point EVENT, flushed between workers in multi-worker
 * shared-table mode (rho_multi.cu).  Same payload as a DPSlot entry; the
 * device appends these to a ring buffer, the host merges them into the
 * shared table (dedup / collision detection).  Unused in single-worker mode. */
typedef struct {
    uint64_t key;            /* low 64 bits of mont affine x          */
    uint64_t xhi;            /* remaining high bits of x (0 when NL<4) */
    sc a, b;
    uint32_t ypar;
} DPRec;

/* ---- RHO_DP_EMIT (STAGE 2b): one persisted DP insert, ordered ----------
 * Appended to the device emit ring at DP_INS time only, so "persisted" and
 * "emitted" are the same predicate and a checkpoint resume re-emits exactly
 * the stored prefix with no gaps and no resubmits.  x,y,a,b are Montgomery
 * domain; the host converts to plain at print time. */
typedef struct { fe x, y; sc a, b; } DPEmitRec;

/* runtime parameters (passed to the kernel by value; host copy for CPU_REF) */
typedef struct {
    uint64_t seed;
    uint64_t dpmask;      /* (1<<dpbits)-1 */
    uint64_t tblmask;     /* table size - 1 */
    uint32_t dpbits;
    uint32_t tblbits;
    /* debug tap: record the first dbg_max DP events of lane 0 (xk, step index).
     * Used for bit-for-bit cross-checks against rho.c / rho_ref on the box. */
    uint64_t *dbg_xk;
    uint64_t *dbg_step;
    uint32_t *dbg_n;
    uint32_t dbg_max;
    /* ---- multi-worker shared-table mode (rho_multi.cu only) ----
     * mp=0 (single mode, the default): dpring* are ignored/NULL and the
     * kernel behaves exactly as before.  mp=1: every DP event is appended
     * to the device ring below so the host can merge it into the shared
     * table; candidate records are resolved host-side from the two flushed
     * halves instead of through the device candidate ring. */
    uint32_t mp;
    uint32_t dpring_bits;
    uint64_t dpring_mask;        /* ring capacity - 1 */
    DPRec *dpring;               /* device ring of DP events          */
    unsigned *dpring_head;       /* device append counter             */
    /* ---- multi-worker PULL inbox (RHO_MULTI only) ----
     * mp=1: the host pushes pulled shared-table records into this device
     * ring and bumps *dpinbox_head as the doorbell; block 0 of the walk
     * kernel drains up to 64 records at every (round&7) checkpoint and
     * inserts them via dp_sync_insert (collisions stay host-side).  This
     * replaces a separate sync_kernel launched on another stream: the
     * endless walk kernel keeps every SM occupied with never-retiring
     * blocks, so a concurrent helper kernel's blocks can starve
     * indefinitely (measured: the first sync_kernel launch never
     * completed and the host then spun forever in the next stream sync).
     * The walk kernel draining its own inbox cannot starve.  mp=0
     * (single-worker): all three fields are NULL and the kernel is
     * unchanged. */
    DPRec *dpinbox;              /* device ring of pulled DP records  */
    unsigned *dpinbox_head;      /* host-written doorbell counter (device
                                  * memory; the host publishes it through a
                                  * pinned-staging copy-engine H2D).  NOTE:
                                  * kernel-side MAPPED host memory was tried
                                  * here and WEDGES the whole device under
                                  * co-residency on sm_120 — every thread of
                                  * the pulling worker's kernel froze at the
                                  * first (round&7) checkpoint after the
                                  * drain touched mapped memory.  Device
                                  * memory only. */
    uint64_t dpinbox_mask;       /* inbox capacity - 1                */
    unsigned *dpin_tail_dev;     /* per-worker drain cursor (device memory;
                                  * MUST be per-worker: with two co-resident
                                  * kernels on one device a shared global
                                  * cursor mixed records across the two
                                  * private inbox rings) */
    /* ---- RHO_DP_EMIT (STAGE 2b): ordered emit ring (single-worker only) --
     * emit=0 (the default): all dplog* fields are NULL/0 and the kernel is
     * unchanged.  emit=1 (env RHO_DP_EMIT=1): every DP_INS appends the full
     * point + scalars to the ring below; the host prints one JSON line per
     * record and appends it to the checkpoint log.  If the host falls behind
     * by more than a full ring the run FAILS LOUDLY (exit 3) rather than
     * silently gapping the emitted stream. */
    uint32_t emit;
    uint32_t dplog_bits;
    uint64_t dplog_mask;         /* ring capacity - 1 */
    DPEmitRec *dplog;            /* device ring of emitted DP records */
    unsigned *dplog_head;        /* device append counter             */
} Cfg;

#define GOLDEN  0x9E3779B97F4A7C15ULL
#define RSV64   0xFFFFFFFFFFFFFFFFULL   /* "claim pending" marker in DP table */

static double g_budget_s = 0;   /* >0: stop the search after this many seconds */
/* RHO_DP_EMIT (STAGE 2b): runtime flag + ring size (log2), set in main */
static int      g_dp_emit    = 0;
static uint32_t g_dplog_bits = 20;

/* ================================================================== */
/* Field arithmetic (Montgomery, 32-bit limbs).  Direct port of the    */
/* FIXED 64-bit-limb logic in rho.c (carry/borrow chains re-derived    */
/* for 32-bit limbs; T arrays are 2*NL 32-bit limbs = same capacity    */
/* as rho.c's 2*NW 64-bit limbs).                                      */
/* ================================================================== */

static HD FI uint64_t low64(const fe *a){ return (uint64_t)a->w[0] | ((uint64_t)a->w[1]<<32); }

static HD FI int fe_is_zero(const fe *a){ uint32_t v=0; for(int i=0;i<NL;i++) v|=a->w[i]; return v==0; }
static HD FI int fe_eq(const fe *a,const fe *b){ for(int i=0;i<NL;i++) if(a->w[i]!=b->w[i]) return 0; return 1; }
/* v6 STAGE 2: canonical-form less-than (limb 0 is the LSB limb) */
static HD FI int fe_lt(const fe *a,const fe *b){
    for(int i=NL-1;i>=0;i--){ if(a->w[i]<b->w[i]) return 1; if(a->w[i]>b->w[i]) return 0; }
    return 0;
}
static HD FI void fe_set(fe *r,const fe *a){ for(int i=0;i<NL;i++) r->w[i]=a->w[i]; }
static HD FI void fe_zero(fe *r){ for(int i=0;i<NL;i++) r->w[i]=0; }
static HD FI int fe_ge_p(const fe *a){
    for(int i=NL-1;i>=0;i--){ if(a->w[i]>KC->P_mod.w[i]) return 1; if(a->w[i]<KC->P_mod.w[i]) return 0; }
    return 1;
}

/* r = a+b mod p ; a,b < p */
static HD void fe_add(fe *r,const fe *a,const fe *b){
    uint32_t c=0; fe t;
    for(int i=0;i<NL;i++){ uint64_t s=(uint64_t)a->w[i]+b->w[i]+c; t.w[i]=(uint32_t)s; c=(uint32_t)(s>>32); }
    if(c){ /* sum >= 2^(32NL): sum-p = (sum mod 2^(32NL)) + (2^(32NL)-p) = rep + P_neg */
        uint32_t cc=0; fe u;
        for(int i=0;i<NL;i++){ uint64_t s2=(uint64_t)t.w[i]+KC->P_neg.w[i]+cc; u.w[i]=(uint32_t)s2; cc=(uint32_t)(s2>>32); }
        fe_set(r,&u); return;
    }
    if(fe_ge_p(&t)){ uint32_t borrow=0; fe u;
        for(int i=0;i<NL;i++){ uint64_t s2=(uint64_t)t.w[i]-KC->P_mod.w[i]-borrow; u.w[i]=(uint32_t)s2; borrow=(uint32_t)((s2>>32)&1); }
        fe_set(r,&u); return; }
    fe_set(r,&t);
}
/* r = a-b mod p ; a,b < p */
static HD void fe_sub(fe *r,const fe *a,const fe *b){
    uint32_t borrow=0; fe t;
    for(int i=0;i<NL;i++){ uint64_t s=(uint64_t)a->w[i]-b->w[i]-borrow; t.w[i]=(uint32_t)s; borrow=(uint32_t)((s>>32)&1); }
    if(borrow){ uint32_t c=0; fe u;
        for(int i=0;i<NL;i++){ uint64_t s2=(uint64_t)t.w[i]+KC->P_mod.w[i]+c; u.w[i]=(uint32_t)s2; c=(uint32_t)(s2>>32); }
        fe_set(r,&u); return; }
    fe_set(r,&t);
}

/*
 * Montgomery multiplication, schoolbook product + REDC (rho.c structure).
 * Requires p < 0.618*R  (i.e. 64*NW64 >= bitlen(p)+2) so that the
 * intermediate T = a*b + m*p < R^2 never overflows the 2*NL-limb array and
 * the REDC carry chain never runs off the top — this is the carry-overflow
 * class of bug rho.c had at p > 2^63; the per-rung NW64 table enforces it.
 */
static HD void mont_mul_generic(fe *r,const fe *a,const fe *b){
    uint32_t T[2*NL];
    for(int i=0;i<2*NL;i++) T[i]=0;
    for(int i=0;i<NL;i++){
        uint32_t carry=0;
        for(int j=0;j<NL;j++){
            uint64_t cur=(uint64_t)a->w[i]*b->w[j]+T[i+j]+carry;
            T[i+j]=(uint32_t)cur; carry=(uint32_t)(cur>>32);
        }
        int k=i+NL;
        while(carry&&k<2*NL){ uint64_t cur=(uint64_t)T[k]+carry; T[k]=(uint32_t)cur; carry=(uint32_t)(cur>>32); k++; }
    }
    for(int i=0;i<NL;i++){
        uint32_t m=T[i]*KC->g_inv, carry=0;
        for(int j=0;j<NL;j++){
            uint64_t cur=(uint64_t)m*KC->P_mod.w[j]+T[i+j]+carry;
            T[i+j]=(uint32_t)cur; carry=(uint32_t)(cur>>32);
        }
        int k=i+NL;
        while(carry&&k<2*NL){ uint64_t cur=(uint64_t)T[k]+carry; T[k]=(uint32_t)cur; carry=(uint32_t)(cur>>32); k++; }
    }
    fe res; for(int i=0;i<NL;i++) res.w[i]=T[NL+i];
    for(int s=0;s<2;s++){
        if(!fe_ge_p(&res)) break;
        uint32_t borrow=0; fe t2;
        for(int i=0;i<NL;i++){ uint64_t s2=(uint64_t)res.w[i]-KC->P_mod.w[i]-borrow; t2.w[i]=(uint32_t)s2; borrow=(uint32_t)((s2>>32)&1); }
        if(!borrow) fe_set(&res,&t2);
    }
    fe_set(r,&res);
}
/* ---- v6 STAGE 1a: special-form Montgomery multiplication --------------------
 * Requires: p = HI*2^(32*SP_PH) - C (verified in mont_init) and CANONICAL
 * inputs (< p, so limbs SP_PL..NL-1 are zero — true for every call site).
 * Schoolbook pass shrinks NL*NL -> SP_PL*SP_PL products; the REDC pass
 *   T += m*p,  m = T[i]*g_inv
 * collapses from a 4-limb schoolbook to
 *   T -= m*C  (one 32xC multiply + borrow ripple; g_inv == C^-1 mod 2^32
 *              because p = -C mod 2^32), then
 *   T += HI*m at limb i+SP_PH (carry ripple).
 * The accumulated T is algebraically identical to the generic pass (same m_i,
 * same integer T after every iteration), so results are BIT-IDENTICAL to
 * mont_mul_generic — verified by 200k-vector fuzz special-vs-generic plus the
 * internal __int128 cross-check on the NL==2 rungs. */
#if RHO_SP_MUL
static HD void mont_mul_sp(fe *r,const fe *a,const fe *b){
    uint32_t T[2*NL];
    for(int i=0;i<2*NL;i++) T[i]=0;
    for(int i=0;i<SP_PL;i++){
        uint32_t carry=0;
        for(int j=0;j<SP_PL;j++){
            uint64_t cur=(uint64_t)a->w[i]*b->w[j]+T[i+j]+carry;
            T[i+j]=(uint32_t)cur; carry=(uint32_t)(cur>>32);
        }
        for(int k=i+SP_PL;carry&&k<2*NL;k++){ uint64_t cur=(uint64_t)T[k]+carry; T[k]=(uint32_t)cur; carry=(uint32_t)(cur>>32); }
    }
    for(int i=0;i<NL;i++){
        uint32_t m=T[i]*KC->g_inv;
        /* T -= m*C  (the limb-i word nets to exactly 0: m*C == T[i] mod 2^32) */
        uint64_t mc=(uint64_t)m*KC->sp_c;
        uint32_t borrow=0;
        { uint64_t s=(uint64_t)T[i]-(uint32_t)mc; T[i]=(uint32_t)s; borrow=(uint32_t)((s>>32)&1); }
        { uint64_t s=(uint64_t)T[i+1]-(uint32_t)(mc>>32)-borrow; T[i+1]=(uint32_t)s; borrow=(uint32_t)((s>>32)&1); }
        for(int k=i+2;borrow&&k<2*NL;k++){ uint64_t s=(uint64_t)T[k]-borrow; T[k]=(uint32_t)s; borrow=(uint32_t)((s>>32)&1); }
        /* T += HI*m at limb i+SP_PH */
        uint32_t carry=0;
#if SP_HI16
        { uint64_t s=(uint64_t)T[i+SP_PH]+((uint32_t)m<<16); T[i+SP_PH]=(uint32_t)s; carry=(uint32_t)(s>>32); }
        { uint64_t s=(uint64_t)T[i+SP_PH+1]+((uint32_t)m>>16)+carry; T[i+SP_PH+1]=(uint32_t)s; carry=(uint32_t)(s>>32); }
        for(int k=i+SP_PH+2;carry&&k<2*NL;k++){ uint64_t cur=(uint64_t)T[k]+carry; T[k]=(uint32_t)cur; carry=(uint32_t)(cur>>32); }
#else
        { uint64_t s=(uint64_t)T[i+SP_PH]+m; T[i+SP_PH]=(uint32_t)s; carry=(uint32_t)(s>>32); }
        for(int k=i+SP_PH+1;carry&&k<2*NL;k++){ uint64_t cur=(uint64_t)T[k]+carry; T[k]=(uint32_t)cur; carry=(uint32_t)(cur>>32); }
#endif
    }
    fe res; for(int i=0;i<NL;i++) res.w[i]=T[NL+i];
    for(int s=0;s<2;s++){
        if(!fe_ge_p(&res)) break;
        uint32_t borrow=0; fe t2;
        for(int i=0;i<NL;i++){ uint64_t s2=(uint64_t)res.w[i]-KC->P_mod.w[i]-borrow; t2.w[i]=(uint32_t)s2; borrow=(uint32_t)((s2>>32)&1); }
        if(!borrow) fe_set(&res,&t2);
    }
    fe_set(r,&res);
}
#endif

/* dispatcher: special-form path by default, generic kept for A/B fuzz gates */
static HD FI void mont_mul(fe *r,const fe *a,const fe *b){
#if RHO_SP_MUL
    mont_mul_sp(r,a,b);
#else
    mont_mul_generic(r,a,b);
#endif
}
static HD FI void mont_sqr(fe *r,const fe *a){ mont_mul(r,a,a); }

/* Montgomery inverse: returns mont(a^-1) = a^-1 * R mod p (Fermat, p prime).
 * Skips leading zero bits of the exponent p-2 (p is ~RUNG_BITS bits). */
static HD void fe_inv_fermat(fe *r,const fe *a){
    fe e,base,result,t;
    fe_zero(&e); e.w[0]=2; fe_sub(&e,&KC->P_mod,&e);   /* e = p-2 */
    fe_set(&base,a); fe_set(&result,&KC->R1_mont);
    int top=-1;
    for(int i=NL-1;i>=0&&top<0;i--)
        if(e.w[i]){ for(int b=31;b>=0;b--) if((e.w[i]>>b)&1){ top=i*32+b; break; } }
    for(int i=top;i>=0;i--){
        mont_sqr(&result,&result);
        if((e.w[i/32]>>(i%32))&1) mont_mul(&result,&result,&base);
    }
    fe_set(r,&result);
}

/* ---- division-step (TCHES) inversion, SASS-PORT patch ----------------------
 * Port of RCKangaroo's InvModP (RCGpuUtils.h; hand-SASS twin = mod_inv.asm,
 * priced by RC at "about 56 MulMod256").  Generalized from secp256k1's sparse
 * p to our per-rung primes: the only p-special constant in the algorithm,
 * 0xD2253531 = -p^-1 mod 2^30, becomes KC->pinv30, and mul_P_by_32 (which RC
 * gets for free from p = 2^256 - 2^32 - 977) is a plain m*p schoolbook mul.
 * Requires: a in [1,p), gcd(a,p)=1.  Returns the canonical mont(a^-1) — the
 * same value fe_inv_fermat returns, so the walk is bit-identical either way.
 * Word count NLL = NL+2 gives the same +32-bit headroom per word that RC's
 * 288-bit window gives his 256-bit p (our values are < 2^(RUNG_BITS+2)). */
#define NLL (NL+2)

static HD FI uint32_t fsr30(uint32_t lo,uint32_t hi){
#if defined(__CUDA_ARCH__)
    return __funnelshift_r(lo,hi,30);
#else
    return (lo>>30)|(hi<<2);
#endif
}
static HD FI int ffsi32(int32_t v){
#if defined(__CUDA_ARCH__)
    return __ffs(v);
#else
    return __builtin_ffs(v);
#endif
}
static HD FI void addnll(uint32_t *r,const uint32_t *a,const uint32_t *b){
    uint32_t c=0;
    for(int i=0;i<NLL;i++){ uint64_t s=(uint64_t)a[i]+b[i]+c; r[i]=(uint32_t)s; c=(uint32_t)(s>>32); }
}
static HD FI void negnll(uint32_t *r){
    uint32_t c=1;
    for(int i=0;i<NLL;i++){ uint64_t s=(uint64_t)(~r[i])+c; r[i]=(uint32_t)s; c=(uint32_t)(s>>32); }
}
static HD FI void shr30nll(uint32_t *r){
    for(int i=0;i<NLL-1;i++) r[i]=fsr30(r[i],r[i+1]);
    r[NLL-1]=(uint32_t)(((int32_t)r[NLL-1])>>30);   /* arithmetic: keep the sign */
}
/* r = a*m, two's complement (|m| < 2^31; a < 2^(32*(NLL-1)) so it fits) */
static HD FI void mulnll_by_i32(uint32_t *r,const uint32_t *a,int32_t m){
    uint32_t mag=(m<0)?(uint32_t)(-(int64_t)m):(uint32_t)m;
    uint32_t carry=0;
    for(int i=0;i<NLL;i++){
        uint64_t cur=(uint64_t)a[i]*mag+carry;
        r[i]=(uint32_t)cur; carry=(uint32_t)(cur>>32);
    }
    if(m<0) negnll(r);
}
/* r = p*m for 0 <= m < 2^30 (the TCHES reduction step) */
static HD FI void mulP_by_32(uint32_t *r,uint32_t m){
    uint32_t carry=0;
    for(int i=0;i<NL;i++){
        uint64_t cur=(uint64_t)m*KC->P_mod.w[i]+carry;
        r[i]=(uint32_t)cur; carry=(uint32_t)(cur>>32);
    }
    r[NL]=carry;
    for(int i=NL+1;i<NLL;i++) r[i]=0;
}
/* r += a*m, two's complement (|m| < 2^31; the true product fits in NLL words) */
static HD FI void mulnll_acc(uint32_t *r,const uint32_t *a,int32_t m){
    uint32_t mag=(m<0)?(uint32_t)(-(int64_t)m):(uint32_t)m;
    uint64_t p=0;
    if(m>=0){
        uint32_t c=0;
        for(int i=0;i<NLL;i++){
            p+=(uint64_t)a[i]*mag;
            uint64_t s=(uint64_t)(uint32_t)r[i]+(uint32_t)p+c;
            r[i]=(uint32_t)s; c=(uint32_t)(s>>32); p>>=32;
        }
    }else{
        uint32_t br=0;
        for(int i=0;i<NLL;i++){
            p+=(uint64_t)a[i]*mag;
            uint64_t t=(uint64_t)(uint32_t)r[i]-(uint32_t)p-br;
            r[i]=(uint32_t)t; br=(uint32_t)((t>>32)&1); p>>=32;
        }
    }
}
/* r += p*m for 0 <= m < 2^30 (accumulate form of mulP_by_32) */
static HD FI void mulP_acc(uint32_t *r,uint32_t m){
    uint32_t carry=0;
    for(int i=0;i<NL;i++){
        uint64_t cur=(uint64_t)m*KC->P_mod.w[i]+carry+(uint64_t)(uint32_t)r[i];
        r[i]=(uint32_t)cur; carry=(uint32_t)(cur>>32);
    }
    for(int i=NL;i<NLL;i++){ uint64_t s=(uint64_t)(uint32_t)r[i]+carry; r[i]=(uint32_t)s; carry=(uint32_t)(s>>32); }
}
static HD FI void setnll_i32(uint32_t *r,int32_t v){
    uint32_t s=(v<0)?0xFFFFFFFFu:0u;
    r[0]=(uint32_t)v;
    for(int i=1;i<NLL;i++) r[i]=s;
}
/* r += p (signed fixup) */
static HD FI void addnll_P(uint32_t *r){
    uint32_t c=0;
    for(int i=0;i<NL;i++){ uint64_t s=(uint64_t)r[i]+KC->P_mod.w[i]+c; r[i]=(uint32_t)s; c=(uint32_t)(s>>32); }
    for(int i=NL;i<NLL;i++){ uint64_t s=(uint64_t)r[i]+c; r[i]=(uint32_t)s; c=(uint32_t)(s>>32); }
}
static HD FI void subnll_P(uint32_t *r){
    uint32_t borrow=0;
    for(int i=0;i<NL;i++){ uint64_t s=(uint64_t)r[i]-KC->P_mod.w[i]-borrow; r[i]=(uint32_t)s; borrow=(uint32_t)((s>>32)&1); }
    for(int i=NL;i<NLL;i++){ uint64_t s=(uint64_t)r[i]-borrow; r[i]=(uint32_t)s; borrow=(uint32_t)((s>>32)&1); }
}

/* explicit two's-complement i32 ops (no signed-overflow UB on host or device) */
#define NEG32(x)   ((int32_t)(0u-(uint32_t)(x)))
#define MUL32(a,b) ((int32_t)((uint32_t)(a)*(uint32_t)(b)))
#define ADD32(a,b) ((int32_t)((uint32_t)(a)+(uint32_t)(b)))
#define SHL32(a,k) ((int32_t)((uint32_t)(a)<<(k)))

static HD void fe_inv_gcd(fe *r,const fe *a){
    /* NOTE (nvcc 13.0 / sm_120 ptxas bug): this function was repeatedly
     * miscompiled inside rho_kernel (values silently wrong at -O1..-O3, clean
     * in standalone probe kernels and always correct on the host).  Two
     * structural mitigations are in place:
     *   (1) no value is carried across the main loop's exit: every update is
     *       committed inside the iteration (the original computed t0/t1 before
     *       the termination test and consumed them after the break);
     *   (2) the word-array working set is minimal (5 arrays, in-place fused
     *       updates) so ptxas can hold it in registers instead of the local
     *       window whose slot allocation was the corruption vector. */
    uint32_t modp[NLL],val[NLL],res[NLL],acc[NLL],t1[NLL];
    int matrix[4],_val,_modp,index,cnt,mx,kbnt;
    const uint32_t g30=KC->pinv30;   /* -p^-1 mod 2^30 */
    for(int i=0;i<NLL;i++){
        val[i]=(i<NL)?a->w[i]:0u;
        modp[i]=(i<NL)?KC->P_mod.w[i]:0u;
    }
    matrix[0]=matrix[3]=1; matrix[1]=matrix[2]=0;
    kbnt=-1;
    _val=(int32_t)val[0]; _modp=(int32_t)modp[0];
    index=ffsi32(_val|0x40000000)-1;
    matrix[0]=SHL32(matrix[0],index); matrix[1]=SHL32(matrix[1],index);
    kbnt-=index; _val>>=index;
    cnt=30-index;
    while(cnt>0){
        if(kbnt<0){ kbnt=-kbnt; int t=NEG32(_modp); _modp=_val; _val=t;
                    t=NEG32(matrix[0]); matrix[0]=matrix[2]; matrix[2]=t;
                    t=NEG32(matrix[1]); matrix[1]=matrix[3]; matrix[3]=t; }
        mx=(kbnt+1<cnt)?31-kbnt:32-cnt;
        int32_t mul=MUL32(NEG32(_modp),_val)&7;
        mul&=(int32_t)(0xFFFFFFFFu>>mx);
        _val=ADD32(_val,MUL32(_modp,mul));
        matrix[2]=ADD32(matrix[2],MUL32(matrix[0],mul));
        matrix[3]=ADD32(matrix[3],MUL32(matrix[1],mul));
        index=ffsi32(_val|(1<<cnt))-1;
        matrix[0]=SHL32(matrix[0],index); matrix[1]=SHL32(matrix[1],index);
        kbnt-=index; _val>>=index;
        cnt-=index;
    }
    /* first pass: val-side sum into t1, modp-side in place */
    mulnll_by_i32(t1,modp,matrix[2]);
    mulnll_acc(t1,val,matrix[3]);
    mulnll_by_i32(modp,modp,matrix[0]);
    mulnll_acc(modp,val,matrix[1]);
    shr30nll(modp);
    shr30nll(t1);
    for(int i=0;i<NLL;i++) val[i]=t1[i];
    /* res tracks the transform applied to val; acc tracks modp's chain */
    setnll_i32(t1,matrix[1]);
    mulP_by_32(res,(t1[0]*g30)&0x3FFFFFFFu); addnll(res,res,t1); shr30nll(res);
    setnll_i32(t1,matrix[3]);
    mulP_by_32(acc,(t1[0]*g30)&0x3FFFFFFFu); addnll(acc,acc,t1); shr30nll(acc);
    for(int guard=0;guard<256;guard++){
        matrix[0]=matrix[3]=1; matrix[1]=matrix[2]=0;
        _val=(int32_t)val[0];
        _modp=(int32_t)modp[0];
        index=ffsi32(_val|0x40000000)-1;
        matrix[0]=SHL32(matrix[0],index); matrix[1]=SHL32(matrix[1],index);
        kbnt-=index; _val>>=index;
        cnt=30-index;
        while(cnt>0){
            if(kbnt<0){ kbnt=-kbnt; int t=NEG32(_modp); _modp=_val; _val=t;
                        t=NEG32(matrix[0]); matrix[0]=matrix[2]; matrix[2]=t;
                        t=NEG32(matrix[1]); matrix[1]=matrix[3]; matrix[3]=t; }
            mx=(kbnt+1<cnt)?31-kbnt:32-cnt;
            int32_t mul=MUL32(NEG32(_modp),_val)&7;
            mul&=(int32_t)(0xFFFFFFFFu>>mx);
            _val=ADD32(_val,MUL32(_modp,mul));
            matrix[2]=ADD32(matrix[2],MUL32(matrix[0],mul));
            matrix[3]=ADD32(matrix[3],MUL32(matrix[1],mul));
            index=ffsi32(_val|(1<<cnt))-1;
            matrix[0]=SHL32(matrix[0],index); matrix[1]=SHL32(matrix[1],index);
            kbnt-=index; _val>>=index;
            cnt-=index;
        }
        /* modp/val pair: val-side sum first, then modp in place */
        mulnll_by_i32(t1,modp,matrix[2]);
        mulnll_acc(t1,val,matrix[3]);
        mulnll_by_i32(modp,modp,matrix[0]);
        mulnll_acc(modp,val,matrix[1]);
        shr30nll(modp);
        shr30nll(t1);
        for(int i=0;i<NLL;i++) val[i]=t1[i];
        /* res/acc side: acc-side sum first (reads OLD res and OLD acc) */
        mulnll_by_i32(t1,res,matrix[2]);
        mulnll_acc(t1,acc,matrix[3]);
        mulnll_by_i32(res,res,matrix[0]);
        mulnll_acc(res,acc,matrix[1]);
        mulP_acc(res,((uint32_t)res[0]*g30)&0x3FFFFFFFu);
        shr30nll(res);
        { uint32_t nz=0; for(int i=0;i<NLL-1;i++) nz|=val[i]; if(!nz) break; }
        mulP_acc(t1,((uint32_t)t1[0]*g30)&0x3FFFFFFFu);
        shr30nll(t1);
        for(int i=0;i<NLL;i++) acc[i]=t1[i];
    }
    if((int32_t)modp[NLL-1]<0) negnll(res);
    /* sign fixups on the top word (TCHES bound keeps |res| within a few p) */
    for(int k=0;k<4 && (int32_t)res[NLL-1]<0;k++) addnll_P(res);
    for(int k=0;k<4 && (int32_t)res[NLL-1]>0;k++) subnll_P(res);
    /* canonicalize into [0,p): res may sit in [p,2p) with a zero top word,
       which the top-word-only fixups above cannot see */
    for(int k=0;k<4;k++){
        uint32_t hi=0; for(int i=NL;i<NLL;i++) hi|=res[i];
        int ge=hi?1:0;
        if(!ge){
            for(int i=NL-1;i>=0;i--){
                if(res[i]!=KC->P_mod.w[i]){ ge=(res[i]>KC->P_mod.w[i]); break; }
                if(i==0) ge=1; /* res == p */
            }
        }
        if(!ge) break;
        subnll_P(res);
    }
    /* The TCHES division-step algorithm returns the PLAIN inverse of its
       input. Our contract (matching fe_inv_fermat) is the Montgomery-form
       inverse a^-1*R. Input here is a_mont = a*R, so the raw result is
       (a*R)^-1 = a^-1*R^-1; each mont_mul(.,R2_mont) multiplies by R
       (R2_mont holds R^2 as a residue, mont_mul = x*y*R^-1), so fold twice. */
    {
        fe tf,tg;
        for(int i=0;i<NL;i++) tf.w[i]=res[i];
        mont_mul(&tg,&tf,&KC->R2_mont);
        mont_mul(r,&tg,&KC->R2_mont);
    }
}

static HD void fe_inv(fe *r,const fe *a){
#ifndef RHO_FASTINV
#define RHO_FASTINV 1
#endif
#if RHO_FASTINV
    fe_inv_gcd(r,a);
#else
    fe_inv_fermat(r,a);
#endif
}

/* ================================================================== */
/* Scalar arithmetic mod n                                             */
/* ================================================================== */

static HD FI const sc *N_as_sc(void){ return (const sc*)&KC->N_mod; }
static HD FI int sc_is_zero(const sc *a){ uint32_t v=0; for(int i=0;i<NL;i++) v|=a->w[i]; return v==0; }
static HD FI int sc_eq(const sc *a,const sc *b){ for(int i=0;i<NL;i++) if(a->w[i]!=b->w[i]) return 0; return 1; }
static HD FI void sc_set(sc *r,const sc *a){ for(int i=0;i<NL;i++) r->w[i]=a->w[i]; }
static HD FI void sc_zero(sc *r){ for(int i=0;i<NL;i++) r->w[i]=0; }
static HD FI int sc_ge_n(const sc *a){
    for(int i=NL-1;i>=0;i--){ if(a->w[i]>KC->N_mod.w[i]) return 1; if(a->w[i]<KC->N_mod.w[i]) return 0; }
    return 1;
}
static HD void sc_subn_(sc *r){
    uint32_t borrow=0; sc t;
    for(int i=0;i<NL;i++){ uint64_t s=(uint64_t)r->w[i]-KC->N_mod.w[i]-borrow; t.w[i]=(uint32_t)s; borrow=(uint32_t)((s>>32)&1); }
    sc_set(r,&t);
}
/* r = a+b mod n ; a,b < n */
static HD void sc_add_n(sc *r,const sc *a,const sc *b){
    uint32_t c=0; sc t;
    for(int i=0;i<NL;i++){ uint64_t s=(uint64_t)a->w[i]+b->w[i]+c; t.w[i]=(uint32_t)s; c=(uint32_t)(s>>32); }
    if(c){ /* sum >= 2^(32NL): sum-n = (sum mod 2^(32NL)) + (2^(32NL)-n) */
        uint32_t cc=0; sc u;
        for(int i=0;i<NL;i++){ uint64_t s2=(uint64_t)t.w[i]+KC->N_neg.w[i]+cc; u.w[i]=(uint32_t)s2; cc=(uint32_t)(s2>>32); }
        sc_set(r,&u); return; }
    if(sc_ge_n(&t)) sc_subn_(&t);
    sc_set(r,&t);
}
/* r = a-b mod n ; a,b < n */
static HD void sc_sub_n(sc *r,const sc *a,const sc *b){
    uint32_t borrow=0; sc t;
    for(int i=0;i<NL;i++){ uint64_t s=(uint64_t)a->w[i]-b->w[i]-borrow; t.w[i]=(uint32_t)s; borrow=(uint32_t)((s>>32)&1); }
    if(borrow){ uint32_t c=0; sc u;
        for(int i=0;i<NL;i++){ uint64_t s2=(uint64_t)t.w[i]+KC->N_mod.w[i]+c; u.w[i]=(uint32_t)s2; c=(uint32_t)(s2>>32); }
        sc_set(r,&u); return; }
    sc_set(r,&t);
}
/* v6 STAGE 2: Montgomery multiplication mod n.  a,b and r live in the
 * Montgomery-scalar domain (v*Rn mod n, Rn = 2^(32*NL)); each call costs
 * a generic NL x NL schoolbook plus REDC with the per-rung constant
 * ninv32 = -n^-1 mod 2^32.  Inputs must be canonical (< n), which the walk
 * maintains: mont-scalar constants and reseed conversions are canonical,
 * and sc_add_n/sc_sub_n keep results canonical.  Reduction adds m*N at
 * limb i, so T stays < n^2 + R*n < R*n + R*n -> at most one conditional
 * subtract (loop bound 2 is belt-and-braces). */
static HD void sc_mulred(sc *r,const sc *a,const sc *b){
    uint32_t T[2*NL];
    for(int i=0;i<2*NL;i++) T[i]=0;
    for(int i=0;i<NL;i++){
        uint32_t carry=0;
        for(int j=0;j<NL;j++){
            uint64_t cur=(uint64_t)a->w[i]*b->w[j]+T[i+j]+carry;
            T[i+j]=(uint32_t)cur; carry=(uint32_t)(cur>>32);
        }
        for(int k=i+NL;carry&&k<2*NL;k++){ uint64_t cur=(uint64_t)T[k]+carry; T[k]=(uint32_t)cur; carry=(uint32_t)(cur>>32); }
    }
    for(int i=0;i<NL;i++){
        uint32_t m=T[i]*KC->ninv32, carry=0;
        for(int j=0;j<NL;j++){
            uint64_t cur=(uint64_t)m*KC->N_mod.w[j]+T[i+j]+carry;
            T[i+j]=(uint32_t)cur; carry=(uint32_t)(cur>>32);
        }
        for(int k=i+NL;carry&&k<2*NL;k++){ uint64_t cur=(uint64_t)T[k]+carry; T[k]=(uint32_t)cur; carry=(uint32_t)(cur>>32); }
    }
    sc res;
    for(int i=0;i<NL;i++) res.w[i]=T[NL+i];
    for(int s=0;s<2;s++){
        if(!sc_ge_n(&res)) break;
        sc_subn_(&res);
    }
    sc_set(r,&res);
}

/* product mod n, slow path (host-only use: candidate recovery, verification) */
static HD void sc_mul(sc *r,const sc *a,const sc *b){
    uint32_t T[2*NL];
    for(int i=0;i<2*NL;i++) T[i]=0;
    for(int i=0;i<NL;i++){
        uint32_t carry=0;
        for(int j=0;j<NL;j++){
            uint64_t cur=(uint64_t)a->w[i]*b->w[j]+T[i+j]+carry;
            T[i+j]=(uint32_t)cur; carry=(uint32_t)(cur>>32);
        }
        int k=i+NL; while(carry&&k<2*NL){ uint64_t cur=(uint64_t)T[k]+carry; T[k]=(uint32_t)cur; carry=(uint32_t)(cur>>32); k++; }
    }
    sc res; sc_zero(&res);
    for(int bit=2*NL*32-1;bit>=0;bit--){
        uint32_t c=0;
        for(int i=0;i<NL;i++){ uint32_t nc=res.w[i]>>31; res.w[i]=(res.w[i]<<1)|c; c=nc; }
        if((T[bit/32]>>(bit%32))&1){ uint64_t s=(uint64_t)res.w[0]+1; res.w[0]=(uint32_t)s;
            uint32_t cc=(uint32_t)(s>>32); for(int i=1;cc&&i<NL;i++){ uint64_t s2=(uint64_t)res.w[i]+cc; res.w[i]=(uint32_t)s2; cc=(uint32_t)(s2>>32);} }
        if(c){ /* res wrapped: value-n = res + (2^(32NL)-n) */
            uint32_t cc=0; sc t2;
            for(int i=0;i<NL;i++){ uint64_t s2=(uint64_t)res.w[i]+KC->N_neg.w[i]+cc; t2.w[i]=(uint32_t)s2; cc=(uint32_t)(s2>>32); }
            sc_set(&res,&t2);
        } else if(sc_ge_n(&res)) sc_subn_(&res);
    }
    sc_set(r,&res);
}
static HD void sc_pow(sc *r,const sc *base_,const sc *e){
    sc result,base;
    sc_zero(&result); result.w[0]=1;
    sc_set(&base,base_);
    for(int bit=NL*32-1;bit>=0;bit--){
        if((e->w[bit/32]>>(bit%32))&1){ sc t; sc_mul(&t,&result,&base); sc_set(&result,&t); }
        if(bit){ sc t; sc_mul(&t,&result,&result); sc_set(&result,&t); }
    }
    sc_set(r,&result);
}
/* binary extended gcd mod n */
static HD void sc_inv(sc *r,const sc *a){
    sc u,v,x1,x2,t;
    sc_set(&u,a); sc_set(&v,N_as_sc());
    sc_zero(&x1); sc_zero(&x2); x1.w[0]=1;
    while(!sc_is_zero(&v)){
        while(!(u.w[0]&1)){
            uint32_t carry=0;
            for(int i=NL-1;i>=0;i--){ uint32_t nc=u.w[i]&1; u.w[i]=(u.w[i]>>1)|(carry<<31); carry=nc; }
            if(x1.w[0]&1) sc_add_n(&x1,&x1,N_as_sc());
            uint32_t c2=0;
            for(int i=NL-1;i>=0;i--){ uint32_t nc=x1.w[i]&1; x1.w[i]=(x1.w[i]>>1)|(c2<<31); c2=nc; }
        }
        while(!(v.w[0]&1)){
            uint32_t carry=0;
            for(int i=NL-1;i>=0;i--){ uint32_t nc=v.w[i]&1; v.w[i]=(v.w[i]>>1)|(carry<<31); carry=nc; }
            if(x2.w[0]&1) sc_add_n(&x2,&x2,N_as_sc());
            uint32_t c2=0;
            for(int i=NL-1;i>=0;i--){ uint32_t nc=x2.w[i]&1; x2.w[i]=(x2.w[i]>>1)|(c2<<31); c2=nc; }
        }
        int uge=0;
        for(int i=NL-1;i>=0;i--){ if(u.w[i]>v.w[i]){uge=1;break;} if(u.w[i]<v.w[i]){uge=0;break;} }
        if(uge){ uint32_t borrow=0;
                 for(int i=0;i<NL;i++){ uint64_t s=(uint64_t)u.w[i]-v.w[i]-borrow; t.w[i]=(uint32_t)s; borrow=(uint32_t)((s>>32)&1); }
                 sc_set(&u,&t); sc_sub_n(&t,&x1,&x2); sc_set(&x1,&t); }
        else   { uint32_t borrow=0;
                 for(int i=0;i<NL;i++){ uint64_t s=(uint64_t)v.w[i]-u.w[i]-borrow; t.w[i]=(uint32_t)s; borrow=(uint32_t)((s>>32)&1); }
                 sc_set(&v,&t); sc_sub_n(&t,&x2,&x1); sc_set(&x2,&t); }
    }
    sc_set(r,&x1);
}

/* splitmix64 stream, identical to rho.c's rnd64 */
static HD uint64_t rnd64(uint64_t *s){
    uint64_t z=(*s += GOLDEN);
    z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL;
    z=(z^(z>>27))*0x94D049BB133111EBULL;
    return z^(z>>31);
}
/* NW64 random 64-bit words reduced mod n by binary shift-subtract —
 * identical value stream to rho.c's random_sc (same draws, same reduction). */
static HD void random_sc(sc *r,uint64_t *rng){
    uint64_t rw[NW64];
    for(int i=0;i<NW64;i++) rw[i]=rnd64(rng);
    sc res; sc_zero(&res);
    for(int bit=NW64*64-1;bit>=0;bit--){
        uint32_t c=0;
        for(int i=0;i<NL;i++){ uint32_t nc=res.w[i]>>31; res.w[i]=(res.w[i]<<1)|c; c=nc; }
        if((rw[bit/64]>>(bit%64))&1){
            uint64_t s=(uint64_t)res.w[0]+1; res.w[0]=(uint32_t)s;
            uint32_t cc=(uint32_t)(s>>32);
            for(int i=1;cc&&i<NL;i++){ uint64_t s2=(uint64_t)res.w[i]+cc; res.w[i]=(uint32_t)s2; cc=(uint32_t)(s2>>32); }
        }
        /* c!=0 only when n > 2^(32NL-1); cannot happen for the rungs here
         * (NW64 is sized so n < 2^(64*NW64-1)), handled for safety anyway */
        if(c){ uint32_t cc=0; sc t2;
            for(int i=0;i<NL;i++){ uint64_t s2=(uint64_t)res.w[i]+KC->N_neg.w[i]+cc; t2.w[i]=(uint32_t)s2; cc=(uint32_t)(s2>>32); }
            sc_set(&res,&t2);
        } else if(sc_ge_n(&res)) sc_subn_(&res);
    }
    sc_set(r,&res);
    if(sc_is_zero(r)) r->w[0]=1;
}

/* ================================================================== */
/* Jacobian point arithmetic (direct port of rho.c)                    */
/* ================================================================== */

static HD FI void jac_set_inf(jac *R){ fe_zero(&R->X); fe_set(&R->Y,&KC->R1_mont); fe_zero(&R->Z); }

static HD void jac_dbl(jac *R,const jac *P1){
    if(fe_is_zero(&P1->Z)||fe_is_zero(&P1->Y)){ jac_set_inf(R); return; }
    fe A,B,C,D,E,F,X3,Y3,Z3,t,u;
    mont_sqr(&A,&P1->X);
    mont_sqr(&B,&P1->Y);
    mont_sqr(&C,&B);
    fe_add(&t,&P1->X,&B); mont_sqr(&t,&t); fe_sub(&t,&t,&A); fe_sub(&t,&t,&C);
    fe_add(&D,&t,&t);
    fe_add(&E,&A,&A); fe_add(&E,&E,&A);
    mont_sqr(&F,&E);
    fe_add(&X3,&D,&D); fe_sub(&X3,&F,&X3);
    fe_add(&t,&C,&C); fe_add(&t,&t,&t); fe_add(&t,&t,&t);
    fe_sub(&u,&D,&X3); mont_mul(&Y3,&E,&u); fe_sub(&Y3,&Y3,&t);
    mont_mul(&Z3,&P1->Y,&P1->Z); fe_add(&Z3,&Z3,&Z3);
    R->X=X3; R->Y=Y3; R->Z=Z3;
}
/* mixed: R = P1(jac) + (x2,y2)(affine) */
static HD void jac_madd(jac *R,const jac *P1,const fe *x2,const fe *y2){
    if(fe_is_zero(&P1->Z)){ fe_set(&R->X,x2); fe_set(&R->Y,y2); fe_set(&R->Z,&KC->R1_mont); return; }
    fe Z1Z1,U2,S2,H,HH,HHH,V,rr,X3,Y3,Z3,t,u;
    mont_sqr(&Z1Z1,&P1->Z);
    mont_mul(&U2,x2,&Z1Z1);
    mont_mul(&S2,y2,&Z1Z1); mont_mul(&S2,&S2,&P1->Z);
    fe_sub(&H,&U2,&P1->X);
    if(fe_is_zero(&H)){
        fe neg; fe_sub(&neg,&KC->P_mod,y2);
        if(fe_eq(&P1->Y,y2)){ jac d; jac_dbl(&d,P1); *R=d; }
        else jac_set_inf(R);
        return;
    }
    mont_sqr(&HH,&H);
    mont_mul(&HHH,&H,&HH);
    mont_mul(&V,&P1->X,&HH);
    fe_sub(&rr,&S2,&P1->Y);
    mont_sqr(&X3,&rr);
    fe_add(&t,&HHH,&V); fe_add(&t,&t,&V); fe_sub(&X3,&X3,&t);
    fe_sub(&u,&V,&X3);
    mont_mul(&t,&P1->Y,&HHH);
    mont_mul(&Y3,&rr,&u); fe_sub(&Y3,&Y3,&t);
    mont_mul(&Z3,&P1->Z,&H);
    R->X=X3; R->Y=Y3; R->Z=Z3;
}
static HD void jac_add(jac *R,const jac *A,const jac *B){
    if(fe_is_zero(&A->Z)){ *R=*B; return; }
    if(fe_is_zero(&B->Z)){ *R=*A; return; }
    fe Z1Z1,Z2Z2,U1,U2,S1,S2,H,I,J,r,V,X3,Y3,Z3,t,u;
    mont_sqr(&Z1Z1,&A->Z); mont_sqr(&Z2Z2,&B->Z);
    mont_mul(&U1,&A->X,&Z2Z2); mont_mul(&U2,&B->X,&Z1Z1);
    mont_mul(&S1,&A->Y,&Z2Z2); mont_mul(&S1,&S1,&B->Z);
    mont_mul(&S2,&B->Y,&Z1Z1); mont_mul(&S2,&S2,&A->Z);
    fe_sub(&H,&U2,&U1);
    fe_sub(&r,&S2,&S1); fe_add(&r,&r,&r);
    if(fe_is_zero(&H)){
        if(fe_is_zero(&r)){ jac d; jac_dbl(&d,A); *R=d; }
        else jac_set_inf(R);
        return;
    }
    fe_add(&t,&H,&H); mont_sqr(&I,&t);
    mont_mul(&J,&H,&I);
    mont_mul(&V,&U1,&I);
    mont_sqr(&X3,&r); fe_sub(&X3,&X3,&J);
    fe_add(&t,&V,&V); fe_sub(&X3,&X3,&t);
    mont_mul(&Y3,&S1,&J); fe_add(&Y3,&Y3,&Y3);
    fe_sub(&u,&V,&X3); mont_mul(&t,&r,&u); fe_sub(&Y3,&t,&Y3);
    fe_add(&t,&A->Z,&B->Z); mont_sqr(&t,&t); fe_sub(&t,&t,&Z1Z1); fe_sub(&t,&t,&Z2Z2);
    mont_mul(&Z3,&t,&H);
    R->X=X3; R->Y=Y3; R->Z=Z3;
}
static HD void scalar_mul_point(jac *R,const sc *k,const fe *xa,const fe *ya){
    jac_set_inf(R);
    jac base; base.X=*xa; base.Y=*ya; fe_set(&base.Z,&KC->R1_mont);
    for(int bit=NL*32-1;bit>=0;bit--){
        jac d; jac_dbl(&d,R);
        if((k->w[bit/32]>>(bit%32))&1){ jac t; jac_add(&t,&d,&base); d=t; }
        *R=d;
    }
}
/* Jacobian (mont domain) -> affine (mont domain): x = X*Z^-2, y = Y*Z^-3 */
static HD void jac_to_affine(const jac *J,fe *xa,fe *ya){
    fe Zinv,z2,z3;
    fe_inv(&Zinv,&J->Z);
    mont_mul(&z2,&Zinv,&Zinv);
    mont_mul(&z3,&z2,&Zinv);
    mont_mul(xa,&J->X,&z2);
    mont_mul(ya,&J->Y,&z3);
}
/* verify [k]Q == P (both affine mont) — reads KC->P_montX/Y */
static HD int verify_kappa(const sc *k){
    jac R; scalar_mul_point(&R,k,&KC->Q_montX,&KC->Q_montY);
    if(fe_is_zero(&R.Z)) return 0;
    fe x,y; jac_to_affine(&R,&x,&y);
    return fe_eq(&x,&KC->P_montX) && fe_eq(&y,&KC->P_montY);
}

/* ================================================================== */
/* Setup: hex parsing, Montgomery constants, jump table, walk reseed   */
/* ================================================================== */

static void parse_fe(fe *r,const char *s){
    fe_zero(r);
    int len=(int)strlen(s);
    if(len>NL*8){ fprintf(stderr,"hex constant too long (%d limbs)\n",NL); exit(1); }
    for(int i=0;i<len;i++){
        char c=s[i]; uint32_t v=(c>='0'&&c<='9')?(uint32_t)(c-'0'):((c>='a'&&c<='f')?(uint32_t)(c-'a'+10):(uint32_t)(c-'A'+10));
        uint32_t carry=v;
        for(int w=0;w<NL;w++){ uint32_t nc=r->w[w]>>28; r->w[w]=(r->w[w]<<4)|carry; carry=nc; }
    }
}
static void print_fe_words(const fe *a){ /* 32-bit limbs, big-endian order */
    for(int i=NL-1;i>=0;i--) printf("%08x",a->w[i]);
}
static void print_sc_words(const sc *a){
    for(int i=NL-1;i>=0;i--) printf("%08x",a->w[i]);
}
/* print a fe/sc as NW64 big-endian 64-bit words (rho.c print style) */
static void print_val64(const uint32_t *w){
    for(int k=NW64-1;k>=0;k--)
        printf("%016llx",(unsigned long long)(((uint64_t)w[2*k+1]<<32)|w[2*k]));
}


static void mont_init(void){
    parse_fe(&hK.P_mod,P_HEX);
    parse_fe((fe*)&hK.N_mod,N_HEX);
    /* two's complements: 2^(32NL) - p and -n */
    { uint32_t bb=1;
      for(int i=0;i<NL;i++){ uint64_t s=(uint64_t)0-hK.P_mod.w[i]-bb; hK.P_neg.w[i]=(uint32_t)s; bb=(uint32_t)((s>>32)&1); } }
    { uint32_t bb=1;
      for(int i=0;i<NL;i++){ uint64_t s=(uint64_t)0-hK.N_mod.w[i]-bb; hK.N_neg.w[i]=(uint32_t)s; bb=(uint32_t)((s>>32)&1); } }
    /* g_inv = -p^-1 mod 2^32 (Newton; 5 iterations reach 32 bits) */
    uint32_t inv=1;
    for(int i=0;i<5;i++) inv*=2u-inv*hK.P_mod.w[0];
    if((uint32_t)(hK.P_mod.w[0]*inv)!=1){ fprintf(stderr,"inv check failed\n"); exit(1); }
    hK.g_inv=0u-inv;
    /* pinv30 = -p^-1 mod 2^30 : the TCHES division-step inversion's reduction
     * constant (RC hardcodes 0xD2253531 for secp256k1).  Newton from 1 bit. */
    { uint32_t inv30=1;
      for(int i=0;i<5;i++) inv30*=2u-inv30*hK.P_mod.w[0];   /* p^-1 mod 2^32 */
      inv30&=0x3FFFFFFFu;                                   /* p^-1 mod 2^30 */
      hK.pinv30=0u-inv30;                                   /* -p^-1 mod 2^30 */
      if((((hK.pinv30*(hK.P_mod.w[0]&0x3FFFFFFFu))+1u)&0x3FFFFFFFu)!=0){
        fprintf(stderr,"pinv30 check failed\n"); exit(1);
      }
    }
    /* R1 = 2^(32NL) mod p ; R2 = R1^2 mod p */
    fe Rmod; fe_zero(&Rmod); Rmod.w[0]=1;
    for(int i=0;i<32*NL;i++) fe_add(&Rmod,&Rmod,&Rmod);
    fe_set(&hK.R1_mont,&Rmod);
    fe_set(&hK.R2_mont,&Rmod);
    for(int i=0;i<32*NL;i++) fe_add(&hK.R2_mont,&hK.R2_mont,&hK.R2_mont);
#if RHO_SP_MUL
    /* v6 STAGE 1a: derive C = HI*2^(32*SP_PH) - p and verify the special form.
     * g_inv (= -p^-1 mod 2^32) doubles as C^-1 mod 2^32 because p = -C mod 2^32. */
    {
        uint32_t W[NL]; for(int i=0;i<NL;i++) W[i]=0;
#if SP_HI16
        W[SP_PH]=1u<<16;
#else
        W[SP_PH]=1;
#endif
        uint32_t borrow=0, Cw[NL];
        for(int i=0;i<NL;i++){ uint64_t s=(uint64_t)W[i]-hK.P_mod.w[i]-borrow; Cw[i]=(uint32_t)s; borrow=(uint32_t)((s>>32)&1); }
        if(borrow||Cw[0]==0){ fprintf(stderr,"FATAL: special-form shape failed (C underflow)\n"); exit(1); }
        for(int i=1;i<NL;i++) if(Cw[i]){ fprintf(stderr,"FATAL: special-form shape failed (C >= 2^32)\n"); exit(1); }
        hK.sp_c=Cw[0];
        /* cross-check: p == HI*2^(32*SP_PH) - C must reproduce P_mod */
        uint32_t V[NL]; for(int i=0;i<NL;i++) V[i]=0;
#if SP_HI16
        V[SP_PH]=1u<<16;
#else
        V[SP_PH]=1;
#endif
        uint32_t b2=0;
        for(int i=0;i<NL;i++){ uint64_t s=(uint64_t)V[i]-((i==0)?hK.sp_c:0u)-b2; V[i]=(uint32_t)s; b2=(uint32_t)((s>>32)&1); }
        for(int i=0;i<NL;i++) if(V[i]!=hK.P_mod.w[i]){ fprintf(stderr,"FATAL: special-form shape failed (reconstruction)\n"); exit(1); }
        fprintf(stderr,"sp-mul: p = %s*2^%d - %u (PL=%d)\n",SP_HI16?"2^16":"1",32*SP_PH,hK.sp_c,SP_PL);
    }
#endif
    fe Px,Py,Qx,Qy;
    parse_fe(&Px,PX_HEX); parse_fe(&Py,PY_HEX);
    parse_fe(&Qx,QX_HEX); parse_fe(&Qy,QY_HEX);
    if(fe_ge_p(&Px)||fe_ge_p(&Py)||fe_ge_p(&Qx)||fe_ge_p(&Qy)){
        fprintf(stderr,"point coordinate not reduced mod p\n"); exit(1);
    }
    mont_mul(&hK.P_montX,&Px,&hK.R2_mont); mont_mul(&hK.P_montY,&Py,&hK.R2_mont);
    mont_mul(&hK.Q_montX,&Qx,&hK.R2_mont); mont_mul(&hK.Q_montY,&Qy,&hK.R2_mont);
}

/* curve sanity: y^2 = x^3+7 for both P and Q (mont domain).
 * v6: HD so the kernel can run the STAGE 0b per-DP on-curve guard. */
static HD int on_curve(const fe *xm,const fe *ym){
    fe lhs,rhs,xx,seven;
    mont_sqr(&lhs,ym);
    mont_mul(&xx,xm,xm); mont_mul(&xx,&xx,xm);
    fe_zero(&seven); seven.w[0]=7; mont_mul(&seven,&seven,&KC->R2_mont);
    fe_add(&rhs,&xx,&seven);
    return fe_eq(&lhs,&rhs);
}

/* 32-jump r-adding walk table: M_j = [s_j]Q + [t_j]P  (rho.c order/RNG) */
static void build_jumptable(uint64_t seed){
    uint64_t rs=seed?seed:0xDEADBEEFCAFEBABEULL;
    for(int j=0;j<NJUMPS;j++){
        sc s_,t_; random_sc(&s_,&rs); random_sc(&t_,&rs);
        jac A,B,M;
        scalar_mul_point(&A,&s_,&KC->Q_montX,&KC->Q_montY);
        scalar_mul_point(&B,&t_,&KC->P_montX,&KC->P_montY);
        jac_add(&M,&A,&B);
        if(fe_is_zero(&M.Z)){ j--; continue; }
        jac_to_affine(&M,&h_jx[j],&h_jy[j]);
        h_js[j]=s_; h_jt[j]=t_;
    }
#if RHO_ENDO
    /* v6 STAGE 2: mont-scalar copies (s*Rn mod n).  hK.scR2 is built by
     * endo_init(), which runs before build_jumptable in every mode. */
    for(int j=0;j<NJUMPS;j++){
        sc t; sc_mulred(&t,&h_js[j],&hK.scR2); h_jsm[j]=t;
        sc_mulred(&t,&h_jt[j],&hK.scR2); h_jtm[j]=t;
    }
#endif
}

/* re-randomize a walk from a fresh random start:  S = [a0]Q + [b0]P */
static HD void walk_reseed(fe *x,fe *y,sc *a,sc *b,uint64_t *rng){
    sc a0,b0; random_sc(&a0,rng); random_sc(&b0,rng);
    jac A,B,S;
    scalar_mul_point(&A,&a0,&KC->Q_montX,&KC->Q_montY);
    scalar_mul_point(&B,&b0,&KC->P_montX,&KC->P_montY);
    jac_add(&S,&A,&B);
    jac_to_affine(&S,x,y);
    sc_set(a,&a0); sc_set(b,&b0);
}

#if RHO_ENDO
/* ---- v6 STAGE 2: order-6 endomorphism canonicalization --------------------
 * E: y^2 = x^3 + 7 has the CM endomorphism zeta: (x,y) -> (beta*x, y) with
 * beta^3 = 1, beta != 1 in F_p, acting as [lambda] on the order-n subgroup
 * (lambda^2 + lambda + 1 = 0 mod n).  Together with negation this gives the
 * order-6 group {+-1, +-zeta, +-zeta^2}.  Every element maps R = [a]Q + [b]P
 * to a point in the same orbit class, so we walk CANONICAL orbit reps only:
 *   x*  = min(x, beta*x, beta^2*x) as canonical integers
 *         (beta^2 = -1 - beta, so beta*x + beta^2*x = -x in F_p)
 *   y*  = the y-sign whose Montgomery form is odd  (mont(-y) = p - mont(y))
 *   (a,b) <- mu*(a,b) mod n with mu in {+-1, +-lambda, +-lambda^2} the group
 *            element carrying R onto (x*,y*)   (zeta^k = [lambda^k])
 * All six group elements then land on the SAME state, which is what makes
 * distinct walks merge sqrt(6) sooner.  Walk scalars are held in the
 * Montgomery-scalar domain (v*Rn mod n) so mu-application is one sc_mulred;
 * resolve_candidate converts back to plain scalars with sc_mulred(.,1).
 * Cost: 3 mont_mul (beta*x, its canonical, beta^2*x's canonical) + ~1.67
 * sc_mulred average per step, plus 2-3 canonical compares.
 * Returns nothing; writes canonical x (mont), xc (canonical integer), y, a, b.
 * `one` = the plain constant 1 as a mont fe (mont_mul(.,one) = canonicalize). */
static HD void endo_canon(const fe *one,fe *x,fe *xc,fe *y,sc *a,sc *b){
    fe x2m,x3m,x2c,x3c,zero,t;
    mont_mul(&x2m,x,&KC->beta_m);          /* mont(beta*x) */
    fe_zero(&zero);
    fe_add(&t,x,&x2m);                     /* mont(x + beta*x) */
    fe_sub(&x3m,&zero,&t);                 /* mont(beta^2*x) = p - (x + beta*x) */
    mont_mul(xc,x,one);                    /* canonical(x) */
    mont_mul(&x2c,&x2m,one);               /* canonical(beta*x) */
    mont_mul(&x3c,&x3m,one);               /* canonical(beta^2*x) */
    const fe *wm=x,*wc=xc; int k=0;        /* k = winner among {1, zeta, zeta^2} */
    if(fe_lt(&x2c,wc)){ k=1; wm=&x2m; wc=&x2c; }
    if(fe_lt(&x3c,wc)){ k=2; wm=&x3m; wc=&x3c; }
    if(!(y->w[0]&1u)){                     /* y-sign: force odd Montgomery form */
        fe_sub(y,&KC->P_mod,y);            /* mont(-y) = p - mont(y); y != 0 */
        k+=3;
    }
    /* (a,b) <- mu*(a,b):  0:+1 1:+lam 2:+lam2 3:-1 4:-lam 5:-lam2 */
    if(k){
        sc tt;
        if(k==3){
            if(!sc_is_zero(a)){ sc_sub_n(&tt,N_as_sc(),a); sc_set(a,&tt); }
            if(!sc_is_zero(b)){ sc_sub_n(&tt,N_as_sc(),b); sc_set(b,&tt); }
        }else{
            const sc *mu = (k==1)?&KC->lam_m:(k==2)?&KC->lam2_m:(k==4)?&KC->nlam_m:&KC->nlam2_m;
            sc_mulred(&tt,a,mu); sc_set(a,&tt);
            sc_mulred(&tt,b,mu); sc_set(b,&tt);
        }
    }
    fe_set(x,wm);
    fe_set(xc,wc);
}

/* v6 STAGE 2: 64-bit hash of the canonical x for the fruitless-cycle window.
 * A true state repeat implies a hash repeat; the 2^-64-scale collision case
 * only costs one harmless extra reseed. */
static HD uint64_t cyc_hash(const fe *xc){
    uint64_t h=low64(xc);
    h^=h>>33; h*=0xff51afd7ed558ccdULL; h^=h>>33; h*=0xc4ceb9fe1a85ec53ULL; h^=h>>33;
    return h;
}
/* reseed + endo post-process (STAGE 2).  xc is the canonical-x walk state;
 * cycw/cyc_n/cyc_i are the per-lane fruitless-cycle history window (reset on
 * every reseed).  In non-endo builds this is exactly walk_reseed. */
static HD void walk_reseed_full(fe *x,fe *y,sc *a,sc *b,uint64_t *rng,fe *xc,
                                uint64_t *cycw,int *cyc_n,int *cyc_i){
    walk_reseed(x,y,a,b,rng);
#if RHO_ENDO
    { sc t; sc_mulred(&t,a,&KC->scR2); sc_set(a,&t);   /* plain a -> a*Rn mod n */
      sc_mulred(&t,b,&KC->scR2); sc_set(b,&t);
      fe fone; fe_zero(&fone); fone.w[0]=1;            /* plain 1: mont_mul(.,1) = /R */
      endo_canon(&fone,x,xc,y,a,b);
      *cyc_n=0; *cyc_i=0; }
#else
    (void)cycw; (void)cyc_n; (void)cyc_i;
#endif
}
/* ---- v6 STAGE 2 init: derive + verify the endomorphism constants ----------
 * Host-only, runs once after mont_init.  Everything is verified numerically
 * before the walk is allowed to start (mirrors endo_check.py):
 *   beta:    g^((p-1)/3) for small g until beta^3 = 1, beta != 1
 *   lambda:  Tonelli-Shanks sqrt(-3) mod n -> (sqrt-1)/2, root of
 *            x^2 + x + 1 = 0 mod n; eigenvalue picked so [lambda]Q = (beta*Qx, Qy)
 *   final:   for 8 random (a,b): canon([a]Q+[b]P) == ([a*]Q+[b*]P) with the
 *            canonical x match and on-curve check.  Any failure exits(1). */
static sc sc_one_static;   /* plain 1 as sc (host init use) */

static void fe_pow_mont(fe *r,const fe *base_m,const fe *e){
    fe result,t;
    fe_set(&result,&hK.R1_mont);
    int top=-1;
    for(int i=NL-1;i>=0&&top<0;i--) if(e->w[i]){ for(int b=31;b>=0;b--) if((e->w[i]>>b)&1){ top=i*32+b; break; } }
    for(int i=top;i>=0;i--){
        mont_mul(&result,&result,&result);
        if((e->w[i/32]>>(i%32))&1){ mont_mul(&t,&result,base_m); fe_set(&result,&t); }
    }
    fe_set(r,&result);
}
/* (p-1)/3 by long division; p-1 is divisible by 3 for every rung
 * (2^(2k) = 1 mod 3 and C = 1119..7245 = 0 mod 3). */
static void fe_div3(fe *r,const fe *a){
    uint32_t rem=0;
    for(int i=NL-1;i>=0;i--){
        uint64_t cur=((uint64_t)rem<<32)|a->w[i];
        r->w[i]=(uint32_t)(cur/3);
        rem=(uint32_t)(cur%3);
    }
}
static int sc_is_one(const sc *a){
    if(a->w[0]!=1) return 0;
    for(int i=1;i<NL;i++) if(a->w[i]) return 0;
    return 1;
}
static void sc_shr1(sc *r,const uint32_t *a){
    uint32_t carry=0;
    for(int i=NL-1;i>=0;i--){ uint32_t nc=a[i]&1u; r->w[i]=(a[i]>>1)|(carry<<31); carry=nc; }
}
/* Tonelli-Shanks sqrt mod n (n prime, host-only).  Returns 0 on failure. */
static int sc_ts_sqrt(sc *r,const sc *a){
    sc q; sc_sub_n(&q,N_as_sc(),&sc_one_static);   /* q = n-1 */
    int s=0;
    while(!(q.w[0]&1u)){ sc_shr1(&q,q.w); s++; }
    /* z: smallest quadratic non-residue (Legendre z^((n-1)/2) = n-1) */
    sc z,pm1_half,zn;
    sc_shr1(&pm1_half,hK.N_mod.w);       /* n>>1 = (n-1)/2, n odd */
    int found=0;
    for(uint32_t v=2;v<256&&!found;v++){
        sc_zero(&z); z.w[0]=v;
        sc_pow(&zn,&z,&pm1_half);
        int isneg1=1;
        for(int i=0;i<NL;i++){ uint32_t want=(i==0)?(hK.N_mod.w[0]-1):hK.N_mod.w[i];
            if(zn.w[i]!=want){ isneg1=0; break; } }
        if(isneg1) found=1;
    }
    if(!found) return 0;
    /* m=s; c=z^q; t=a^q; r=a^((q+1)/2); q is odd so (q+1)/2 = (q>>1)+1 */
    sc c,t,rr,bb,e2;
    sc_pow(&c,&z,&q);
    sc_pow(&t,a,&q);
    sc_shr1(&e2,q.w);
    sc_add_n(&e2,&e2,&sc_one_static);
    sc_pow(&rr,a,&e2);
    int m=s;
    while(!sc_is_one(&t)){
        int i=0; sc tt; sc_set(&tt,&t);
        while(!sc_is_one(&tt)&&i<m){ sc_mul(&tt,&tt,&tt); i++; }
        if(i>=m) return 0;
        sc_set(&bb,&c);
        for(int k=0;k<m-i-1;k++) sc_mul(&bb,&bb,&bb);
        m=i;
        sc_mul(&c,&bb,&bb);
        sc_mul(&t,&t,&c);
        sc_mul(&rr,&rr,&bb);
    }
    sc_set(r,&rr);
    return 1;
}

static void endo_init(void){
    /* --- ninv32 = -n^-1 mod 2^32 (Newton) --- */
    uint32_t n0=hK.N_mod.w[0], inv=1;
    for(int i=0;i<6;i++) inv*=2u-inv*n0;
    hK.ninv32=0u-inv;
    if(n0*hK.ninv32!=0xFFFFFFFFu){ fprintf(stderr,"endo: ninv32 check failed\n"); exit(1); }
    sc_zero(&sc_one_static); sc_one_static.w[0]=1;

    /* --- beta = nontrivial cube root of unity in F_p --- */
    fe pm1,e3;
    fe_set(&pm1,&hK.P_mod);
    { fe one; fe_zero(&one); one.w[0]=1; fe_sub(&pm1,&pm1,&one); }      /* p-1 */
    fe_div3(&e3,&pm1);
    int gotb=0;
    for(uint32_t v=2;v<256&&!gotb;v++){
        fe base,bm,b2,b3;
        fe_zero(&base); base.w[0]=v;
        mont_mul(&bm,&base,&hK.R2_mont);
        fe_pow_mont(&bm,&bm,&e3);
        if(fe_eq(&bm,&hK.R1_mont)) continue;               /* g^((p-1)/3) == 1 */
        mont_mul(&b2,&bm,&bm); mont_mul(&b3,&b2,&bm);
        if(fe_eq(&b3,&hK.R1_mont)){ fe_set(&hK.beta_m,&bm); gotb=1; }
    }
    if(!gotb){ fprintf(stderr,"endo: no nontrivial cube root of unity in F_p\n"); exit(1); }

    /* --- lambda: roots of x^2+x+1 = 0 mod n are (sqrt(-3)-1)/2 and its neg-1 --- */
    sc m3,sq,lam,lamalt,eig,lam2;
    { sc three; sc_zero(&three); three.w[0]=3;
      sc_sub_n(&m3,N_as_sc(),&three); }                    /* -3 mod n */
    if(!sc_ts_sqrt(&sq,&m3)){ fprintf(stderr,"endo: no sqrt(-3) mod n\n"); exit(1); }
    { sc chk; sc_mul(&chk,&sq,&sq);
      if(memcmp(chk.w,m3.w,sizeof(sc))){ fprintf(stderr,"endo: sqrt(-3) check failed\n"); exit(1); } }
    sc half; sc_shr1(&half,hK.N_mod.w);                     /* (n-1)/2 */
    sc_add_n(&half,&half,&sc_one_static);                  /* (n+1)/2 = inv2 */
    sc t1;
    sc_sub_n(&t1,&sq,&sc_one_static);                      /* sqrt-1 */
    sc_mul(&lam,&t1,&half);
    sc_sub_n(&t1,N_as_sc(),&sq);                           /* n-sqrt */
    sc_sub_n(&t1,&t1,&sc_one_static);                      /* n-sqrt-1 */
    sc_mul(&lamalt,&t1,&half);
    /* both must satisfy lam^2+lam+1 = 0 mod n */
    for(int which=0;which<2;which++){
        sc lm = which?lamalt:lam, l2, s1;
        sc_mul(&l2,&lm,&lm);
        sc_add_n(&s1,&l2,&lm); sc_add_n(&s1,&s1,&sc_one_static);
        if(!sc_is_zero(&s1)){ fprintf(stderr,"endo: lambda root check failed (%d)\n",which); exit(1); }
    }

    /* --- eigenvalue: [lam]Q == (beta*Qx, Qy) picks lam vs lam^2 --- */
    fe tx,ty,rx,ry;
    mont_mul(&tx,&hK.Q_montX,&hK.beta_m);                  /* mont(beta*Qx) */
    ty=hK.Q_montY;
    jac R;
    scalar_mul_point(&R,&lam,&hK.Q_montX,&hK.Q_montY);
    if(fe_is_zero(&R.Z)){ fprintf(stderr,"endo: [lam]Q = O\n"); exit(1); }
    jac_to_affine(&R,&rx,&ry);
    if(fe_eq(&rx,&tx)&&fe_eq(&ry,&ty)) eig=lam;
    else{
        scalar_mul_point(&R,&lamalt,&hK.Q_montX,&hK.Q_montY);
        if(fe_is_zero(&R.Z)){ fprintf(stderr,"endo: [lam^2]Q = O\n"); exit(1); }
        jac_to_affine(&R,&rx,&ry);
        if(!(fe_eq(&rx,&tx)&&fe_eq(&ry,&ty))){ fprintf(stderr,"endo: no eigenvalue match on Q\n"); exit(1); }
        eig=lamalt;
    }
    /* [eig]P == (beta*Px, Py) */
    mont_mul(&tx,&hK.P_montX,&hK.beta_m);
    ty=hK.P_montY;
    scalar_mul_point(&R,&eig,&hK.P_montX,&hK.P_montY);
    jac_to_affine(&R,&rx,&ry);
    if(!(fe_eq(&rx,&tx)&&fe_eq(&ry,&ty))){ fprintf(stderr,"endo: eigenvalue mismatch on P\n"); exit(1); }
    sc_mul(&lam2,&eig,&eig);                               /* lam^2 = the other root */
    { sc e2,e3s;                                           /* lam^3 == 1 mod n */
      sc_mul(&e2,&eig,&eig); sc_mul(&e3s,&e2,&eig);
      if(!sc_is_one(&e3s)){ fprintf(stderr,"endo: lam^3 != 1\n"); exit(1); } }

    /* --- Rn mod n and Rn^2 mod n, mont-scalar constants --- */
    sc scR1; sc_zero(&scR1); scR1.w[0]=1;
    for(int i=0;i<32*NL;i++){ sc t; sc_add_n(&t,&scR1,&scR1); sc_set(&scR1,&t); }
    sc_mul(&hK.scR2,&scR1,&scR1);
    sc_mulred(&hK.lam_m,&eig,&hK.scR2);                    /* eig*Rn mod n */
    sc_mulred(&hK.lam2_m,&lam2,&hK.scR2);
    sc_sub_n(&hK.nlam_m,N_as_sc(),&hK.lam_m);
    sc_sub_n(&hK.nlam2_m,N_as_sc(),&hK.lam2_m);
    /* mont-scalar round trips */
    { sc chk; sc_mulred(&chk,&hK.lam_m,&sc_one_static);
      if(memcmp(chk.w,eig.w,sizeof(sc))){ fprintf(stderr,"endo: lam_m round trip failed\n"); exit(1); }
      sc_mulred(&chk,&hK.lam2_m,&sc_one_static);
      if(memcmp(chk.w,lam2.w,sizeof(sc))){ fprintf(stderr,"endo: lam2_m round trip failed\n"); exit(1); } }

    /* --- random-point canonicalization self-check (8 trials) --- */
    { uint64_t rs=0xE6D0BEEF1234ULL;
      fe fone; fe_zero(&fone); fone.w[0]=1;
      for(int trial=0;trial<8;trial++){
        sc a0,b0,am,bm,ap,bp; random_sc(&a0,&rs); random_sc(&b0,&rs);
        jac A,B,S; scalar_mul_point(&A,&a0,&hK.Q_montX,&hK.Q_montY);
        scalar_mul_point(&B,&b0,&hK.P_montX,&hK.P_montY);
        jac_add(&S,&A,&B);
        if(fe_is_zero(&S.Z)){ trial--; continue; }
        fe x,y,xc; jac_to_affine(&S,&x,&y);
        sc_set(&am,&a0); sc_set(&bm,&b0);
        { sc t; sc_mulred(&t,&am,&hK.scR2); sc_set(&am,&t);
          sc_mulred(&t,&bm,&hK.scR2); sc_set(&bm,&t); }
        endo_canon(&fone,&x,&xc,&y,&am,&bm);
        if(!on_curve(&x,&y)){ fprintf(stderr,"endo: canonical point off curve (trial %d)\n",trial); exit(1); }
        /* plain scalars back out of the mont-scalar domain */
        sc_mulred(&ap,&am,&sc_one_static);
        sc_mulred(&bp,&bm,&sc_one_static);
        scalar_mul_point(&A,&ap,&hK.Q_montX,&hK.Q_montY);
        scalar_mul_point(&B,&bp,&hK.P_montX,&hK.P_montY);
        jac_add(&S,&A,&B);
        if(fe_is_zero(&S.Z)){ fprintf(stderr,"endo: canonical scalars gave O (trial %d)\n",trial); exit(1); }
        jac_to_affine(&S,&rx,&ry);
        /* rx,ry are mont; x,y from endo_canon are mont; xc is canonical */
        mont_mul(&tx,&rx,&fone);
        if(!fe_eq(&tx,&xc)||!fe_eq(&ry,&y)){
            fprintf(stderr,"endo: canon([a]Q+[b]P) != [a*]Q+[b*]P (trial %d)\n",trial); exit(1); }
        /* mu must be applied: at least one of a,b changed in 5 of 6 cases --
           verify scalars stayed canonical */
        if(sc_ge_n(&am)||sc_ge_n(&bm)){ fprintf(stderr,"endo: mont-scalar not canonical (trial %d)\n",trial); exit(1); }
      } }
    fprintf(stderr,"endo: constants verified (beta, lambda, eigenvalue, 8 canon trials)\n");
}
#endif /* RHO_ENDO */

/* ================================================================== */
/* Distinguished points: sharded hash table + collision candidates     */
/* ================================================================== */

#define NCAND  8192          /* candidate ring size (records) */
#define NPROBE 64            /* probes per 64-slot shard (rho.c semantics)  */

typedef struct {
    uint64_t key;            /* low 64 bits of mont affine x; 0 = empty */
    uint64_t xhi;            /* remaining high bits of x (0 when NL==2) */
    sc a, b;
    uint32_t ypar;
} DPSlot;

typedef struct {
    sc a1, b1, a2, b2;
    uint32_t ypar1, ypar2;
} CandRec;

#define DP_INS     0        /* inserted — keep walking                    */
#define DP_RESEED  1        /* merged/degenerate walk — restart          */
#define DP_CAND    2        /* real collision candidate — report, restart */
#define DP_DROP    3        /* shard full, DP dropped (rho.c behaviour)   */
#define DP_DUP     4        /* multi-worker: record already known (no-op) */

static HD FI uint64_t hashx(uint64_t x){ return x*GOLDEN; }

/* ---- RHO_DP_EMIT (STAGE 2b) host-side helpers ------------------------------
 * print_hex_min: minimal lowercase hex of an NL-limb little-endian value
 * (limb 0 is the LSB), matching Python's format(v,"x") in stub_walker.py.
 * dp_emit_line: one JSON line per persisted DP record; mont -> plain here. */
static void print_hex_min(const uint32_t *w,int nlimbs){
    int k=nlimbs-1; while(k>0 && w[k]==0) k--;
    if(k==0 && w[0]==0){ printf("0"); return; }
    printf("%x",w[k]);
    for(int i=k-1;i>=0;i--) printf("%08x",w[i]);
}
static void dp_emit_line(unsigned rung,const fe *x,const fe *y,const sc *a,const sc *b){
    fe one,px,py; sc pa,pb,one_s;
    fe_zero(&one); one.w[0]=1;                 /* plain 1: mont_mul(.,1) = /R */
    memset(&one_s,0,sizeof(one_s)); one_s.w[0]=1;   /* same plain-1 trick */
    mont_mul(&px,x,&one); mont_mul(&py,y,&one);
#if RHO_ENDO
    sc_mulred(&pa,a,&one_s); sc_mulred(&pb,b,&one_s); /* mont-scalar -> plain */
#else
    pa=*a; pb=*b;                                   /* plain-scalar builds  */
#endif
    printf("{\"dp\":1,\"rung\":%u,\"x\":\"",rung); print_hex_min(px.w,NL);
    printf("\",\"y\":\"");                    print_hex_min(py.w,NL);
    printf("\",\"a\":\"");                     print_hex_min(pa.w,NL);
    printf("\",\"b\":\"");                     print_hex_min(pb.w,NL);
    printf("\"}\n");
    fflush(stdout);
}
/* ordered host log of emitted records (checkpoint v3 payload) */
static DPEmitRec *g_emit_log=NULL;
static uint64_t   g_emit_n=0, g_emit_cap=0;
static void emit_log_append(const DPEmitRec *r){
    if(g_emit_n==g_emit_cap){
        g_emit_cap=g_emit_cap?g_emit_cap*2:1024;
        g_emit_log=(DPEmitRec*)realloc(g_emit_log,g_emit_cap*sizeof(DPEmitRec));
        if(!g_emit_log){ fprintf(stderr,"dp-emit: host log alloc failed\n"); exit(1); }
    }
    g_emit_log[g_emit_n++]=*r;
}

static HD FI uint64_t hi64(const fe *x){ return (NL>2)?((uint64_t)x->w[2]|((uint64_t)x->w[3]<<32)):0ULL; }

/*
 * One DP event.  Mirrors rho.c's shard probing; on the device the slot key is
 * manipulated with atomics (claim -> payload -> publish).  xhi is an extra
 * guard rho.c does not have: it rejects "same low-64 x, different point"
 * false positives when the Montgomery x is wider than 64 bits.
 */
static HD int dp_check(const Cfg *C, DPSlot *tbl, const fe *x, const fe *y,
                       const sc *a, const sc *b, CandRec *out)
{
    uint64_t xk  = low64(x);
    uint64_t xhi = hi64(x);
    uint32_t ypar = y->w[0] & 1u;
    uint64_t h = hashx(xk >> C->dpbits);
    uint64_t base = (h & C->tblmask) & ~63ULL;
    int probe=0;
    long guard=0;   /* total slot-visit bound (RSV spins can revisit one slot) */
    while(probe<NPROBE){
        if(++guard>64L*NPROBE) return DP_DROP;   /* pathological: bounded */
        uint64_t slot = base + (uint64_t)((h+(uint64_t)probe)&63);
        DPSlot *s = &tbl[slot];
#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
        /* v6 STAGE 0a fix: RSV64 is always transient (claim -> payload ->
         * publish is straight-line, no early exit in between), so WAIT for the
         * publish instead of probing past it.  Two walks of a merged pair hit
         * a shared DP in the same round; the loser of the claim race used to
         * probe on and insert a duplicate record elsewhere, so the pair never
         * compared records and detection waited for ~1/p later shared DPs
         * (p = publish window / thread-scheduling skew ~ 2e-4 measured via the
         * dpb=14 detection delay: 91.7M vs 8.5M expected steps).  The delay
         * scales with dpbits and was the dpb>=9 near-non-termination hazard.
         * Losing the insert CAS now also re-evaluates the SAME slot (the
         * winner's record may be our own collision partner). */
        unsigned long long k = atomicAdd((unsigned long long*)&s->key, 0ULL);
        if(k==RSV64){
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
            __nanosleep(32);
#endif
            continue;                     /* wait for the publish, same slot */
        }
#else
        uint64_t k = s->key;
#endif
        if(k==0){
#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
            unsigned long long prev = atomicCAS((unsigned long long*)&s->key, 0ULL, RSV64);
            if(prev!=0) continue;         /* lost the claim race: re-evaluate */
            s->xhi=xhi; s->ypar=ypar; s->a=*a; s->b=*b;
            __threadfence();
            atomicExch((unsigned long long*)&s->key, xk);
            return DP_INS;
#else
            s->xhi=xhi; s->ypar=ypar; s->a=*a; s->b=*b;
            s->key=xk;
            return DP_INS;
#endif
        }else if(k==xk){
#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
            __threadfence();
#endif
            if(s->xhi!=xhi){ probe++; continue; } /* different point, same low-64 x */
            if(s->ypar==ypar){
                int asame=1,bsame=1;
                for(int i=0;i<NL;i++){ if(s->a.w[i]!=a->w[i]) asame=0; if(s->b.w[i]!=b->w[i]) bsame=0; }
                if(asame&&bsame) return DP_RESEED;         /* walk merged */
                if(!bsame){
                    out->a1=*a; out->b1=*b; out->a2=s->a; out->b2=s->b;
                    out->ypar1=ypar; out->ypar2=s->ypar;
                    return DP_CAND;
                }
                return DP_RESEED;                          /* degenerate  */
            }else{
                out->a1=*a; out->b1=*b; out->a2=s->a; out->b2=s->b;
                out->ypar1=ypar; out->ypar2=s->ypar;
                return DP_CAND;                            /* negation case */
            }
        }else{
            probe++;      /* slot holds a different point: next probe */
        }
    }
    return DP_DROP;  /* shard full: drop (rho.c behaviour) */
}

/* Host-side collision resolution (identical math to rho.c).  Returns 1 and
 * sets *kap to the kappa candidate that should be verified. */
static int resolve_candidate(const sc *a1,const sc *b1,const sc *a2,const sc *b2,
                             int same_pt, sc *kap)
{
    sc one; sc_zero(&one); one.w[0]=1;
#if RHO_ENDO
    /* v6 STAGE 2: candidate scalars are mont-domain (v*Rn mod n); convert to
     * plain (sc_mulred by 1 divides out Rn) before the recovery math. */
    sc pa1,pb1,pa2,pb2;
    sc_mulred(&pa1,a1,&one); sc_mulred(&pb1,b1,&one);
    sc_mulred(&pa2,a2,&one); sc_mulred(&pb2,b2,&one);
    a1=&pa1; b1=&pb1; a2=&pa2; b2=&pb2;
#endif
    if(same_pt){
        if(sc_eq(a1,a2)&&sc_eq(b1,b2)) return 0;
        if(sc_eq(b1,b2)) return 0;
        sc da,db,nm1,nm2,invdb;
        sc_sub_n(&da,a1,a2);
        sc_sub_n(&db,b2,b1);
        sc_sub_n(&nm1,N_as_sc(),&one);
        sc_sub_n(&nm2,&nm1,&one);
        sc_pow(&invdb,&db,&nm2);
        sc_mul(kap,&da,&invdb);
        return 1;
    }else{
        sc s,t,invt,nm1,nm2,neg;
        sc_add_n(&s,a1,a2);
        sc_add_n(&t,b1,b2);
        if(sc_is_zero(&t)||sc_is_zero(&s)) return 0;
        sc_sub_n(&nm1,N_as_sc(),&one);
        sc_sub_n(&nm2,&nm1,&one);
        sc_pow(&invt,&t,&nm2);
        sc_mul(kap,&s,&invt);
        sc_sub_n(&neg,N_as_sc(),kap);
        *kap=neg;
        return 1;
    }
}

/* ================================================================== */
/* Walk step (shared by CUDA kernel and CPU_REF lanes)                 */
/* ================================================================== */

/* One r-adding walk step.  Inputs: current affine point (x,y) [mont] and
 * scalars (a,b).  Outputs the pending Jacobian point (Xo,Yo,Zo) whose
 * affine coordinates are recovered by the caller's batch inversion.
 * Sets *reseed_pending when the walk must be restarted before the next step. */
static HD void walk_step(fe *x, fe *y, sc *a, sc *b,
                         fe *Xo, fe *Yo, fe *Zo, int *reseed_pending, uint64_t *rng,
                         int esc)
{
    if(*reseed_pending){
        walk_reseed(x,y,a,b,rng);
        *reseed_pending=0;
        fe_set(Xo,x); fe_set(Yo,y); fe_set(Zo,&KC->R1_mont);
        return;
    }
    /* v6 STAGE 2b: esc=1 breaks fruitless 2-cycles: the cycle window fired on
     * the previous state, so this step takes the NEXT jump instead of the one
     * the state maps to.  Deterministic in (x, esc) => merge-safe. */
    int j = (int)(((low64(x) >> 8) + esc) & (NJUMPS-1));
    jac P1,R;
    fe_set(&P1.X,x); fe_set(&P1.Y,y); fe_set(&P1.Z,&KC->R1_mont);
    jac_madd(&R,&P1,&JX[j],&JY[j]);
    if(fe_is_zero(&R.Z)){
        *reseed_pending=1;
        fe_set(Xo,x); fe_set(Yo,y); fe_set(Zo,&KC->R1_mont);
        return;
    }
    sc_add_n(a,a,&JS[j]);
    sc_add_n(b,b,&JT[j]);
    fe_set(Xo,&R.X); fe_set(Yo,&R.Y); fe_set(Zo,&R.Z);
}

/* convert pending (X,Y,zi) to affine (x,y) [mont] */
static HD FI void pending_to_affine(const fe *X,const fe *Y,const fe *zi,fe *x,fe *y){
    fe z2,z3;
    mont_sqr(&z2,zi);
    mont_mul(&z3,&z2,zi);
    mont_mul(x,X,&z2);
    mont_mul(y,Y,&z3);
}

/* ---- SASS-PORT patch: affine r-adding walk --------------------------------
 * Same walk as walk_step (bit-identical point/scalar stream — the affine
 * formulas produce the same canonical field elements the Jacobian path
 * recovers), but the point add is done directly on affine coordinates:
 *   lambda = (jy-y)/(jx-x),  x' = lambda^2 - x - jx,  y' = lambda*(x-x') - y
 * which costs 3 mont_muls instead of jac_madd's 10 + pending_to_affine's 4.
 * The (jx-x) denominators of the whole warp/lane-group are inverted by the
 * SAME batch inversion the Jacobian path already amortizes.  Split into a
 * prep phase (jump select + denominators, before the batch inversion) and a
 * finish phase (after it), exactly like the old X,Y,Z + pending_to_affine.
 *
 * walk_prep return codes (per lane/walk):
 *   0 normal          — contrib = dx = jx-x; caller batch-inverts, then calls
 *                       walk_finish to produce the new (x,y,a,b).
 *   1 reseeded now    — walk_reseed already ran; (x,y,a,b) are final; caller
 *                       still must give the lane a nonzero batch contribution
 *                       (R1) but must NOT call walk_finish.  The fresh point
 *                       IS DP-checked (old behaviour).
 *   2 infinity        — point + jump = O; reseed_pending set; (x,y) untouched;
 *                       no walk_finish, no DP check this round (old behaviour).
 * Degenerate dx==0: point == jump.  dy==0 -> point doubling, contrib = 2y
 * (zi = inv(2y), lambda = 3x^2*zi); dy!=0 -> infinity (case 2).  contrib is
 * never zero, so the batch product never vanishes. */
#ifndef RHO_AFFINE
#define RHO_AFFINE 1
#endif
#if RHO_ENDO && !RHO_AFFINE
#error "RHO_ENDO (v6 STAGE 2) requires RHO_AFFINE=1"
#endif
#ifndef RHO_SM_JT
#define RHO_SM_JT 1
#endif
static HD int walk_prep(const fe *jxarr,const fe *jyarr,
                        fe *x,fe *y,sc *a,sc *b,int *reseed_pending,uint64_t *rng,
                        int *jout,fe *dx,fe *dy,fe *contrib,int *isdbl,int esc)
{
    if(*reseed_pending){
        walk_reseed(x,y,a,b,rng);
        *reseed_pending=0;
        fe_set(contrib,&KC->R1_mont);
        return 1;
    }
    /* v6 STAGE 2b: esc=1 breaks fruitless 2-cycles: the cycle window fired on
     * the previous state, so this step takes the NEXT jump instead of the one
     * the state maps to.  Deterministic in (x, esc) => merge-safe: two walks
     * retracing the same path hold the same 8-hash window, so the escape fires
     * (and is chosen) identically for both.  A double-negation + same-jump
     * 2-cycle of the canonical map otherwise kills the walk every ~100-500
     * steps; merged walks retracing a dead walk's path then die in that dead
     * walk's cycle before reaching a stored DP (detection delay ~2^(dpb/2),
     * measured 24x at dpb=14).  Escaping keeps walks alive so merges survive
     * to their shared DP. */
    int j=(int)(((low64(x)>>8)+esc)&(NJUMPS-1));
    *jout=j;
    fe_sub(dx,&jxarr[j],x);
    fe_sub(dy,&jyarr[j],y);
    if(fe_is_zero(dx)){
        if(fe_is_zero(dy)){
            *isdbl=1;
            fe_add(contrib,y,y);          /* 2y: invert it for the doubling lambda */
            return 0;
        }
        *reseed_pending=1;                /* point == -jump: walk dies */
        fe_set(contrib,&KC->R1_mont);
        return 2;
    }
    *isdbl=0;
    fe_set(contrib,dx);
    return 0;
}
static HD FI void walk_finish(fe *x,fe *y,sc *a,sc *b,int j,const fe *jxarr,
                              const fe *dy,int isdbl,const fe *zi)
{
    fe lam,l2,t,ox,oy;
    ox=*x; oy=*y;
    if(isdbl){
        fe s;
        mont_sqr(&s,&ox);
        fe_add(&lam,&s,&s); fe_add(&lam,&lam,&s);   /* 3x^2 */
        mont_mul(&lam,&lam,zi);                     /* lambda = 3x^2/(2y) */
    }else{
        mont_mul(&lam,dy,zi);                       /* lambda = dy/dx */
    }
    mont_sqr(&l2,&lam);
    fe_sub(&t,&l2,&ox);                             /* lam^2 - x */
    if(isdbl) fe_sub(x,&t,&ox);                     /* x' = lam^2 - 2x   */
    else      fe_sub(x,&t,&jxarr[j]);               /* x' = lam^2 - x - jx */
    fe_sub(&t,&ox,x);                               /* x - x' */
    mont_mul(&t,&t,&lam);
    fe_sub(y,&t,&oy);                               /* y' = lam*(x-x') - y */
    sc_add_n(a,a,&JS[j]);
    sc_add_n(b,b,&JT[j]);
}


#ifdef RHO_MULTI
/* ================================================================== */
/* Sync insert: put a PULLED record into a worker's local table        */
/* ================================================================== */
/* Mirrors dp_check's probe/claim protocol (incl. the RSV64 in-flight-claim
 * handling).  Never overwrites an existing entry; identical records (a
 * worker's own event echoed back through the log) are no-ops; a
 * different-scalar match is a collision — on the host we resolve it on the
 * spot, on the device we leave it to the host (both halves are in the shared
 * log already).  Lives here (not in rho_multi.cu) because the walk kernel's
 * inbox-drain block calls it on the device. */
static HD int dp_sync_insert(const Cfg *C, DPSlot *tbl, const DPRec *r, DPRec *stored)
{
    uint64_t h=hashx(r->key >> C->dpbits);
    uint64_t base=(h & C->tblmask) & ~63ULL;
    for(int probe=0;probe<NPROBE;probe++){
        uint64_t slot=base+(uint64_t)((h+(uint64_t)probe)&63);
        DPSlot *s=&tbl[slot];
#if defined(__CUDA_ARCH__)
        unsigned long long k=atomicAdd((unsigned long long*)&s->key,0ULL);
#else
        uint64_t k=s->key;
#endif
        if(k==0){
#if defined(__CUDA_ARCH__)
            unsigned long long prev=atomicCAS((unsigned long long*)&s->key,0ULL,RSV64);
            if(prev!=0) continue;              /* lost the claim race */
            s->xhi=r->xhi; s->ypar=r->ypar; s->a=r->a; s->b=r->b;
            __threadfence();
            atomicExch((unsigned long long*)&s->key,r->key);
#else
            s->xhi=r->xhi; s->ypar=r->ypar; s->a=r->a; s->b=r->b; s->key=r->key;
#endif
            return DP_INS;
        }
        if(k==r->key){
#if defined(__CUDA_ARCH__)
            __threadfence();
#endif
            if(s->xhi!=r->xhi) return DP_DUP;
            if(s->ypar==r->ypar){
                int asame=1,bsame=1;
                for(int i=0;i<NL;i++){ if(s->a.w[i]!=r->a.w[i]) asame=0; if(s->b.w[i]!=r->b.w[i]) bsame=0; }
                if(asame&&bsame) return DP_DUP;    /* own record echoed back */
            }
            if(stored){ stored->key=k; stored->xhi=s->xhi; stored->ypar=s->ypar; stored->a=s->a; stored->b=s->b; }
            return DP_CAND;
        }
        /* k==RSV64 (an in-flight claim by the walk kernel or another sync
         * batch): skip this slot; the record will be retried on a later pull
         * if it matters — the shared table still holds it either way. */
    }
    return DP_DROP;   /* local shard full: the shared table still has it */
}
#endif /* RHO_MULTI */

#if defined(__CUDACC__) && !defined(CPU_REF)
/* ================================================================== */
/* CUDA kernel: one walk per thread, warp-wide batch inversion         */
/* ================================================================== */

#define ALLL 0xffffffffu

DI volatile int g_found_flag = 0;   /* host sets via cudaMemcpyToSymbol */
DI volatile unsigned g_cand_pend = 0; /* v6 STAGE 0a: a candidate is parked on */
DI unsigned g_dphits = 0;
DI unsigned g_dpins  = 0;
DI unsigned g_badcurve = 0;         /* v6 STAGE 0b: off-curve DP events seen */
#ifdef RHO_MULTI
DI unsigned g_drain_n = 0;          /* RHO_MULTI debug: records drained     */
#endif
#ifdef RHO_INVTEST
/* in-kernel batch-inversion verifier: mont_mul(invt,total) must equal R1_mont.
 * Mismatches are counted and the first 64 are recorded for host dump. */
DI unsigned g_invmis = 0;
#define INVBAD_N 64
#define INVBAD_W (2*NL+3)
DI uint32_t g_invbad[INVBAD_N*INVBAD_W];
#endif

DI FI fe shfl_fe(const fe &v,int src){ fe r; for(int i=0;i<NL;i++) r.w[i]=__shfl_sync(ALLL,v.w[i],src); return r; }
DI FI fe shfl_fe_up(const fe &v,int d){ fe r; for(int i=0;i<NL;i++) r.w[i]=__shfl_up_sync(ALLL,v.w[i],d); return r; }
DI FI fe shfl_fe_down(const fe &v,int d){ fe r; for(int i=0;i<NL;i++) r.w[i]=__shfl_down_sync(ALLL,v.w[i],d); return r; }

__global__ __launch_bounds__(256,1) void rho_kernel(const Cfg C, DPSlot *tbl, CandRec *cands,
                                                  unsigned *cand_head,
                                                  unsigned long long *stepctr)
{
    const int tid  = blockIdx.x*blockDim.x + threadIdx.x;
    const int lane = threadIdx.x & 31;
    uint64_t rng = C.seed ^ (0xABCDEF1234567890ULL*(uint64_t)(tid+1)) ^ GOLDEN;
    fe x,y; sc a,b;
    uint64_t steps=0; int reseed_pending=0;
#if RHO_ENDO
    /* v6 STAGE 2: canonical-x state for the quotient walk + fruitless-cycle
     * history window: a canonical state repeating ANY of the last RHO_CYCWIN
     * states sets cycle_pending (same-jump runs close structurally under the
     * automorphism action; see the RHO_ENDO header note). */
    fe xc;
    uint64_t cycw[RHO_CYCWIN];
    int cyc_n=0, cyc_i=0, cyc_def=0;
    int cycle_pending=0, dp_ev=0, esc_pending=0;
    fe fone; fe_zero(&fone); fone.w[0]=1;   /* plain 1: mont_mul(.,1) = /R */
    walk_reseed_full(&x,&y,&a,&b,&rng,&xc,cycw,&cyc_n,&cyc_i);
#else
    walk_reseed(&x,&y,&a,&b,&rng);
#endif
#if RHO_SM_JT
    /* SASS-PORT: jump table staged in shared memory once per block (1 KB);
     * the 32-lane jump gather then hits shared, not L2-cached global. */
    __shared__ fe s_jx[NJUMPS], s_jy[NJUMPS];
    for(int i=threadIdx.x;i<NJUMPS;i+=blockDim.x){ s_jx[i]=c_jx[i]; s_jy[i]=c_jy[i]; }
    __syncthreads();
    const fe *JXs=s_jx, *JYs=s_jy;
#else
    const fe *JXs=JX, *JYs=JY;
#endif
    for(uint64_t round=0;;round++){
        steps++;
#if RHO_AFFINE
        /* ---- SASS-PORT affine walk: 3 mont_muls/step instead of 14 ---- */
        fe dx,dy,contrib; int j=0,isdbl=0;
        int esc=esc_pending; esc_pending=0;
        int act=walk_prep(JXs,JYs,&x,&y,&a,&b,&reseed_pending,&rng,&j,&dx,&dy,&contrib,&isdbl,esc);
        /* ---- warp-wide batch inversion (Montgomery's trick via 2 shfl scans) ----
         * pref = prod_{j<lane} Z_j ; suff = prod_{j>lane} Z_j ; total = prod_all
         * zi = pref * suff * total^-1 = Z_lane^-1   (all values Montgomery form) */
        fe incl=contrib, suff=contrib, t, total, invt, pref, zi;
        for(int off=1;off<32;off<<=1){                 /* inclusive prefix scan */
            t = shfl_fe_up(incl,off);
            if(lane>=off) mont_mul(&incl,&incl,&t);
        }
        total=shfl_fe(incl,31);
        fe_inv(&invt,&total);          /* all lanes, same value: divergence-free */
#ifdef RHO_INVTEST
        { fe chkv; mont_mul(&chkv,&invt,&total);
          unsigned bad=0; for(int i=0;i<NL;i++) bad|=(unsigned)(chkv.w[i]!=KC->R1_mont.w[i]);
          if(bad){ unsigned k=atomicAdd(&g_invmis,1u);
            if(k<INVBAD_N){ for(int i=0;i<NL;i++){ g_invbad[k*INVBAD_W+i]=total.w[i];
                                                   g_invbad[k*INVBAD_W+NL+i]=invt.w[i]; }
                             g_invbad[k*INVBAD_W+2*NL+0]=(unsigned)blockIdx.x;
                             g_invbad[k*INVBAD_W+2*NL+1]=(unsigned)(threadIdx.x>>5);
                             g_invbad[k*INVBAD_W+2*NL+2]=((unsigned)round)&0xFFFFFFu; } } }
#endif
        for(int off=1;off<32;off<<=1){                 /* inclusive suffix scan */
            t = shfl_fe_down(suff,off);
            if(lane+off<32) mont_mul(&suff,&suff,&t);
        }
        /* excl prefix/suffix.  NOTE: do NOT write these as
         *   pref = (lane==0) ? KC->R1_mont : shfl_fe_up(incl,1);
         * nvcc (sm_120, CUDA 13.0) miscompiles a shfl result selected against
         * a __constant__ load in a ternary: the selected fe comes out with
         * limb-wise garbage (observed: correct middle limb, own-lane top limb,
         * constant 0x7ffffffe low limb), which zeroes/garbles zi for the whole
         * warp and poisons every walk.  Shfl into a temporary, then override
         * the edge lane — bit-identical math, no miscompile. */
        pref = shfl_fe_up(incl,1);
        if(lane==0) fe_set(&pref,&KC->R1_mont);
        suff = shfl_fe_down(suff,1);
        if(lane==31) fe_set(&suff,&KC->R1_mont);
        mont_mul(&zi,&pref,&suff);
        mont_mul(&zi,&zi,&invt);
        if(!reseed_pending){
            if(act==0) walk_finish(&x,&y,&a,&b,j,JXs,&dy,isdbl,&zi);
#else
        fe X,Y,Z;
        walk_step(&x,&y,&a,&b,&X,&Y,&Z,&reseed_pending,&rng,esc_pending);
        /* ---- warp-wide batch inversion (Montgomery's trick via 2 shfl scans) ----
         * pref = prod_{j<lane} Z_j ; suff = prod_{j>lane} Z_j ; total = prod_all
         * zi = pref * suff * total^-1 = Z_lane^-1   (all values Montgomery form) */
        fe incl=Z, suff=Z, t, total, invt, pref, zi;
        for(int off=1;off<32;off<<=1){                 /* inclusive prefix scan */
            t = shfl_fe_up(incl,off);
            if(lane>=off) mont_mul(&incl,&incl,&t);
        }
        total=shfl_fe(incl,31);
        fe_inv(&invt,&total);          /* all lanes, same value: divergence-free */
#ifdef RHO_INVTEST
        { fe chkv; mont_mul(&chkv,&invt,&total);
          unsigned bad=0; for(int i=0;i<NL;i++) bad|=(unsigned)(chkv.w[i]!=KC->R1_mont.w[i]);
          if(bad){ unsigned k=atomicAdd(&g_invmis,1u);
            if(k<INVBAD_N){ for(int i=0;i<NL;i++){ g_invbad[k*INVBAD_W+i]=total.w[i];
                                                   g_invbad[k*INVBAD_W+NL+i]=invt.w[i]; }
                             g_invbad[k*INVBAD_W+2*NL+0]=(unsigned)blockIdx.x;
                             g_invbad[k*INVBAD_W+2*NL+1]=(unsigned)(threadIdx.x>>5);
                             g_invbad[k*INVBAD_W+2*NL+2]=((unsigned)round)&0xFFFFFFu; } } }
#endif
        for(int off=1;off<32;off<<=1){                 /* inclusive suffix scan */
            t = shfl_fe_down(suff,off);
            if(lane+off<32) mont_mul(&suff,&suff,&t);
        }
        /* excl prefix/suffix.  NOTE: do NOT write these as
         *   pref = (lane==0) ? KC->R1_mont : shfl_fe_up(incl,1);
         * nvcc (sm_120, CUDA 13.0) miscompiles a shfl result selected against
         * a __constant__ load in a ternary: the selected fe comes out with
         * limb-wise garbage (observed: correct middle limb, own-lane top limb,
         * constant 0x7ffffffe low limb), which zeroes/garbles zi for the whole
         * warp and poisons every walk.  Shfl into a temporary, then override
         * the edge lane — bit-identical math, no miscompile. */
        pref = shfl_fe_up(incl,1);
        if(lane==0) fe_set(&pref,&KC->R1_mont);
        suff = shfl_fe_down(suff,1);
        if(lane==31) fe_set(&suff,&KC->R1_mont);
        mont_mul(&zi,&pref,&suff);
        mont_mul(&zi,&zi,&invt);
        pending_to_affine(&X,&Y,&zi,&x,&y);
        if(!reseed_pending){
#if RHO_ENDO
            /* ---- v6 STAGE 2: canonicalize the new state (order-6 quotient
             * walk).  act==0: point moved by the jump; act==1: walk_prep
             * reseeded (plain scalars -> mont-scalar domain first).
             * Fruitless-cycle guard: the new canonical state repeating ANY
             * hash in the RHO_CYCWIN window sets cycle_pending; the reseed
             * fires AFTER the DP phase so a cycle whose state is itself a DP
             * still produces its (genuine) collision candidate via dp_check
             * — only DP-free cycles are reseeded. */
            if(act==0){
                endo_canon(&fone,&x,&xc,&y,&a,&b);
                uint64_t ch=cyc_hash(&xc);
                for(int k=0;k<cyc_n;k++)
                    if(cycw[k]==ch){ cycle_pending=1; break; }
                cycw[cyc_i]=ch; cyc_i=(cyc_i+1)%RHO_CYCWIN;
                if(cyc_n<RHO_CYCWIN) cyc_n++;
            }else if(act==1){
                sc t5; sc_mulred(&t5,&a,&KC->scR2); sc_set(&a,&t5);
                sc_mulred(&t5,&b,&KC->scR2); sc_set(&b,&t5);
                endo_canon(&fone,&x,&xc,&y,&a,&b);
                cyc_n=0; cyc_i=0; cyc_def=0;
            }
#endif
#endif
            uint64_t xk = low64(&x);
            if((xk & C.dpmask)==0){
                atomicAdd(&g_dphits,1u);
                int dpk = 1;
#if RHO_CURVE_GUARD
                /* ---- v6 STAGE 0b: on-curve guard at DP insertion ------------
                 * ~0.13% of warps drift permanently off-curve (whole 32-lane
                 * runs, init-time).  Catch the warp at its first DP event and
                 * re-init the lane from a fresh random point instead of
                 * inserting a garbage DP and burning the warp forever.  The
                 * fresh point is DP-checked like any other reseed (act==1
                 * semantics).  CPU_REF runs the identical check (it never
                 * fires there), so healthy-lane streams are unchanged. */
                if(!on_curve(&x,&y)){
                    atomicAdd(&g_badcurve,1u);
#if RHO_ENDO
                    walk_reseed_full(&x,&y,&a,&b,&rng,&xc,cycw,&cyc_n,&cyc_i);
#else
                    walk_reseed(&x,&y,&a,&b,&rng);
#endif
                    xk = low64(&x);
                    dpk = ((xk & C.dpmask)==0);
                }
#endif
                if(dpk){
#if RHO_ENDO
                dp_ev=1;
#endif
                if(C.dbg_max && tid==0){
                    uint32_t n=*C.dbg_n;
                    if(n<C.dbg_max){ C.dbg_xk[n]=xk; C.dbg_step[n]=steps; *C.dbg_n=n+1; }
                }
                CandRec cand;
                int act = dp_check(&C,tbl,&x,&y,&a,&b,&cand);
                if(act==DP_INS){
                    atomicAdd(&g_dpins,1u);
                    /* STAGE 2b: persist == emit.  Ordered by the ring's
                     * append counter; the host prints in drain order. */
                    if(C.emit){
                        unsigned li=atomicAdd(C.dplog_head,1u);
                        DPEmitRec r; r.x=x; r.y=y; r.a=a; r.b=b;
                        C.dplog[li & (unsigned)C.dplog_mask]=r;
                    }
                }
                else if(act==DP_RESEED) reseed_pending=1;
                else if(act==DP_CAND){
                    /* single-worker mode reports the candidate through the
                     * device candidate ring; multi-worker mode leaves the
                     * collision to the host, which sees BOTH halves (each
                     * half was flushed when it was inserted). */
                    if(!C.mp){
                        unsigned idx = atomicAdd(cand_head,1u);
                        cands[idx & (NCAND-1)] = cand;
                        /* v6 STAGE 0a: publish-then-park.  The fence makes the
                         * record visible before the park request; the host
                         * clears g_cand_pend and tri-states g_found_flag every
                         * poll, so the grid stops burning steps within ~8
                         * rounds of the candidate instead of ~1.5 s worth. */
                        __threadfence();
                        atomicExch((unsigned*)&g_cand_pend,1u);
                    }
                    /* v5 semantics (restored): the publishing lane restarts.
                     * After a collision the two walks share the same canonical
                     * state and would stay merged forever, re-publishing a
                     * degenerate candidate at every joint DP; the reseed breaks
                     * the merge.  Dropping this in the stage-0a rewrite caused
                     * a 1.4-3.3x GPU endo step inflation (measured). */
                    reseed_pending=1;
                }
                if(C.mp){
                    /* multi-worker: flush EVERY DP event (insert, merge,
                     * candidate, dropped) to the host shared table.  Two
                     * walks that meet at a non-DP point tail together to the
                     * next DP, where both halves arrive at the host with
                     * different (a,b) — the host resolves the collision from
                     * the pair, so no cross-worker collision is lost even
                     * when both walkers pass the DP simultaneously. */
                    DPRec r;
                    r.key=xk; r.xhi=hi64(&x);
                    r.ypar=(uint32_t)(y.w[0]&1u); r.a=a; r.b=b;
                    unsigned ridx = atomicAdd(C.dpring_head,1u);
                    C.dpring[ridx & (unsigned)C.dpring_mask] = r;
                }
                }
            }
        }
#if RHO_ENDO
        /* STAGE 2b: a detected fruitless cycle is ESCAPED (next step takes
         * jump j+1) rather than reseeding.  The structural 2-cycle of the
         * canonical map (double negation + repeated jump index) otherwise
         * kills the walk every ~100-500 steps; merged walks retracing a dead
         * walk's path then die in that dead walk's cycle before reaching a
         * stored DP, which at dpb>=9 delays detection by orders of magnitude
         * (measured 24x at dpb=14).  RHO_CYC_ESC consecutive failed escapes
         * fall back to a reseed (stubborn true loop).  Reseeding still wins
         * when the DP phase itself requested one (candidate/merge/degenerate). */
        if(cycle_pending && !reseed_pending){
            if(++cyc_def<=RHO_CYC_ESC){ esc_pending=1; }
            else { reseed_pending=1; cyc_def=0; }
        } else if(!cycle_pending) cyc_def=0;
        cycle_pending=0; dp_ev=0;
#endif
        if((round & 255)==255) stepctr[tid]=steps;
        if((round & 7)==7){
#ifdef RHO_MULTI
            /* ---- RHO_MULTI: drain the host->device pull inbox ----
             * mp=1: the host appends pulled shared-table records to
             * C.dpinbox and bumps *dpinbox_head as the doorbell AFTER the
             * record copies complete (stream-ordered).  Block 0 / thread 0
             * consumes up to 64 new records per checkpoint via
             * dp_sync_insert — collisions are resolved host-side from the
             * shared log, so the return value is deliberately ignored.
             * *dpin_tail_dev is the drain cursor — per-worker device memory
             * (a shared module global was a bug: two co-resident kernels on
             * one device would each advance the SAME cursor against their
             * own private inbox ring, corrupting both sides).
             * This runs IN the walk kernel: a separate helper kernel would
             * starve, because every SM is held by a never-retiring walk
             * block for the whole run.  mp=0: dpinbox is NULL, skipped. */
            if(C.mp && C.dpinbox && blockIdx.x==0 && threadIdx.x==0){
                unsigned h=*(volatile unsigned*)C.dpinbox_head;
                unsigned t=*(volatile unsigned*)C.dpin_tail_dev;
                unsigned n=h-t;                /* monotonic; guard anyway */
                if(h<t) n=0; else if(n>64) n=64;
                for(unsigned i=0;i<n;i++){
                    DPRec r=C.dpinbox[(uint64_t)(t+i) & C.dpinbox_mask];
                    dp_sync_insert(&C,tbl,&r,NULL);
                }
                *(volatile unsigned*)C.dpin_tail_dev=t+n;
                g_drain_n+=n;
            }
#endif
#if RHO_CAND_HANDSHAKE
            /* ---- v6 STAGE 0a: candidate handshake (single-worker mode) ----
             * g_cand_pend != 0: some lane published a collision candidate and
             * the host must verify it.  Park (spin) while the park request is
             * set AND no stop was signalled; the host clears g_cand_pend when
             * it has drained the ring, which releases the parked threads, and
             * writes g_found_flag = 1 only to stop the run (candidate genuine,
             * or budget expiry).  g_found_flag is 0 or 1 ONLY: the old tri-
             * state value 2 ("rejected, resume") was written on every poll,
             * and the no-pend threads below exited on any nonzero flag — the
             * grid died at the first park boundary after the first poll.
             * mp mode never parks: its candidates are resolved host-side from
             * the flushed DP stream. */
            if(!C.mp){
                if(g_cand_pend){
                    while(g_found_flag==0 && g_cand_pend){
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
                        __nanosleep(64);
#endif
                    }
                    if(g_found_flag==1){ stepctr[tid]=steps; return; }
                } else if(g_found_flag){ stepctr[tid]=steps; return; }
                /* (write before return in BOTH exit paths: a thread that
                 * reaches the round-7 check after the host has already
                 * cleared the park request but before/while g_found_flag
                 * lands used to return without publishing its final step
                 * count -- sub-10s solves then reported steps=0 depending
                 * on the arch/timing race. */
            } else
#endif
            if(g_found_flag){ stepctr[tid]=steps; return; }
        }
    }
}

#if defined(__CUDACC__) && !defined(CPU_REF)
#if RHO_ENDO
/* v6 STAGE 2 debug kernel: canonicalize [a]Q+[b]P for random plain (a,b) on
 * device through the exact production configuration (mont_mul_sp, sc_mulred,
 * endo_canon); the host byte-compares against its own endo_canon result. */
__global__ __launch_bounds__(256) void endotest_kernel(const sc *in,fe *out,sc *outs,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){
        jac A,B,S; fe x,y,xc;
        scalar_mul_point(&A,&in[2*i],&KC->Q_montX,&KC->Q_montY);
        scalar_mul_point(&B,&in[2*i+1],&KC->P_montX,&KC->P_montY);
        jac_add(&S,&A,&B);
        jac_to_affine(&S,&x,&y);
        sc am,bm;
        sc_mulred(&am,&in[2*i],&KC->scR2);
        sc_mulred(&bm,&in[2*i+1],&KC->scR2);
        fe fone; fe_zero(&fone); fone.w[0]=1;
        endo_canon(&fone,&x,&xc,&y,&am,&bm);
        out[3*i]=x; out[3*i+1]=xc; out[3*i+2]=y;
        outs[2*i]=am; outs[2*i+1]=bm;
    }
}
#endif
#endif

#ifdef RHO_INVTEST
/* SASS-PORT debug kernel: fe_inv_gcd vs fe_inv_fermat on device. */
__global__ __launch_bounds__(256) void invtest_kernel(const fe *in,fe *outg,fe *outf,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){ fe g,f; fe_inv_gcd(&g,&in[i]); fe_inv_fermat(&f,&in[i]); outg[i]=g; outf[i]=f; }
}
/* SASS-PORT debug kernel 2: batch-inversion context with shfl scans around
 * fe_inv, per-round feedback, and a caller-state canary around the call. */
__global__ __launch_bounds__(256) void invtest2_kernel(const fe *in,fe *out,
                                                       unsigned *badctr,int n,int rounds){
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    int lane=threadIdx.x&31;
    fe contrib=in[tid & (n-1)];
    for(int r=0;r<rounds;r++){
        fe canary; fe_set(&canary,&contrib);
        fe incl=contrib, suff=contrib, t, total, invt, pref, zi;
        for(int off=1;off<32;off<<=1){ t=shfl_fe_up(incl,off); if(lane>=off) mont_mul(&incl,&incl,&t); }
        total=shfl_fe(incl,31);
        fe_inv(&invt,&total);
        for(int off=1;off<32;off<<=1){ t=shfl_fe_down(suff,off); if(lane+off<32) mont_mul(&suff,&suff,&t); }
        pref=shfl_fe_up(incl,1); if(lane==0) fe_set(&pref,&KC->R1_mont);
        suff=shfl_fe_down(suff,1); if(lane==31) fe_set(&suff,&KC->R1_mont);
        mont_mul(&zi,&pref,&suff); mont_mul(&zi,&zi,&invt);
        fe chk; mont_mul(&chk,&contrib,&zi);
        if(!fe_eq(&chk,&KC->R1_mont)){
            unsigned k=atomicAdd(badctr,1u);
            if(k<16){ out[k*4+0]=total; out[k*4+1]=contrib; out[k*4+2]=zi; out[k*4+3]=chk; }
        }
        if(!fe_eq(&canary,&contrib)){
            unsigned k=atomicAdd(badctr,1u);
            if(k<16){ out[k*4+0]=canary; out[k*4+1]=contrib; out[k*4+2]=total; out[k*4+3]=zi; }
        }
        mont_mul(&contrib,&zi,&contrib);   /* feedback: evolve input */
    }
}
#endif

#endif /* CUDA */

/* ================================================================== */
/* Backends: CPU_REF lane loop (host) and CUDA launch/poll (device)    */
/* ================================================================== */

#include <time.h>
#include <math.h>
#if !defined(_WIN32)
#include <unistd.h>
#endif

static double now_s(void){
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return (double)ts.tv_sec + 1e-9*(double)ts.tv_nsec;
}

typedef struct {
    fe x,y; sc a,b;
    uint64_t rng, steps;
    int reseed_pending;
#if RHO_ENDO
    /* v6 STAGE 2: canonical-x state + fruitless-cycle window (kernel mirror) */
    fe xc;
    uint64_t cycw[RHO_CYCWIN];
    int cyc_n, cyc_i, cyc_def;
    int cycle_pending, dp_ev;
#endif
    int esc_pending;   /* STAGE 2b: next step breaks a fruitless cycle (j+1) */
} Lane;

typedef struct {
    uint64_t steps, dphits, dpins, cands;
    uint64_t badcurve;   /* v6 STAGE 0b: off-curve DP events caught+reseeded */
    double   secs;
    int      found;
} SolveStats;

/* ================================================================== */
/* Checkpointing: --ckpt <file>, RHO_CKPT_EVERY_S (default 300 s)      */
/*                                                                     */
/* Purpose: long (or time-shared) runs survive a kill / box reboot.    */
/* The DP table is the accumulated asset of a Pollard run, so it is    */
/* serialized; the CPU_REF build additionally saves the per-lane walk  */
/* state, which makes a resumed run a bit-for-bit continuation of the  */
/* interrupted one (same kappa, same final step count).                */
/*                                                                     */
/* Binary format (native little-endian, fields written individually    */
/* so the layout does not depend on struct padding):                   */
/*   header : "RHOCKPT", version u32, rung u32, dpbits u32,            */
/*            tblbits u32, seed u64, nrec u64, cum_steps u64,          */
/*            cum_dphits u64, cum_dpins u64, cum_ms u64,               */
/*            nlanes u32, rsvd u32                                     */
/*   DP rec: slot u64, key u64, xhi u64, a u32[NL], b u32[NL],         */
/*           ypar u32                        (only slots key!=0)       */
/*   lane rec (CPU_REF, nlanes>0): rng u64, steps u64, x u32[NL],      */
/*           y u32[NL], a u32[NL], b u32[NL], reseed u32               */
/*   trailer: rec_hash u64 (FNV-1a over all DP record payloads)        */
/*                                                                     */
/* DP records carry their table slot index, so a reloaded table has    */
/* exactly the live layout (probing stays inside one 64-slot shard,    */
/* slot-exact restore therefore also reproduces DP_DROP behaviour).    */
/* Written atomically: <file>.tmp + fsync + rename.                    */
/* ================================================================== */

#define CKPT_MAGIC_STR "RHOCKPT"
/* v6 STAGE 2: RHO_ENDO builds write version 2 (lane records carry the
 * canonical-x state xc plus the fruitless-cycle window cycw/cyc_n/cyc_i)
 * and REFUSE version 1: a v1 ckpt's lane
 * states are plain-scalar walks, incompatible with the quotient walk.
 * DP records are mode-independent, but lane-state loss changes the stream,
 * so a mixed load is not silently accepted either.  Non-endo v6 builds
 * keep version 1 and remain checkpoint-compatible with v5. */
#ifdef RHO_ENDO
#if RHO_ENDO
#define CKPT_VERSION   3u
#define CKPT_ENDO      1
#else
#define CKPT_VERSION   3u
#endif
#else
#define CKPT_VERSION   3u
#endif
/* STAGE 2b: version 3 in ALL builds -- the file now ends with the ordered
 * RHO_DP_EMIT record log (possibly empty when emit is off).  v1/v2 files are
 * REFUSED loudly: the emit contract (re-emit stored records first, in order)
 * needs the log, and a v1/v2 resume would silently produce an emit stream
 * with a missing prefix.  This intentionally ends non-endo v6 ckpt
 * compatibility with v5 files; a v5 ckpt must be restarted under v6. */

static const char *g_ckpt_path  = NULL;   /* --ckpt, set in main          */
static double      g_ckpt_every = 300.0;  /* RHO_CKPT_EVERY_S             */
/* rho_multi.cu sets this: in shared-table mode the authoritative table is
 * the host-side shared table, so the per-GPU device-table mirror that the
 * single-worker checkpoint path uses is pure waste (~235 MB pinned/GPU). */
static int         g_no_htbl_mirror = 0;

typedef struct {
    int      enabled;
    double   every;
    double   next;
    uint64_t cum_steps, cum_dph, cum_dpi, cum_ms;  /* carried across restarts */
} CkptState;

static int ck_bad;
static void cw(FILE *f,const void *p,size_t n){ if(!ck_bad && fwrite(p,1,n,f)!=n) ck_bad=1; }
static void cw32(FILE *f,uint32_t v){ cw(f,&v,4); }
static void cw64(FILE *f,uint64_t v){ cw(f,&v,8); }
static int  rd(FILE *f,void *p,size_t n){ return fread(p,1,n,f)==(size_t)n; }
static int  rd32(FILE *f,uint32_t *v){ return rd(f,v,4); }
static int  rd64(FILE *f,uint64_t *v){ return rd(f,v,8); }
static void ck_hash(uint64_t *h,const void *p,size_t n){
    const unsigned char *b=(const unsigned char*)p;
    for(size_t i=0;i<n;i++){ *h^=b[i]; *h*=0x100000001B3ULL; }
}

static int ckpt_write(const char *path,const Cfg *C,const DPSlot *tbl,size_t tblslots,
                      const Lane *lanes,int nlanes,
                      uint64_t cum_steps,uint64_t cum_dph,uint64_t cum_dpi,
                      uint64_t cum_ms)
{
    if(!path||!path[0]) return 0;
    uint64_t nrec=0;
    if(!g_dp_emit){
        for(size_t i=0;i<tblslots;i++)
            if(tbl[i].key && tbl[i].key!=RSV64) nrec++;   /* RSV64 = in-flight insert */
    }
    /* g_dp_emit: the table section is written EMPTY -- the ordered emit log
     * below IS the insert history (persist == emit) and the loader rebuilds
     * the table from it, so the racy device-table snapshot is not needed. */
    char tmp[1024];
    if(snprintf(tmp,sizeof(tmp),"%s.tmp",path)>=(int)sizeof(tmp)) return 0;
    FILE *f=fopen(tmp,"wb");
    if(!f){ fprintf(stderr,"ckpt: cannot write %s (%s); checkpointing disabled\n",tmp,strerror(errno)); return 0; }
    ck_bad=0;
    uint64_t h=0xCBF29CE484222325ULL;
    cw(f,CKPT_MAGIC_STR,8);
    cw32(f,CKPT_VERSION);
    cw32(f,(uint32_t)RUNG_BITS);
    cw32(f,C->dpbits);
    cw32(f,C->tblbits);
    cw64(f,C->seed);
    cw64(f,nrec);
    cw64(f,cum_steps); cw64(f,cum_dph); cw64(f,cum_dpi); cw64(f,cum_ms);
    cw32(f,(uint32_t)(lanes?nlanes:0));
    cw32(f,0u);
    for(size_t i=0;i<tblslots && !g_dp_emit;i++){
        const DPSlot *s=&tbl[i];
        if(!s->key || s->key==RSV64) continue;
        cw64(f,(uint64_t)i);
        cw64(f,s->key); cw64(f,s->xhi);
        for(int k=0;k<NL;k++) cw32(f,s->a.w[k]);
        for(int k=0;k<NL;k++) cw32(f,s->b.w[k]);
        cw32(f,s->ypar);
        ck_hash(&h,&s->key,8); ck_hash(&h,&s->xhi,8);
        ck_hash(&h,s->a.w,4*NL); ck_hash(&h,s->b.w,4*NL);
        ck_hash(&h,&s->ypar,4);
    }
    if(lanes) for(int i=0;i<nlanes;i++){
        cw64(f,lanes[i].rng); cw64(f,lanes[i].steps);
        for(int k=0;k<NL;k++) cw32(f,lanes[i].x.w[k]);
        for(int k=0;k<NL;k++) cw32(f,lanes[i].y.w[k]);
        for(int k=0;k<NL;k++) cw32(f,lanes[i].a.w[k]);
        for(int k=0;k<NL;k++) cw32(f,lanes[i].b.w[k]);
        cw32(f,(uint32_t)lanes[i].reseed_pending);
#ifdef CKPT_ENDO
        for(int k=0;k<NL;k++) cw32(f,lanes[i].xc.w[k]);
        for(int k=0;k<RHO_CYCWIN;k++) cw64(f,lanes[i].cycw[k]);
        cw32(f,(uint32_t)lanes[i].cyc_n); cw32(f,(uint32_t)lanes[i].cyc_i);
#endif
    }
    /* v3: the ordered RHO_DP_EMIT record log (mont-domain, in emit order) */
    cw64(f,g_emit_n);
    for(uint64_t i=0;i<g_emit_n;i++){
        const DPEmitRec *r=&g_emit_log[i];
        for(int k=0;k<NL;k++) cw32(f,r->x.w[k]);
        for(int k=0;k<NL;k++) cw32(f,r->y.w[k]);
        for(int k=0;k<NL;k++) cw32(f,r->a.w[k]);
        for(int k=0;k<NL;k++) cw32(f,r->b.w[k]);
    }
    cw64(f,h);
    if(fflush(f)!=0) ck_bad=1;
#if !defined(_WIN32)
    if(!ck_bad && fsync(fileno(f))!=0) ck_bad=1;
#endif
    fclose(f);
    if(ck_bad){ fprintf(stderr,"ckpt: write failed; checkpointing disabled\n"); remove(tmp); return 0; }
    if(rename(tmp,path)!=0){ fprintf(stderr,"ckpt: rename %s -> %s failed (%s)\n",tmp,path,strerror(errno)); return 0; }
    return 1;
}

/* Host-side replay of one RHO_DP_EMIT log record into a fresh table.  Mirrors
 * the kernel's DP_INS placement exactly (shard base from hashx(key>>dpbits),
 * first empty of the 64 slots) so a replayed log reproduces the layout the
 * original inserts produced.  Used only on emit-mode resume: the log is the
 * authoritative insert history (persist == emit), which removes the race
 * between the drained log and the device-table snapshot at save time. */
static void tbl_replay_insert(DPSlot *tbl,const Cfg *C,const DPEmitRec *r){
    uint64_t xk=low64(&r->x);
    if(!xk) return;                                   /* never persisted */
    uint64_t h=hashx(xk>>C->dpbits);
    uint64_t base=(h & C->tblmask) & ~63ULL;
    /* the probe sequence starts at shard offset (h & 63), not 0 -- mirroring
     * dp_check's `slot = base + ((h+probe)&63)` exactly is what makes a
     * replayed log reproduce the original layout slot for slot. */
    uint64_t off=h & 63ULL;
    for(int p=0;p<64;p++){
        uint64_t slot=base+((off+(uint64_t)p)&63ULL);
        if(!tbl[slot].key){
            tbl[slot].key=xk; tbl[slot].xhi=hi64(&r->x);
            tbl[slot].ypar=(uint32_t)(r->y.w[0]&1u);
            tbl[slot].a=r->a; tbl[slot].b=r->b;
            return;
        }
    }
    /* shard full at insert time cannot happen for an emitted record (the
     * kernel emits only on DP_INS); if it somehow does, drop like the
     * kernel would have. */
}

/* Returns 1 if the checkpoint was loaded, 0 if the file does not exist
 * (fresh start).  Refuses — exit(1) — on version/rung/dpbits/tblbits/seed
 * mismatch or on corruption: a DP entry from a different jump table would
 * silently produce wrong collision arithmetic, so this must never pass. */
static int ckpt_load(const char *path,const Cfg *C,DPSlot *tbl,size_t tblslots,
                     Lane *lanes,int nlanes,CkptState *ck)
{
    if(!path||!path[0]) return 0;
    FILE *f=fopen(path,"rb");
    if(!f) return 0;
    char magic[8]; uint32_t ver,rung,dpb,tbb,nl_saved,resv;
    uint64_t seed,nrec,cs,cdh,cdi,cms,h;
    if(!rd(f,magic,8)||memcmp(magic,CKPT_MAGIC_STR,8)!=0){
        fprintf(stderr,"ckpt: %s is not a %s file\n",path,CKPT_MAGIC_STR); exit(1); }
    if(!rd32(f,&ver)||!rd32(f,&rung)||!rd32(f,&dpb)||!rd32(f,&tbb)||!rd64(f,&seed)||
       !rd64(f,&nrec)||!rd64(f,&cs)||!rd64(f,&cdh)||!rd64(f,&cdi)||!rd64(f,&cms)||
       !rd32(f,&nl_saved)||!rd32(f,&resv)){
        fprintf(stderr,"ckpt: truncated header in %s\n",path); exit(1); }
    if(ver!=CKPT_VERSION){
#ifdef CKPT_ENDO
        fprintf(stderr,"ckpt: %s has version %u but this is an RHO_ENDO (v6 STAGE 2) build "
                       "expecting version %u -- endo walks cannot resume from non-endo lane state\n",
                path,ver,CKPT_VERSION);
#else
        fprintf(stderr,"ckpt: %s has unsupported version %u (expected %u)\n",path,ver,CKPT_VERSION);
#endif
        exit(1);
    }
    if(rung!=(uint32_t)RUNG_BITS||dpb!=C->dpbits||tbb!=C->tblbits||seed!=C->seed){
        fprintf(stderr,"ckpt: refusing %s: file(rung=%u dpbits=%u tblbits=%u seed=%llu) "
                       "!= requested(rung=%d dpbits=%u tblbits=%u seed=%llu)\n",path,
                rung,dpb,tbb,(unsigned long long)seed,
                RUNG_BITS,C->dpbits,C->tblbits,(unsigned long long)C->seed);
        exit(1);
    }
    memset(tbl,0,tblslots*sizeof(DPSlot));
    uint64_t got=0, sum=0xCBF29CE484222325ULL;
    for(uint64_t r=0;r<nrec;r++){
        uint64_t slot,key,xhi; uint32_t ypar; sc a,b;
        if(!rd64(f,&slot)||!rd64(f,&key)||!rd64(f,&xhi)||
           !rd(f,a.w,4*NL)||!rd(f,b.w,4*NL)||!rd32(f,&ypar)){
            fprintf(stderr,"ckpt: truncated record %llu in %s\n",(unsigned long long)r,path); exit(1); }
        ck_hash(&sum,&key,8); ck_hash(&sum,&xhi,8);
        ck_hash(&sum,a.w,4*NL); ck_hash(&sum,b.w,4*NL); ck_hash(&sum,&ypar,4);
        if(slot>=tblslots||key==0||key==RSV64){
            fprintf(stderr,"ckpt: corrupt record %llu in %s (slot=%llu key=%llu)\n",
                    (unsigned long long)r,path,(unsigned long long)slot,(unsigned long long)key); exit(1); }
        uint64_t base=(hashx(key>>C->dpbits) & C->tblmask) & ~63ULL;
        if(slot<base||slot>=base+64){
            fprintf(stderr,"ckpt: corrupt record %llu in %s (slot %llu outside its shard)\n",
                    (unsigned long long)r,path,(unsigned long long)slot); exit(1); }
        DPSlot *s=&tbl[slot];
        s->xhi=xhi; s->ypar=ypar; s->a=a; s->b=b; s->key=key;
        got++;
    }
    int lanes_restored=0;
    if(nl_saved>0){
        if(lanes && (int)nl_saved==nlanes){
            for(int i=0;i<nlanes;i++){
                uint32_t rs;
                if(!rd64(f,&lanes[i].rng)||!rd64(f,&lanes[i].steps)||
                   !rd(f,lanes[i].x.w,4*NL)||!rd(f,lanes[i].y.w,4*NL)||
                   !rd(f,lanes[i].a.w,4*NL)||!rd(f,lanes[i].b.w,4*NL)||!rd32(f,&rs)){
                    fprintf(stderr,"ckpt: truncated lane record in %s\n",path); exit(1); }
                lanes[i].reseed_pending=(int)rs;
#ifdef CKPT_ENDO
                uint32_t ckn_,cki_;
                if(!rd(f,lanes[i].xc.w,4*NL)||!rd(f,lanes[i].cycw,8*RHO_CYCWIN)||
                   !rd32(f,&ckn_)||!rd32(f,&cki_)){
                    fprintf(stderr,"ckpt: truncated endo lane record in %s\n",path); exit(1); }
                lanes[i].cyc_n=(int)ckn_; lanes[i].cyc_i=(int)cki_;
                lanes[i].cycle_pending=0; lanes[i].dp_ev=0; lanes[i].cyc_def=0;
                lanes[i].esc_pending=0;
#endif
            }
            lanes_restored=1;
        }else{
            size_t lb=(size_t)nl_saved*(16+(size_t)NL*16+4
#ifdef CKPT_ENDO
                +(size_t)NL*4+(size_t)RHO_CYCWIN*8+8
#endif
            );
            if(fseek(f,(long)lb,SEEK_CUR)!=0){
                fprintf(stderr,"ckpt: cannot skip lane records in %s\n",path); exit(1); }
        }
    }
    /* v3: the ordered RHO_DP_EMIT record log (mont-domain, in emit order) */
    { uint64_t emit_n=0;
      if(!rd64(f,&emit_n)){
          fprintf(stderr,"ckpt: truncated emit-log count in %s\n",path); exit(1); }
      if(emit_n>((uint64_t)1<<30)){
          fprintf(stderr,"ckpt: absurd emit-log count %llu in %s\n",
                  (unsigned long long)emit_n,path); exit(1); }
      for(uint64_t i=0;i<emit_n;i++){
          DPEmitRec r;
          if(!rd(f,r.x.w,4*NL)||!rd(f,r.y.w,4*NL)||!rd(f,r.a.w,4*NL)||!rd(f,r.b.w,4*NL)){
              fprintf(stderr,"ckpt: truncated emit record %llu in %s\n",
                      (unsigned long long)i,path); exit(1); }
          if(g_dp_emit) emit_log_append(&r);
      }
      if(emit_n&&!g_dp_emit){
          fprintf(stderr,"ckpt: %s carries %llu RHO_DP_EMIT records; resume with "
                         "RHO_DP_EMIT=1 (or delete the ckpt) -- an emit-mode run must "
                         "re-emit its stored prefix\n",
                  path,(unsigned long long)emit_n);
          exit(1);
      }
      if(g_dp_emit){
          /* emit-mode resume: the LOG is the authoritative insert history
           * (persist == emit), so rebuild the DP table from it instead of the
           * (racy) table snapshot and skip the table section entirely at
           * write time.  Replaying in order reproduces the exact shard
           * layout the original inserts produced. */
          memset(tbl,0,tblslots*sizeof(DPSlot));
          for(uint64_t i=0;i<g_emit_n;i++)
              tbl_replay_insert(tbl,C,&g_emit_log[i]);
          fprintf(stderr,"dp-emit: rebuilt DP table from %llu log records\n",
                  (unsigned long long)g_emit_n);
      }
    }
    if(!rd64(f,&h)||h!=sum){
        fprintf(stderr,"ckpt: record hash mismatch in %s (corrupt or partially written file)\n",path);
        exit(1);
    }
    fclose(f);
    if(nl_saved>0&&lanes_restored)
        fprintf(stderr,"ckpt: loaded %llu DP records + %u lane states "
                       "(cum_steps=%llu cum_dphits=%llu cum_dpins=%llu)\n",
                (unsigned long long)got,nl_saved,(unsigned long long)cs,
                (unsigned long long)cdh,(unsigned long long)cdi);
    else if(nl_saved>0)
        fprintf(stderr,"ckpt: loaded %llu DP records (lane count %u != %d: walks restart fresh)\n",
                (unsigned long long)got,nl_saved,nlanes);
    else
        fprintf(stderr,"ckpt: loaded %llu DP records (no lane states: walks restart fresh)\n",
                (unsigned long long)got);
    ck->cum_steps=cs; ck->cum_dph=cdh; ck->cum_dpi=cdi; ck->cum_ms=cms;
    return 1;
}

/*
 * CPU_REF solver: nlanes independent walks, each advanced one r-adding step
 * per round, batch-inverted across the lane group exactly like the kernel's
 * warp (the serial Montgomery trick yields the very same zi = Z_i^-1 values,
 * so lane 0 reproduces the kernel's lane 0 walk bit-for-bit).
 */
static int solve_host(const Cfg *C, DPSlot *tbl, int nlanes, sc *kap_out,
                      SolveStats *st, CkptState *ck)
{
    Lane *L=(Lane*)calloc((size_t)nlanes,sizeof(Lane));
    fe *Xa=(fe*)malloc(sizeof(fe)*nlanes), *Ya=(fe*)malloc(sizeof(fe)*nlanes);
    fe *Za=(fe*)malloc(sizeof(fe)*nlanes), *pre=(fe*)malloc(sizeof(fe)*nlanes);
    fe *suf=(fe*)malloc(sizeof(fe)*nlanes);
    /* SASS-PORT affine-walk scratch (see walk_prep) */
    int *jout=(int*)malloc(sizeof(int)*(size_t)nlanes);
    int *isdbl=(int*)malloc(sizeof(int)*(size_t)nlanes);
    int *act=(int*)malloc(sizeof(int)*(size_t)nlanes);
    for(int i=0;i<nlanes;i++){
#if RHO_ENDO
        L[i].cyc_def=0;
#endif
        L[i].rng = C->seed ^ (0xABCDEF1234567890ULL*(uint64_t)(i+1)) ^ GOLDEN;
#if RHO_ENDO
        walk_reseed_full(&L[i].x,&L[i].y,&L[i].a,&L[i].b,&L[i].rng,&L[i].xc,
                         L[i].cycw,&L[i].cyc_n,&L[i].cyc_i);
#else
        walk_reseed(&L[i].x,&L[i].y,&L[i].a,&L[i].b,&L[i].rng);
#endif
    }
    uint64_t gsteps=0,dph=0,dpi=0,dpc=0,badc=0;
    double t0=now_s(), last=t0;
    size_t tblslots=(size_t)1<<C->tblbits;
    if(ck->enabled){
        /* resume: DP table always; lane states when the lane count matches.
         * With lane states restored, the run continues bit-for-bit where the
         * checkpoint left off (same kappa, same final step count). */
        if(ckpt_load(g_ckpt_path,C,tbl,tblslots,L,nlanes,ck)){
            gsteps=ck->cum_steps; dph=ck->cum_dph; dpi=ck->cum_dpi;
            /* RHO_DP_EMIT resume contract: the stored DP prefix is re-emitted
             * verbatim, in original order, BEFORE any new walk step.  The
             * loader already appended these records to g_emit_log (it is the
             * authoritative insert history), so only the lines are printed
             * here.  A client that acked the pre-ckpt stream skips an exact
             * prefix -- no gap, no resubmit. */
            if(g_dp_emit){
                for(uint64_t i=0;i<g_emit_n;i++){
                    const DPEmitRec *r=&g_emit_log[i];
                    dp_emit_line((unsigned)RUNG,&r->x,&r->y,&r->a,&r->b);
                }
                fprintf(stderr,"dp-emit: re-emitted %llu stored records\n",
                        (unsigned long long)g_emit_n);
            }
        }
        ck->next=t0+ck->every;
    }
    if(getenv("WALKDBG") && nlanes>0){
        printf("init walk0: x="); print_val64(L[0].x.w);
        printf("\n y="); print_val64(L[0].y.w);
        printf("\n a="); print_val64(L[0].a.w);
        printf("\n b="); print_val64(L[0].b.w);
        printf("\n"); fflush(stdout);
    }
    int found=0;
    for(uint64_t round=0;!found;round++){
#if RHO_AFFINE
        /* SASS-PORT affine walk: per lane compute the jump and the batch
         * denominator; serial Montgomery trick over contrib[] gives the very
         * same zi values the warp scans produce on the GPU. */
        {
            for(int i=0;i<nlanes;i++){
                act[i]=walk_prep(JX,JY,&L[i].x,&L[i].y,&L[i].a,&L[i].b,
                                 &L[i].reseed_pending,&L[i].rng,
                                 &jout[i],&Xa[i],&Ya[i],&Za[i],&isdbl[i],
                                 L[i].esc_pending);
                L[i].esc_pending=0;
            }
            /* batch inversion across the lane group (Montgomery's trick, same
             * formula as the kernel: zi = pref*suff*total^-1 = Z_i^-1) */
            fe_set(&pre[0],&Za[0]);
            for(int i=1;i<nlanes;i++) mont_mul(&pre[i],&pre[i-1],&Za[i]);
            fe_set(&suf[nlanes-1],&Za[nlanes-1]);
            for(int i=nlanes-2;i>=0;i--) mont_mul(&suf[i],&suf[i+1],&Za[i]);
            fe invt; fe_inv(&invt,&pre[nlanes-1]);
            for(int i=0;i<nlanes;i++){
                if(L[i].reseed_pending) continue;      /* infinity case */
                if(act[i]==0){
                    fe zi,pref,su;
                    if(i==0) fe_set(&pref,&KC->R1_mont); else mont_mul(&pref,&pre[i-1],&KC->R1_mont);
                    if(i==nlanes-1) fe_set(&su,&KC->R1_mont); else fe_set(&su,&suf[i+1]);
                    mont_mul(&zi,&pref,&su);
                    mont_mul(&zi,&zi,&invt);
                    walk_finish(&L[i].x,&L[i].y,&L[i].a,&L[i].b,jout[i],JX,&Ya[i],isdbl[i],&zi);
                }
                /* act[i]==1: reseeded — (x,y) already final, DP-checked below */
            }
        }
#if RHO_ENDO
        /* v6 STAGE 2: canonicalize each lane's new state (order-6 quotient
         * walk) + fruitless-cycle detection.  Mirrors the kernel exactly. */
        {
            fe fone; fe_zero(&fone); fone.w[0]=1;
            for(int i=0;i<nlanes;i++){
                if(L[i].reseed_pending) continue;      /* infinity: no step */
                if(act[i]==0){
                    endo_canon(&fone,&L[i].x,&L[i].xc,&L[i].y,&L[i].a,&L[i].b);
                    uint64_t ch=cyc_hash(&L[i].xc);
                    for(int k=0;k<L[i].cyc_n;k++)
                        if(L[i].cycw[k]==ch){ L[i].cycle_pending=1; break; }
                    L[i].cycw[L[i].cyc_i]=ch;
                    L[i].cyc_i=(L[i].cyc_i+1)%RHO_CYCWIN;
                    if(L[i].cyc_n<RHO_CYCWIN) L[i].cyc_n++;
                }else{                                 /* act[i]==1: fresh start */
                    sc t5; sc_mulred(&t5,&L[i].a,&KC->scR2); sc_set(&L[i].a,&t5);
                    sc_mulred(&t5,&L[i].b,&KC->scR2); sc_set(&L[i].b,&t5);
                    endo_canon(&fone,&L[i].x,&L[i].xc,&L[i].y,&L[i].a,&L[i].b);
                    L[i].cyc_n=0; L[i].cyc_i=0; L[i].cyc_def=0;
                }
            }
        }
#endif
#else
        for(int i=0;i<nlanes;i++)
            walk_step(&L[i].x,&L[i].y,&L[i].a,&L[i].b,&Xa[i],&Ya[i],&Za[i],
                      &L[i].reseed_pending,&L[i].rng,L[i].esc_pending);
        /* batch inversion across the lane group (Montgomery's trick, same
         * formula as the kernel: zi = pref*suff*total^-1 = Z_i^-1) */
        fe_set(&pre[0],&Za[0]);
        for(int i=1;i<nlanes;i++) mont_mul(&pre[i],&pre[i-1],&Za[i]);
        fe_set(&suf[nlanes-1],&Za[nlanes-1]);
        for(int i=nlanes-2;i>=0;i--) mont_mul(&suf[i],&suf[i+1],&Za[i]);
        fe invt; fe_inv(&invt,&pre[nlanes-1]);
        for(int i=0;i<nlanes;i++){
            fe zi,pref,su;
            if(i==0) fe_set(&pref,&KC->R1_mont); else mont_mul(&pref,&pre[i-1],&KC->R1_mont);
            if(i==nlanes-1) fe_set(&su,&KC->R1_mont); else fe_set(&su,&suf[i+1]);
            mont_mul(&zi,&pref,&su);
            mont_mul(&zi,&zi,&invt);
            pending_to_affine(&Xa[i],&Ya[i],&zi,&L[i].x,&L[i].y);
        }
#endif
        /* DP phase, in lane order (kernel order) */
        for(int i=0;i<nlanes;i++){
            if(L[i].reseed_pending) continue;
            uint64_t xk=low64(&L[i].x);
            if((xk & C->dpmask)==0){
                dph++;
                int dpk=1;
#if RHO_CURVE_GUARD
                /* v6 STAGE 0b: identical guard to the kernel's (never fires on
                 * the host — 0 in 188k DP events — but keeps the two streams
                 * definitionally identical). */
                if(!on_curve(&L[i].x,&L[i].y)){
                    badc++;
#if RHO_ENDO
                    walk_reseed_full(&L[i].x,&L[i].y,&L[i].a,&L[i].b,&L[i].rng,&L[i].xc,
                                     L[i].cycw,&L[i].cyc_n,&L[i].cyc_i);
#else
                    walk_reseed(&L[i].x,&L[i].y,&L[i].a,&L[i].b,&L[i].rng);
#endif
                    xk=low64(&L[i].x);
                    dpk=((xk & C->dpmask)==0);
                }
#endif
                if(dpk){
#if RHO_ENDO
                L[i].dp_ev=1;
#endif
                if(i==0 && C->dbg_max){
                    uint32_t n=*C->dbg_n;
                    if(n<C->dbg_max){ C->dbg_xk[n]=xk; C->dbg_step[n]=L[i].steps+1; *C->dbg_n=n+1; }
                }
                CandRec cand;
                int act=dp_check(C,tbl,&L[i].x,&L[i].y,&L[i].a,&L[i].b,&cand);
                if(act==DP_INS){
                    dpi++;
                    if(g_dp_emit){
                        DPEmitRec r; r.x=L[i].x; r.y=L[i].y; r.a=L[i].a; r.b=L[i].b;
                        dp_emit_line((unsigned)RUNG,&r.x,&r.y,&r.a,&r.b);
                        emit_log_append(&r);
                    }
                }
                else if(act==DP_RESEED) L[i].reseed_pending=1;
                else if(act==DP_CAND){
                    dpc++; L[i].reseed_pending=1;
                    sc kap;
                    if(resolve_candidate(&cand.a1,&cand.b1,&cand.a2,&cand.b2,
                                         cand.ypar1==cand.ypar2,&kap) && verify_kappa(&kap)){
                        *kap_out=kap; found=1;
                    }
                }
                }
            }
        }
#if RHO_ENDO
        for(int i=0;i<nlanes;i++){
            /* STAGE 2b: escape (j+1 next step) instead of reseed; see the
             * kernel comment for the structural-2-cycle rationale. */
            if(L[i].cycle_pending && !L[i].reseed_pending){
                if(++L[i].cyc_def<=RHO_CYC_ESC){ L[i].esc_pending=1; }
                else { L[i].reseed_pending=1; L[i].cyc_def=0; }
            } else if(!L[i].cycle_pending) L[i].cyc_def=0;
            L[i].cycle_pending=0; L[i].dp_ev=0;
        }
#endif
        for(int i=0;i<nlanes;i++) L[i].steps++;
        gsteps += (uint64_t)nlanes;
        double now=now_s();
        if(now-last>10.0){
            last=now;
            fprintf(stderr,"[%0.0fs] steps=%llu rate=%.2f M/s dphits=%llu ins=%llu cand=%llu badcurve=%llu\n",
                now-t0,(unsigned long long)gsteps,gsteps/(now-t0)/1e6,
                (unsigned long long)dph,(unsigned long long)dpi,(unsigned long long)dpc,
                (unsigned long long)badc);
            fflush(stderr);
        }
        if(ck->enabled && now>=ck->next){
            ck->next=now+ck->every;
            if(ckpt_write(g_ckpt_path,C,tbl,tblslots,L,nlanes,gsteps,dph,dpi,
                          ck->cum_ms+(uint64_t)((now-t0)*1000.0)))
                fprintf(stderr,"[ckpt] %s written (steps=%llu, %llu DPs)\n",g_ckpt_path,
                        (unsigned long long)gsteps,(unsigned long long)dpi);
        }
        if(g_budget_s>0 && now_s()-t0>g_budget_s){
            if(ck->enabled) ckpt_write(g_ckpt_path,C,tbl,tblslots,L,nlanes,gsteps,dph,dpi,
                                       ck->cum_ms+(uint64_t)((now_s()-t0)*1000.0));
            break;
        }
    }
    st->steps=gsteps; st->dphits=dph; st->dpins=dpi; st->cands=dpc; st->badcurve=badc;
    st->secs=now_s()-t0+(double)ck->cum_ms/1000.0;   /* cumulative across restarts */
    st->found=found;
    free(L); free(Xa); free(Ya); free(Za); free(pre); free(suf);
    free(jout); free(isdbl); free(act);
    return found;
}



#if RHO_ENDO
/* v6 STAGE 2: endotest — canonicalization validation over ntrial random
 * plain (a,b).  CPU: [a*]Q+[b*]P must equal the canonical point and stay on
 * curve.  CUDA builds additionally byte-compare the device endo_canon
 * (mont_mul_sp + sc_mulred in production configuration) against the host. */
static void copy_constants(void);
static int endo_endotest(uint64_t ntrial){
    sc *in=(sc*)malloc(2*sizeof(sc)*ntrial);
    uint64_t rs=0xE6D0BEEF9999ULL;
    for(uint64_t i=0;i<ntrial;i++){ random_sc(&in[2*i],&rs); random_sc(&in[2*i+1],&rs); }
    fe fone; fe_zero(&fone); fone.w[0]=1;
    fe *cx=(fe*)malloc(3*sizeof(fe)*ntrial);
    sc *cs=(sc*)malloc(2*sizeof(sc)*ntrial);
    for(uint64_t i=0;i<ntrial;i++){
        jac A,B,S; fe x,y,xc;
        scalar_mul_point(&A,&in[2*i],&hK.Q_montX,&hK.Q_montY);
        scalar_mul_point(&B,&in[2*i+1],&hK.P_montX,&hK.P_montY);
        jac_add(&S,&A,&B);
        if(fe_is_zero(&S.Z)){ --i; random_sc(&in[2*i],&rs); random_sc(&in[2*i+1],&rs); continue; }
        jac_to_affine(&S,&x,&y);
        sc_set(&cs[2*i],&in[2*i]); sc_set(&cs[2*i+1],&in[2*i+1]);
        { sc t; sc_mulred(&t,&cs[2*i],&hK.scR2); sc_set(&cs[2*i],&t);
          sc_mulred(&t,&cs[2*i+1],&hK.scR2); sc_set(&cs[2*i+1],&t); }
        endo_canon(&fone,&x,&xc,&y,&cs[2*i],&cs[2*i+1]);
        cx[3*i]=x; cx[3*i+1]=xc; cx[3*i+2]=y;
    }
    int bad=0;
    for(uint64_t i=0;i<ntrial;i++){
        sc ap,bp; sc_mulred(&ap,&cs[2*i],&sc_one_static); sc_mulred(&bp,&cs[2*i+1],&sc_one_static);
        jac A,B,S; fe rx,ry,tx,ty;
        scalar_mul_point(&A,&ap,&hK.Q_montX,&hK.Q_montY);
        scalar_mul_point(&B,&bp,&hK.P_montX,&hK.P_montY);
        jac_add(&S,&A,&B);
        if(fe_is_zero(&S.Z)){ bad++; continue; }
        jac_to_affine(&S,&rx,&ry);
        mont_mul(&tx,&rx,&fone);
        if(!fe_eq(&tx,&cx[3*i+1])||!fe_eq(&ry,&cx[3*i+2])||!on_curve(&cx[3*i],&cx[3*i+2])) bad++;
    }
    printf("endotest: CPU self-consistency %s (%llu trials, %d bad)\n",
           bad?"FAILED":"ok",(unsigned long long)ntrial,bad);
#if defined(__CUDACC__) && !defined(CPU_REF)
#ifndef CUDAOK
#define CUDAOK(x) do{ cudaError_t _e=(x); if(_e!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s @%d\n",cudaGetErrorString(_e),__LINE__); exit(1);} }while(0)
#endif
    if(bad==0){
        sc *din,*douts_c; fe *dout;
        CUDAOK(cudaMalloc((void**)&din,2*sizeof(sc)*ntrial));
        CUDAOK(cudaMalloc((void**)&dout,3*sizeof(fe)*ntrial));
        CUDAOK(cudaMalloc((void**)&douts_c,2*sizeof(sc)*ntrial));
        CUDAOK(cudaMemcpy(din,in,2*sizeof(sc)*ntrial,cudaMemcpyHostToDevice));
        copy_constants();   /* endotest runs standalone: device constants (cK etc.)
                               are only otherwise uploaded in gpu_run() */
        endotest_kernel<<<(int)((ntrial+255)/256),256>>>(din,dout,douts_c,(int)ntrial);
        CUDAOK(cudaGetLastError());
        fe *gx=(fe*)malloc(3*sizeof(fe)*ntrial); sc *gs=(sc*)malloc(2*sizeof(sc)*ntrial);
        CUDAOK(cudaMemcpy(gx,dout,3*sizeof(fe)*ntrial,cudaMemcpyDeviceToHost));
        CUDAOK(cudaMemcpy(gs,douts_c,2*sizeof(sc)*ntrial,cudaMemcpyDeviceToHost));
        int gbad=0; uint64_t first=(uint64_t)-1;
        for(uint64_t i=0;i<ntrial;i++){
            int m=0;
            for(int k=0;k<NL;k++){
                if(gx[3*i].w[k]!=cx[3*i].w[k]) m=1;
                if(gx[3*i+1].w[k]!=cx[3*i+1].w[k]) m=1;
                if(gx[3*i+2].w[k]!=cx[3*i+2].w[k]) m=1;
                if(gs[2*i].w[k]!=cs[2*i].w[k]) m=1;
                if(gs[2*i+1].w[k]!=cs[2*i+1].w[k]) m=1;
            }
            if(m){ gbad++; if(first==(uint64_t)-1){ first=i;
                printf("endotest: MISMATCH trial %llu\n",(unsigned long long)i);
                printf("  CPU x ="); for(int k=NL-1;k>=0;k--) printf(" %08x",cx[3*i].w[k]); printf("\n");
                printf("  GPU x ="); for(int k=NL-1;k>=0;k--) printf(" %08x",gx[3*i].w[k]); printf("\n");
                printf("  CPU xc="); for(int k=NL-1;k>=0;k--) printf(" %08x",cx[3*i+1].w[k]); printf("\n");
                printf("  GPU xc="); for(int k=NL-1;k>=0;k--) printf(" %08x",gx[3*i+1].w[k]); printf("\n");
                printf("  CPU y ="); for(int k=NL-1;k>=0;k--) printf(" %08x",cx[3*i+2].w[k]); printf("\n");
                printf("  GPU y ="); for(int k=NL-1;k>=0;k--) printf(" %08x",gx[3*i+2].w[k]); printf("\n");
                printf("  CPU am="); for(int k=NL-1;k>=0;k--) printf(" %08x",cs[2*i].w[k]); printf("\n");
                printf("  GPU am="); for(int k=NL-1;k>=0;k--) printf(" %08x",gs[2*i].w[k]); printf("\n");
                printf("  CPU bm="); for(int k=NL-1;k>=0;k--) printf(" %08x",cs[2*i+1].w[k]); printf("\n");
                printf("  GPU bm="); for(int k=NL-1;k>=0;k--) printf(" %08x",gs[2*i+1].w[k]); printf("\n");
            } }
        }
        printf("endotest: GPU vs CPU canon %s (%llu trials, %d mismatch%s)\n",
               gbad?"FAILED":"ok",(unsigned long long)ntrial,gbad,gbad==1?"":"s");
        if(gbad) printf("endotest: first mismatch at trial %llu\n",(unsigned long long)first);
        bad+=gbad;
        free(gx); free(gs);
        cudaFree(din); cudaFree(dout); cudaFree(douts_c);
    }
#endif
    free(in); free(cx); free(cs);
    return bad==0;
}
#endif

#ifndef CPU_REF
/* ------------------------------ CUDA backend ------------------------------ */
#ifndef CUDAOK
#define CUDAOK(x) do{ cudaError_t _e=(x); if(_e!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s @%d\n",cudaGetErrorString(_e),__LINE__); exit(1);} }while(0)
#endif

typedef struct {
    DPSlot *tbl; size_t tbl_bytes;
    CandRec *cands; unsigned *head; unsigned long long *stepctr;
    uint64_t *dbg_xk; uint64_t *dbg_step; uint32_t *dbg_n;
    int blocks, threads;
    /* host-side pinned staging + a non-blocking side stream.  All host<->device
     * traffic that has to happen WHILE the kernel runs (candidate polling, step
     * counters, stop flag, DP-table checkpoint snapshot) goes through here: a
     * plain cudaMemcpy on the legacy default stream is stream-ordered behind
     * the endless kernel and would never complete. */
    cudaStream_t io;
    void *p_found,*p_dph,*p_dpi;      /* device addresses of the g_* symbols */
    void *p_cand,*p_bad;              /* v6: g_cand_pend / g_badcurve symbols */
#ifdef RHO_INVTEST
    void *p_inv,*p_invbad;
#endif
    unsigned long long *sc_pin;
    CandRec *cand_pin;
    DPSlot *htbl;                /* host mirror of the DP table (ckpt only) */
    /* RHO_DP_EMIT (STAGE 2b) */
    DPEmitRec *dplog; unsigned *dplog_head;
    DPEmitRec *dplog_pin;        /* pinned staging for drained records */
    unsigned emit_consumed;      /* host drain position in the emit ring */
} GpuRes;

/* All host<->device traffic that must happen while the kernel runs goes
 * through this non-blocking side stream (pinned staging buffers). */
static void io_copy(GpuRes *g,void *dst,const void *src,size_t n,cudaMemcpyKind kind){
    CUDAOK(cudaMemcpyAsync(dst,src,n,kind,g->io));
    CUDAOK(cudaStreamSynchronize(g->io));
}

static void gpu_init(GpuRes *g,int tblbits,uint32_t dbg_max){
    CUDAOK(cudaFree(0));
    /* properties of the CURRENT device (cudaSetDevice in multi-GPU workers) */
    int curdev=0; cudaDeviceProp pr;
    if(cudaGetDevice(&curdev)!=cudaSuccess) curdev=0;
    CUDAOK(cudaGetDeviceProperties(&pr,curdev));
    fprintf(stderr,"device %d: %s SMs=%d cap=%d.%d\n",curdev,pr.name,pr.multiProcessorCount,pr.major,pr.minor);
    g->threads=256;
    /* SM cap for time-sharing: RHO_SM_FRAC in [0.05,1.0] (default 1.0) scales
     * the launch grid to that fraction of the device's SMs, so the card can be
     * shared with another tenant (e.g. a vLLM serving lane at ~99% SM util).
     * Only the NUMBER of concurrent walk lanes changes — walk rule, jump
     * tables, DP mask and the DP table are untouched, so the search remains
     * bit-comparable to rho.c / a full-grid run (just proportionally slower). */
    double frac=1.0; const char *sf=getenv("RHO_SM_FRAC");
    if(sf&&*sf){
        frac=atof(sf);
        if(!(frac>=0.05&&frac<=1.0)){
            fprintf(stderr,"RHO_SM_FRAC=%s outside [0.05,1.0]; using 1.0\n",sf);
            frac=1.0;
        }
    }
    const int per_sm=8;                       /* ~2048 resident threads/SM     */
    g->blocks=(int)(frac*pr.multiProcessorCount*per_sm+0.5);
    if(g->blocks<1) g->blocks=1;
    g->tbl_bytes=((size_t)1<<tblbits)*sizeof(DPSlot);
    CUDAOK(cudaMalloc(&g->tbl,g->tbl_bytes));
    CUDAOK(cudaMalloc(&g->cands,NCAND*sizeof(CandRec)));
    CUDAOK(cudaMalloc(&g->head,4));
    CUDAOK(cudaMalloc(&g->stepctr,(size_t)g->blocks*g->threads*sizeof(unsigned long long)));
    CUDAOK(cudaMalloc(&g->dbg_xk,(size_t)(dbg_max?dbg_max:1)*8));
    CUDAOK(cudaMalloc(&g->dbg_step,(size_t)(dbg_max?dbg_max:1)*8));
    CUDAOK(cudaMalloc(&g->dbg_n,4));
    CUDAOK(cudaStreamCreateWithFlags(&g->io,cudaStreamNonBlocking));
    CUDAOK(cudaGetSymbolAddress(&g->p_found,g_found_flag));
    CUDAOK(cudaGetSymbolAddress(&g->p_dph,g_dphits));
    CUDAOK(cudaGetSymbolAddress(&g->p_dpi,g_dpins));
    CUDAOK(cudaGetSymbolAddress(&g->p_cand,g_cand_pend));
    CUDAOK(cudaGetSymbolAddress(&g->p_bad,g_badcurve));
#ifdef RHO_INVTEST
    CUDAOK(cudaGetSymbolAddress(&g->p_inv,g_invmis));
    CUDAOK(cudaGetSymbolAddress(&g->p_invbad,g_invbad));
#endif
    CUDAOK(cudaMallocHost((void**)&g->sc_pin,(size_t)g->blocks*g->threads*8));
    CUDAOK(cudaMallocHost((void**)&g->cand_pin,NCAND*sizeof(CandRec)));
    g->htbl=NULL;
    /* RHO_DP_EMIT (STAGE 2b): device emit ring + pinned staging */
    g->dplog=NULL; g->dplog_head=NULL; g->dplog_pin=NULL; g->emit_consumed=0;
    if(g_dp_emit){
        size_t dn=(size_t)1<<g_dplog_bits;
        CUDAOK(cudaMalloc(&g->dplog,dn*sizeof(DPEmitRec)));
        CUDAOK(cudaMalloc(&g->dplog_head,4));
        CUDAOK(cudaMallocHost((void**)&g->dplog_pin,dn*sizeof(DPEmitRec)));
        fprintf(stderr,"dp-emit: ring %u records (%zu MB), RHO_DP_EMIT=1\n",
                (unsigned)dn,dn*sizeof(DPEmitRec)>>20);
    }
    if(!g_no_htbl_mirror&&g_ckpt_path&&g_ckpt_path[0]&&g_ckpt_every>0){
        /* pinned host mirror of the DP table for the periodic checkpoint
         * snapshot (async D2H while the kernel keeps running) */
        if(cudaMallocHost((void**)&g->htbl,g->tbl_bytes)!=cudaSuccess){
            cudaGetLastError();
            fprintf(stderr,"ckpt: pinned host mirror (%zu MB) failed, using pageable\n",
                    g->tbl_bytes>>20);
            g->htbl=(DPSlot*)malloc(g->tbl_bytes);
        }
        if(!g->htbl){ fprintf(stderr,"ckpt: host table mirror alloc failed\n"); exit(1); }
    }
    fprintf(stderr,"grid: %d blocks x %d threads = %d walks of %d (%.0f%% of SMs); table %zu MB\n",
        g->blocks,g->threads,g->blocks*g->threads,pr.multiProcessorCount*per_sm,
        100.0*frac,g->tbl_bytes>>20);
}

static void copy_constants(void){
    CUDAOK(cudaMemcpyToSymbol(cK,&hK,sizeof(FeConst)));
    CUDAOK(cudaMemcpyToSymbol(c_jx,h_jx,sizeof(h_jx)));
    CUDAOK(cudaMemcpyToSymbol(c_jy,h_jy,sizeof(h_jy)));
    CUDAOK(cudaMemcpyToSymbol(c_js,h_js,sizeof(h_js)));
    CUDAOK(cudaMemcpyToSymbol(c_jt,h_jt,sizeof(h_jt)));
#if RHO_ENDO
    CUDAOK(cudaMemcpyToSymbol(c_jsm,h_jsm,sizeof(h_jsm)));
    CUDAOK(cudaMemcpyToSymbol(c_jtm,h_jtm,sizeof(h_jtm)));
#endif
}

static int gpu_run(GpuRes *g,const Cfg *C,sc *kap_out,SolveStats *st,CkptState *ck)
{
    unsigned zero=0; int zeroi=0;
    /* config N>1 (selftest/sattest): a sticky CUDA error left over from a
     * previous run in this process must surface HERE with a clear label,
     * not lazily inside a random later io_copy. */
    { cudaError_t e=cudaDeviceSynchronize();
      if(e!=cudaSuccess)
          fprintf(stderr,"gpu_run: PRE-Run sync error (leftover from a previous run in this process): %s\n",cudaGetErrorString(e)); }
    CUDAOK(cudaMemset(g->tbl,0,g->tbl_bytes));
    io_copy(g,g->head,&zero,4,cudaMemcpyHostToDevice);
    io_copy(g,g->dbg_n,&zero,4,cudaMemcpyHostToDevice);
    CUDAOK(cudaMemset(g->cands,0,NCAND*sizeof(CandRec)));
    CUDAOK(cudaMemset(g->stepctr,0,(size_t)g->blocks*g->threads*sizeof(unsigned long long)));
    io_copy(g,g->p_found,&zeroi,4,cudaMemcpyHostToDevice);
    io_copy(g,g->p_dph,&zero,4,cudaMemcpyHostToDevice);
    io_copy(g,g->p_dpi,&zero,4,cudaMemcpyHostToDevice);
    io_copy(g,g->p_cand,&zero,4,cudaMemcpyHostToDevice);   /* v6 STAGE 0a */
    io_copy(g,g->p_bad,&zero,4,cudaMemcpyHostToDevice);    /* v6 STAGE 0b */
    if(g->dplog_head){ io_copy(g,g->dplog_head,&zero,4,cudaMemcpyHostToDevice); g->emit_consumed=0; }
#ifdef RHO_INVTEST
    io_copy(g,g->p_inv,&zero,4,cudaMemcpyHostToDevice);
#endif
    copy_constants();
    double t0=now_s(), last=t0;
    size_t tblslots=(size_t)1<<C->tblbits;
    uint64_t base_steps=0,base_dh=0,base_di=0,last_tot=0;
    if(ck->enabled){
        /* resume: reload the DP table into the host mirror and upload it.
         * Walk states are NOT checkpointed on the CUDA build (they live in
         * registers), so walks restart from fresh seeds — the stored DPs stay
         * valid collision targets, so the accumulated work is kept.  The seed
         * must match (validated in ckpt_load): the stored (a,b) pairs are only
         * meaningful for the same jump table. */
        if(ckpt_load(g_ckpt_path,C,g->htbl,tblslots,NULL,0,ck)){
            base_steps=ck->cum_steps; base_dh=ck->cum_dph; base_di=ck->cum_dpi;
            CUDAOK(cudaMemcpy(g->tbl,g->htbl,g->tbl_bytes,cudaMemcpyHostToDevice));
            /* RHO_DP_EMIT resume contract (see the CPU twin below): re-emit the
             * stored DP prefix, in original order, before the kernel starts. */
            if(g_dp_emit){
                for(uint64_t i=0;i<g_emit_n;i++){
                    const DPEmitRec *r=&g_emit_log[i];
                    dp_emit_line((unsigned)RUNG,&r->x,&r->y,&r->a,&r->b);
                }
                fprintf(stderr,"dp-emit: re-emitted %llu stored records\n",
                        (unsigned long long)g_emit_n);
            }
        }
    }
    ck->next=t0+ck->every;
    int found=0; unsigned consumed=0; int first_poll=1;
    uint64_t total_cands=0;   /* v6: ring-derived candidate count (GPU st.cands) */
#ifdef CPU_STUB
    launch_stub(rho_kernel,g->blocks,g->threads,*C,g->tbl,g->cands,g->head,g->stepctr);
#else
    rho_kernel<<<g->blocks,g->threads>>>(*C,g->tbl,g->cands,g->head,g->stepctr);
#endif
    CUDAOK(cudaGetLastError());
    while(!found){
        /* v6 STAGE 0a: the kernel parks itself on a published candidate, so a
         * short first poll catches fast solves (rung-48 selftests complete in
         * milliseconds); afterwards the steady 1.5 s cadence is unchanged. */
        struct timespec ts = first_poll ? (struct timespec){0,100000000}
                                        : (struct timespec){1,500000000};
        nanosleep(&ts,0); first_poll=0;
        unsigned head=0;
        io_copy(g,&head,g->head,4,cudaMemcpyDeviceToHost);
        if(head>consumed){
            if(head-consumed>NCAND) consumed=head-NCAND;      /* ring wrapped */
            unsigned start=consumed&(unsigned)(NCAND-1), cnt=head-consumed;
            if(start+cnt<=NCAND){
                io_copy(g,g->cand_pin,g->cands+start,sizeof(CandRec)*cnt,cudaMemcpyDeviceToHost);
                for(unsigned i=0;i<cnt&&!found;i++){
                    sc kap;
                    if(resolve_candidate(&g->cand_pin[i].a1,&g->cand_pin[i].b1,
                                         &g->cand_pin[i].a2,&g->cand_pin[i].b2,
                                         g->cand_pin[i].ypar1==g->cand_pin[i].ypar2,&kap)
                                       && verify_kappa(&kap)){
                        *kap_out=kap; found=1;
                    }
                }
            }
            consumed=head;
            total_cands+=cnt;
        }
        /* ---- RHO_DP_EMIT (STAGE 2b): drain the emit ring in order ---- */
        if(g->dplog_head){
            unsigned lh=0;
            io_copy(g,&lh,g->dplog_head,4,cudaMemcpyDeviceToHost);
            unsigned dn=(unsigned)1<<C->dplog_bits;
            if(lh-g->emit_consumed>dn){
                fprintf(stderr,"dp-emit: ring overflow (%u records lost); "
                               "raise RHO_DPLOG_N or increase dpbits\n",
                        lh-g->emit_consumed-dn);
                exit(3);   /* loud failure: a silent gap would corrupt the client's ack stream */
            }
            unsigned start=g->emit_consumed&(dn-1), cnt=lh-g->emit_consumed;
            if(cnt){
                if(start+cnt<=dn){
                    io_copy(g,g->dplog_pin,g->dplog+start,sizeof(DPEmitRec)*cnt,cudaMemcpyDeviceToHost);
                    for(unsigned i=0;i<cnt;i++){
                        dp_emit_line((unsigned)RUNG,&g->dplog_pin[i].x,&g->dplog_pin[i].y,
                                     &g->dplog_pin[i].a,&g->dplog_pin[i].b);
                        emit_log_append(&g->dplog_pin[i]);
                    }
                }else{
                    unsigned c1=dn-start;
                    io_copy(g,g->dplog_pin,g->dplog+start,sizeof(DPEmitRec)*c1,cudaMemcpyDeviceToHost);
                    io_copy(g,g->dplog_pin+c1,g->dplog,sizeof(DPEmitRec)*(cnt-c1),cudaMemcpyDeviceToHost);
                    for(unsigned i=0;i<cnt;i++){
                        dp_emit_line((unsigned)RUNG,&g->dplog_pin[i].x,&g->dplog_pin[i].y,
                                     &g->dplog_pin[i].a,&g->dplog_pin[i].b);
                        emit_log_append(&g->dplog_pin[i]);
                    }
                }
                g->emit_consumed=lh;
            }
        }
        /* v6 STAGE 0a (fix in 2b): release the parked grid every poll.  Clear
         * the park request first, then write g_found_flag = 1 only when the
         * run must stop (candidate verified, or budget expiry) and 0
         * otherwise.  The old tri-state wrote 2 ("rejected, resume") on EVERY
         * poll — including polls that drained nothing — and the kernel's
         * no-pend threads exited on any nonzero flag, killing the whole grid
         * at the first park boundary after the first poll (~1.5 s into every
         * single-worker run; all longer GPU solves silently died there).
         * Parked threads now resume because the park condition itself
         * (g_cand_pend != 0) is cleared, so no "resume" value is needed.  A
         * candidate published between the head read and this release stays in
         * the ring and is picked up at the next poll (same ring semantics as
         * v5 — nothing is lost, it just parks one interval later). */
        { unsigned zc=0; io_copy(g,g->p_cand,&zc,4,cudaMemcpyHostToDevice); }
        { int fv = found?1:0; io_copy(g,g->p_found,&fv,4,cudaMemcpyHostToDevice); }
        double now=now_s();
        if(now-last>10.0){
            last=now;
            io_copy(g,g->sc_pin,g->stepctr,(size_t)g->blocks*g->threads*8,cudaMemcpyDeviceToHost);
            uint64_t tot=0; for(long i=0;i<(long)g->blocks*g->threads;i++) tot+=g->sc_pin[i];
            last_tot=tot;
            unsigned dh=0,di=0,bd=0;
            io_copy(g,&dh,g->p_dph,4,cudaMemcpyDeviceToHost);
            io_copy(g,&di,g->p_dpi,4,cudaMemcpyDeviceToHost);
            io_copy(g,&bd,g->p_bad,4,cudaMemcpyDeviceToHost);
#ifdef RHO_INVTEST
            { unsigned im=0; io_copy(g,&im,g->p_inv,4,cudaMemcpyDeviceToHost);
              fprintf(stderr,"[%0.0fs] steps=%llu rate=%.2f M/s dphits=%llu ins=%llu badcurve=%u invmis=%u\n",
                now-t0,(unsigned long long)(base_steps+tot),
                (double)(base_steps+tot)/(now-t0)/1e6,
                (unsigned long long)(base_dh+dh),(unsigned long long)(base_di+di),bd,im); }
#else
            fprintf(stderr,"[%0.0fs] steps=%llu rate=%.2f M/s dphits=%llu ins=%llu badcurve=%u\n",
                now-t0,(unsigned long long)(base_steps+tot),
                (double)(base_steps+tot)/(now-t0)/1e6,
                (unsigned long long)(base_dh+dh),(unsigned long long)(base_di+di),bd);
#endif
            fflush(stderr);
        }
        if(ck->enabled && now>=ck->next){
            ck->next=now+ck->every;
            /* snapshot the device DP table into the host mirror.  The insert
             * rate is low (dphits ~1e4 over a full run), so this whole-table
             * copy every few minutes is negligible; slots still claimed by an
             * in-flight insert (key==RSV64) are skipped. */
            io_copy(g,g->htbl,g->tbl,g->tbl_bytes,cudaMemcpyDeviceToHost);
            unsigned dh=0,di=0;
            io_copy(g,&dh,g->p_dph,4,cudaMemcpyDeviceToHost);
            io_copy(g,&di,g->p_dpi,4,cudaMemcpyDeviceToHost);
            if(ckpt_write(g_ckpt_path,C,g->htbl,tblslots,NULL,0,
                          base_steps+last_tot,base_dh+dh,base_di+di,
                          ck->cum_ms+(uint64_t)((now-t0)*1000.0)))
                fprintf(stderr,"[ckpt] %s written (steps=%llu)\n",g_ckpt_path,
                        (unsigned long long)(base_steps+last_tot));
        }
        if(g_budget_s>0 && now_s()-t0>g_budget_s){
            int one=1; io_copy(g,g->p_found,&one,4,cudaMemcpyHostToDevice);
            if(ck->enabled&&g->htbl){
                io_copy(g,g->htbl,g->tbl,g->tbl_bytes,cudaMemcpyDeviceToHost);
                ckpt_write(g_ckpt_path,C,g->htbl,tblslots,NULL,0,
                           base_steps+last_tot,base_dh,base_di,
                           ck->cum_ms+(uint64_t)((now_s()-t0)*1000.0));
            }
            break;
        }
    }
    int one=1; io_copy(g,g->p_found,&one,4,cudaMemcpyHostToDevice);
    /* RHO_DP_EMIT (STAGE 2b): final drain so the emitted stream ends exactly
     * at the last persisted insert (no lost tail before SOLVED). */
    if(g->dplog_head){
        CUDAOK(cudaDeviceSynchronize());
        unsigned lh=0;
        io_copy(g,&lh,g->dplog_head,4,cudaMemcpyDeviceToHost);
        unsigned dn=(unsigned)1<<C->dplog_bits;
        unsigned start=g->emit_consumed&(dn-1), cnt=lh-g->emit_consumed;
        if(start+cnt<=dn){
            io_copy(g,g->dplog_pin,g->dplog+start,sizeof(DPEmitRec)*cnt,cudaMemcpyDeviceToHost);
        }else{
            unsigned c1=dn-start;
            io_copy(g,g->dplog_pin,g->dplog+start,sizeof(DPEmitRec)*c1,cudaMemcpyDeviceToHost);
            io_copy(g,g->dplog_pin+c1,g->dplog,sizeof(DPEmitRec)*(cnt-c1),cudaMemcpyDeviceToHost);
        }
        for(unsigned i=0;i<cnt;i++){
            dp_emit_line((unsigned)RUNG,&g->dplog_pin[i].x,&g->dplog_pin[i].y,
                         &g->dplog_pin[i].a,&g->dplog_pin[i].b);
            emit_log_append(&g->dplog_pin[i]);
        }
        g->emit_consumed=lh;
    }
    CUDAOK(cudaDeviceSynchronize());
    { cudaError_t e=cudaGetLastError();
      if(e!=cudaSuccess)
          fprintf(stderr,"gpu_run: POST-Run error (this run's kernel did an illegal access): %s\n",cudaGetErrorString(e)); }
    double t1=now_s();
    io_copy(g,g->sc_pin,g->stepctr,(size_t)g->blocks*g->threads*8,cudaMemcpyDeviceToHost);
    uint64_t tot=0; for(long i=0;i<(long)g->blocks*g->threads;i++) tot+=g->sc_pin[i];
    unsigned dh=0,di=0,bd=0;
    io_copy(g,&dh,g->p_dph,4,cudaMemcpyDeviceToHost);
    io_copy(g,&di,g->p_dpi,4,cudaMemcpyDeviceToHost);
    io_copy(g,&bd,g->p_bad,4,cudaMemcpyDeviceToHost);
    st->steps=base_steps+tot; st->dphits=base_dh+dh; st->dpins=base_di+di;
    st->badcurve=bd; st->cands=total_cands;
#ifdef RHO_INVTEST
    { unsigned im=0; io_copy(g,&im,g->p_inv,4,cudaMemcpyDeviceToHost);
      fprintf(stderr,"INVMIS total=%u\n",im);
      if(im){
          static uint32_t recs[INVBAD_N*INVBAD_W];
          io_copy(g,recs,g->p_invbad,sizeof(recs),cudaMemcpyDeviceToHost);
          int nrec=(im<INVBAD_N)?(int)im:INVBAD_N;
          for(int r=0;r<nrec;r++){
              const uint32_t *q=recs+r*INVBAD_W;
              fprintf(stderr,"  invbad[%d] blk=%u wid=%u rnd=%u total=",r,q[2*NL+0],q[2*NL+1],q[2*NL+2]);
              for(int i=NL-1;i>=0;i--) fprintf(stderr,"%08x",q[i]);
              fprintf(stderr," invt=");
              for(int i=NL-1;i>=0;i--) fprintf(stderr,"%08x",q[NL+i]);
              fprintf(stderr,"\n");
          }
      } }
#endif
    st->secs=t1-t0+(double)ck->cum_ms/1000.0;   /* cumulative across restarts */
    st->found=found;
    return found;
}
#endif /* !CPU_REF */

/* ================================================================== */
/* Modes: fuzz / jumptest / selftest / smoke / solve                   */
/* ================================================================== */

/* ------------------------- field fuzz (CPU_REF) ------------------------- */
static void draw_fe(fe *r,uint64_t *rng){
    for(int i=0;i<NL;i++){
        uint32_t v=(uint32_t)rnd64(rng);
        int b=RUNG_BITS-32*i;
        if(b<=0) r->w[i]=0;
        else r->w[i]=(b>=32)?v:(v&((1u<<b)-1u));
    }
    while(fe_ge_p(r)) fe_sub(r,r,&KC->P_mod);
}

static void fuzz_field(uint64_t n){
    printf("FUZZ rung=%d NL=%d NW64=%d p=%s n=%s\n",RUNG,NL,NW64,P_HEX,N_HEX);
    uint64_t rng=0xF00DC0DEULL;
    fe xp,yp,xm,ym,pr,s_,d_,iv,chk;
    long ninv=0;
    for(uint64_t it=0;it<n;it++){
        draw_fe(&xp,&rng); draw_fe(&yp,&rng);
        mont_mul(&xm,&xp,&KC->R2_mont);      /* mont(x) = x*R^2/R = x*R mod p */
        mont_mul(&ym,&yp,&KC->R2_mont);
        mont_mul(&pr,&xm,&ym);               /* = mont(x*y) */
        fe_add(&s_,&xp,&yp);
        fe_sub(&d_,&xp,&yp);
        printf("M "); print_val64(xp.w); printf(" "); print_val64(yp.w); printf(" "); print_val64(pr.w); printf("\n");
        printf("A "); print_val64(xp.w); printf(" "); print_val64(yp.w); printf(" "); print_val64(s_.w); printf("\n");
        printf("S "); print_val64(xp.w); printf(" "); print_val64(yp.w); printf(" "); print_val64(d_.w); printf("\n");
        if((it&7)==0){
            fe_inv(&iv,&xm);                 /* mont(x^-1) */
            mont_mul(&chk,&xm,&iv);          /* must equal mont(1) = R1 */
            printf("I "); print_val64(xp.w); printf(" "); print_val64(iv.w); printf(" "); print_val64(chk.w); printf("\n");
            ninv++;
        }
    }
    printf("END vectors=%llu inversions=%ld\n",(unsigned long long)n,ninv);
#if defined(__SIZEOF_INT128__)
    if(NL==2){
        /* internal cross-check against unsigned __int128 (valid because the
         * Montgomery domain is exactly 64 bits wide for NW64==1 rungs) */
        uint64_t p64=((uint64_t)hK.P_mod.w[1]<<32)|hK.P_mod.w[0];
        uint64_t r164=((uint64_t)hK.R1_mont.w[1]<<32)|hK.R1_mont.w[0];
        long bad=0;
        for(uint64_t it=0;it<3000000;it++){
            uint64_t x=rnd64(&rng)%p64, y=rnd64(&rng)%p64;
            fe a,b2,r;
            a.w[0]=(uint32_t)x; a.w[1]=(uint32_t)(x>>32);
            mont_mul(&a,&a,&hK.R2_mont);                 /* mont(x) */
            b2.w[0]=(uint32_t)y; b2.w[1]=(uint32_t)(y>>32);
            mont_mul(&b2,&b2,&hK.R2_mont);               /* mont(y) */
            mont_mul(&r,&a,&b2);
            uint64_t got=low64(&r);
            unsigned __int128 xy=(__uint128_t)x*y;
            unsigned __int128 refm=((xy%p64)*(unsigned __int128)r164)%p64;
            if(got!=(uint64_t)refm){
                if(bad<5) fprintf(stderr,"MONT MISMATCH x=%016llx y=%016llx got=%016llx want=%016llx\n",
                    (unsigned long long)x,(unsigned long long)y,(unsigned long long)got,(unsigned long long)refm);
                bad++;
            }
            fe a2,b3,su,di;
            a2.w[0]=(uint32_t)x; a2.w[1]=(uint32_t)(x>>32);
            b3.w[0]=(uint32_t)y; b3.w[1]=(uint32_t)(y>>32);
            fe_add(&su,&a2,&b3); fe_sub(&di,&a2,&b3);
            unsigned __int128 refd=((unsigned __int128)x+p64-y)%p64;
            if(low64(&su)!=(x+y)%p64 || low64(&di)!=(uint64_t)refd){
                if(bad<10) fprintf(stderr,"ADDSUB MISMATCH x=%016llx y=%016llx\n",(unsigned long long)x,(unsigned long long)y);
                bad++;
            }
        }
        printf("INTERNAL fuzz (3e6 mont_mul + 3e6 add/sub): mismatches=%ld\n",bad);
    }
#endif
}

/* ----------------------------- run_search ------------------------------ */
static int run_search(int dpbits,int tblbits,uint64_t seed,int nlanes,
                      uint32_t dbg_max,sc *kap_out,SolveStats *st)
{
    Cfg C; memset(&C,0,sizeof(C));
    if(dpbits<1||dpbits>60){ fprintf(stderr,"dpbits must be in [1,60]\n"); exit(1); }
    if(tblbits<6||tblbits>30){ fprintf(stderr,"tblbits must be in [6,30]\n"); exit(1); }
    C.seed=seed; C.dpbits=(uint32_t)dpbits; C.tblbits=(uint32_t)tblbits;
    C.dpmask =(dpbits>=64)?~0ULL:(((uint64_t)1<<dpbits)-1);
    C.tblmask=((uint64_t)1<<tblbits)-1;
    C.dbg_max=dbg_max;
    uint32_t dbg_n=0; uint64_t *dx=NULL,*ds=NULL;
    if(dbg_max){ dx=(uint64_t*)calloc(dbg_max,sizeof(uint64_t)); ds=(uint64_t*)calloc(dbg_max,sizeof(uint64_t)); }
    CkptState ck; memset(&ck,0,sizeof(ck));
    ck.enabled=(g_ckpt_path&&g_ckpt_path[0]&&g_ckpt_every>0);
    ck.every=g_ckpt_every;
#ifdef CPU_REF
    C.dbg_xk=dx; C.dbg_step=ds; C.dbg_n=&dbg_n;
    DPSlot *tbl=(DPSlot*)calloc((size_t)1<<tblbits,sizeof(DPSlot));
    if(!tbl){ fprintf(stderr,"table alloc failed (%d bits)\n",tblbits); exit(1); }
    int ok=solve_host(&C,tbl,nlanes,kap_out,st,&ck);
    if(dbg_max){
        printf("DBG lane0 first %u DPs:\n",dbg_n);
        for(uint32_t i=0;i<dbg_n;i++)
            printf("  DP %u step=%llu xk=%016llx\n",i,(unsigned long long)ds[i],(unsigned long long)dx[i]);
    }
    free(tbl); free(dx); free(ds);
    return ok;
#else
    static GpuRes g; static int inited=0;
    if(!inited){ gpu_init(&g,tblbits,dbg_max); inited=1; }
    C.dbg_xk=g.dbg_xk; C.dbg_step=g.dbg_step; C.dbg_n=g.dbg_n;
    /* RHO_DP_EMIT (STAGE 2b): wire the emit ring into the device Cfg (NULL
     * ring -> emit=0 -> the kernel's emit branch never fires). */
    C.emit=(g_dp_emit&&g.dplog)?1u:0u;
    C.dplog_bits=g_dplog_bits;
    C.dplog_mask=((uint64_t)1<<g_dplog_bits)-1;
    C.dplog=g.dplog; C.dplog_head=g.dplog_head;
    int ok=gpu_run(&g,&C,kap_out,st,&ck);
    if(dbg_max){
        cudaMemcpy(&dbg_n,g.dbg_n,4,cudaMemcpyDeviceToHost);
        if(dbg_n>dbg_max) dbg_n=dbg_max;
        cudaMemcpy(dx,g.dbg_xk,(size_t)dbg_n*sizeof(uint64_t),cudaMemcpyDeviceToHost);
        cudaMemcpy(ds,g.dbg_step,(size_t)dbg_n*sizeof(uint64_t),cudaMemcpyDeviceToHost);
        printf("DBG lane0 first %u DPs:\n",(unsigned)dbg_n);
        for(uint32_t i=0;i<dbg_n;i++)
            printf("  DP %u step=%llu xk=%016llx\n",i,(unsigned long long)ds[i],(unsigned long long)dx[i]);
    }
    free(dx); free(ds);
    return ok;
#endif
}

/* ------------------------------- selftest ------------------------------ */
static void do_selftest(int nself,uint64_t seed,int dpbits,int tblbits,uint32_t dbg_max)
{
    printf("SELFTEST rung=%d: %d random-kappa DLPs on the same curve\n",RUNG,nself);
    if(RUNG_BITS>=64)
        fprintf(stderr,"note: rung %d selftest costs ~%.3g group ops per instance\n",
                RUNG,1.25*pow(2.0,(double)RUNG_BITS/2.0));
    uint64_t rng=seed^0x51CE5EEDULL;
    int allok=1;
    for(int t=0;t<nself;t++){
        sc kappa0; random_sc(&kappa0,&rng);
        jac R; scalar_mul_point(&R,&kappa0,&hK.Q_montX,&hK.Q_montY);
        fe px,py; jac_to_affine(&R,&px,&py);
        fe sPX=hK.P_montX, sPY=hK.P_montY;
        hK.P_montX=px; hK.P_montY=py;
        uint64_t ts=seed^(0xC0FFEEULL*(uint64_t)(t+1));
        build_jumptable(ts);
        sc kap; SolveStats st; memset(&st,0,sizeof(st));
        int ok=run_search(dpbits,tblbits,ts,32,dbg_max,&kap,&st);
        int pass = ok && sc_eq(&kap,&kappa0) && verify_kappa(&kap);
        printf("selftest %d: expected 0x",t); print_val64(kappa0.w);
        printf("  got 0x"); print_val64(kap.w);
        printf("  -> %s (steps=%llu, %.1fs)\n",pass?"PASS":"FAIL",
               (unsigned long long)st.steps,st.secs);
        allok&=pass;
        hK.P_montX=sPX; hK.P_montY=sPY;
    }
    printf("SELFTEST %s\n",allok?"PASSED":"FAILED");
}

/* --------------------------------- main -------------------------------- */
/* rho_multi.cu includes this file and provides its own main() with the
 * multi-worker modes (msolve / sharetest); everything above is shared. */
#ifndef RHO_MULTI
int main(int argc,char**argv){
    const char *mode=(argc>1)?argv[1]:"solve";
    const uint64_t DEFSEED=0x123456789ABCDEFull;
    /* --ckpt <file> may appear anywhere in argv; interval from
     * RHO_CKPT_EVERY_S (seconds, default 300, <=0 disables). */
    for(int i=1;i<argc;i++)
        if(!strcmp(argv[i],"--ckpt")&&i+1<argc) g_ckpt_path=argv[++i];
    { const char *ce=getenv("RHO_CKPT_EVERY_S");
      if(ce&&*ce) g_ckpt_every=atof(ce);
      if(g_ckpt_path&&g_ckpt_path[0]&&g_ckpt_every<=0)
          fprintf(stderr,"ckpt: RHO_CKPT_EVERY_S<=0 -- checkpointing disabled\n"); }
    /* RHO_DP_EMIT (STAGE 2b): off by default.  =1 -> every distinguished-point
     * insert prints one JSON line to stdout (see dp_emit_line) and is appended
     * to the checkpoint's ordered record log; a resumed run re-emits its
     * stored prefix first, so a client that acks what it has seen never sees
     * a gap or a resubmit.  RHO_DPLOG_N = log2 of the device ring capacity
     * (default 20 = 1M records; a slower client can raise it). */
    { const char *de=getenv("RHO_DP_EMIT");
      if(de&&*de) g_dp_emit=atoi(de);
      const char *dn=getenv("RHO_DPLOG_N");
      if(dn&&*dn){ long v=atol(dn); if(v>0&&v<32) g_dplog_bits=(uint32_t)v; }
      if(g_dp_emit&&(!strcmp(mode,"msolve")||!strcmp(mode,"sharetest"))){
          fprintf(stderr,"dp-emit: not supported in multi-worker modes (%s)\n",mode);
          return 1;
      }
      if(g_dp_emit)
          fprintf(stderr,"dp-emit: ON (ring 2^%u records, jsonl on stdout)\n",g_dplog_bits); }
    mont_init();
    if(!on_curve(&hK.P_montX,&hK.P_montY)||!on_curve(&hK.Q_montX,&hK.Q_montY)){
        fprintf(stderr,"FATAL: P or Q not on y^2=x^3+7 (rung %d)\n",RUNG); return 1;
    }
    { sc nn; for(int i=0;i<NL;i++) nn.w[i]=hK.N_mod.w[i];
      jac R; scalar_mul_point(&R,&nn,&hK.Q_montX,&hK.Q_montY);
      if(!fe_is_zero(&R.Z)){ fprintf(stderr,"FATAL: [n]Q != O\n"); return 1; } }
#if RHO_ENDO
    endo_init();
#endif

    if(!strcmp(mode,"fuzz")){
        uint64_t n=(argc>2)?strtoull(argv[2],NULL,10):200000;
        fuzz_field(n); return 0;
    }
    if(!strcmp(mode,"jumptest")){
        uint64_t s=(argc>2)?strtoull(argv[2],NULL,10):DEFSEED;
        build_jumptable(s);
        for(int j=0;j<NJUMPS;j++){
            fe one; fe_zero(&one); one.w[0]=1;
            fe px,py; mont_mul(&px,&h_jx[j],&one); mont_mul(&py,&h_jy[j],&one);
            printf("%d s=",j); print_val64(h_js[j].w);
            printf(" t=");      print_val64(h_jt[j].w);
            printf(" Mx=");     print_val64(px.w);
            printf(" My=");     print_val64(py.w);
            printf("\n");
        }
        { fe one; fe_zero(&one); one.w[0]=1; fe px,py;
          mont_mul(&px,&hK.Q_montX,&one); mont_mul(&py,&hK.Q_montY,&one);
          printf("Q x="); print_val64(px.w); printf(" y="); print_val64(py.w); printf("\n"); }
        return 0;
    }
    if(!strcmp(mode,"selftest")){
        int nself=(argc>2)?atoi(argv[2]):3;
        uint64_t s=(argc>3)?strtoull(argv[3],NULL,10):DEFSEED;
        int dpb=(argc>4)?atoi(argv[4]):DPBITS;
        int tb =(argc>5)?atoi(argv[5]):TBLBITS;
        uint32_t dm=(argc>6)?(uint32_t)atoi(argv[6]):0;
        g_ckpt_path=NULL;   /* selftest swaps targets/seeds: never resume */
        do_selftest(nself,s,dpb,tb,dm);
        return 0;
    }
    /* v6 STAGE 2b regression (coordinator hazard): high-dpb configs against a
     * saturated/default table used to NEVER terminate -- the structural
     * 2-cycle of the canonical map killed walks before they reached a stored
     * DP, so merged walks retraced dead paths and detection waited for
     * ~sqrt(2n/theta) steps (measured 207M vs 4.4M here at dpb=14).  The
     * escape-step fix (RHO_CYC_ESC) removed the death spiral.  PASS = the
     * config SOLVES within the time budget (all three solve in seconds when
     * healthy; pre-fix they ran forever).  NOTE a table whose capacity is far
     * below the DP count needed for detection (e.g. dpb=2/tbl=12) is
     * unsolvable by construction in ANY rho variant -- not a regression. */
    if(!strcmp(mode,"sattest")){
        /* All configs share tblbits: the CUDA build allocates the DP table
         * once per process (static GpuRes sized on the first run_search), so
         * varying tblbits within one sattest run is not supported.
         * RUNG-RELATIVE: the dpb/tbl pairs below are calibrated for rung 48,
         * where each config SOLVES in seconds when the escape fix is healthy.
         * On rung 96 no (dpb,tbl) pair solves quickly (a real solve is a
         * multi-hour campaign), so there the configs degenerate to a
         * saturated-table STORM and the regression property is "survives the
         * budget cleanly: keeps processing DP events, budget break works, no
         * CUDA error" -- a FAIL there does NOT indicate an arch divergence. */
        struct { int dpb,tb,ln; uint64_t seed; const char *name; } cfgs[]={
            { 9,14,32,777,   "dpb9-tbl14-default-saturated" },
            {12,14,32,777,   "dpb12-tbl14-default" },
            {14,14,32,777,   "dpb14-tbl14-formerly-47x-delay" },
        };
        int storm=(RUNG_BITS>=90);   /* configs can't solve quickly: storm mode */
        double budget=(argc>2)?atof(argv[2]):120.0;
        int fails=0;
        for(size_t c=0;c<sizeof(cfgs)/sizeof(cfgs[0]);c++){
            build_jumptable(cfgs[c].seed);
            sc kap; SolveStats st; memset(&st,0,sizeof(st));
            g_budget_s=budget;
            double t0=now_s();
            int ok=run_search(cfgs[c].dpb,cfgs[c].tb,cfgs[c].seed,cfgs[c].ln,0,&kap,&st);
            int pass=ok;
            const char *how="solved";
            if(!ok&&storm&&st.dphits>0){ pass=1; how="storm-survived"; }
            printf("sattest rung=%d %-28s %s(%s) steps=%llu dphits=%llu cands=%llu (%.1fs)\n",
                   RUNG,cfgs[c].name,pass?"PASS":"FAIL",how,
                   (unsigned long long)st.steps,(unsigned long long)st.dphits,
                   (unsigned long long)st.cands,now_s()-t0);
            fflush(stdout);
            if(!pass) fails++;
        }
        printf("sattest: %s (%zu/%zu configs %s within %.0fs)\n",
               fails?"FAIL":"PASS",sizeof(cfgs)/sizeof(cfgs[0])-fails,
               sizeof(cfgs)/sizeof(cfgs[0]),storm?"storm-survived":"solved",budget);
        return fails?1:0;
    }
#if RHO_ENDO
    /* v6 STAGE 2: canonicalization validation (CPU always; CUDA builds also
     * byte-compare the device endo_canon against the host). */
    if(!strcmp(mode,"endotest")){
        uint64_t n=(argc>2)?strtoull(argv[2],NULL,10):256;
        return endo_endotest(n)?0:1;
    }
#endif
#ifdef RHO_INVTEST
    /* SASS-PORT debug mode (only in -DRHO_INVTEST builds): run fe_inv_gcd and
     * fe_inv_fermat side by side ON DEVICE over random Montgomery inputs and
     * report mismatches.  Production builds never see this mode. */
    if(!strcmp(mode,"invtest")){
        int n=(argc>2)?atoi(argv[2]):65536;
        fe *hin=(fe*)malloc(sizeof(fe)*n),*hg=(fe*)malloc(sizeof(fe)*n),*hf=(fe*)malloc(sizeof(fe)*n);
        uint64_t rng=0x0DDF00DULL;
        for(int i=0;i<n;i++){
            fe xp; draw_fe(&xp,&rng);
            mont_mul(&hin[i],&xp,&hK.R2_mont);   /* random Montgomery value */
        }
        fe *din,*dg,*df;
        CUDAOK(cudaMalloc(&din,sizeof(fe)*n)); CUDAOK(cudaMalloc(&dg,sizeof(fe)*n)); CUDAOK(cudaMalloc(&df,sizeof(fe)*n));
        CUDAOK(cudaMemcpy(din,hin,sizeof(fe)*n,cudaMemcpyHostToDevice));
        invtest_kernel<<<(n+255)/256,256>>>(din,dg,df,n);
        CUDAOK(cudaGetLastError());
        CUDAOK(cudaMemcpy(hg,dg,sizeof(fe)*n,cudaMemcpyDeviceToHost));
        CUDAOK(cudaMemcpy(hf,df,sizeof(fe)*n,cudaMemcpyDeviceToHost));
        long bad=0;
        for(int i=0;i<n;i++){
            if(!fe_eq(&hg[i],&hf[i])){
                if(bad<5){
                    printf("MISMATCH i=%d in=",i); print_val64(hin[i].w);
                    printf(" gcd="); print_val64(hg[i].w);
                    printf(" fermat="); print_val64(hf[i].w); printf("\n");
                }
                bad++;
            }
        }
        printf("invtest n=%d mismatches=%ld\n",n,bad);
        free(hin); free(hg); free(hf);
        cudaFree(din); cudaFree(dg); cudaFree(df);
        return 0;
    }
    if(!strcmp(mode,"invtest2")){
        int n=1<<20, rounds=(argc>2)?atoi(argv[2]):2000;
        fe *hin=(fe*)malloc(sizeof(fe)*n),*hout=(fe*)malloc(sizeof(fe)*64);
        uint64_t rng=0x0DDF00DULL;
        for(int i=0;i<n;i++){ fe xp; draw_fe(&xp,&rng); mont_mul(&hin[i],&xp,&hK.R2_mont); }
        fe *din,*dout; unsigned *dbad,hbad=0;
        CUDAOK(cudaMalloc(&din,sizeof(fe)*n)); CUDAOK(cudaMalloc(&dout,sizeof(fe)*64)); CUDAOK(cudaMalloc(&dbad,4));
        CUDAOK(cudaMemcpy(din,hin,sizeof(fe)*n,cudaMemcpyHostToDevice));
        CUDAOK(cudaMemset(dbad,0,4));
        invtest2_kernel<<<(n*2+255)/256,256>>>(din,dout,dbad,n,rounds);
        CUDAOK(cudaGetLastError());
        CUDAOK(cudaMemcpy(&hbad,dbad,4,cudaMemcpyDeviceToHost));
        CUDAOK(cudaMemcpy(hout,dout,sizeof(fe)*64,cudaMemcpyDeviceToHost));
        printf("invtest2 rounds=%d bad=%u\n",rounds,hbad);
        for(int k=0;k<(int)(hbad<8?hbad:8);k++){
            printf("bad[%d] total=",k); print_val64(hout[k*4+0].w);
            printf(" contrib="); print_val64(hout[k*4+1].w);
            printf(" zi="); print_val64(hout[k*4+2].w);
            printf(" chk="); print_val64(hout[k*4+3].w); printf("\n");
        }
        free(hin); free(hout); cudaFree(din); cudaFree(dout); cudaFree(dbad);
        return 0;
    }
#endif
    printf("rung %d (secp%dk1): p=%s n=%s | %d x 32-bit limbs, Montgomery domain 2^%d\n",
           RUNG,RUNG_BITS,P_HEX,N_HEX,NL,32*NL);
    /* solve (default) and smoke */
    int dpb=(argc>2)?atoi(argv[2]):DPBITS;
    int tb =(argc>3)?atoi(argv[3]):TBLBITS;
    int nlanes=(argc>4)?atoi(argv[4]):32;
    uint64_t s=(argc>5)?strtoull(argv[5],NULL,10):DEFSEED;
    uint32_t dm=(argc>6)?(uint32_t)atoi(argv[6]):0;
    if(!strcmp(mode,"smoke")) g_budget_s=20.0;
    build_jumptable(s);
    printf("target P=%s,%s  Q=%s,%s | dpbits=%d tblbits=%d seed=%llu\n",
           PX_HEX,PY_HEX,QX_HEX,QY_HEX,dpb,tb,(unsigned long long)s);
    if(g_ckpt_path&&g_ckpt_path[0]&&g_ckpt_every>0)
        printf("checkpoint: %s every %.0fs (resume: rerun with the same seed and --ckpt)\n",
               g_ckpt_path,g_ckpt_every);
    sc kap; SolveStats st; memset(&st,0,sizeof(st));
    int ok=run_search(dpb,tb,s,nlanes,dm,&kap,&st);
    if(ok){
        int v=verify_kappa(&kap);
        printf("SOLVED kappa=0x"); print_val64(kap.w);
        printf(" verified=%d\n",v);
        printf("steps=%llu time=%.1fs rate=%.2f Mops/s dphits=%llu ins=%llu badcurve=%llu cands=%llu\n",
               (unsigned long long)st.steps,st.secs,
               st.secs>0?(double)st.steps/st.secs/1e6:0.0,
               (unsigned long long)st.dphits,(unsigned long long)st.dpins,
               (unsigned long long)st.badcurve,(unsigned long long)st.cands);
        return v?0:2;
    }
    fprintf(stderr,"no solution%s\n",g_budget_s>0?" within budget":"");
    printf("steps=%llu time=%.1fs rate=%.2f Mops/s dphits=%llu ins=%llu badcurve=%llu\n",
           (unsigned long long)st.steps,st.secs,
           st.secs>0?(double)st.steps/st.secs/1e6:0.0,
           (unsigned long long)st.dphits,(unsigned long long)st.dpins,
           (unsigned long long)st.badcurve);
    return 2;
}

#endif /* RHO_MULTI */
