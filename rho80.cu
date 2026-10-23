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
 * Expected cost (rho with distinguished points): ~1.25*sqrt(n) group ops.
 * Rung 80: n ~ 2^80 -> ~1.4e12 group ops; at 0.5-2 Gops/s on one Ada-class GPU
 * that is roughly 12-50 minutes (see GPU80.md for the throughput math).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

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
#else
#error "No constants for this RUNG; add a block to the rung table."
#endif

/* 32-bit limbs; NL = 2*NW64 so R = 2^(64*NW64) = 2^(32*NL), matching rho.c. */
#define NL      (2*NW64)
#define NJUMPS  32

#ifndef DPBITS
#define DPBITS  DPBITS_DEF
#endif
#ifndef TBLBITS
#define TBLBITS TBLBITS_DEF
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
} FeConst;

#if defined(__CUDACC__)
DC  FeConst cK;
DI  fe c_jx[NJUMPS], c_jy[NJUMPS];
DI  sc c_js[NJUMPS], c_jt[NJUMPS];
#endif
static FeConst hK;
static fe h_jx[NJUMPS], h_jy[NJUMPS];
static sc h_js[NJUMPS], h_jt[NJUMPS];

#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
#define KC   (&cK)
#define JX   (c_jx)
#define JY   (c_jy)
#define JS   (c_js)
#define JT   (c_jt)
#else
#define KC   (&hK)
#define JX   (h_jx)
#define JY   (h_jy)
#define JS   (h_js)
#define JT   (h_jt)
#endif

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
} Cfg;

#define GOLDEN  0x9E3779B97F4A7C15ULL
#define RSV64   0xFFFFFFFFFFFFFFFFULL   /* "claim pending" marker in DP table */

static double g_budget_s = 0;   /* >0: stop the search after this many seconds */

/* ================================================================== */
/* Field arithmetic (Montgomery, 32-bit limbs).  Direct port of the    */
/* FIXED 64-bit-limb logic in rho.c (carry/borrow chains re-derived    */
/* for 32-bit limbs; T arrays are 2*NL 32-bit limbs = same capacity    */
/* as rho.c's 2*NW 64-bit limbs).                                      */
/* ================================================================== */

static HD FI uint64_t low64(const fe *a){ return (uint64_t)a->w[0] | ((uint64_t)a->w[1]<<32); }

static HD FI int fe_is_zero(const fe *a){ uint32_t v=0; for(int i=0;i<NL;i++) v|=a->w[i]; return v==0; }
static HD FI int fe_eq(const fe *a,const fe *b){ for(int i=0;i<NL;i++) if(a->w[i]!=b->w[i]) return 0; return 1; }
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
static HD void mont_mul(fe *r,const fe *a,const fe *b){
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
static HD FI void mont_sqr(fe *r,const fe *a){ mont_mul(r,a,a); }

/* Montgomery inverse: returns mont(a^-1) = a^-1 * R mod p (Fermat, p prime).
 * Skips leading zero bits of the exponent p-2 (p is ~RUNG_BITS bits). */
static HD void fe_inv(fe *r,const fe *a){
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
    /* R1 = 2^(32NL) mod p ; R2 = R1^2 mod p */
    fe Rmod; fe_zero(&Rmod); Rmod.w[0]=1;
    for(int i=0;i<32*NL;i++) fe_add(&Rmod,&Rmod,&Rmod);
    fe_set(&hK.R1_mont,&Rmod);
    fe_set(&hK.R2_mont,&Rmod);
    for(int i=0;i<32*NL;i++) fe_add(&hK.R2_mont,&hK.R2_mont,&hK.R2_mont);
    fe Px,Py,Qx,Qy;
    parse_fe(&Px,PX_HEX); parse_fe(&Py,PY_HEX);
    parse_fe(&Qx,QX_HEX); parse_fe(&Qy,QY_HEX);
    if(fe_ge_p(&Px)||fe_ge_p(&Py)||fe_ge_p(&Qx)||fe_ge_p(&Qy)){
        fprintf(stderr,"point coordinate not reduced mod p\n"); exit(1);
    }
    mont_mul(&hK.P_montX,&Px,&hK.R2_mont); mont_mul(&hK.P_montY,&Py,&hK.R2_mont);
    mont_mul(&hK.Q_montX,&Qx,&hK.R2_mont); mont_mul(&hK.Q_montY,&Qy,&hK.R2_mont);
}

/* curve sanity: y^2 = x^3+7 for both P and Q (mont domain) */
static int on_curve(const fe *xm,const fe *ym){
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

static HD FI uint64_t hashx(uint64_t x){ return x*GOLDEN; }

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
    for(int probe=0;probe<NPROBE;probe++){
        uint64_t slot = base + (uint64_t)((h+(uint64_t)probe)&63);
        DPSlot *s = &tbl[slot];
#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
        unsigned long long k = atomicAdd((unsigned long long*)&s->key, 0ULL);
#else
        uint64_t k = s->key;
#endif
        if(k==0){
#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
            unsigned long long prev = atomicCAS((unsigned long long*)&s->key, 0ULL, RSV64);
            if(prev!=0) continue;         /* lost the claim race */
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
            if(s->xhi!=xhi) continue;     /* different point, same low-64 x */
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
                         fe *Xo, fe *Yo, fe *Zo, int *reseed_pending, uint64_t *rng)
{
    if(*reseed_pending){
        walk_reseed(x,y,a,b,rng);
        *reseed_pending=0;
        fe_set(Xo,x); fe_set(Yo,y); fe_set(Zo,&KC->R1_mont);
        return;
    }
    int j = (int)((low64(x) >> 8) & (NJUMPS-1));
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

#if defined(__CUDACC__) && !defined(CPU_REF)
/* ================================================================== */
/* CUDA kernel: one walk per thread, warp-wide batch inversion         */
/* ================================================================== */

#define ALLL 0xffffffffu

DI volatile int g_found_flag = 0;   /* host sets via cudaMemcpyToSymbol */
DI unsigned g_dphits = 0;
DI unsigned g_dpins  = 0;

DI FI fe shfl_fe(const fe &v,int src){ fe r; for(int i=0;i<NL;i++) r.w[i]=__shfl_sync(ALLL,v.w[i],src); return r; }
DI FI fe shfl_fe_up(const fe &v,int d){ fe r; for(int i=0;i<NL;i++) r.w[i]=__shfl_up_sync(ALLL,v.w[i],d); return r; }
DI FI fe shfl_fe_down(const fe &v,int d){ fe r; for(int i=0;i<NL;i++) r.w[i]=__shfl_down_sync(ALLL,v.w[i],d); return r; }

__global__ __launch_bounds__(256) void rho_kernel(const Cfg C, DPSlot *tbl, CandRec *cands,
                                                  unsigned *cand_head,
                                                  unsigned long long *stepctr)
{
    const int tid  = blockIdx.x*blockDim.x + threadIdx.x;
    const int lane = threadIdx.x & 31;
    uint64_t rng = C.seed ^ (0xABCDEF1234567890ULL*(uint64_t)(tid+1)) ^ GOLDEN;
    fe x,y; sc a,b;
    walk_reseed(&x,&y,&a,&b,&rng);
    uint64_t steps=0; int reseed_pending=0;
    for(uint64_t round=0;;round++){
        fe X,Y,Z;
        walk_step(&x,&y,&a,&b,&X,&Y,&Z,&reseed_pending,&rng);
        steps++;
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
        for(int off=1;off<32;off<<=1){                 /* inclusive suffix scan */
            t = shfl_fe_down(suff,off);
            if(lane+off<32) mont_mul(&suff,&suff,&t);
        }
        pref = (lane==0)    ? KC->R1_mont : shfl_fe_up(incl,1);   /* excl prefix */
        suff = (lane==31)   ? KC->R1_mont : shfl_fe_down(suff,1); /* excl suffix */
        mont_mul(&zi,&pref,&suff);
        mont_mul(&zi,&zi,&invt);
        pending_to_affine(&X,&Y,&zi,&x,&y);
        if(!reseed_pending){
            uint64_t xk = low64(&x);
            if((xk & C.dpmask)==0){
                atomicAdd(&g_dphits,1u);
                if(C.dbg_max && tid==0){
                    uint32_t n=*C.dbg_n;
                    if(n<C.dbg_max){ C.dbg_xk[n]=xk; C.dbg_step[n]=steps; *C.dbg_n=n+1; }
                }
                CandRec cand;
                int act = dp_check(&C,tbl,&x,&y,&a,&b,&cand);
                if(act==DP_INS) atomicAdd(&g_dpins,1u);
                else if(act==DP_RESEED) reseed_pending=1;
                else if(act==DP_CAND){
                    unsigned idx = atomicAdd(cand_head,1u);
                    cands[idx & (NCAND-1)] = cand;
                    reseed_pending=1;
                }
            }
        }
        if((round & 255)==255) stepctr[tid]=steps;
        if((round & 7)==7 && g_found_flag) return;
    }
}
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
} Lane;

typedef struct {
    uint64_t steps, dphits, dpins, cands;
    double   secs;
    int      found;
} SolveStats;

/*
 * CPU_REF solver: nlanes independent walks, each advanced one r-adding step
 * per round, batch-inverted across the lane group exactly like the kernel's
 * warp (the serial Montgomery trick yields the very same zi = Z_i^-1 values,
 * so lane 0 reproduces the kernel's lane 0 walk bit-for-bit).
 */
static int solve_host(const Cfg *C, DPSlot *tbl, int nlanes, sc *kap_out, SolveStats *st)
{
    Lane *L=(Lane*)calloc((size_t)nlanes,sizeof(Lane));
    fe *Xa=(fe*)malloc(sizeof(fe)*nlanes), *Ya=(fe*)malloc(sizeof(fe)*nlanes);
    fe *Za=(fe*)malloc(sizeof(fe)*nlanes), *pre=(fe*)malloc(sizeof(fe)*nlanes);
    fe *suf=(fe*)malloc(sizeof(fe)*nlanes);
    for(int i=0;i<nlanes;i++){
        L[i].rng = C->seed ^ (0xABCDEF1234567890ULL*(uint64_t)(i+1)) ^ GOLDEN;
        walk_reseed(&L[i].x,&L[i].y,&L[i].a,&L[i].b,&L[i].rng);
    }
    if(getenv("WALKDBG") && nlanes>0){
        printf("init walk0: x="); print_val64(L[0].x.w);
        printf("\n y="); print_val64(L[0].y.w);
        printf("\n a="); print_val64(L[0].a.w);
        printf("\n b="); print_val64(L[0].b.w);
        printf("\n"); fflush(stdout);
    }
    uint64_t gsteps=0,dph=0,dpi=0,dpc=0;
    double t0=now_s(), last=t0;
    int found=0;
    for(uint64_t round=0;!found;round++){
        for(int i=0;i<nlanes;i++)
            walk_step(&L[i].x,&L[i].y,&L[i].a,&L[i].b,&Xa[i],&Ya[i],&Za[i],
                      &L[i].reseed_pending,&L[i].rng);
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
        /* DP phase, in lane order (kernel order) */
        for(int i=0;i<nlanes;i++){
            if(L[i].reseed_pending) continue;
            uint64_t xk=low64(&L[i].x);
            if((xk & C->dpmask)==0){
                dph++;
                if(i==0 && C->dbg_max){
                    uint32_t n=*C->dbg_n;
                    if(n<C->dbg_max){ C->dbg_xk[n]=xk; C->dbg_step[n]=L[i].steps+1; *C->dbg_n=n+1; }
                }
                CandRec cand;
                int act=dp_check(C,tbl,&L[i].x,&L[i].y,&L[i].a,&L[i].b,&cand);
                if(act==DP_INS) dpi++;
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
        for(int i=0;i<nlanes;i++) L[i].steps++;
        gsteps += (uint64_t)nlanes;
        double now=now_s();
        if(now-last>10.0){
            last=now;
            fprintf(stderr,"[%0.0fs] steps=%llu rate=%.2f M/s dphits=%llu ins=%llu cand=%llu\n",
                now-t0,(unsigned long long)gsteps,gsteps/(now-t0)/1e6,
                (unsigned long long)dph,(unsigned long long)dpi,(unsigned long long)dpc);
            fflush(stderr);
        }
        if(g_budget_s>0 && now_s()-t0>g_budget_s) break;
    }
    st->steps=gsteps; st->dphits=dph; st->dpins=dpi; st->cands=dpc; st->secs=now_s()-t0;
    st->found=found;
    free(L); free(Xa); free(Ya); free(Za); free(pre); free(suf);
    return found;
}

#ifndef CPU_REF
/* ------------------------------ CUDA backend ------------------------------ */
#define CUDAOK(x) do{ cudaError_t _e=(x); if(_e!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s @%d\n",cudaGetErrorString(_e),__LINE__); exit(1);} }while(0)

typedef struct {
    DPSlot *tbl; size_t tbl_bytes;
    CandRec *cands; unsigned *head; unsigned long long *stepctr;
    uint64_t *dbg_xk; uint64_t *dbg_step; uint32_t *dbg_n;
    int blocks, threads;
} GpuRes;

static void gpu_init(GpuRes *g,int tblbits,uint32_t dbg_max){
    CUDAOK(cudaFree(0));
    cudaDeviceProp pr; CUDAOK(cudaGetDeviceProperties(&pr,0));
    fprintf(stderr,"device: %s SMs=%d cap=%d.%d\n",pr.name,pr.multiProcessorCount,pr.major,pr.minor);
    g->threads=256;
    g->blocks=pr.multiProcessorCount*8;          /* ~2048 resident threads/SM */
    g->tbl_bytes=((size_t)1<<tblbits)*sizeof(DPSlot);
    CUDAOK(cudaMalloc(&g->tbl,g->tbl_bytes));
    CUDAOK(cudaMalloc(&g->cands,NCAND*sizeof(CandRec)));
    CUDAOK(cudaMalloc(&g->head,4));
    CUDAOK(cudaMalloc(&g->stepctr,(size_t)g->blocks*g->threads*sizeof(unsigned long long)));
    CUDAOK(cudaMalloc(&g->dbg_xk,(size_t)(dbg_max?dbg_max:1)*8));
    CUDAOK(cudaMalloc(&g->dbg_step,(size_t)(dbg_max?dbg_max:1)*8));
    CUDAOK(cudaMalloc(&g->dbg_n,4));
    fprintf(stderr,"grid: %d blocks x %d threads = %d walks; table %zu MB\n",
        g->blocks,g->threads,g->blocks*g->threads,g->tbl_bytes>>20);
}

static void copy_constants(void){
    CUDAOK(cudaMemcpyToSymbol(cK,&hK,sizeof(FeConst)));
    CUDAOK(cudaMemcpyToSymbol(c_jx,h_jx,sizeof(h_jx)));
    CUDAOK(cudaMemcpyToSymbol(c_jy,h_jy,sizeof(h_jy)));
    CUDAOK(cudaMemcpyToSymbol(c_js,h_js,sizeof(h_js)));
    CUDAOK(cudaMemcpyToSymbol(c_jt,h_jt,sizeof(h_jt)));
}

static int gpu_run(GpuRes *g,const Cfg *C,sc *kap_out,SolveStats *st)
{
    unsigned zero=0; int zeroi=0;
    CUDAOK(cudaMemset(g->tbl,0,g->tbl_bytes));
    CUDAOK(cudaMemcpy(g->head,&zero,4,cudaMemcpyHostToDevice));
    CUDAOK(cudaMemcpy(g->dbg_n,&zero,4,cudaMemcpyHostToDevice));
    CUDAOK(cudaMemset(g->cands,0,NCAND*sizeof(CandRec)));
    CUDAOK(cudaMemset(g->stepctr,0,(size_t)g->blocks*g->threads*sizeof(unsigned long long)));
    CUDAOK(cudaMemcpyToSymbol(g_found_flag,&zeroi,sizeof(int)));
    CUDAOK(cudaMemcpyToSymbol(g_dphits,&zero,sizeof(unsigned)));
    CUDAOK(cudaMemcpyToSymbol(g_dpins,&zero,sizeof(unsigned)));
    copy_constants();
    double t0=now_s(), last=t0;
    int found=0; unsigned consumed=0;
#ifdef CPU_STUB
    launch_stub(rho_kernel,g->blocks,g->threads,*C,g->tbl,g->cands,g->head,g->stepctr);
#else
    rho_kernel<<<g->blocks,g->threads>>>(*C,g->tbl,g->cands,g->head,g->stepctr);
#endif
    CUDAOK(cudaGetLastError());
    while(!found){
        struct timespec ts={1,500000000}; nanosleep(&ts,0);   /* 1.5 s */
        unsigned head=0; CUDAOK(cudaMemcpy(&head,g->head,4,cudaMemcpyDeviceToHost));
        if(head>consumed){
            if(head-consumed>NCAND) consumed=head-NCAND;      /* ring wrapped */
            unsigned start=consumed&(unsigned)(NCAND-1), cnt=head-consumed;
            if(start+cnt<=NCAND){
                CandRec *tmp=(CandRec*)malloc(sizeof(CandRec)*cnt);
                CUDAOK(cudaMemcpy(tmp,g->cands+start,sizeof(CandRec)*cnt,cudaMemcpyDeviceToHost));
                for(unsigned i=0;i<cnt&&!found;i++){
                    sc kap;
                    if(resolve_candidate(&tmp[i].a1,&tmp[i].b1,&tmp[i].a2,&tmp[i].b2,
                                         tmp[i].ypar1==tmp[i].ypar2,&kap) && verify_kappa(&kap)){
                        *kap_out=kap; found=1;
                    }
                }
                free(tmp);
            }
            consumed=head;
        }
        double now=now_s();
        if(now-last>10.0){
            last=now;
            unsigned long long *sc_buf=(unsigned long long*)malloc((size_t)g->blocks*g->threads*8);
            CUDAOK(cudaMemcpy(sc_buf,g->stepctr,(size_t)g->blocks*g->threads*8,cudaMemcpyDeviceToHost));
            uint64_t tot=0; for(long i=0;i<(long)g->blocks*g->threads;i++) tot+=sc_buf[i];
            free(sc_buf);
            unsigned dh=0,di=0;
            cudaMemcpyFromSymbol(&dh,g_dphits,sizeof(unsigned));
            cudaMemcpyFromSymbol(&di,g_dpins,sizeof(unsigned));
            fprintf(stderr,"[%0.0fs] steps=%llu rate=%.2f M/s dphits=%u ins=%u\n",
                now-t0,(unsigned long long)tot,tot/(now-t0)/1e6,dh,di);
            fflush(stderr);
        }
        if(g_budget_s>0 && now_s()-t0>g_budget_s){
            int one=1; CUDAOK(cudaMemcpyToSymbol(g_found_flag,&one,sizeof(int)));
            break;
        }
    }
    int one=1; CUDAOK(cudaMemcpyToSymbol(g_found_flag,&one,sizeof(int)));
    CUDAOK(cudaDeviceSynchronize());
    double t1=now_s();
    unsigned long long *sc_buf=(unsigned long long*)malloc((size_t)g->blocks*g->threads*8);
    CUDAOK(cudaMemcpy(sc_buf,g->stepctr,(size_t)g->blocks*g->threads*8,cudaMemcpyDeviceToHost));
    uint64_t tot=0; for(long i=0;i<(long)g->blocks*g->threads;i++) tot+=sc_buf[i];
    free(sc_buf);
    unsigned dh=0,di=0;
    cudaMemcpyFromSymbol(&dh,g_dphits,sizeof(unsigned));
    cudaMemcpyFromSymbol(&di,g_dpins,sizeof(unsigned));
    st->steps=tot; st->dphits=dh; st->dpins=di; st->secs=t1-t0; st->found=found;
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
#ifdef CPU_REF
    C.dbg_xk=dx; C.dbg_step=ds; C.dbg_n=&dbg_n;
    DPSlot *tbl=(DPSlot*)calloc((size_t)1<<tblbits,sizeof(DPSlot));
    if(!tbl){ fprintf(stderr,"table alloc failed (%d bits)\n",tblbits); exit(1); }
    int ok=solve_host(&C,tbl,nlanes,kap_out,st);
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
    int ok=gpu_run(&g,&C,kap_out,st);
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
int main(int argc,char**argv){
    const char *mode=(argc>1)?argv[1]:"solve";
    const uint64_t DEFSEED=0x123456789ABCDEFull;
    mont_init();
    if(!on_curve(&hK.P_montX,&hK.P_montY)||!on_curve(&hK.Q_montX,&hK.Q_montY)){
        fprintf(stderr,"FATAL: P or Q not on y^2=x^3+7 (rung %d)\n",RUNG); return 1;
    }
    { sc nn; for(int i=0;i<NL;i++) nn.w[i]=hK.N_mod.w[i];
      jac R; scalar_mul_point(&R,&nn,&hK.Q_montX,&hK.Q_montY);
      if(!fe_is_zero(&R.Z)){ fprintf(stderr,"FATAL: [n]Q != O\n"); return 1; } }

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
        do_selftest(nself,s,dpb,tb,dm);
        return 0;
    }
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
    sc kap; SolveStats st; memset(&st,0,sizeof(st));
    int ok=run_search(dpb,tb,s,nlanes,dm,&kap,&st);
    if(ok){
        int v=verify_kappa(&kap);
        printf("SOLVED kappa=0x"); print_val64(kap.w);
        printf(" verified=%d\n",v);
        printf("steps=%llu time=%.1fs rate=%.2f Mops/s dphits=%llu ins=%llu\n",
               (unsigned long long)st.steps,st.secs,
               st.secs>0?(double)st.steps/st.secs/1e6:0.0,
               (unsigned long long)st.dphits,(unsigned long long)st.dpins);
        return v?0:2;
    }
    fprintf(stderr,"no solution%s\n",g_budget_s>0?" within budget":"");
    printf("steps=%llu time=%.1fs rate=%.2f Mops/s dphits=%llu ins=%llu\n",
           (unsigned long long)st.steps,st.secs,
           st.secs>0?(double)st.steps/st.secs/1e6:0.0,
           (unsigned long long)st.dphits,(unsigned long long)st.dpins);
    return 2;
}
