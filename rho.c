/*
 * Parallel Pollard rho with distinguished points for
 *   E/F_p : y^2 = x^3 + 7     (short Weierstrass, A=0, B=7)
 * solving  P = [kappa] * Q  over the prime-order group <Q>, ord(Q) = n.
 * Full-range DLP (no interval assumption).
 *
 * Field arithmetic: Montgomery, NW 64-bit limbs (NW=1: p<2^64, NW=2: p<2^128).
 * Walk: r-adding walk, jumps M_j = [s_j]Q + [t_j]P, index from low bits of x.
 * Each thread runs W independent walks in lockstep; projective denominators
 * are batch-inverted (Montgomery's trick) once per round.
 * Collisions via distinguished points (low bits of x zero), shared sharded table.
 *
 * Build:  cc -O3 -march=native -fopenmp -DNW=1 -o rho1 rho.c
 *         cc -O3 -march=native -fopenmp -DNW=2 -o rho2 rho.c
 *
 * Usage:
 *   ./rhoX <p> <n> <Px> <Py> <Qx> <Qy> <dpbits> <jumps> <walks> <threads> <tblbits> [seed]
 * Px..Qy are plain (non-Montgomery) affine hex, big-endian.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdatomic.h>
#include <omp.h>

#ifndef NW
#define NW 1
#endif

typedef struct { uint64_t w[NW]; } fe;
typedef struct { uint64_t w[NW]; } sc;

static fe P_mod, P_neg, N_neg, R1_mont, R2_mont;
static fe P_montX, P_montY, Q_montX, Q_montY;
static sc N_mod;
static uint64_t g_inv;

static inline uint64_t addw(uint64_t a,uint64_t b,uint64_t *c){ __uint128_t t=(__uint128_t)a+b; *c=(uint64_t)(t>>64); return (uint64_t)t; }
static inline uint64_t subw(uint64_t a,uint64_t b,uint64_t *bo){ __uint128_t t=(__uint128_t)a-b; *bo=(uint64_t)((t>>64)&1); return (uint64_t)t; }

static inline int fe_is_zero(const fe *a){ uint64_t v=0; for(int i=0;i<NW;i++) v|=a->w[i]; return v==0; }
static inline int fe_eq(const fe *a,const fe *b){ for(int i=0;i<NW;i++) if(a->w[i]!=b->w[i]) return 0; return 1; }
static inline void fe_set(fe *r,const fe *a){ for(int i=0;i<NW;i++) r->w[i]=a->w[i]; }
static inline void fe_zero(fe *r){ for(int i=0;i<NW;i++) r->w[i]=0; }
static inline int fe_ge_p(const fe *a){ for(int i=NW-1;i>=0;i--){ if(a->w[i]>P_mod.w[i]) return 1; if(a->w[i]<P_mod.w[i]) return 0; } return 1; }

/* r = a+b mod p ; a,b < p */
static inline void fe_add(fe *r,const fe *a,const fe *b){
    uint64_t c=0;
    for(int i=0;i<NW;i++){ __uint128_t t=(__uint128_t)a->w[i]+b->w[i]+c; r->w[i]=(uint64_t)t; c=(uint64_t)(t>>64); }
    if(c){ /* sum >= 2^(64NW): sum-p = rep + (2^(64NW)-p) */
        uint64_t cc=0; fe t;
        for(int i=0;i<NW;i++){ __uint128_t s2=(__uint128_t)r->w[i]+P_neg.w[i]+cc; t.w[i]=(uint64_t)s2; cc=(uint64_t)(s2>>64); }
        fe_set(r,&t);
        return;
    }
    if(fe_ge_p(r)){ uint64_t borrow=0; fe t;
        for(int i=0;i<NW;i++){ __uint128_t s2=(__uint128_t)r->w[i]-P_mod.w[i]-borrow; t.w[i]=(uint64_t)s2; borrow=(uint64_t)((s2>>64)&1); }
        fe_set(r,&t); }
}
/* r = a-b mod p ; a,b < p */
static inline void fe_sub(fe *r,const fe *a,const fe *b){
    uint64_t borrow=0; fe t;
    for(int i=0;i<NW;i++){ __uint128_t t2=(__uint128_t)a->w[i]-b->w[i]-borrow; t.w[i]=(uint64_t)t2; borrow=(uint64_t)((t2>>64)&1); }
    if(borrow){ uint64_t c=0;
        for(int i=0;i<NW;i++){ __uint128_t s2=(__uint128_t)t.w[i]+P_mod.w[i]+c; r->w[i]=(uint64_t)s2; c=(uint64_t)(s2>>64); }
    }
    else fe_set(r,&t);
}

static inline void mont_mul(fe *r,const fe *a,const fe *b){
#if NW==1
    __uint128_t t=(__uint128_t)a->w[0]*b->w[0];
    uint64_t m=(uint64_t)t*g_inv;
    __uint128_t mp=(__uint128_t)m*P_mod.w[0];
    t+=mp;
    uint64_t wrapped = (t<mp);
    uint64_t res=(uint64_t)(t>>64);
    if(wrapped){ /* true (t+mp)>>64 = res + 2^64 ; mod p: res + (2^64 mod p) */
        __uint128_t rr=(__uint128_t)res + R1_mont.w[0];
        if(rr>=P_mod.w[0]) rr-=P_mod.w[0];
        res=(uint64_t)rr;
    }
    if(res>=P_mod.w[0]) res-=P_mod.w[0];
    r->w[0]=res;
#else
    uint64_t T[2*NW]; memset(T,0,sizeof(T));
    for(int i=0;i<NW;i++){
        uint64_t carry=0;
        for(int j=0;j<NW;j++){
            __uint128_t cur=(__uint128_t)a->w[i]*b->w[j]+T[i+j]+carry;
            T[i+j]=(uint64_t)cur; carry=(uint64_t)(cur>>64);
        }
        int k=i+NW;
        while(carry&&k<2*NW){ __uint128_t cur=(__uint128_t)T[k]+carry; T[k]=(uint64_t)cur; carry=(uint64_t)(cur>>64); k++; }
    }
    for(int i=0;i<NW;i++){
        uint64_t m=T[i]*g_inv, carry=0;
        for(int j=0;j<NW;j++){
            __uint128_t cur=(__uint128_t)m*P_mod.w[j]+T[i+j]+carry;
            T[i+j]=(uint64_t)cur; carry=(uint64_t)(cur>>64);
        }
        int k=i+NW;
        while(carry&&k<2*NW){ __uint128_t cur=(__uint128_t)T[k]+carry; T[k]=(uint64_t)cur; carry=(uint64_t)(cur>>64); k++; }
    }
    fe res; for(int i=0;i<NW;i++) res.w[i]=T[NW+i];
    for(int s=0;s<2;s++){
        if(!fe_ge_p(&res)) break;
        uint64_t borrow=0; fe t2;
        for(int i=0;i<NW;i++) t2.w[i]=subw(res.w[i],P_mod.w[i],&borrow);
        if(!borrow) fe_set(&res,&t2);
    }
    fe_set(r,&res);
#endif
}
static inline void mont_sqr(fe *r,const fe *a){ mont_mul(r,a,a); }

/* Montgomery inverse: returns a^{-1}*R mod p */
static void fe_inv(fe *r,const fe *a){
    fe e; fe_sub(&e,&P_mod,&P_mod); fe_zero(&e); e.w[0]=2; fe_sub(&e,&P_mod,&e); /* e = p-2 */
    fe base,result; fe_set(&base,a); fe_set(&result,&R1_mont);
    for(int i=NW*64-1;i>=0;i--){
        mont_sqr(&result,&result);
        if((e.w[i/64]>>(i%64))&1) mont_mul(&result,&result,&base);
    }
    fe_set(r,&result);
}

static inline int sc_is_zero(const sc *a){ uint64_t v=0; for(int i=0;i<NW;i++) v|=a->w[i]; return v==0; }
static inline int sc_eq(const sc *a,const sc *b){ for(int i=0;i<NW;i++) if(a->w[i]!=b->w[i]) return 0; return 1; }
static inline void sc_set(sc *r,const sc *a){ for(int i=0;i<NW;i++) r->w[i]=a->w[i]; }
static inline void sc_zero(sc *r){ for(int i=0;i<NW;i++) r->w[i]=0; }
static inline int sc_ge_n(const sc *a){ for(int i=NW-1;i>=0;i--){ if(a->w[i]>N_mod.w[i]) return 1; if(a->w[i]<N_mod.w[i]) return 0; } return 1; }

static inline void sc_subn_(sc *r);
static inline void sc_add_n(sc *r,const sc *a,const sc *b){
    uint64_t c=0;
    for(int i=0;i<NW;i++){ __uint128_t t=(__uint128_t)a->w[i]+b->w[i]+c; r->w[i]=(uint64_t)t; c=(uint64_t)(t>>64); }
    if(c){ uint64_t cc=0; sc t;
        for(int i=0;i<NW;i++){ __uint128_t s2=(__uint128_t)r->w[i]+N_neg.w[i]+cc; t.w[i]=(uint64_t)s2; cc=(uint64_t)(s2>>64); }
        sc_set(r,&t); return; }
    if(sc_ge_n(r)) sc_subn_(r);
}
/* helper: r -= n (r >= n known) */
static inline void sc_subn_(sc *r){ uint64_t borrow=0; sc t;
    for(int i=0;i<NW;i++){ __uint128_t t2=(__uint128_t)r->w[i]-N_mod.w[i]-borrow; t.w[i]=(uint64_t)t2; borrow=(uint64_t)((t2>>64)&1); }
    sc_set(r,&t); }
static inline void sc_sub_n(sc *r,const sc *a,const sc *b){
    uint64_t borrow=0; sc t;
    for(int i=0;i<NW;i++){ __uint128_t t2=(__uint128_t)a->w[i]-b->w[i]-borrow; t.w[i]=(uint64_t)t2; borrow=(uint64_t)((t2>>64)&1); }
    if(borrow){ uint64_t c=0;
        for(int i=0;i<NW;i++){ __uint128_t s2=(__uint128_t)t.w[i]+N_mod.w[i]+c; r->w[i]=(uint64_t)s2; c=(uint64_t)(s2>>64); }
    }
    else sc_set(r,&t);
}
static void sc_mul(sc *r,const sc *a,const sc *b){ /* product mod n, slow path */
    uint64_t T[2*NW]; memset(T,0,sizeof(T));
    for(int i=0;i<NW;i++){
        uint64_t carry=0;
        for(int j=0;j<NW;j++){
            __uint128_t cur=(__uint128_t)a->w[i]*b->w[j]+T[i+j]+carry;
            T[i+j]=(uint64_t)cur; carry=(uint64_t)(cur>>64);
        }
        int k=i+NW; while(carry&&k<2*NW){ __uint128_t cur=(__uint128_t)T[k]+carry; T[k]=(uint64_t)cur; carry=(uint64_t)(cur>>64); k++; }
    }
    sc res; sc_zero(&res);
    for(int bit=2*NW*64-1;bit>=0;bit--){
        uint64_t c=0;
        for(int i=0;i<NW;i++){ uint64_t nc=res.w[i]>>63; res.w[i]=(res.w[i]<<1)|c; c=nc; }
        if((T[bit/64]>>(bit%64))&1){ uint64_t cc=0; res.w[0]=addw(res.w[0],1,&cc); }
        if(c){ /* res wrapped: value = res + 2^(64NW); value-n = res + (2^(64NW)-n) */
            uint64_t cc=0; sc t2;
            for(int i=0;i<NW;i++){ __uint128_t s2=(__uint128_t)res.w[i]+N_neg.w[i]+cc; t2.w[i]=(uint64_t)s2; cc=(uint64_t)(s2>>64); }
            sc_set(&res,&t2);
        } else if(sc_ge_n(&res)) sc_subn_(&res);
    }
    sc_set(r,&res);
}
static void sc_pow(sc *r,const sc *base_,const sc *e){ /* r = base^e mod n via sc_mul */
    sc result,base,e_ ;
    sc_zero(&result); result.w[0]=1;
    sc_set(&base,base_); sc_set(&e_,e);
    for(int bit=NW*64-1;bit>=0;bit--){
        if((e_.w[bit/64]>>(bit%64))&1){ sc t; sc_mul(&t,&result,&base); sc_set(&result,&t); }
        if(bit){ sc t; sc_mul(&t,&result,&result); sc_set(&result,&t); }
    }
    sc_set(r,&result);
}
static void sc_inv(sc *r,const sc *a){ /* binary ext gcd mod n */
    sc u,v,x1,x2,t;
    sc_set(&u,a); sc_set(&v,&N_mod);
    sc_zero(&x1); sc_zero(&x2); x1.w[0]=1;
    while(!sc_is_zero(&v)){
        while(!(u.w[0]&1)){
            uint64_t carry=0;
            for(int i=NW-1;i>=0;i--){ uint64_t nc=u.w[i]&1; u.w[i]=(u.w[i]>>1)|(carry<<63); carry=nc; }
            if(x1.w[0]&1) sc_add_n(&x1,&x1,&N_mod);
            uint64_t c2=0;
            for(int i=NW-1;i>=0;i--){ uint64_t nc=x1.w[i]&1; x1.w[i]=(x1.w[i]>>1)|(c2<<63); c2=nc; }
        }
        while(!(v.w[0]&1)){
            uint64_t carry=0;
            for(int i=NW-1;i>=0;i--){ uint64_t nc=v.w[i]&1; v.w[i]=(v.w[i]>>1)|(carry<<63); carry=nc; }
            if(x2.w[0]&1) sc_add_n(&x2,&x2,&N_mod);
            uint64_t c2=0;
            for(int i=NW-1;i>=0;i--){ uint64_t nc=x2.w[i]&1; x2.w[i]=(x2.w[i]>>1)|(c2<<63); c2=nc; }
        }
        int uge=0;
        for(int i=NW-1;i>=0;i--){ if(u.w[i]>v.w[i]){uge=1;break;} if(u.w[i]<v.w[i]){uge=0;break;} }
        if(uge){ uint64_t borrow=0;
                 for(int i=0;i<NW;i++){ __uint128_t s2=(__uint128_t)u.w[i]-v.w[i]-borrow; t.w[i]=(uint64_t)s2; borrow=(uint64_t)((s2>>64)&1); }
                 sc_set(&u,&t); sc_sub_n(&t,&x1,&x2); sc_set(&x1,&t); }
        else   { uint64_t borrow=0;
                 for(int i=0;i<NW;i++){ __uint128_t s2=(__uint128_t)v.w[i]-u.w[i]-borrow; t.w[i]=(uint64_t)s2; borrow=(uint64_t)((s2>>64)&1); }
                 sc_set(&v,&t); sc_sub_n(&t,&x2,&x1); sc_set(&x2,&t); }
    }
    sc_set(r,&x1);
}

typedef struct { fe X,Y,Z; } jac;
static inline void jac_set_inf(jac *R){ fe_zero(&R->X); fe_set(&R->Y,&R1_mont); fe_zero(&R->Z); }

static void jac_dbl(jac *R,const jac *P1){
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
static void jac_madd(jac *R,const jac *P1,const fe *x2,const fe *y2){
    if(fe_is_zero(&P1->Z)){ R->X=*x2; R->Y=*y2; fe_set(&R->Z,&R1_mont); return; }
    fe Z1Z1,U2,S2,H,HH,HHH,V,rr,X3,Y3,Z3,t,u;
    mont_sqr(&Z1Z1,&P1->Z);
    mont_mul(&U2,x2,&Z1Z1);
    mont_mul(&S2,y2,&Z1Z1); mont_mul(&S2,&S2,&P1->Z);
    fe_sub(&H,&U2,&P1->X);
    if(fe_is_zero(&H)){
        fe neg; fe_sub(&neg,&P_mod,y2);
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
static void jac_add(jac *R,const jac *A,const jac *B){
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
static void scalar_mul_point(jac *R,const sc *k,const fe *xa,const fe *ya){
    jac_set_inf(R);
    jac base; base.X=*xa; base.Y=*ya; fe_set(&base.Z,&R1_mont);
    for(int bit=NW*64-1;bit>=0;bit--){
        jac d; jac_dbl(&d,R);
        if((k->w[bit/64]>>(bit%64))&1){ jac t; jac_add(&t,&d,&base); d=t; }
        *R=d;
    }
}
/* Jacobian (mont domain) -> affine (mont domain): x = X*Z^-2, y = Y*Z^-3 */
static void jac_to_affine(const jac *J,fe *xa,fe *ya){
    fe Zinv; fe_inv(&Zinv,&J->Z);
    fe z2,z3;
    mont_mul(&z2,&Zinv,&Zinv);
    mont_mul(&z3,&z2,&Zinv);
    mont_mul(xa,&J->X,&z2);
    mont_mul(ya,&J->Y,&z3);
}
/* verify [k]Q == P (both affine mont) */
static int verify_kappa(const sc *k){
    jac R; scalar_mul_point(&R,k,&Q_montX,&Q_montY);
    if(fe_is_zero(&R.Z)) return 0;
    fe x,y; jac_to_affine(&R,&x,&y);
    return fe_eq(&x,&P_montX) && fe_eq(&y,&P_montY);
}

static void parse_fe(fe *r,const char *s){
    fe_zero(r);
    int len=(int)strlen(s);
    for(int i=0;i<len;i++){
        char c=s[i]; int v=(c>='0'&&c<='9')?c-'0':((c>='a'&&c<='f')?c-'a'+10:c-'A'+10);
        uint64_t carry=(uint64_t)v;
        for(int w=0;w<NW;w++){ uint64_t nc=r->w[w]>>60; r->w[w]=(r->w[w]<<4)|carry; carry=nc; }
    }
}
static void print_sc(const sc *a){ for(int i=NW-1;i>=0;i--) fprintf(stderr,"%016llx",(unsigned long long)a->w[i]); fprintf(stderr,"\n"); }

/* DP table: key = mont-form affine x (64-bit) */
typedef struct { uint64_t x; uint64_t a[NW]; uint64_t b[NW]; uint64_t ypar; } dpent;
static dpent *tbl; static uint64_t tblmask;
static omp_lock_t *locks; static int nlocks=1024; static int shardbits=6;
static inline uint64_t hashx(uint64_t x){ return x*0x9E3779B97F4A7C15ull; }

static atomic_int g_found;
static sc g_kap;
static atomic_ulong g_steps;
static atomic_int g_dbg;
static atomic_ulong g_dphits, g_inserts, g_exact, g_negcase, g_drops;

static inline uint64_t rnd64(uint64_t *s){ uint64_t z=(*s+=0x9E3779B97F4A7C15ull); z=(z^(z>>30))*0xBF58476D1CE4E5B9ull; z=(z^(z>>27))*0x94D049BB133111EBull; return z^(z>>31); }

static fe *jx,*jy; static sc *js,*jt; static int njumps;

typedef struct { fe x,y; sc a,b; } walkpt;

static void random_sc(sc *r,uint64_t *rng){
    /* NW random words, then binary reduction mod n (NW*64 shift-subtract steps) */
    for(int i=0;i<NW;i++) r->w[i]=rnd64(rng);
    sc res; sc_zero(&res);
    for(int bit=NW*64-1;bit>=0;bit--){
        uint64_t c=0;
        for(int i=0;i<NW;i++){ uint64_t nc=res.w[i]>>63; res.w[i]=(res.w[i]<<1)|c; c=nc; }
        if((r->w[bit/64]>>(bit%64))&1){ uint64_t cc=0; res.w[0]=addw(res.w[0],1,&cc); }
        if(sc_ge_n(&res)) sc_subn_(&res);
    }
    sc_set(r,&res);
    if(sc_is_zero(r)) r->w[0]=1;
}

/* re-randomize walk i from a random start */
static void reseed_walk(walkpt *Wk,uint64_t *rng){
    sc a0,b0; random_sc(&a0,rng); random_sc(&b0,rng);
    jac A,B,S;
    scalar_mul_point(&A,&a0,&Q_montX,&Q_montY);
    scalar_mul_point(&B,&b0,&P_montX,&P_montY);
    jac_add(&S,&A,&B);
    jac_to_affine(&S,&Wk->x,&Wk->y);
    sc_set(&Wk->a,&a0); sc_set(&Wk->b,&b0);
}

int main(int argc,char**argv){
    if(argc<12){ fprintf(stderr,"args: p n Px Py Qx Qy dpbits jumps walks threads tblbits [seed]\n"); return 1; }
    parse_fe(&P_mod,argv[1]);
    parse_fe((fe*)&N_mod,argv[2]);
    fe Px,Py,Qx,Qy;
    parse_fe(&Px,argv[3]); parse_fe(&Py,argv[4]);
    parse_fe(&Qx,argv[5]); parse_fe(&Qy,argv[6]);
    int dpbits=atoi(argv[7]); njumps=atoi(argv[8]);
    int walks=atoi(argv[9]); int nthreads=atoi(argv[10]); int tblbits=atoi(argv[11]);
    uint64_t seed=(argc>12)?strtoull(argv[12],NULL,10):0x123456789ABCDEFull;
    if((njumps&(njumps-1))!=0){ fprintf(stderr,"jumps must be power of two\n"); return 1; }

    /* two's complements (for carry-fix): 2^(64NW) - p and - n */
    { uint64_t bb=0; for(int i=0;i<NW;i++) P_neg.w[i]=subw(0,P_mod.w[i],&bb); }
    { uint64_t bb=0; for(int i=0;i<NW;i++) N_neg.w[i]=subw(0,N_mod.w[i],&bb); }

    /* g_inv = -p^{-1} mod 2^64 (Newton) */
    uint64_t inv=1;
    for(int i=0;i<6;i++) inv*=2-inv*P_mod.w[0];
    if((uint64_t)(P_mod.w[0]*inv)!=1){ fprintf(stderr,"inv check failed\n"); return 1; }
    g_inv=-inv;

    /* R1 = 2^(64NW) mod p : double 1, 64NW times */
    fe Rmod; fe_zero(&Rmod); Rmod.w[0]=1;
    for(int i=0;i<64*NW;i++) fe_add(&Rmod,&Rmod,&Rmod);
    fe_set(&R1_mont,&Rmod);
    /* R2 = R^2 mod p : double R1, 64NW more times */
    fe_set(&R2_mont,&Rmod);
    for(int i=0;i<64*NW;i++) fe_add(&R2_mont,&R2_mont,&R2_mont);

    mont_mul(&P_montX,&Px,&R2_mont); mont_mul(&P_montY,&Py,&R2_mont);
    mont_mul(&Q_montX,&Qx,&R2_mont); mont_mul(&Q_montY,&Qy,&R2_mont);

    /* sanity: Q on curve */
    { fe lhs,rhs,xx,seven;
      mont_sqr(&lhs,&Q_montY);
      mont_mul(&xx,&Q_montX,&Q_montX); mont_mul(&xx,&xx,&Q_montX);
      fe_zero(&seven); seven.w[0]=7; mont_mul(&seven,&seven,&R2_mont);
      fe_add(&rhs,&xx,&seven);
      if(!fe_eq(&lhs,&rhs)){ fprintf(stderr,"Q not on curve\n"); return 1; }
      mont_sqr(&lhs,&P_montY);
      mont_mul(&xx,&P_montX,&P_montX); mont_mul(&xx,&xx,&P_montX);
      fe_add(&rhs,&xx,&seven);
      if(!fe_eq(&lhs,&rhs)){ fprintf(stderr,"P not on curve\n"); return 1; }
    }

    /* selftest mode: argv[1]=="selftest" -> hardcode 6-bit, kappa=23 */
    if(argc>12 && strtoull(argv[12],NULL,10)==99){
        sc k; sc_zero(&k); k.w[0]=23;
        jac R; scalar_mul_point(&R,&k,&Q_montX,&Q_montY);
        fe x,y; jac_to_affine(&R,&x,&y);
        printf("23*Q affine mont: x="); print_sc((sc*)&x); printf("y="); print_sc((sc*)&y);
        printf("P mont:           x="); print_sc((sc*)&P_montX); printf("y="); print_sc((sc*)&P_montY);
        { fe Zinv,x,y;
          for(int kk=2;kk<=5;kk++){
            sc kk_; sc_zero(&kk_); kk_.w[0]=kk;
            jac R; scalar_mul_point(&R,&kk_,&Q_montX,&Q_montY);
            jac_to_affine(&R,&x,&y);
            printf("k=%d: x=%016llx y=%016llx  Z=",kk,(unsigned long long)x.w[0],(unsigned long long)y.w[0]);
            print_sc((sc*)&R.Z);
          }
          /* fe_inv sanity: mont(2)*inv(mont(2)) == mont(1)=R1 */
          fe two_m, inv2, chk; mont_mul(&two_m,&R2_mont,&R2_mont); /* arbitrary nonzero */
          two_m.w[0]=8; mont_mul(&inv2,&two_m,&R2_mont); fe_inv(&inv2,&inv2);
          mont_mul(&chk,&two_m,&inv2);
          printf("inv check: "); print_sc((sc*)&chk); printf("R1       : "); print_sc((sc*)&R1_mont); }
        { sc a,inv_,prod;
          sc_zero(&a); a.w[0]=0x5d6c04824df0ull&0xffffffffffffffffull;
          { sc b; sc_zero(&b); b.w[0]=0xc57daa03038full; sc_mul(&prod,&a,&b);
            printf("prod1 = "); print_sc(&prod); }
          sc_zero(&a); a.w[0]=0x40d3f62c2666ull;
          { sc b; sc_zero(&b); b.w[0]=0x5fd0c9fed159ull; sc_mul(&prod,&a,&b);
            printf("prod2 = "); print_sc(&prod); }
          sc_zero(&a); a.w[0]=4; sc_inv(&inv_,&a);
          sc_mul(&prod,&a,&inv_);
          printf("4*inv(4) mod n = "); print_sc(&prod); printf("(expect 1)\n");
          sc_zero(&a); a.w[0]=0xdeadbeefcafe1234ull; sc_inv(&inv_,&a);
          sc_mul(&prod,&a,&inv_);
          printf("big*inv(big) mod n = "); print_sc(&prod); printf("(expect 1)\n");
          sc_zero(&a); a.w[0]=30; sc_zero(&inv_); inv_.w[0]=8; sc_mul(&prod,&a,&inv_);
          printf("30*8 mod n    = "); print_sc(&prod);
          sc_zero(&a); a.w[0]=2; sc_zero(&inv_); inv_.w[0]=16; sc_add_n(&prod,&a,&inv_);
          printf("2+16 mod 31   = "); print_sc(&prod);
          sc_sub_n(&prod,&a,&inv_);
          printf("2-16 mod 31   = "); print_sc(&prod); }
        { /* fuzz mont_mul against __uint128 reference */
          uint64_t rng=12345; long bad=0;
          fe R2ref; fe_set(&R2ref,&R2_mont);
          for(long it=0;it<3000000;it++){
            uint64_t x=rnd64(&rng)%P_mod.w[0], y=rnd64(&rng)%P_mod.w[0];
            fe xm,ym,prod;
            mont_mul(&xm,&R1_mont,&R1_mont); /* dummy */
            /* mont(x) = mont_mul(x_plain, R2) */
            fe xp,yp; fe_zero(&xp); fe_zero(&yp); xp.w[0]=x; yp.w[0]=y;
            mont_mul(&xm,&xp,&R2ref); mont_mul(&ym,&yp,&R2ref);
            mont_mul(&prod,&xm,&ym);
            /* reference: x*y mod p (plain) */
            __uint128_t ref=((__uint128_t)x*y)%P_mod.w[0];
            /* mont-domain result corresponds to plain x*y: mont_mul(mont(x),mont(y)) = mont(x*y) = (x*y)*R mod p */
            __uint128_t refm=((__uint128_t)ref*R1_mont.w[0])%P_mod.w[0];
            if(prod.w[0]!=(uint64_t)refm){ if(bad<5) fprintf(stderr,"MISMATCH x=%016llx y=%016llx got=%016llx want=%016llx\n",(unsigned long long)x,(unsigned long long)y,(unsigned long long)prod.w[0],(uint64_t)refm); bad++; }
          }
          printf("mont_mul fuzz mismatches: %ld / 3000000\n",bad);
          /* fuzz fe_add/fe_sub */
          bad=0;
          for(long it=0;it<3000000;it++){
            uint64_t x=rnd64(&rng)%P_mod.w[0], y=rnd64(&rng)%P_mod.w[0];
            fe xr,yr,s_,d_; fe_zero(&xr); fe_zero(&yr);
            xr.w[0]=x; yr.w[0]=y;
            fe_add(&s_,&xr,&yr); fe_sub(&d_,&xr,&yr);
            uint64_t refs=(x+y)%P_mod.w[0], refd=(x-y)%P_mod.w[0];
            if(s_.w[0]!=refs||d_.w[0]!=refd){ if(bad<5) fprintf(stderr,"ADDSUB MISMATCH x=%016llx y=%016llx got=%016llx/%016llx want=%016llx/%016llx\n",(unsigned long long)x,(unsigned long long)y,(unsigned long long)s_.w[0],(unsigned long long)d_.w[0],(unsigned long long)refs,(unsigned long long)refd); bad++; }
          }
          printf("fe_add/sub fuzz mismatches: %ld / 3000000\n",bad);
        }
        { fe xx,yy,ss;
          parse_fe(&xx,"2b43765e3745"); parse_fe(&yy,"ccac889b6252");
          printf("p words: "); for(int i=NW-1;i>=0;i--) printf("%016llx ",(unsigned long long)P_mod.w[i]); printf("\n");
          printf("x words: "); for(int i=NW-1;i>=0;i--) printf("%016llx ",(unsigned long long)xx.w[i]); printf("\n");
          printf("y words: "); for(int i=NW-1;i>=0;i--) printf("%016llx ",(unsigned long long)yy.w[i]); printf("\n");
          printf("ge_p(x)? %d  ge_p(y)? %d\n", fe_ge_p(&xx), fe_ge_p(&yy));
          fe_add(&ss,&xx,&yy);
          printf("x+y words: "); for(int i=NW-1;i>=0;i--) printf("%016llx ",(unsigned long long)ss.w[i]); printf("\n");
          fe_sub(&ss,&xx,&yy);
          printf("x-y words: "); for(int i=NW-1;i>=0;i--) printf("%016llx ",(unsigned long long)ss.w[i]); printf("\n");
        }
        printf("verify_kappa(23) = %d\n", verify_kappa(&k));
        sc k2; sc_zero(&k2); k2.w[0]=24;
        printf("verify_kappa(24) = %d\n", verify_kappa(&k2));
        return 0;
    }
    /* jump table */
    jx=malloc(sizeof(fe)*njumps); jy=malloc(sizeof(fe)*njumps);
    js=malloc(sizeof(sc)*njumps); jt=malloc(sizeof(sc)*njumps);
    uint64_t rs=seed?seed:0xDEADBEEFCAFEBABEull;
    for(int j=0;j<njumps;j++){
        sc s_,t_; random_sc(&s_,&rs); random_sc(&t_,&rs);
        js[j]=s_; jt[j]=t_;
        jac A,B,M;
        scalar_mul_point(&A,&s_,&Q_montX,&Q_montY);
        scalar_mul_point(&B,&t_,&P_montX,&P_montY);
        jac_add(&M,&A,&B);
        if(fe_is_zero(&M.Z)){ j--; continue; }
        jac_to_affine(&M,&jx[j],&jy[j]);
    }

    shardbits=tblbits-6; if(shardbits<0) shardbits=0;
    nlocks=1<<shardbits;
    if(argc>12 && strtoull(argv[12],NULL,10)==98){
        /* jumptest: dump s_j,t_j and M_j affine (plain) for external verification */
        fe one; fe_zero(&one); one.w[0]=1;
        for(int j=0;j<njumps;j++){
            fe px,py; mont_mul(&px,&jx[j],&one); mont_mul(&py,&jy[j],&one);
            printf("%d s=",j); for(int i=NW-1;i>=0;i--) printf("%016llx",(unsigned long long)js[j].w[i]);
            printf(" t="); for(int i=NW-1;i>=0;i--) printf("%016llx",(unsigned long long)jt[j].w[i]);
            printf(" Mx="); for(int i=NW-1;i>=0;i--) printf("%016llx",(unsigned long long)px.w[i]);
            printf(" My="); for(int i=NW-1;i>=0;i--) printf("%016llx",(unsigned long long)py.w[i]);
            printf("\n");
        }
        { fe px,py; mont_mul(&px,&Q_montX,&one); mont_mul(&py,&Q_montY,&one);
          printf("Q x="); for(int i=NW-1;i>=0;i--) printf("%016llx",(unsigned long long)px.w[i]);
          printf(" y="); for(int i=NW-1;i>=0;i--) printf("%016llx",(unsigned long long)py.w[i]); printf("\n"); }
        return 0;
    }
    tbl=calloc((size_t)1<<tblbits,sizeof(dpent));
    tblmask=((uint64_t)1<<tblbits)-1;
    locks=malloc(sizeof(omp_lock_t)*nlocks);
    for(int i=0;i<nlocks;i++) omp_init_lock(&locks[i]);

    fprintf(stderr,"tblbits=%d capacity=%llu nlocks=%d\n",tblbits,(unsigned long long)((uint64_t)1<<tblbits),nlocks);
    atomic_store(&g_found,0); atomic_store(&g_steps,0);
    double t0=omp_get_wtime();

    #pragma omp parallel num_threads(nthreads)
    {
        int tid=omp_get_thread_num();
        uint64_t rng=seed ^ (0xABCDEF1234567890ull*(uint64_t)(tid+1)) ^ 0x9E3779B97F4A7C15ull;
        walkpt *Wk=malloc(sizeof(walkpt)*walks);
        fe *pendX=malloc(sizeof(fe)*walks), *pendY=malloc(sizeof(fe)*walks);
        fe *Zarr=malloc(sizeof(fe)*walks);
        unsigned char *reseed=calloc(walks,1);
        uint64_t dpbitmask=((uint64_t)1<<dpbits)-1;

        for(int i=0;i<walks;i++) reseed_walk(&Wk[i],&rng);

        double last=t0;
        while(!atomic_load(&g_found)){
            /* step each walk projectively */
            for(int i=0;i<walks;i++){
                int j=(int)((Wk[i].x.w[0]>>8)&(uint64_t)(njumps-1));
                jac P1,R2_;
                P1.X=Wk[i].x; P1.Y=Wk[i].y; fe_set(&P1.Z,&R1_mont);
                jac_madd(&R2_,&P1,&jx[j],&jy[j]);
                if(fe_is_zero(&R2_.Z)){
                    reseed_walk(&Wk[i],&rng);
                    fe_set(&pendX[i],&Wk[i].x); fe_set(&pendY[i],&Wk[i].y);
                    fe_set(&Zarr[i],&R1_mont);
                    continue;
                }
                sc_add_n(&Wk[i].a,&Wk[i].a,&js[j]);
                sc_add_n(&Wk[i].b,&Wk[i].b,&jt[j]);
                pendX[i]=R2_.X; pendY[i]=R2_.Y; Zarr[i]=R2_.Z;
            }
            atomic_fetch_add(&g_steps,(unsigned long)walks);
            {
                fe *pre=malloc(sizeof(fe)*walks);
                fe_set(&pre[0],&Zarr[0]);
                for(int i=1;i<walks;i++) mont_mul(&pre[i],&pre[i-1],&Zarr[i]);
                fe allinv; fe_inv(&allinv,&pre[walks-1]);
                fe cur; fe_set(&cur,&allinv);
                for(int i=walks-1;i>=0;i--){
                    fe zi; /* Zinv[i] = cur * pre[i-1] (or allinv for i=0) */
                    if(i>0){ mont_mul(&zi,&cur,&pre[i-1]); }
                    else fe_set(&zi,&cur);
                    /* convert pending to affine (mont): x=X*zi^2, y=Y*zi^3 */
                    fe z2,z3,xa,ya;
                    mont_mul(&z2,&zi,&zi); mont_mul(&z3,&z2,&zi);
                    mont_mul(&xa,&pendX[i],&z2); mont_mul(&ya,&pendY[i],&z3);
                    Wk[i].x=xa; Wk[i].y=ya;
                    /* prepare cur for next (i-1): cur *= Z_orig[i] */
                    mont_mul(&cur,&cur,&Zarr[i]);
                    Zarr[i]=zi; /* store inverse back (not needed further) */
                }
                free(pre);
            }
            /* DP checks */
            for(int i=0;i<walks;i++){
                if(Wk[i].x.w[0]&dpbitmask) continue;
                atomic_fetch_add(&g_dphits,1ul);
                uint64_t xk=Wk[i].x.w[0];
                uint64_t ypar=Wk[i].y.w[0]&1ull;
                uint64_t h=hashx(xk>>dpbits);   /* low dpbits of xk are all zero -> would bias shards */
                uint64_t base=(h&tblmask)&~63ull;   /* shard base: 64 slots */
                if(getenv("RHODEBUG") && atomic_load(&g_dphits)<25){
                    fprintf(stderr,"DP xk=%016llx h=%016llx base=%016llu slot0=%llu ins=%lu drops=%lu\n",
                        (unsigned long long)xk,(unsigned long long)h,(unsigned long long)base,
                        (unsigned long long)((h)&63),atomic_load(&g_inserts),atomic_load(&g_drops));
                }
                omp_set_lock(&locks[base>>6]);
                int done=0;
                for(int probe=0;probe<64&&!done;probe++){
                    uint64_t slot=base+(uint64_t)((h+probe)&63);
                    if(tbl[slot].x==0){
                        atomic_fetch_add(&g_inserts,1ul);
                        tbl[slot].x=xk; tbl[slot].ypar=ypar;
                        for(int w=0;w<NW;w++){ tbl[slot].a[w]=Wk[i].a.w[w]; tbl[slot].b[w]=Wk[i].b.w[w]; }
                        done=1; break;
                    }
                    if(tbl[slot].x==xk){
                        sc da,db,kap;
                        if(tbl[slot].ypar==ypar){
                            /* same point: (a1-a2)Q = (b2-b1)P */
                            sc a2,b2;
                            for(int w=0;w<NW;w++){ a2.w[w]=tbl[slot].a[w]; b2.w[w]=tbl[slot].b[w]; }
                            if(sc_eq(&Wk[i].a,&a2)&&sc_eq(&Wk[i].b,&b2)){
                                atomic_fetch_add(&g_exact,1ul);
                                reseed[i]=1;   /* walk merged: dead weight, restart */
                            } else if(!sc_eq(&Wk[i].b,&b2)){
                                sc_sub_n(&da,&Wk[i].a,&a2);
                                sc_sub_n(&db,&b2,&Wk[i].b);
                                { sc onem, nm1, nm2, invdb;
  sc_zero(&onem); onem.w[0]=1;
  sc_sub_n(&nm1,&N_mod,&onem);      /* n-1 */
  sc_sub_n(&nm2,&nm1,&onem);        /* n-2 */
  sc_pow(&invdb,&db,&nm2); sc_mul(&kap,&da,&invdb); }
                                if(verify_kappa(&kap)){ g_kap=kap; atomic_store(&g_found,1); }
                                else { reseed[i]=1;
                                    if(atomic_fetch_add(&g_dbg,1)<2){
                                        jac A,B,S; sc tmp;
                                        sc_set(&tmp,&Wk[i].a);
                                        scalar_mul_point(&A,&tmp,&Q_montX,&Q_montY);
                                        sc_set(&tmp,&Wk[i].b);
                                        scalar_mul_point(&B,&tmp,&P_montX,&P_montY);
                                        jac_add(&S,&A,&B);
                                        fe x,y; jac_to_affine(&S,&x,&y);
                                        fprintf(stderr,"SAMEPTFAIL\nrecomputed a1Q+b1P: x="); print_sc((sc*)&x);
                                        fprintf(stderr,"y="); print_sc((sc*)&y);
                                        fprintf(stderr,"walk actual:       x="); print_sc((sc*)&Wk[i].x);
                                        fprintf(stderr,"y="); print_sc((sc*)&Wk[i].y);
                                        fprintf(stderr,"a1="); print_sc(&Wk[i].a);
                                        fprintf(stderr,"b1="); print_sc(&Wk[i].b);
                                        fprintf(stderr,"a2="); print_sc(&a2);
                                        fprintf(stderr,"b2="); print_sc(&b2);
                                        fprintf(stderr,"kap="); print_sc(&kap);
                                        fprintf(stderr,"da="); print_sc(&da); fprintf(stderr,"db="); print_sc(&db);
                                    } }
                            } else reseed[i]=1;
                        } else {
                            atomic_fetch_add(&g_negcase,1ul);
                            /* X1 = -X2: kappa = -(a1+a2)*(b1+b2)^{-1} */
                            sc a2,b2,s,t2,kap;
                            for(int w=0;w<NW;w++){ a2.w[w]=tbl[slot].a[w]; b2.w[w]=tbl[slot].b[w]; }
                            if(0&&atomic_load(&g_dbg)==0){
                                atomic_store(&g_dbg,1);
                                jac A,B,S; sc tmp;
                                sc_set(&tmp,&Wk[i].a);
                                scalar_mul_point(&A,&tmp,&Q_montX,&Q_montY);
                                sc_set(&tmp,&Wk[i].b);
                                scalar_mul_point(&B,&tmp,&P_montX,&P_montY);
                                jac_add(&S,&A,&B);
                                fe x,y; jac_to_affine(&S,&x,&y);
                                fprintf(stderr,"recomputed walk point: x="); print_sc((sc*)&x);
                                fprintf(stderr,"                y="); print_sc((sc*)&y);
                                fprintf(stderr,"actual walk point:     x="); print_sc((sc*)&Wk[i].x);
                                fprintf(stderr,"                y="); print_sc((sc*)&Wk[i].y);
                                fprintf(stderr,"a1="); print_sc(&Wk[i].a); fprintf(stderr,"b1="); print_sc(&Wk[i].b);
                                fprintf(stderr,"a2="); print_sc(&a2); fprintf(stderr,"b2="); print_sc(&b2);
                            }
                            sc_add_n(&s,&Wk[i].a,&a2);
                            sc_add_n(&t2,&Wk[i].b,&b2);
                            if(!sc_is_zero(&t2)&&!sc_is_zero(&s)){
                                { sc nm2, invt;
  sc onem2; sc_zero(&onem2); onem2.w[0]=1;
  sc nm1; sc_sub_n(&nm1,&N_mod,&onem2);
  sc_sub_n(&nm2,&nm1,&onem2);
  sc_pow(&invt,&t2,&nm2); sc_mul(&kap,&s,&invt); }
                                /* negate */
                                sc negkap; sc_sub_n(&negkap,&N_mod,&kap);
                                if(!sc_is_zero(&negkap)&&verify_kappa(&negkap)){ g_kap=negkap; atomic_store(&g_found,1); }
                                else if(atomic_fetch_add(&g_dbg,1)<3){
                                    fprintf(stderr,"NEGFAIL x=%llx ypar1=%llu a1=",(unsigned long long)xk,(unsigned long long)ypar); print_sc(&Wk[i].a);
                                    fprintf(stderr," b1="); print_sc(&Wk[i].b);
                                    fprintf(stderr," a2="); print_sc(&a2); fprintf(stderr," b2="); print_sc(&b2);
                                }
                            }
                        }
                        done=1; break;
                    }
                }
                if(!done){ /* shard full: drop this DP */
                    atomic_fetch_add(&g_drops,1ul);
                    if(atomic_load(&g_drops)<4){
                        fprintf(stderr,"DROP xk=%016llx base=%llu occ:",(unsigned long long)xk,(unsigned long long)(base>>6));
                        for(int probe2=0;probe2<64;probe2++) fprintf(stderr,"%d",tbl[base+(uint64_t)((h+probe2)&63)].x!=0);
                        fprintf(stderr,"\n");
                    }
                }
                omp_unset_lock(&locks[base>>6]);
            }
            for(int i=0;i<walks;i++) if(reseed[i]){ reseed[i]=0; reseed_walk(&Wk[i],&rng); }
            /* progress */
            if(tid==0){
                double now=omp_get_wtime();
                if(now-last>10.0){
                    last=now;
                    unsigned long st=atomic_load(&g_steps);
                    fprintf(stderr,"[%.0fs] steps=%lu rate=%.1f M/s dphits=%lu ins=%lu drops=%lu neg=%lu\n",now-t0,st,(double)st/(now-t0)/1e6,atomic_load(&g_dphits),atomic_load(&g_inserts),atomic_load(&g_drops),atomic_load(&g_negcase));
                    fflush(stderr);
                }
            }
        }
        free(Wk); free(pendX); free(pendY); free(Zarr);
    }
    double t1=omp_get_wtime();
    if(atomic_load(&g_found)){
        printf("FOUND kappa = ");
        for(int i=NW-1;i>=0;i--) printf("%016llx",(unsigned long long)g_kap.w[i]);
        printf("\n");
        fprintf(stderr,"steps=%lu time=%.1fs rate=%.2f Mops/s\n",atomic_load(&g_steps),t1-t0,(double)atomic_load(&g_steps)/(t1-t0)/1e6);
    } else {
        fprintf(stderr,"exited without solution\n");
    }
    return atomic_load(&g_found)?0:2;
}
