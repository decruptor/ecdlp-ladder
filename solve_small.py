import json, math, time
data=json.load(open("/Users/mkunak/Devs/ecdlp-ladder/challenges.json"))
def uncompress(hexstr,p):
    prefix=int(hexstr[:2],16); x=int(hexstr[2:],16)
    assert pow(x*x*x+7,(p-1)//2,p)==1, "x not on curve"
    y=pow(x*x*x+7,(p+1)//4,p) if p%4==3 else None
    if y is None:
        # Tonelli-Shanks
        q,s=p-1,0
        while q%2==0: q//=2; s+=1
        z=2
        while pow(z,(p-1)//2,p)!=p-1: z+=1
        m,c,t,r=s,pow(z,q,p),pow(x*x*x+7,q,p),pow(x*x*x+7,(q+1)//2,p)
        while t!=1:
            i,t2=0,t
            while t2!=1: t2=t2*t2%p; i+=1
            b=pow(c,1<<(m-i-1),p); m,c=i,b*b%p; t=t*c%p; r=r*b%p
        y=r
    if y%2!=prefix%2: y=p-y
    return (x,y)
def add(P1,P2,p):
    if P1 is None: return P2
    if P2 is None: return P1
    x1,y1=P1; x2,y2=P2
    if x1==x2:
        if (y1+y2)%p==0: return None
        l=3*x1*x1*pow(2*y1,-1,p)%p
    else: l=(y2-y1)*pow(x2-x1,-1,p)%p
    x3=(l*l-x1-x2)%p
    return x3,(l*(x1-x3)-y1)%p
def mul(k,P,p):
    R=None
    while k:
        if k&1: R=add(R,P,p)
        P=add(P,P,p); k>>=1
    return R

results={}
for r in data["rungs"]:
    k=r["bits"]
    if k>32: continue
    p=int(r["p"]); n=int(r["n"])
    Q=uncompress(r["Q_compressed"],p); P=uncompress(r["P_compressed"],p)
    t0=time.time()
    # BSGS: P = kappa*Q, kappa in [1,n-1]
    m=1<<((n.bit_length()+1)//2)
    tbl={}
    cur=None
    for j in range(m):
        tbl[cur]=(j) if cur is not None else 0
        cur=add(cur,Q,p)   # cur = j*Q ; store index j
    # recompute: store mapping point->j for j*Q
    tbl={}; cur=None
    for j in range(m):
        if cur is not None: tbl[cur]=j
        cur=add(cur,Q,p)
    factor=mul(m,Q,p)   # m*Q
    gamma=P; kappa=None
    for i in range(m):
        if gamma is None:          # gamma = O = 0*Q -> kappa = -i*m
            kappa=(-i*m)%n; break
        if gamma in tbl:
            kappa=(tbl[gamma]-i*m)%n; break
        gamma=add(gamma,factor,p)
    dt=time.time()-t0
    ok = kappa is not None and mul(kappa,Q,p)==P
    print(f"secp{k}k1: kappa={kappa} (hex {hex(kappa) if kappa is not None else '-'})  verified={ok}  time={dt:.2f}s")
    results[k]={"kappa":str(kappa) if kappa else None,"hex":hex(kappa) if kappa else None,"verified":ok,"time_s":round(dt,3),"method":"BSGS"}
json.dump(results, open("/Users/mkunak/Devs/ecdlp-ladder/results_small.json","w"), indent=2)
