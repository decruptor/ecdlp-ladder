import json, subprocess, sys, time
data=json.load(open("challenges.json"))
def uncompress(hexstr,p):
    prefix=int(hexstr[:2],16); x=int(hexstr[2:],16)
    assert pow(x*x*x+7,(p-1)//2,p)==1
    if p%4==3: y=pow(x*x*x+7,(p+1)//4,p)
    else:
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
    return x,y
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

bits=int(sys.argv[1])
exe=sys.argv[2] if len(sys.argv)>2 else "./rho1"
dpbits=int(sys.argv[3]) if len(sys.argv)>3 else None
threads=int(sys.argv[4]) if len(sys.argv)>4 else 10
rung=[r for r in data["rungs"] if r["bits"]==bits][0]
p=int(rung["p"]); n=int(rung["n"])
Px,Py=uncompress(rung["P_compressed"],p)
Qx,Qy=uncompress(rung["Q_compressed"],p)
import math
if dpbits is None:
    # target ~2^18 DPs: dpbits = log2(1.25*sqrt(n)) - 18
    dpbits=max(1,int(math.log2(1.25*math.isqrt(n)))-18)
exp_dps=1.25*math.isqrt(n)*(2.0**-dpbits)
tblbits=max(10,int(math.ceil(math.log2(max(exp_dps,2))))+3)
args=[exe,str(hex(p)[2:]),hex(n)[2:],hex(Px)[2:],hex(Py)[2:],hex(Qx)[2:],hex(Qy)[2:],
      str(dpbits),"32","64",str(threads),str(tblbits)]
print("dpbits",dpbits,"tblbits",tblbits)
t0=time.time()
res=subprocess.run(args,capture_output=True,text=True)
dt=time.time()-t0
print(res.stderr[-2000:] if res.stderr else "")
out=res.stdout.strip()
print(out)
if out.startswith("FOUND"):
    kap=int(out.split("=")[1].strip(),16)
    ok=mul(kap,(Qx,Qy),p)==(Px,Py)
    print(f"VERIFIED kappa={kap} hex={hex(kap)} time={dt:.1f}s ok={ok}")
    json.dump({"bits":bits,"kappa":str(kap),"hex":hex(kap),"verified":ok,"time_s":round(dt,1),
               "method":"parallel Pollard rho (DP, OpenMP C, Montgomery)"},
              open(f"result_{bits}.json","w"),indent=2)
