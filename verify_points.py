import json, hashlib, sys
sys.setrecursionlimit(10000)
data = json.load(open("/Users/mkunak/Devs/ecdlp-ladder/challenges.json"))

def is_probable_prime(n, k=20):
    if n < 2: return False
    for p in (2,3,5,7,11,13,17,19,23,29,31,37): 
        if n % p == 0: return n == p
    d, r = n-1, 0
    while d % 2 == 0: d //= 2; r += 1
    import random
    for _ in range(k):
        a = random.randrange(2, n-1)
        x = pow(a, d, n)
        if x in (1, n-1): continue
        for _ in range(r-1):
            x = x*x % n
            if x == n-1: break
        else: return False
    return True

def tonelli(a, p):
    if a == 0: return 0
    assert pow(a, (p-1)//2, p) == 1
    if p % 4 == 3: return pow(a, (p+1)//4, p)
    q, s = p-1, 0
    while q % 2 == 0: q //= 2; s += 1
    z = 2
    while pow(z, (p-1)//2, p) != p-1: z += 1
    m, c, t, r = s, pow(z, q, p), pow(a, q, p), pow(a, (q+1)//2, p)
    while t != 1:
        i, t2 = 0, t
        while t2 != 1: t2 = t2*t2 % p; i += 1
        b = pow(c, 1 << (m-i-1), p)
        m, c = i, b*b % p
        t = t*c % p; r = r*b % p
    return r

def uncompress(hexstr, p):
    prefix = int(hexstr[:2], 16); x = int(hexstr[2:], 16)
    rhs = (x*x*x + 7) % p
    y = tonelli(rhs, p)
    if y % 2 != prefix % 2: y = p - y
    assert (prefix in (2,3))
    return x, y

def compress(x, y, p):
    nbytes=(p.bit_length()+7)//8
    return "%02x%0*x" % (2 + (y & 1), 2*nbytes, x)

# group ops
def add(P1, P2, p):
    if P1 is None: return P2
    if P2 is None: return P1
    x1,y1 = P1; x2,y2 = P2
    if x1 == x2:
        if (y1 + y2) % p == 0: return None
        l = 3*x1*x1 * pow(2*y1, -1, p) % p
    else:
        l = (y2-y1) * pow(x2-x1, -1, p) % p
    x3 = (l*l - x1 - x2) % p
    y3 = (l*(x1-x3) - y1) % p
    return x3, y3

def mul(k, P, p):
    R = None
    while k:
        if k & 1: R = add(R, P, p)
        P = add(P, P, p); k >>= 1
    return R

def nums_point(label, p):
    h = int.from_bytes(hashlib.sha256(label.encode()).digest(), 'big') % p
    d = 0
    while True:
        x = (h + d) % p
        rhs = (x*x*x + 7) % p
        if pow(rhs, (p-1)//2, p) == 1:
            y = tonelli(rhs, p)
            if y > (p-1)//2: y = p - y   # canonical root 0<=y<=(p-1)/2
            return compress(x, y, p), x, y, d
        d += 1

report = []
for rung in data["rungs"]:
    k = rung["bits"]; p = int(rung["p"]); n = int(rung["n"])
    Px,Py = uncompress(rung["P_compressed"], p)
    Qx,Qy = uncompress(rung["Q_compressed"], p)
    on_curve = (Py*Py - (Px**3+7)) % p == 0 and (Qy*Qy - (Qx**3+7)) % p == 0
    p_prime = is_probable_prime(p); n_prime = is_probable_prime(n)
    # order check: [n]Q == O and [n+1]Q == Q  => order of Q is exactly n (n prime)
    nQ = mul(n, (Qx,Qy), p); n1Q = mul(n+1, (Qx,Qy), p)
    orderQ = (nQ is None) and (n1Q == (Qx,Qy))
    nP = mul(n, (Px,Py), p)
    orderP = (nP is None)
    # Hasse
    hasse = abs(n - (p+1)) <= 2*int((4*p)**0.5)+2
    # recompute NUMS points from labels
    Pc,Pxr,Pyr,dP = nums_point("Quantum", p)
    Qc,Qxr,Qyr,dQ = nums_point("Challenge", p)
    P_match = (Pc.lower() == rung["P_compressed"].lower())
    Q_match = (Qc.lower() == rung["Q_compressed"].lower())
    anomalous = (p == n)
    report.append((k, on_curve, p_prime, n_prime, orderQ, orderP, hasse, P_match, Q_match, anomalous, dP, dQ))
    print(f"secp{k}k1: on_curve={on_curve} p'={p_prime} n'={n_prime} ord(Q)=n:{orderQ} ord(P)=n:{orderP} hasse={hasse} P_recomputed={P_match} Q_recomputed={Q_match} anomalous={anomalous} deltaP={dP} deltaQ={dQ}")

bad = [r for r in report if not all(r[1:9])]   # fields 1..8 must be True; anomalous flag is informational
print("\nALL CHECKS PASS" if not bad else f"\nFAILURES: {bad}")
