#!/usr/bin/env python3
"""Independent verification of all solved rungs of the Brace-for-impact ECDLP ladder.

Uses pure-Python short-Weierstrass arithmetic (independent of the C solver).
Checks kappa*Q == P exactly for every solved rung.
"""
import json, sys, os

D = os.path.dirname(os.path.abspath(__file__))

def uncompress(hexstr, p):
    prefix = int(hexstr[:2], 16); x = int(hexstr[2:], 16)
    assert pow(x*x*x + 7, (p-1)//2, p) == 1, "x not on curve"
    if p % 4 == 3:
        y = pow(x*x*x + 7, (p+1)//4, p)
    else:
        q, s = p-1, 0
        while q % 2 == 0: q //= 2; s += 1
        z = 2
        while pow(z, (p-1)//2, p) != p-1: z += 1
        m, c, t, r = s, pow(z, q, p), pow(x*x*x+7, q, p), pow(x*x*x+7, (q+1)//2, p)
        while t != 1:
            i, t2 = 0, t
            while t2 != 1: t2 = t2*t2 % p; i += 1
            b = pow(c, 1 << (m-i-1), p); m, c = i, b*b % p; t = t*c % p; r = r*b % p
        y = r
    if y % 2 != prefix % 2: y = p - y
    return (x, y)

def add(P1, P2, p):
    if P1 is None: return P2
    if P2 is None: return P1
    x1, y1 = P1; x2, y2 = P2
    if x1 == x2:
        if (y1 + y2) % p == 0: return None
        l = 3*x1*x1 * pow(2*y1, -1, p) % p
    else:
        l = (y2-y1) * pow(x2-x1, -1, p) % p
    x3 = (l*l - x1 - x2) % p
    return x3, (l*(x1-x3) - y1) % p

def mul(k, P, p):
    R = None
    while k:
        if k & 1: R = add(R, P, p)
        P = add(P, P, p); k >>= 1
    return R

def main():
    data = json.load(open(os.path.join(D, "challenges.json")))
    kappas = {}
    small = json.load(open(os.path.join(D, "results_small.json")))
    for k, v in small.items():
        kappas[int(k)] = int(v["kappa"])
    for f in ("result_48.json", "result_64.json", "result_80.json", "result_96.json"):
        path = os.path.join(D, f)
        if os.path.exists(path):
            v = json.load(open(path))
            if v.get("verified"): kappas[v["bits"]] = int(v["kappa"])
    allok = True
    for bits in sorted(kappas):
        r = [x for x in data["rungs"] if x["bits"] == bits][0]
        p, n = int(r["p"]), int(r["n"])
        P = uncompress(r["P_compressed"], p)
        Q = uncompress(r["Q_compressed"], p)
        kap = kappas[bits]
        ok = mul(kap, Q, p) == P
        print(f"secp{bits}k1: kappa = {hex(kap)}  check [k]Q == P: {'PASS' if ok else 'FAIL'}")
        allok &= ok
    print("ALL VERIFIED" if allok else "VERIFICATION FAILURE")

if __name__ == "__main__":
    main()
