# GPU fleet prep — remaining "Brace for impact" ladder rungs

Status: PREPARED, NOT EXECUTED. No publishing.

## Situation

Rungs 6..64 are solved classically (see RESULTS.md). The next rungs are 80, 96,
112, 128, 144, 160, ... The secret κ is full-range (no interval), so the right
GPU algorithm is either

* Pollard **rho** with distinguished points (cost ≈ 1.25·√n point ops), or
* Pollard **kangaroo** over the full interval [0, n) (cost ≈ 2·√n jumps).

Both are √n; the kangaroo has better GPU constants (batched x-only differential
adds, no DP coordination).

## Tool: RCKangaroo (github.com/RetiredC/RCKangaroo)

* NVIDIA CUDA, solves interval ECDLP; README-quoted throughput
  **≈ 14.5 G jump/s on RTX 4090, 19.3 G jump/s on RTX 5090** (turbo kernels).
* CLI: `-gpu <mask>` `-pubkey <key>` `-start <hex>` `-range <bits 32..170>`
  `-dp <bits 14..32>` `-max <x>` `-tames <file>`.
* Curve constants (field prime p, group order n, generator G) are hard-coded for
  secp256k1 in `secp256k1.c` / `secp256k1.h`. **Patch required for our rungs**:
  replace p, n and G with the rung's (p, n, Q) from challenges.json, rebuild.
  We only need the compressed target P on the command line; the tool then finds
  privkey = κ with pubkey = κ·G = κ·Q = P.
  (If the code uses secp256k1 endomorphism/glv shortcuts, disable them — our rung
  curves have no efficient endomorphism.)
* Range cap 170 bits is fine: every rung we would classically attack is ≤ 160 bits.

## Expected cost (kangaroo, K≈1.15 → ≈1.15·2^(k/2) jumps)

| rung | jumps | 1× RTX 4090 @14.5G/s | 8× 4090 | comment |
|------|-------|----------------------|---------|---------|
| 80   | 1.26e12 | ~1.5 min  | seconds  | trivial for a GPU |
| 96   | 3.2e14  | ~6.1 h    | ~46 min  | easy |
| 112  | 8.3e16  | ~66 days  | ~8 days  | serious fleet job |
| 128  | 2.1e19  | ~1.5e4 years | ~190 years (8×) | record territory (needs ~10^4 GPU-class setup, cf. 129-bit 2025 record) |
| 144  | 3.4e21  | — | — | Shor-target scale, classical full attack unrealistic |

(The paper's own Sec. 3 notes the public record moved 114→129 bits between 2020
and 2025 on secp256k1 puzzles.)

## Command lines (after patching constants per rung)

For rung k, with Q as generator and P as target (values in challenges.json):

```
# rung 80 (secp80k1)
./RCKangaroo -dp 20 -range 80 -start 0 -pubkey 030D08430A6C00FAE05651

# rung 96 (secp96k1)
./RCKangaroo -dp 22 -range 96 -start 0 -pubkey 029E1F276C17BCE3D8DF9D22BB

# rung 112 (secp112k1)
./RCKangaroo -dp 24 -range 112 -start 0 -pubkey 02949E2914E4863781500B38122BFD
```

Notes:
* `-dp` should be raised with range so DP-table memory stays sane
  (rule of thumb: dp ≈ range/2 − 20, clamp to [14,32]).
* `-start 0 -range k` covers [0, 2^k) ⊇ [1, n); the kangaroo cost is unchanged.
* Verify every result independently: κ·Q == P (see verify_all.py).
* If patching RCKangaroo turns out messy, porting our `rho.c` DP-rho design to
  CUDA is a fallback (the walk/DP structure is already written and verified);
  expected throughput ~2-5 G ops/s per 4090 with a x-only Montgomery ladder.

## CPU fallback (already running)

secp80k1 is currently being attacked on the M4 with the 2-limb rho solver
(rho2, launched in the background; log: rung80_err.txt). Expected ~1.4e12 group
ops at ~15 M ops/s ⇒ ~26 h on this machine. If the GPU fleet runs first, just
kill it.
