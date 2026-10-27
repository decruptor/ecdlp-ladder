#!/usr/bin/env bash
# rho80v6b_gen.sh — generate the minimal production diff (rho80v6b.cu) from the
# RESTORED validated source, applying ONLY the winning launch geometry.
#
# Usage: rho80v6b_gen.sh <restored-source.cu> <TPB> [LBMIN]
#   TPB   = winning threads/block (128/256/512/1024)
#   LBMIN = launch_bounds minBlocksPerSM (default 1)
# If TPB == 256 and LBMIN == 1, nothing to change in source (winner was a
# build-flag-only result, e.g. -maxrregcount) — script prints that instead.
set -euo pipefail
SRC=$(readlink -f "${1:?restored source}")
TPB="${2:?TPB}"
LBMIN="${3:-1}"
OUT=/root/rho80v6b.cu
if [ "$TPB" = "256" ] && [ "$LBMIN" = "1" ]; then
  echo "Winner is build-flag-only (e.g. -maxrregcount); no source diff needed."
  echo "Production build: nvcc -O3 -DRUNG=96 -arch=sm_89 <WINNING-FLAGS> -o rho96v6b $SRC"
  exit 0
fi
cp "$SRC" "$OUT"
python3 - "$TPB" "$LBMIN" <<'EOF'
import re,sys
tpb,lbmin=sys.argv[1],sys.argv[2]
s=open('/root/rho80v6b.cu').read()
def sub1(pat,rep):
    global s
    n=len(re.findall(pat,s,re.M))
    if n!=1: print(f"ANCHOR FAIL ({n}x): {pat}"); sys.exit(1)
    s=re.sub(pat,rep,s,count=1,flags=re.M)
sub1(r"__global__ __launch_bounds__\(256,1\) void rho_kernel\(",
     f"__global__ __launch_bounds__({tpb},{lbmin}) void rho_kernel(")
sub1(r"    g->threads=256;\n", f"    g->threads={tpb};\n")
open('/root/rho80v6b.cu','w').write(s)
print(f"rho80v6b.cu written: TPB={tpb} LBMIN={lbmin}")
EOF
diff "$SRC" "$OUT" || true
