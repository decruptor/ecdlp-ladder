# rho80.cu — CUDA port of rho.c for the ECDLP ladder
#
# CPU reference builds (no GPU required) — used for all local validation:
#   make ref48 ref80 ref32 ref64
# CUDA build (on the GPU box):
#   make cuda RUNG=80 ARCH=sm_89
#
# Arch table:
#   RTX 6000 Ada / RTX 6000D / L40S / RTX 4090 : sm_89
#   A100                                        : sm_80
#   H100                                        : sm_90
#   RTX PRO 6000 (Blackwell)                    : sm_120  (nvcc >= 12.8; else sm_90)
CXX      ?= clang++
NVCC     ?= nvcc
RUNG     ?= 80
ARCH     ?= sm_89

all: rho_ref80

ref%: rho80.cu
	$(CXX) -x c++ -O3 -DCPU_REF -DRUNG=$* -o rho_ref$* rho80.cu

cuda: rho80.cu
	$(NVCC) -O3 -DRUNG=$(RUNG) -arch=$(ARCH) -o rho80 rho80.cu

stubcheck: rho80.cu cuda_stubs.h
	clang++ -x c++ -O1 -include cuda_stubs.h -DCPU_STUB -D__CUDACC__ \
	  -D__CUDA_ARCH__=890 -DRUNG=$(RUNG) -o /tmp/rho80_stub rho80.cu && echo "STUB OK"

rho_ref32 rho_ref48 rho_ref64 rho_ref80: ;

transcript: rho_ref32 rho_ref48 rho_ref64 rho_ref80
	python3 make_transcript.py

clean:
	rm -f rho_ref32 rho_ref48 rho_ref64 rho_ref80 rho80

.PHONY: all cuda stubcheck transcript clean
