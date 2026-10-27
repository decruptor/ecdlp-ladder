#include <cstdio>
#include <cstdint>
__global__ void kern(const uint32_t* in, uint32_t* out, int mode) {
    int lane = threadIdx.x & 31;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t v = in[idx];
    uint32_t r;
    if (mode==0) r = (lane==0) ? 0x7ffffffeu : __shfl_up_sync(0xffffffffu, v, 1);
    else { r = __shfl_up_sync(0xffffffffu, v, 1); if (lane==0) r = 0x7ffffffeu; }
    out[idx] = r;
}
int main(){
    const int N=32*1024; uint32_t *din,*dout; uint32_t *h_in=new uint32_t[N], *h_out=new uint32_t[N];
    for (int i=0;i<N;i++) h_in[i]=(uint32_t)(i*2654435761u);
    cudaMalloc(&din,N*4); cudaMalloc(&dout,N*4);
    cudaMemcpy(din,h_in,N*4,cudaMemcpyHostToDevice);
    for (int mode=0;mode<2;mode++){
        kern<<<N/32,32>>>(din,dout,mode);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) { printf("mode %d: CUDA error: %s\n", mode, cudaGetErrorString(e)); continue; }
        cudaMemcpy(h_out,dout,N*4,cudaMemcpyDeviceToHost);
        uint32_t bad=0;
        for (int w=0;w<N/32;w++) for (int l=0;l<32;l++){
            uint32_t want=(l==0)?0x7ffffffeu:h_in[w*32+l-1];
            if (h_out[w*32+l]!=want) bad++;
        }
        printf("mode %d: bad=%u ; warp0:", mode, bad);
        for (int l=0;l<4;l++) printf(" l%d=%u(want %u)", l, h_out[l], l?h_in[l-1]:0x7ffffffeu);
        printf(" ; warp1:");
        for (int l=0;l<4;l++) printf(" l%d=%u(want %u)", l, h_out[32+l], l?h_in[32+l-1]:0x7ffffffeu);
        printf("\n");
    }
    return 0;
}
