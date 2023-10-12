#include <cuda_runtime.h>

#define N 512
#define THREADS_PER_BLOCK 256

__global__ void butterflyStepKernelOptimized2(cufftComplex *a, int n, bool invert) {
    __shared__ cufftComplex shared_data[THREADS_PER_BLOCK];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= n/2) return;

    int opposite = tid + n/2;
    
    float ang = 2 * PI * tid / n * (invert ? -1 : 1);
    cufftComplex w = make_cuComplex(__cosf(ang), __sinf(ang)); // 使用快速数学函数

    shared_data[threadIdx.x] = a[tid];
    __syncthreads();

    // 展开蝶形操作的循环
    cufftComplex a0 = shared_data[threadIdx.x];
    cufftComplex a1 = a[opposite];
    
    a[tid] = cuCaddf(a0, cuCmulf(w, a1));
    a[opposite] = cuCsubf(a0, cuCmulf(w, a1));
}

void parallelFFTOptimized2(cufftComplex *data, int n, bool invert) {
    cudaSetDevice(0);
    
    cufftComplex *d_data;
    cudaMalloc((void**)&d_data, sizeof(cufftComplex) * n);
    cudaMemcpyAsync(d_data, data, sizeof(cufftComplex) * n, cudaMemcpyHostToDevice); // 使用异步内存传输

    int numBlocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    for(int len = 2; len <= n; len <<= 1) {
        butterflyStepKernelOptimized2<<<numBlocks, THREADS_PER_BLOCK>>>(d_data, len, invert);
    }

    cudaMemcpyAsync(data, d_data, sizeof(cufftComplex) * n, cudaMemcpyDeviceToHost); // 使用异步内存传输
    cudaFree(d_data);
}
