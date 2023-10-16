#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>

#define N 512
#define THREADS_PER_BLOCK 256
const float PI = 3.14159265358979323846f;

__global__ void butterflyStepKernelOptimized3(cufftComplex *a, int n, bool invert) {
    __shared__ cufftComplex shared_data[THREADS_PER_BLOCK];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= n/2) return;

    int opposite = tid + n/2;
    
    float ang = 2 * PI * tid / n * (invert ? -1 : 1);
    cufftComplex w = make_cuComplex(__cosf(ang), __sinf(ang));

    shared_data[threadIdx.x] = a[tid];
    __syncthreads();

    cufftComplex a0 = shared_data[threadIdx.x];
    cufftComplex a1 = a[opposite];
    
    a[tid] = cuCaddf(a0, cuCmulf(w, a1));
    a[opposite] = cuCsubf(a0, cuCmulf(w, a1));
}

void parallelFFTOptimized3(cufftComplex *data, int n, bool invert) {
    cudaSetDevice(0);
    
    cufftComplex *d_data;
    cudaMalloc((void**)&d_data, sizeof(cufftComplex) * n);
    cudaMemcpyAsync(d_data, data, sizeof(cufftComplex) * n, cudaMemcpyHostToDevice);

    int numBlocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    for(int len = 2; len <= n; len <<= 1) {
        butterflyStepKernelOptimized3<<<numBlocks, THREADS_PER_BLOCK>>>(d_data, len, invert);
    }

    cudaMemcpyAsync(data, d_data, sizeof(cufftComplex) * n, cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}

int main() {
    int n = N;
    cufftComplex *data = new cufftComplex[n];

    // Initialize the data (e.g. with random values or some test pattern)
    for(int i = 0; i < n; i++) {
        data[i].x = (float)i;
        data[i].y = 0.0f;
    }

    parallelFFTOptimized3(data, n, false);

    // Optional: Display results or further processing
    for(int i = 0; i < n; i++) {
        std::cout << "(" << data[i].x << ", " << data[i].y << ")" << std::endl;
    }

    delete[] data;
    return 0;
}
