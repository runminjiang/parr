#include <cufft.h>
#include <cuda_runtime.h>
#include <iostream>

const float PI = 3.14159265358979323846f;

#define N 512

__global__ void butterflyStepKernel(cufftComplex *a, int n, bool invert) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= n/2) return;
    
    int opposite = tid + n/2;
    float ang = 2 * PI * tid / n * (invert ? -1 : 1);
    cufftComplex w = make_cuComplex(cos(ang), sin(ang));
    
    cufftComplex a0 = a[tid];
    cufftComplex a1 = a[opposite];
    
    a[tid] = cuCaddf(a0, cuCmulf(w, a1));
    a[opposite] = cuCsubf(a0, cuCmulf(w, a1));
}

void parallelFFT(cufftComplex *data, int n, bool invert) {
    cudaSetDevice(0);
    
    cufftComplex *d_data;
    cudaMalloc((void**)&d_data, sizeof(cufftComplex) * n);
    cudaMemcpy(d_data, data, sizeof(cufftComplex) * n, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    for(int len = 2; len <= n; len <<= 1) {
        butterflyStepKernel<<<numBlocks, blockSize>>>(d_data, len, invert);
    }

    cudaMemcpy(data, d_data, sizeof(cufftComplex) * n, cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}

int main() {
    int n = N; // Number of points
    cufftComplex *data = new cufftComplex[n];

    // Initialize the data here if needed

    parallelFFT(data, n, false);

    // Display results or further processing

    delete[] data;
    return 0;
}
