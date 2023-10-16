#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "/home/opc/imaging/parr/HW4_Image/mnist/include/mnist/mnist_reader.hpp"

__global__ void thresholdingKernel(uint8_t *images, int numImages, int imageSize, uint8_t threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numImages * imageSize) {
        images[idx] = (images[idx] > threshold) ? 255 : 0;  // 基于阈值的分割
    }
}

int main() {
    auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("..");

    int imageSize = dataset.training_images[0].size();
    int totalImagesSize = dataset.training_images.size() * imageSize;

    uint8_t *h_all_images = new uint8_t[totalImagesSize];
    for (int i = 0; i < dataset.training_images.size(); i++) {
        memcpy(h_all_images + i * imageSize, dataset.training_images[i].data(), imageSize);
    }

    uint8_t *d_images;
    cudaMalloc(&d_images, totalImagesSize * sizeof(uint8_t));
    cudaMemcpy(d_images, h_all_images, totalImagesSize * sizeof(uint8_t), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (totalImagesSize + threadsPerBlock - 1) / threadsPerBlock;
    uint8_t threshold = 128;

    // Define CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start, NULL);

    thresholdingKernel<<<blocks, threadsPerBlock>>>(d_images, dataset.training_images.size(), imageSize, threshold);

    // Record the stop event
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

    cudaMemcpy(h_all_images, d_images, totalImagesSize * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    for (int i = 0; i < dataset.training_images.size(); i++) {
        memcpy(dataset.training_images[i].data(), h_all_images + i * imageSize, imageSize);
    }

    cudaFree(d_images);
    delete[] h_all_images;

    // Clean up the events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // ... 你可以保存处理后的图像或进行其他操作 ...

    return 0;
}
