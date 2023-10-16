#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "/home/opc/imaging/parr/HW4_Image/mnist/include/mnist/mnist_reader.hpp"


__global__ void thresholdingKernel(uint8_t *images, int imageSize, uint8_t threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < imageSize) {
        images[idx] = (images[idx] > threshold) ? 255 : 0;  // 基于阈值的分割
    }
}

int main() {
    auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("..");

    int imageSize = dataset.training_images[0].size();

    uint8_t *d_images;
    cudaMalloc(&d_images, dataset.training_images.size() * imageSize * sizeof(uint8_t));

    for (int i = 0; i < dataset.training_images.size(); i++) {
        cudaMemcpy(d_images + i * imageSize, dataset.training_images[i].data(), imageSize * sizeof(uint8_t), cudaMemcpyHostToDevice);
    }

    int blocks = (dataset.training_images.size() * imageSize + 255) / 256;
    uint8_t threshold = 128;

    thresholdingKernel<<<blocks, 256>>>(d_images, dataset.training_images.size() * imageSize, threshold);

    for (int i = 0; i < dataset.training_images.size(); i++) {
        cudaMemcpy(dataset.training_images[i].data(), d_images + i * imageSize, imageSize * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    }

    cudaFree(d_images);

    // ... 你可以保存处理后的图像或进行其他操作 ...

    return 0;
}
