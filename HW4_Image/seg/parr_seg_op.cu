#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "/home/opc/imaging/parr/HW4_Image/include/mnist/mnist_reader.hpp"

__global__ void thresholdingKernel(uint8_t *images, int numImages, int imageSize, uint8_t threshold) {
    // 使用共享内存
    __shared__ uint8_t sharedImages[1024];  // 假设每个线程块有1024个线程

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numImages * imageSize) {
        // 将数据从全局内存加载到共享内存
        sharedImages[threadIdx.x] = images[idx];
        __syncthreads();  // 确保所有线程都完成了数据加载

        // 在共享内存中进行处理
        sharedImages[threadIdx.x] = (sharedImages[threadIdx.x] > threshold) ? 255 : 0;
        __syncthreads();  // 确保所有线程都完成了数据处理

        // 将处理后的数据写回全局内存
        images[idx] = sharedImages[threadIdx.x];
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
    cudaMemcpyAsync(d_images, h_all_images, totalImagesSize * sizeof(uint8_t), cudaMemcpyHostToDevice);

    int threadsPerBlock = 1024;
    int blocks = (totalImagesSize + threadsPerBlock - 1) / threadsPerBlock;
    uint8_t threshold = 128;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    thresholdingKernel<<<blocks, threadsPerBlock>>>(d_images, dataset.training_images.size(), imageSize, threshold);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

    cudaMemcpyAsync(h_all_images, d_images, totalImagesSize * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    for (int i = 0; i < dataset.training_images.size(); i++) {
        memcpy(dataset.training_images[i].data(), h_all_images + i * imageSize, imageSize);
    }

    cudaFree(d_images);
    delete[] h_all_images;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
