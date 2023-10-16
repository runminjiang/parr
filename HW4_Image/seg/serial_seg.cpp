#include <iostream>
#include <vector>
#include "/home/opc/imaging/parr/HW4_Image/mnist/include/mnist/mnist_reader.hpp"


int main() {
    // 读取MNIST数据
    auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("..");

    uint8_t threshold = 128;  // 设置阈值

    // 串行处理训练图像
    for (auto& image : dataset.training_images) {
        for (auto& pixel : image) {
            pixel = (pixel > threshold) ? 255 : 0;  // 基于阈值的分割
        }
    }

    // ... 你可以保存处理后的图像或进行其他操作 ...

    return 0;
}
