# How to run
## Pre-processing
以下是将MNIST数据集处理成能被C++程序处理的预处理步骤。\
1. 进入example文件夹
``cd /HW4_Image/example``
2. 编译main.cpp程序: \
``g++ -o mnist_example main.cpp -I../include -std=c++11``
3. 运行实例程序: \
``./mnist_example``

## Serial（串行）
进入seg文件夹
``cd /HW4_Image/seg``
编译：
``g++ serial.cpp -o serial`` \
测试：
``time ./serial``

## Parr（CUDA并行）
编译：
``nvcc program_name.cu -o program_name`` \
测试：
``time ./program_name`` \
内核测试：
``nvprof ./program_name``


## Reference
[1] MNIST: http://yann.lecun.com/exdb/mnist/ \
[2] Simple C++ reader for MNIST dataset: https://github.com/wichtounet/mnist