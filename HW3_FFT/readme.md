# How to run
## Serial（串行）
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


