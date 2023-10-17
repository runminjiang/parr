#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

int parallel_partition(int arr[], int low, int high) {
    int pivot = arr[high];
    int i = low - 1;
    int j;

#pragma omp parallel for private(j) schedule(dynamic) reduction(+:i)
    for (j = low; j <= high - 1; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}

void parallel_quicksort(int arr[], int low, int high) {
    if (low < high) {
        int pivot = parallel_partition(arr, low, high);

#pragma omp task
        parallel_quicksort(arr, low, pivot - 1);

#pragma omp task
        parallel_quicksort(arr, pivot + 1, high);

#pragma omp taskwait
    }
}

int main() {
    const int N = 100000;  // 设置数组大小为十万
    int arr[N];

    // 使用随机数填充数组
    for (int i = 0; i < N; i++) {
        arr[i] = rand() % (N * 10);  // 假设数字范围为0到1000000
    }

    printf("Original array (first 20 numbers): \n");
    // 打印前20个数字
    for (int i = 0; i < 20; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n... (and so on for 100,000 numbers)\n");

#pragma omp parallel
    {
#pragma omp single
        {
            parallel_quicksort(arr, 0, N - 1);
        }
    }

    printf("Sorted array (first 20 numbers): \n");
    // 同样地，只打印排序后的前20个数字
    for (int i = 0; i < 20; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n... (and so on for 100,000 numbers)\n");

    return 0;
}


//并行分区：我们可以通过并行扫描来实现。
//任务切割：使用task而不仅仅是sections进行更细粒度的并行化。
//动态调度：对于不均匀分布的数据，动态调度可能提供更好的负载平衡。