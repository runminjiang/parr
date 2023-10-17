#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

int partition(int arr[], int low, int high) {
    int pivot = arr[high];
    int i = (low - 1);

    for (int j = low; j <= high - 1; j++) {
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
        int pivot = partition(arr, low, high);

#pragma omp parallel sections
        {
#pragma omp section
            {
                parallel_quicksort(arr, low, pivot - 1);
            }

#pragma omp section
            {
                parallel_quicksort(arr, pivot + 1, high);
            }
        }
    }
}

int main() {
    const int N = 1000000;  // 设置数组大小为十万
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

    parallel_quicksort(arr, 0, N - 1);

    printf("Sorted array (first 20 numbers): \n");
    // 同样地，只打印排序后的前20个数字
    for (int i = 0; i < 20; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n... (and so on for 100,000 numbers)\n");

    return 0;
}
