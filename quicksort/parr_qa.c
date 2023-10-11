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
    int arr[] = {10, 7, 8, 9, 1, 5};
    int n = sizeof(arr) / sizeof(arr[0]);

    printf("Original array: \n");
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");

#pragma omp parallel
    {
#pragma omp single
        {
            parallel_quicksort(arr, 0, n - 1);
        }
    }

    printf("Sorted array: \n");
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");

    return 0;
}

//并行分区：我们可以通过并行扫描来实现。
//任务切割：使用task而不仅仅是sections进行更细粒度的并行化。
//动态调度：对于不均匀分布的数据，动态调度可能提供更好的负载平衡。