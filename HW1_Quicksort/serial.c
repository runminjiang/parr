#include <stdio.h>
#include <stdlib.h>
#include <time.h>  // 引入计时库

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

void quicksort(int arr[], int low, int high) {
    if (low < high) {
        int pivot = partition(arr, low, high);

        quicksort(arr, low, pivot - 1);
        quicksort(arr, pivot + 1, high);
    }
}

int main() {
    const int N = 1000000;  // 设置数组大小为百万
    int arr[N];

    // 使用随机数填充数组
    for (int i = 0; i < N; i++) {
        arr[i] = rand() % (N * 10);  // 假设数字范围为0到10000000
    }

    printf("Original array (first 20 numbers): \n");
    // 打印前20个数字，以避免控制台上显示百万个数字
    for (int i = 0; i < 20 && i < N; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n... (and so on for 1,000,000 numbers)\n");

    clock_t start, end;
    start = clock();  // 开始计时
    quicksort(arr, 0, N - 1);
    end = clock();  // 结束计时

    printf("Sorted array (first 20 numbers): \n");
    // 同样地，只打印排序后的前20个数字
    for (int i = 0; i < 20 && i < N; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n... (and so on for 1,000,000 numbers)\n");

    // 输出所花费的时间
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time taken for sorting: %f seconds\n", time_spent);

    return 0;
}
