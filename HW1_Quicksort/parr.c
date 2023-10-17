#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>  // 引入OpenMP库

#define THRESHOLD 10000  // 阈值

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
        
        // 当子数组大小超过阈值时进行并行
        if (high - low > THRESHOLD) {
            #pragma omp task firstprivate(arr, low, pivot)
            quicksort(arr, low, pivot - 1);
            
            #pragma omp task firstprivate(arr, high, pivot)
            quicksort(arr, pivot + 1, high);
        } else {
            quicksort(arr, low, pivot - 1);
            quicksort(arr, pivot + 1, high);
        }
    }
}

int main() {
    const int N = 1000000;
    int arr[N];

    for (int i = 0; i < N; i++) {
        arr[i] = rand() % (N * 10);
    }

    printf("Original array (first 20 numbers): \n");
    for (int i = 0; i < 20 && i < N; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n... (and so on for 1,000,000 numbers)\n");

    double start_time, end_time;
    start_time = omp_get_wtime();

    #pragma omp parallel
    {
        #pragma omp single nowait
        quicksort(arr, 0, N - 1);
    }

    end_time = omp_get_wtime();

    printf("Sorted array (first 20 numbers): \n");
    for (int i = 0; i < 20 && i < N; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n... (and so on for 1,000,000 numbers)\n");

    printf("Time taken for sorting: %f seconds\n", end_time - start_time);

    return 0;
}
