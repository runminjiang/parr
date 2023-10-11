//
// Created by Jerrimy on 2023/10/11.
//
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
    int arr[] = {10, 7, 8, 9, 1, 5};
    int n = sizeof(arr) / sizeof(arr[0]);

    printf("Original array: \n");
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");

    parallel_quicksort(arr, 0, n - 1);

    printf("Sorted array: \n");
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");

    return 0;
}
