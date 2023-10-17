#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define THRESHOLD 10000
#define MAX_LEVELS 300

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
    int stack[MAX_LEVELS];
    int top = -1;

    stack[++top] = low;
    stack[++top] = high;

    while (top >= 0) {
        high = stack[top--];
        low = stack[top--];

        int pivot = partition(arr, low, high);

        if (high - low > THRESHOLD) {
            #pragma omp task default(none) shared(arr, stack) firstprivate(low, high, pivot, top)
            {
                if (pivot - 1 > low) {
                    stack[++top] = low;
                    stack[++top] = pivot - 1;
                }

                if (pivot + 1 < high) {
                    stack[++top] = pivot + 1;
                    stack[++top] = high;
                }
            }
        } else {
            if (pivot - 1 > low) {
                stack[++top] = low;
                stack[++top] = pivot - 1;
            }

            if (pivot + 1 < high) {
                stack[++top] = pivot + 1;
                stack[++top] = high;
            }
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
