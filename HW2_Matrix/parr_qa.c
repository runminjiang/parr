#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define M 1000
#define N 1000
#define P 1000

double A[M][N];
double B[N][P];
double C[M][P];

void matrixMultiplyParallel() {
#pragma omp parallel for collapse(2) schedule(dynamic) firstprivate(C)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < P; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}
void initializeMatrices() {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = rand() % 10;
        }
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < P; j++) {
            B[i][j] = rand() % 10;
        }
    }
}

int main() {
    initializeMatrices();

    double start_time = omp_get_wtime();

    matrixMultiplyParallel();

    double end_time = omp_get_wtime();
    double duration = end_time - start_time;

    printf("Time taken for matrix multiplication: %f seconds\n", duration);

    return 0;
}
