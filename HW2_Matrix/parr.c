// parallel_matrix_multiply.c
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
#pragma omp parallel for
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < P; j++) {
            C[i][j] = 0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
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
    matrixMultiplyParallel();
    // ... Maybe print some results for verification
    return 0;
}
