// parallel_block_matrix_multiply.c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define M 1000
#define N 1000
#define P 1000

#define BLOCK_SIZE 250

double A[M][N];
double B[N][P];
double C[M][P];

void blockMultiply(int rowA, int colA, int rowB, int colB, int size) {
    for (int i = rowA; i < rowA + size; i++) {
        for (int j = colB; j < colB + size; j++) {
            for (int k = colA; k < colA + size; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void matrixMultiplyBlockParallel() {
#pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i += BLOCK_SIZE) {
        for (int j = 0; j < P; j += BLOCK_SIZE) {
            for (int k = 0; k < N; k += BLOCK_SIZE) {
                blockMultiply(i, k, k, j, BLOCK_SIZE);
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
    matrixMultiplyBlockParallel();
    // ... Maybe print some results for verification
    return 0;
}
