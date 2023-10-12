// parallel_matrix_multiply_optimized.c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define M 1000
#define N 1000
#define P 1000

#define TILE_SIZE 50  // Adjust this based on your cache size and performance results

double A[M][N];
double B[N][P];
double C[M][P];

void matrixMultiplyParallelOptimized() {
#pragma omp parallel for collapse(2) schedule(dynamic, TILE_SIZE)
    for (int i = 0; i < M; i+=TILE_SIZE) {
        for (int j = 0; j < P; j+=TILE_SIZE) {
            for (int k = 0; k < N; k+=TILE_SIZE) {
                for (int ii = i; ii < i+TILE_SIZE && ii < M; ++ii) {
                    for (int jj = j; jj < j+TILE_SIZE && jj < P; ++jj) {
                        double sum = C[ii][jj];
                        for (int kk = k; kk < k+TILE_SIZE && kk < N; ++kk) {
                            sum += A[ii][kk] * B[kk][jj];
                        }
                        C[ii][jj] = sum;
                    }
                }
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
    matrixMultiplyParallelOptimized();
    // ... Maybe print some results for verification
    return 0;
}
