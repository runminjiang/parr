#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define M 1000
#define N 1000
#define P 1000
#define TILE_SIZE 4  // 根据实际测试调整这个值，寻找最佳性能

double A[M][N];
double B[N][P];
double C[M][P];

void matrixMultiplyParallel() {
    #pragma omp parallel for schedule(guided)  // 使用guided调度策略
    for (int i = 0; i < M; i += TILE_SIZE) {
        for (int j = 0; j < P; j += TILE_SIZE) {
            for (int k = 0; k < N; k += TILE_SIZE) {
                for (int ii = i; ii < i + TILE_SIZE && ii < M; ++ii) {
                    for (int jj = j; jj < j + TILE_SIZE && jj < P; ++jj) {
                        double sum = C[ii][jj];
                        for (int kk = k; kk < k + TILE_SIZE && kk < N; ++kk) {
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

    double start_time = omp_get_wtime();

    matrixMultiplyParallel();

    double end_time = omp_get_wtime();
    double duration = end_time - start_time;

    printf("Time taken for matrix multiplication: %f seconds\n", duration);

    return 0;
}
