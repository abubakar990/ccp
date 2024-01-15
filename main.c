#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define START_N 1000
#define END_N 10000
#define STEP_N 1000
#define BLOCK_SIZE 32

// Function to initialize a matrix with random values
void initializeMatrix(int** matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[i][j] = rand() % 100;
        }
    }
}

// Serial matrix multiplication
void serialMatrixMultiplication(int** A, int** B, int** result, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            result[i][j] = 0;
            for (int k = 0; k < size; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Parallel matrix multiplication using OpenMP with cache blocking
void parallelMatrixMultiplication(int** A, int** B, int** result, int size) {
    #pragma omp parallel for collapse(3) schedule(static, 1)
    for (int ii = 0; ii < size; ii += BLOCK_SIZE) {
        for (int jj = 0; jj < size; jj += BLOCK_SIZE) {
            for (int kk = 0; kk < size; kk += BLOCK_SIZE) {
                for (int i = ii; i < ii + BLOCK_SIZE && i < size; i++) {
                    for (int j = jj; j < jj + BLOCK_SIZE && j < size; j++) {
                        for (int k = kk; k < kk + BLOCK_SIZE && k < size; k++) {
                            result[i][j] += A[i][k] * B[k][j];
                        }
                    }
                }
            }
        }
    }
}


int main() {
    printf("%-10s%-20s%-20s%-20s\n", "N", "Serial Time (s)", "Parallel Time (s)", "Serial/Parallel Ratio");

    for (int N = START_N; N <= END_N; N += STEP_N) {
        // Allocate memory for matrices
        int** A = (int**)malloc(N * sizeof(int*));
        int** B = (int**)malloc(N * sizeof(int*));
        int** result = (int**)malloc(N * sizeof(int*));

        for (int i = 0; i < N; i++) {
            A[i] = (int*)malloc(N * sizeof(int));
            B[i] = (int*)malloc(N * sizeof(int));
            result[i] = (int*)malloc(N * sizeof(int));
        }

        // Initialize matrices with random values
        initializeMatrix(A, N);
        initializeMatrix(B, N);

        // Measure time for serial execution
        double start_serial = omp_get_wtime();
        serialMatrixMultiplication(A, B, result, N);
        double end_serial = omp_get_wtime();
        double serial_time = end_serial - start_serial;

        // Measure time for parallel execution
        double start_parallel = omp_get_wtime();
        parallelMatrixMultiplication(A, B, result, N);
        double end_parallel = omp_get_wtime();
        double parallel_time = end_parallel - start_parallel;

        // Print the results in a table
        printf("%-10d%-20f%-20f%-20f\n", N, serial_time, parallel_time, serial_time / parallel_time);

        // Free allocated memory
        for (int i = 0; i < N; i++) {
            free(A[i]);
            free(B[i]);
            free(result[i]);
        }
        free(A);
        free(B);
        free(result);
    }

    return 0;
}
