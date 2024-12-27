#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void matmul(int *A, int *B, int *Out, int M, int K, int N) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int sum = 0;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            Out[i * N + j] = sum;
        }
    }
}

void matmul_gpu(int *A, int *B, int *Out, int M, int K, int N) {
    #pragma acc data copyin(A[0 : M * K], B[0 : K * N]) copyout(Out[0 : M * N])
    #pragma acc parallel loop independent collapse(2)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int sum = 0;
            #pragma acc loop reduction(+:sum)
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            Out[i * N + j] = sum;
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s input_file1 input_file2 output_file\n", argv[0]);
        return 1;
    }
    int M, K, K2, N;
    FILE *fp1 = fopen(argv[1], "r"), *fp2 = fopen(argv[2], "r");

    fscanf(fp1, "%d%d", &M, &K);
    fscanf(fp2, "%d%d", &K2, &N);
    if (K != K2) {
        fprintf(stderr, "Matrix dimension does not match!\n");
        return 1;
    }
    int *A = (int*)malloc(M * K * sizeof(int));
    int *B = (int*)malloc(K * N * sizeof(int));
    int *Out = (int*)malloc(M * N * sizeof(int));

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            fscanf(fp1, "%d", A + i * K + j);
        }
    }

    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < N; ++j) {
            fscanf(fp2, "%d", B + i * N + j);
        }
    }
    // Testing performance
    float ratio = 0;
    for (int i = 0; i < 10; ++i) {
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        matmul(A, B, Out, M, K, N);
        clock_gettime(CLOCK_MONOTONIC, &end);
        double elapsed = ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9) * 1000;
        printf("Execution time: %fms\n", elapsed);
        struct timespec start_gpu, end_gpu;
        clock_gettime(CLOCK_MONOTONIC, &start_gpu);
        matmul_gpu(A, B, Out, M, K, N);
        clock_gettime(CLOCK_MONOTONIC, &end_gpu);
        double elapsed_gpu = ((end_gpu.tv_sec - start_gpu.tv_sec) + (end_gpu.tv_nsec - start_gpu.tv_nsec) / 1e9) * 1000;
        printf("Execution time GPU: %fms\n", elapsed_gpu);
        ratio += elapsed / elapsed_gpu;
    }
    printf("1 GPU to 40 CPU ratio: %f\n", ratio / 10);
    fclose(fp1);
    fclose(fp2);
    free(A);
    free(B);
    free(Out);
}