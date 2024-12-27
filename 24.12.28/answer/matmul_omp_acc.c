#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>
#define UNUSED(x) (void)(x)

void matmul(int *A, int *B, int *Out, int M, int K, int N, int st) {
    int upper = N / 16 * 16;
    #pragma omp parallel for
    for (int i = st; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            // Firstly deal it with 16 integers
            int j = 0;
            __m512i a = _mm512_set1_epi32(A[i * K + k]);
            for (j = 0; j < upper; j += 16) {
                __m512i out = _mm512_loadu_epi32(&Out[i * N + j]);
                __m512i b = _mm512_loadu_epi32(&B[k * N + j]);
                _mm512_storeu_epi32(&Out[i * N + j], 
                    _mm512_add_epi32(out, 
                         _mm512_mullo_epi32(a, b)   
                    ));
            }
            // Deal the rest with masking
            if (j < N) {
                unsigned short mask = (1 << (N - j)) - 1;
                __m512i out = _mm512_maskz_loadu_epi32(mask, &Out[i * N + j]);
                __m512i b = _mm512_maskz_loadu_epi32(mask, &B[k * N + j]);
                _mm512_mask_storeu_epi32(&Out[i * N + j], mask, 
                    _mm512_add_epi32(out, 
                         _mm512_mullo_epi32(a, b)   
                    ));
            }
        }
    }
}

void matmul_gpu(int *A, int *B, int *Out, int M, int K, int N, int ed) {
    UNUSED(M);
    #pragma acc parallel loop async(1)
    for (int i = 0; i < ed; ++i) {
        for (int k = 0; k < K; ++k) {
            int a = A[i * K + k];
            #pragma acc loop gang vector
            for (int j = 0; j < N; ++j) {
                Out[i * N + j] += a * B[k * N + j];
            }
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
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    int ed = M / 2;
    #pragma acc enter data copyin(A[0 : ed * K], B[0 : K * N]) create(Out[0 : ed * N]) async(1)
    matmul_gpu(A, B, Out, M, K, N, ed);
    matmul(A, B, Out, M, K, N, ed);
    #pragma acc exit data copyout(Out[0 : ed * N]) async(1)
    #pragma acc wait(1)
    clock_gettime(CLOCK_MONOTONIC, &end);
    FILE *fout = fopen(argv[3], "w");
    fprintf(fout, "%d %d\n", M, N);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            fprintf(fout, "%d ", Out[i * N + j]);
        }
        fputs("\n", fout);
    }
    printf("Execution time: %fms\n", ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9) * 1000);
    fclose(fp1);
    fclose(fp2);
    fclose(fout);
    free(A);
    free(B);
    free(Out);
}