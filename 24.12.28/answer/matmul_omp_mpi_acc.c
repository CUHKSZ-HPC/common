#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <mpi.h>
#include <immintrin.h>
#define UNUSED(x) (void)(x)

void matmul(int *A, int *B, int *Out, int M, int K, int N, int st, int ed) {
    UNUSED(M);
    int upper = N / 16 * 16;
    #pragma omp parallel for
    for (int i = st; i < ed; ++i) {
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

void matmul_gpu(int *A, int *B, int *Out, int M, int K, int N, int st, int ed) {
    UNUSED(M);
    #pragma acc parallel present(A[st * K : (ed - st) * K], B[0 : K * N], Out[st * N : (ed - st) * N]) loop async(1)
    for (int i = st; i < ed; ++i) {
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
    if (argc != 7) {
        fprintf(stderr, "Usage: %s input_file1 input_file2 output_file thread_per_process node_number gpu_per_node\n", argv[0]);
        return 1;
    }
    MPI_Init(&argc, &argv);
    int numtasks, taskid;
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    int M, K, K2, N;
    FILE *fp1 = fopen(argv[1], "r"), *fp2 = fopen(argv[2], "r");
    int nodenum = atoi(argv[5]), gpus = atoi(argv[6]);
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
    int cpu_start = M / 2;
    int gpu_size = (cpu_start - 1) / (nodenum * gpus) + 1;
    int size = (M - cpu_start - 1) / numtasks + 1;
    int st = taskid * size;
    int ed = (taskid + 1) * size > M ? M : (taskid + 1) * size;
    int gpu_st, gpu_ed;
    omp_set_num_threads(atoi(argv[4]));
    for (int i = 0; i < nodenum; ++i) {
        for (int j = 0; j < gpus; ++j) {
            if (taskid == i * (numtasks / nodenum) + j) {
                gpu_st = (i * gpus + j) * gpu_size;
                gpu_ed = (i * gpus + j + 1) * gpu_size > cpu_start ? cpu_start : (i * gpus + j + 1) * gpu_size;
                if (taskid != 0) {
                    #pragma acc enter data copyin(A[gpu_st * K : (gpu_ed - gpu_st) * K], B[0 : K * N]) create(Out[gpu_st * N : (gpu_ed - gpu_st) * N]) async(1)
                    matmul_gpu(A, B, Out, M, K, N, gpu_st, gpu_ed);
                }
                break;
            }
        }
    }
    if (taskid == 0) {
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        #pragma acc enter data copyin(A[gpu_st * K : (gpu_ed - gpu_st) * K], B[0 : K * N]) create(Out[gpu_st * N : (gpu_ed - gpu_st) * N]) async(1)
        matmul_gpu(A, B, Out, M, K, N, gpu_st, gpu_ed);
        matmul(A, B, Out, M, K, N, st + cpu_start, ed + cpu_start);
        for (int i = 1; i < numtasks; ++i) {
            int st = i * size;
            int ed = (i + 1) * size > M ? M : (i + 1) * size;
            if (st < M) {
                MPI_Recv(Out + (cpu_start + st) * N, (ed - st) * N, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        #pragma acc exit data copyout(Out[gpu_st * N : (gpu_ed - gpu_st) * N]) async(1)
        #pragma acc wait(1)
        for (int i = 0; i < nodenum; ++i) {
            for (int j = 0; j < gpus; ++j) {
                if (i != 0 || j != 0) {
                    gpu_st = (i * gpus + j) * gpu_size;
                    gpu_ed = (i * gpus + j + 1) * gpu_size > cpu_start ? cpu_start : (i * gpus + j + 1) * gpu_size;
                    MPI_Recv(Out + gpu_st * N, (gpu_ed - gpu_st) * N, MPI_INT, i * (numtasks / nodenum) + j, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
        }
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
        fclose(fout);
    }
    else if (st < M) {
        matmul(A, B, Out, M, K, N, st + cpu_start, ed + cpu_start);
        MPI_Send(Out + (cpu_start + st) * N, (ed - st) * N, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    for (int i = 0; i < nodenum; ++i) {
        for (int j = 0; j < gpus; ++j) {
            if (taskid != 0 && taskid == i * (numtasks / nodenum) + j) {
                #pragma acc exit data copyout(Out[gpu_st * N : (gpu_ed - gpu_st) * N]) async(1)
                #pragma acc wait(1)
                MPI_Send(Out + gpu_st * N, (gpu_ed - gpu_st) * N, MPI_INT, 0, 1, MPI_COMM_WORLD);
                break;
            }
        }
    }
    MPI_Finalize();
    fclose(fp1);
    fclose(fp2);
    free(A);
    free(B);
    free(Out);
}