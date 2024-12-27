#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
// This program adds array a and array b, and then multiply the result by array c, and store the result into array d.

#define N 1048576

typedef unsigned long long ull;

void add(ull *a, ull *b, ull *out, size_t n) {
    for (int i = 0; i < n; ++i)
        out[i] = a[i] + b[i];
}

void mul(ull *a, ull *b, ull *out, size_t n) {
    for (int i = 0; i < n; ++i)
        out[i] = a[i] * b[i];
}

ull sum(ull *a, size_t n) {
    ull res = 0;
    for (int i = 0; i < n; ++i)
        res += a[i];
    return res;
}
int main(int argc, char** argv) {
    ull *a, *b, *c, *out;
    a = (ull*)malloc(N * sizeof(ull));
    b = (ull*)malloc(N * sizeof(ull));
    c = (ull*)malloc(N * sizeof(ull));
    out = (ull*)malloc(N * sizeof(ull));

    for (int i = 0; i < N; ++i) {
        a[i] = i;
        b[i] = 2 * i;
        c[i] = 3 * i;
    }
    MPI_Init(&argc, &argv);

    int numtasks;
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    int taskid;
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);

    int each = (N - 1) / numtasks + 1;
    int start = taskid * each;
    int end = (taskid + 1) * each > N ? N : (taskid + 1) * each;
    int size = end - start;
    ull result = 0, local_sum = 0;
    add(a + start, b + start, out + start, size);
    mul(out + start, c + start, out + start, size);
    local_sum = sum(out + start, size);
    MPI_Reduce(&local_sum, &result, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (taskid == 0)
        printf("The result is: %llu\n", result);
    MPI_Finalize();
    free(a);
    free(b);
    free(c);
    free(out);
}
