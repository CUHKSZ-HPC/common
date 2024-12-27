#include <stdio.h>
#include <stdlib.h>
// This program adds array a and array b, and then multiply the result by array c, and store the result into array d.

#define N 1024576

typedef unsigned long long ull;

void add(ull *a, ull *b, ull *out, size_t n) {
    #pragma omp parallel for
    for (int i = 0; i < n; ++i)
        out[i] = a[i] + b[i];
}

void mul(ull *a, ull *b, ull *out, size_t n) {
    #pragma omp parallel for
    for (int i = 0; i < n; ++i)
        out[i] = a[i] * b[i];
}

ull sum(ull *a, size_t n) {
    ull res = 0;
    #pragma omp parallel for reduction(+:res)
    for (int i = 0; i < n; ++i)
        res += a[i];
    return res;
}
int main() {
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
    add(a, b, out, N);
    mul(c, out, out, N);
    printf("The result is: %llu\n", sum(out, N));
    free(a);
    free(b);
    free(c);
    free(out);
}