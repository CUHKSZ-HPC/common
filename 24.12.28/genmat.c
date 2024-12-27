#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s M N output_file\n", argv[0]);
        return 1;
    }
    FILE *fp = fopen(argv[3], "w");
    srand(time(NULL));
    const int M = atoi(argv[1]), N = atoi(argv[2]);
    fprintf(fp, "%d %d\n", M, N);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            fprintf(fp, "%d ", rand() % 256);
        }
        fputs("\n", fp);
    }
    fclose(fp);
}