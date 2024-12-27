#include <cstdio>
#include <cstdlib>
#include <cmath>

// Size of array
const int N = 1048576;

int main() {
    
	// Allocate memory for arrays A, B, and C on host
	double *A = new double[N];
	double *B = new double[N];
	double *C = new double[N];

    // Fill host arrays A and B
	for(int i=0; i<N; i++)
	{
		A[i] = 1.0;
		B[i] = 2.0;
	}

    #pragma acc enter data copyin(A[0 : N], B[0 : N], C[0 : N])
    #pragma acc update device(A[0 : N], B[0 : N], C[0 : N])
    #pragma acc parallel present(A[0 : N], B[0 : N], C[0 : N]) num_gangs(1024)
    #pragma acc loop independent
    for (int i = 0; i < N; i++)
        C[i] = A[i] + B[i];
    #pragma acc update self(C[0 : N])
    #pragma acc data copyout(C[0 : N])
    // Verify results
    double tolerance = 1.0e-14;
	for(int i=0; i<N; i++)
	{
		if( fabs(C[i] - 3.0) > tolerance)
		{ 
			printf("\nError: value of C[%d] = %d instead of 3.0\n\n", i, C[i]);
			exit(1);
		}
	}	

	// Free CPU memory
	delete[] A;
	delete[] B;
	delete[] C;

        printf("\n---------------------------\n");
	printf("__SUCCESS__\n");
	printf("---------------------------\n");
	printf("N                 = %d\n", N);
	printf("Numgangs          = %d\n", 1024);
	printf("---------------------------\n\n");

	return 0;

}
