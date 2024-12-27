#include <immintrin.h>

#include <cstddef>
#include <vector>

#include "util.hpp"

using Matrix = std::vector<std::vector<int>>;

Matrix transpose(const Matrix &A) {
  const auto M = A.size();
  const auto N = A[0].size();

  Matrix transposed(N, std::vector<int>(M));

  for (size_t row = 0; row < M; ++row) {
    for (size_t col = 0; col < N; ++col) {
      transposed[col][row] = A[row][col];
    }
  }

  return transposed;
}

Matrix matrixMul(const Matrix &A, const Matrix &B_T) {
  const size_t M = A.size();
  const size_t N = A[0].size();
  const size_t P = B_T.size();

  Matrix C(M, std::vector<int>(P, 0));

  for (size_t row = 0; row < M; ++row) {
    for (size_t col = 0; col < P; ++col) {
      size_t k = 0;
      __m256i sum_vec = _mm256_setzero_si256();

      // Process 8 elements at a time
      for (; k + 8 <= N; k += 8) {
        // Load 8 integers from A and B_T
        __m256i vecA =
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&A[row][k]));
        __m256i vecB =
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&B_T[col][k]));
        __m256i prod = _mm256_mullo_epi32(vecA, vecB);
        sum_vec = _mm256_add_epi32(sum_vec, prod);
      }

      // Sum the elements of sum_vec
      int32_t sum_array[8];
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(sum_array), sum_vec);
      int sum = 0;
      for (int idx = 0; idx < 8; ++idx) {
        sum += sum_array[idx];
      }

      // Handle remaining elements
      for (; k < N; ++k) {
        sum += A[row][k] * B_T[col][k];
      }

      C[row][col] = sum;
    }
  }

  return C;
}

int main(int argc, char *argv[]) {
  std::ios::sync_with_stdio(false);
  Matrix A = readFromFile("A.matrix");
  Matrix B = readFromFile("B.matrix");
  Matrix B_T = transpose(B);

  Matrix C = matrixMul(A, B_T);

  saveToFile(C, "simd.matrix");

  return 0;
}