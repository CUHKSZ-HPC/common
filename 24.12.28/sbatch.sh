#!/bin/bash
#SBATCH --job-name=matmul
#SBATCH --output=MatMul-Output.txt
#SBATCH --partition=WORK
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=40
#SBATCH --gres=gpu:1

echo "OpenMP 40 Cores"
srun -N1 -n1 --cpus-per-task 40 matmul_omp matrix/mat8192x8192_A.txt matrix/mat8192x8192_B.txt out.txt
echo ""

echo "OpenMP + MPI 40 Cores * 8"
srun -N8 -n16 --cpus-per-task 20 --mpi=pmix matmul_omp_mpi matrix/mat8192x8192_A.txt matrix/mat8192x8192_B.txt out.txt 20
echo ""

echo "OpenMP + OpenACC 40 Cores + 1 GPU"
srun -N1 -n1 --cpus-per-task 40 matmul_omp_acc matrix/mat8192x8192_A.txt matrix/mat8192x8192_B.txt out.txt
echo ""

echo "OpenMP + MPI + OpenACC (40 Cores + 1 GPU) * 8"
srun -N8 -n16 --cpus-per-task 20 --mpi=pmix matmul_omp_mpi_acc matrix/mat8192x8192_A.txt matrix/mat8192x8192_B.txt out.txt 20 8 1
echo ""