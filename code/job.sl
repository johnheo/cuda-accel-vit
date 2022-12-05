#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=1:00:00
#SBATCH --partition=gpu 
#SBATCH --output=results/gpujob.out
#SBATCH --gres=gpu:p100:1

module purge
module load nvidia-hpc-sdk


#nsys profile --cuda-memory-usage true ./linear 800 384 6

#ncu --target-processes all --kernel-id ::regex:^.*matrixMultiply.*$:1 --set full -o linear_ncu srun ./linear 800 384 6
#ncu --target-processes all  ./linear 800 384 6
#./go

./go 50 384 6
./go 100 384 6
./go 200 384 6
./go 400 384 6
./go 800 384 6
./go 1000 384 6
