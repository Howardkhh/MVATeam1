#!/bin/bash
#SBATCH -A MST109262            # Account name/project number
#SBATCH -J MVA2023_object_detection
#SBATCH -p gtest             # Partiotion name, gtest is test queue, gp1d, gp2d, and gp4d are queues for overnight jobs (1d = 1 day, 2d = 2 day, and so on)
#SBATCH -N 1                     # Maximum number of nodes to be allocated
#SBATCH --cpus-per-task=4        # Number of cores per srun task
#SBATCH --gres=gpu:4        # allocates each node with 8 GPUs
#SBATCH --ntasks-per-node=4      # allocates each node with 8 srun tasks
#SBATCH --time=00:30:00         ## max time

#SBATCH -o %j.out                # Path to the standard output file
#SBATCH -e %j.err                # Path to the standard error ouput file

module purge

module load miniconda3
module load cuda/10.2

conda activate mva2023

CONFIG=$1

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

srun --kill-on-bad-exit=1 \
    python -u tools/train.py ${CONFIG} --launcher="slurm" ${PY_ARGS}