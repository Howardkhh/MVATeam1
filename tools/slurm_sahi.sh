#!/bin/bash
#SBATCH -A MST109262		# Account name/project number
#SBATCH -J MVA2023_object_detection
#SBATCH -p gp4d             # Partiotion name, gtest is test queue, gp1d, gp2d, and gp4d are queues for overnight jobs (1d = 1 day, 2d = 2 day, and so on)
#SBATCH -N 1                     # Maximum number of nodes to be allocated
#SBATCH --cpus-per-task=4        # Number of cores per srun task
#SBATCH --gres=gpu:1        # allocates each node with 8 GPUs
#SBATCH --ntasks-per-node=1      # allocates each node with 8 srun tasks
## #SBATCH --time=08:00:00         ## max time

#SBATCH -o %j.out                # Path to the standard output file
#SBATCH -e %j.err                # Path to the standard error ouput file

module purge

module load miniconda3
module load cuda/11.4

conda activate mva2023

CONFIG=$1
MODEL=$2
DATADIR=$3
ANNOTATION=$4
# SCORE_THD=$5
# NUM_SIZE=$6
# CROP_SIZE=$7
PY_ARGS=${@:5}

echo $CONFIG

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

srun --kill-on-bad-exit=1 \
    python -u tools/sahi_evaluation.py ${CONFIG} ${MODEL} ${DATADIR} ${ANNOTATION} ${PY_ARGS}

