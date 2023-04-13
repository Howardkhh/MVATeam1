#!/bin/bash
#SBATCH -A MST109262            # Account name/project number
#SBATCH -J MVA2023_object_detection
#SBATCH -p gp4d             # Partiotion name, gtest is test queue, gp1d, gp2d, and gp4d are queues for overnight jobs (1d = 1 day, 2d = 2 day, and so on)
#SBATCH -N 1                     # Maximum number of nodes to be allocated
#SBATCH --cpus-per-task=4        # Number of cores per srun task
#SBATCH --gres=gpu:4        # allocates each node with 8 GPUs
#SBATCH --ntasks-per-node=4      # allocates each node with 8 srun tasks
#SBATCH --time=48:00:00         ## max time

#SBATCH -o %j.out                # Path to the standard output file
#SBATCH -e %j.err                # Path to the standard error ouput file


export MASTER_PORT=9488

module purge

module load miniconda3
module load cuda/11.5
module load gcc10
module load cmake

conda activate mva_team1

CONFIG=$1
#RESUME=$2
CHECKPOINT=$2
HNFILE=$3

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
#export NCCL_IB_DISABLE="1"
export NCCL_SOCKET_IFNAME=ib0
# export NCCL_P2P_DISABLE="1"
export NCCL_P2P_LEVEL=PXB

srun --kill-on-bad-exit=1 \
    python3 hard_neg_example_tools/test_hard_neg_example.py \
    --config ${CONFIG} \
    --checkpoint ${CHECKPOINT} \
    --launcher slurm \
    --generate-hard-negative-samples True \
    --hard-negative-file ${HNFILE}\
    --hard-negative-config num_max_det=10 pos_iou_thr=1e-5 score_thd=0.05 nms_thd=0.05 
                                                                        
