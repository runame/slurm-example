#!/bin/bash
#SBATCH -p gpu-2080ti                     # partition
#SBATCH --gres=gpu:rtx2080ti:1            # type and number of gpus
#SBATCH --time=01:00:00                   # job will be cancelled after max. 72h
#SBATCH --output=example_%A_%a.out        # name of the output file
#SBATCH --array=1-5                       # separate job for all values in {1,2,3,4,5}

# print info about current job
scontrol show job $SLURM_JOB_ID

# Assuming you have activated your conda environment
python train.py --seed $SLURM_ARRAY_TASK_ID
wait
if [ $SLURM_ARRAY_TASK_ID -eq 1 ]
then
    python test.py
fi
