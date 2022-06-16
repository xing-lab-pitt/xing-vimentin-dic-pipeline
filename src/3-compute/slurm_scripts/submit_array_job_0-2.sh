#!/bin/bash

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#                     Slurm Construction Section
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# job name
#SBATCH --job-name=xing-pipeline-segs

#SBATCH --gres=gpu:1
#SBATCH --exclude=g019,g102,g104,g122,g012,g013,g131
#SBATCH --cpus-per-task=16

#SBATCH --mem=40G

# partition (queue) declaration
#SBATCH --partition=dept_gpu

# number of requested nodes
#SBATCH --nodes=1

# number of tasks
#SBATCH --ntasks=1

# standard output & error

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

slurm_array_file_path=$1

file_line=$(sed -n "$SLURM_ARRAY_TASK_ID"p $slurm_array_file_path)

eval $file_line


#sbatch --array=1-2 scripts/3_compute/run_pipe_0-2_all_positions_dante.sh