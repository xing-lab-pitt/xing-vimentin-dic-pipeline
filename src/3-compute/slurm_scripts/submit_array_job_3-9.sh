#!/bin/bash

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#                     Slurm Construction Section
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# job name
#SBATCH --job-name=pcna_step_single_pos

#SBATCH --cpus-per-task=32

#SBATCH --mem=32G

# partition (queue) declaration
#SBATCH --partition=dept_cpu

# number of requested nodes
#SBATCH --nodes=1

# number of tasks
#SBATCH --ntasks=1

# standard output & error
# #SBATCH --output=/net/capricorn/home/xing/tch42/Projects/Nikon_A549/scripts/hj_modify/run_all_out.o

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#                     User Construction Section
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

slurm_array_file_path=$1

file_line=$(sed -n "$SLURM_ARRAY_TASK_ID"p $slurm_array_file_path)

eval $file_line
