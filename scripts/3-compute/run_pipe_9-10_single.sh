#!/bin/bash

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#                     Slurm Construction Section
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# job name
#SBATCH --job-name=mbell-53deb5_run_all_12

#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=16

#SBATCH --mem=32G

# partition (queue) declaration
#SBATCH --partition=camacho_gpu

# number of requested nodes
#SBATCH --nodes=1

# number of tasks
#SBATCH --ntasks=1

# standard output & error
# #SBATCH --output=/net/capricorn/home/xing/tch42/Projects/Nikon_A549/scripts/hj_modify/run_all_out.o

# send email about job start and end
#SBATCH --mail-user=tch42@pitt.edu
#SBATCH --mail-type=ALL

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#                     User Construction Section
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# note:
# 1. modify parameters in config.py.
# Note the variables you set below won't be passed to python scripts, and params such as paths, pos are set in config.py
# 2. modify arguments parameters in this run_all.sh
# 3. remember to get rid of leading 0 of position folders, i.e. change position folder name from 01 to 1
# example of remove leading zeros: note we need "/" at the end of the path arg
# bash remove_dir_left_trailing_zero.sh /net/dali/home/mscbio/ken67/weikang_exp_process/2019-06-21_A549_vim_tgf4ng_3d/vimentin/
# 4. carefully modify move_seg_sample.sh: read the expression of filenames
# 5. do not forget to modify fluor_interval (base) tch42@cluster:~/Projects/Intro_Segment/scripts/weik_pipe$ cd cd nanolive_mbell-53deb5/

# node
echo
echo $SLURM_JOB_NODELIST
echo

# creating directory on /scr folder of compute node & cd into it
user=$(whoami)
job_dir=${user}_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.dcb.private.net
mkdir /scr/$job_dir
cd /scr/$job_dir

# main output directory name
date=08-02-21
pipeline=weik
description=altered_param
job_outdir=${pipeline}_${date}_${description}

# define paths & files
script_dir=/net/capricorn/home/xing/tch42/Projects/Nikon_A549/scripts/3_hj_modify
dat_dir=/net/capricorn/home/xing/tch42/Projects/Nikon_A549/data/A549_vim_rfp_tgfb_livecell
out_dir=${dat_dir}/out/
top_path=${out_dir}/${job_outdir}

# initialize
rsync -ra ${script_dir}/* .
module load anaconda/3-cluster
module load cuda/11.1
eval "$(conda shell.bash hook)"
source activate tf1

### 9 morph pca ###
python pipe_9_morph_pca.py $top_path

### 10 
