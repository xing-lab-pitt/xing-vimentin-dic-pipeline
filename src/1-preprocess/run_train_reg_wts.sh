#!/bin/bash

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#                     Slurm Construction Section
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# job name
#SBATCH --job-name=purity_07

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16

#SBATCH --mem=40G

# partition (queue) declaration
#SBATCH --partition=dept_gpu

# number of requested nodes
#SBATCH --nodes=1

# number of tasks
#SBATCH --ntasks=1

# standard output & error
# #SBATCH --output=/net/capricorn/home/xing/tch42/Projects/Nikon_A549/src/hj_modify/run_all_out.o

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
# 5. do not forget to modify fluor_interval (base) tch42@cluster:~/Projects/Intro_Segment/src/weik_pipe$ cd cd nanolive_mbell-53deb5/

# node
echo
echo $SLURM_JOB_NODELIST
echo

# creating directory on /scr folder of compute node & cd into it
user=$(whoami)
job_dir=${user}_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.dcb.private.net
mkdir /scr/$job_dir
cd /scr/$job_dir

# define paths & files
script_dir=/net/capricorn/home/xing/tch42/Projects/a549_pcna/src/1_preprocess
tools_dir=/net/capricorn/home/xing/tch42/Projects/a549_pcna/src/memes
dat_dir=/net/capricorn/home/xing/tch42/Projects/a549_pcna/data
train_path=${dat_dir}/train/reg/patch/purity_thres_40
wts_path=${dat_dir}/wts/reg
echo ${train_path}

# initialize
rsync -ra ${script_dir}/* .
rsync -ra ${tools_dir}/* .
module load anaconda/3-cluster
module load cuda/11.1
eval "$(conda shell.bash hook)"
source activate tf1

# icnn_train
train_input_path=${train_path}/img
train_gt_path=${train_path}/interior
wts_file=a549_reg_pt40_nextgen.hdf5
mkdir -p wts_path
echo ${wts_file}
python 2_train_reg_wts.py $train_input_path $train_gt_path $wts_path $wts_file
