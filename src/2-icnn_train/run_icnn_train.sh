#!/bin/bash

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#                     Slurm Construction Section
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# job name
#SBATCH --job-name=mbell-53deb5_run_all_12

#SBATCH --gres=gpu:1
#SBATCH --exclude=g019
#SBATCH --cpus-per-task=16

#SBATCH --mem=32G

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
script_dir=/net/capricorn/home/xing/tch42/Projects/A549_144hr/src/2_icnn_train
tools_dir=/net/capricorn/home/xing/tch42/Projects/A549_144hr/src/memes
dat_dir=/net/capricorn/home/xing/tch42/Projects/A549_144hr/data
train_path=${dat_dir}/train/icnn_seg
wts_path=${dat_dir}/wts/icnn_seg

# initialize
rsync -ra ${script_dir}/* .
rsync -ra ${tools_dir}/* .
module load anaconda/3-cluster
module load cuda/11.1
eval "$(conda shell.bash hook)"
source activate tf1

obj_h=128
obj_w=128

# icnn_train
wts_file=seg_gen_on_comb_off_alph_15E-4_run_1-10-26-21.hdf5
nb_class=3
train_mode=seg
python icnn_train.py $train_path $wts_path $wts_file $obj_h $obj_w $nb_class $train_mode
