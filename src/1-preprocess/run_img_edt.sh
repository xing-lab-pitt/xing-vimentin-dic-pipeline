#!/bin/bash

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#                     Slurm Construction Section
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# job name
#SBATCH --job-name=test_a549_pcna_reg_pt40

#SBATCH --gres=gpu:1
#SBATCH --exclude=g019,g012,g013
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
main_path=/net/capricorn/home/xing/tch42/Projects/a549_pcna
script_dir=${main_path}/src/1_preprocess
tools_dir=${main_path}/src/memes

# initialize
rsync -ra ${script_dir}/* .
rsync -ra ${tools_dir}/* .
module load anaconda/3-cluster
module load cuda/11.1
eval "$(conda shell.bash hook)"
source activate tf1

### 1_img_edt ###
img_path=${main_path}/data/test/reg/crop
output_path=${main_path}/data/test/reg/edt_pt40_nextgen
reg_seg_wts_file=${main_path}/data/wts/reg/a549_reg_pt40_nextgen.hdf5
model_mode='reg_seg'
python 3_img_edt.py $img_path $output_path $reg_seg_wts_file $model_mode 
echo 'step1 complete'


