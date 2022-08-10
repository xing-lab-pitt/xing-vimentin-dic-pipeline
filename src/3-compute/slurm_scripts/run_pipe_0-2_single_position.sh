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
#SBATCH --partition=any_gpu

# number of requested nodes
#SBATCH --nodes=1

# number of tasks
#SBATCH --ntasks=1

# standard output & error

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

# check GPU
nvidia-smi -L


# define paths & files
script_dir=$1
tools_dir=$2
dat_dir=$3
img_path=$4
output_path=$5
ori_dir=$6
position_label=$7
reg_seg_wts_file=$8
icnn_am_wts_file=$9
icnn_seg_wts_file=${10}

# creating directory on /scr folder of compute node & cd into it
user=$(whoami)
job_dir=${user}_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.dcb.private.net
mkdir /scr/$job_dir

pwd
rsync -ra ${script_dir}/* /scr/$job_dir/
rsync -ra ${tools_dir}/* /scr/$job_dir/

cd /scr/$job_dir


# experiment settings
dic_channel_label=C1
pcna_chan_label=C2
vim_chan_label=C3

# initialize
module load anaconda/3-cluster
module load cuda/11.1
source activate tf1

#### 0 preprocess ###
#python 0_img_crop.py $ori_dir ${dat_dir}/ori_cropped $position_label
#python 0_img_resize.py ${dat_dir}/ori_cropped $img_path $position_label
#echo 'step0 complete'

### 1_img_edt ###
model_mode='reg_seg'
python pipe_1_img_edt.py $img_path $output_path $reg_seg_wts_file $dic_channel_label $model_mode 
echo 'step1 complete'
tree $output_path


### 2_edt_watershed ###
model_obj_d=128
small_obj_thres=1500
python pipe_2_edt_watershed.py $img_path $output_path $icnn_am_wts_file $icnn_seg_wts_file $dic_channel_label $model_obj_d $small_obj_thres
echo 'step2 complete'
tree $output_path


