#!/bin/bash

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#                     Slurm Construction Section
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# job name
#SBATCH --job-name=mbell-53deb5_run_all_12

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
# #SBATCH --output=/net/capricorn/home/xing/tch42/Projects/Nikon_A549/scripts/hj_modify/run_all_out.o

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

# experiment settings
DIC_chan_label=C1
pcna_chan_label=C2
vim_chan_label=C3

# initialize
rsync -ra ${script_dir}/* .
rsync -ra ${tools_dir}/* .
module load anaconda/3-cluster
module load cuda/11.1
eval "$(conda shell.bash hook)"
source activate tf1

#### 0 preprocess ###
#python 0_img_crop.py $ori_dir ${dat_dir}/ori_cropped $position_label
#python 0_img_resize.py ${dat_dir}/ori_cropped $img_path $position_label
#echo 'step0 complete'

#### 1_img_edt ###
#model_mode='reg_seg'
#python pipe_1_img_edt.py $img_path $output_path $reg_seg_wts_file $DIC_chan_label $model_mode 
#echo 'step1 complete'
#
#### 2_edt_watershed ###
#model_obj_d=128
#small_obj_thres=1500
#python pipe_2_edt_watershed.py $img_path $output_path $icnn_am_wts_file $icnn_seg_wts_file $DIC_chan_label $model_obj_d $small_obj_thres
#echo 'step2 complete'
#
#### cell_profiler ###
#cellprof=/net/capricorn/home/xing/weikang/anaconda3/envs/cellprofiler
#bash run_cp.sh $output_path $script_dir $cellprof
#echo 'step cell profiler complete'
#
#python pipe_3_traj_reorganize_1st.py $img_path $output_path $icnn_seg_wts_file $DIC_chan_label
#echo 'step3 complete'
#python pipe_4_traj_reorganize_2nd.py $output_path
#echo 'step4 complete'
#python pipe_5_traj_reorganize_3rd.py $output_path
#echo 'step5 complete'
#python pipe_6_build_single_cell.py $output_path
#echo 'step6 complete'
#
## remember to modify file names in this script
## move segmentation results to different folders
#
##### calculate mean contours and pca modes ###
##python pipe_meancontour_and_pcamodes.py $output_path
##mkdir -p ${output_path}/contour
##cp /net/capricorn/home/xing/weikang/wwk/210309_2ng_tgf_a549/g1/data/mean_cell_contour ${output_path}/contour
#
## after calculate mean contours
#mean_contour_path=${dat_dir}/contour/mean_cell_contour
#python pipe_7_cell_contours_calculation.py $output_path $mean_contour_path 
#echo 'step7 complete'
#
## modify fluor_interval: check visually or in README.md
#python pipe_8_haralick_calculation.py $img_path $output_path $vim_chan_label
#echo 'step8 complete'

python pipe_9_pcna_calculations.py $img_path $output_path $pcna_chan_label
echo 'step9 complete'
