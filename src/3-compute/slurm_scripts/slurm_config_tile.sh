#!/bin/bash

###################
#  PYTHON INPUTS  #
###################
data_dir=/net/capricorn/home/xing/data/sorted_A549-VIM-PCNA_144hr_4ng-mL_TGFB_YY_Ti2e_2022-5-6/cropped 
#data_dir=/net/capricorn/home/xing/data/sorted_A549-VIM-PCNA_72hr_NoTreat_NA_YY_Ti2e_2022-4-28_complete
#data_dir=/net/capricorn/home/xing/dap182/xing/image_analysis/sample-tutorial

#script_dir=./scripts/3-compute
script_dir=./src/3-compute
tools_dir=./src/3-compute/legacy_utils
out_dir=${data_dir}/out

reg_wts_file=${data_dir}/wts/reg/a549_reg_pt25_no-treat.hdf5 
icnn_am_wts_file=${data_dir}/wts/icnn_am/icnn_am_dc_comb_wk.hdf5
icnn_seg_wts_file=${data_dir}/wts/icnn_seg/icnn_seg_dc_comb_wk.hdf5

#pipeline_steps=0-2
pipeline_steps=3-9