import os
import tensorflow as tf
import os
import argparse
import pandas as pd

# # some TF1 config
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)

# main_path='/home/zoro/Desktop/experiment_data/2019-03-10_HK2_fucci/'
# main_path = '/mnt/data0/Ke/weikang_exp_process/2019-05-05_A549_vim/'
# main_path = '/net/dali/home/mscbio/ken67/weikang_exp_process/2019-06-21_A549_vim_tgf4ng_3d/'
# main_path = '/mnt/data0/weikang/a549_05_time_lapse/a549_05_treatment_time_lapse_72_hours/'
# main_path = '/media/weikang_data/rpe_pcna_p21_72_hr_imaging/tiff_files/rpe_pcna_p21_72hr_time_lapse/'
# main_path = '/media/weikang_data/a549_vim_rfp_20_tgfb_88_hour_experiment/A549_2ng_data/'
# main_path = '/media/weikang_data/rpe_pcna_p21_72_hr_imaging/tiff_files/rpe_pcna_p21_72hr_time_lapse/'
# main_path = '/media/weikang_data/rpe_pcna_p21_72_hr_imaging/tiff_files/c1_raw_tiffs/'
main_path = "/net/capricorn/home/xing/huijing/Segmentation/data/1-13-Incucyte/"
weight_file = (
    "/net/capricorn/home/xing/huijing/Segmentation/scripts/vimentin_DIC_segmentation_pipeline/models/dld1.hdf5"
)

# input path should be path of dic
# input_path=main_path+'a549_tif/dic/' # 'img/'
input_path = main_path + "img/"
# input_path=main_path+'img_samples'
output_path = main_path + "output/"
# input_path = '/mnt/data0/weikang/a549_05_time_lapse/a549_05_treatment_time_lapse_72_hours/img_samples/'
# output_path = '/mnt/data0/weikang/a549_05_time_lapse/a549_05_treatment_time_lapse_72_hours/img_samples_reg/'
if not os.path.exists(output_path):
    os.makedirs(output_path)


# TODO: remove in next version: deprecated
posi_start = 1
posi_end = 6

# step 6 cell path
cells_path = main_path + "cells/"
cell_output_path = main_path + "output/"
# vim_input_path=main_path+'a549_tif/vimentin/'
vim_input_path = main_path + "vimentin/"

fluor_cells_path = main_path + "cells/"
fluor_interval = 3

# mean_cell_contour_path = output_path+'mean_cell_contour'
mean_cell_contour_path = "./A549_emt_mean_cell_contour"
sample_path = main_path + "/output/seg_sample"

# ----traj parameter setting -----------
sct_path = main_path + "single_cell_traj/"

# depend on: cell type, time interval-------
# 1/2 max distance between two trajectory ends: cell type and time interval
mitosis_max_dist = 75
# size_similarity=1-abs(size1-size2)/(size1+size2) size similarity between
# two possible sister cells
size_similarity_threshold = 0.7
# for judging possible mitosis, if exceed this value, probably a single
# cell: time interval
traj_len_threshold = 6
# size correlation between two sister cells, if exceed this value
# ,probably mitosis
size_corr_threshold = 0.5
mature_time = 60  # time from the cell born to divide: depend on cell type and time interval


args = None


def run_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="./pipeline_results/")
    parser.add_argument("--icnn_seg_weights_path", type=str, default="C0")
    parser.add_argument("--mean_contour_path", type=str, default="./data/mean_contour")
    parser.add_argument("--vim_chan_label", type=str, default="C1")
    parser.add_argument("--pcna_chan_label", type=str, default="C2")
    parser.add_argument("--segmentation_weights_path", type=str, required=True)

    global args
    args = parser.parse_args()
    update_globals_by_args(args)


def update_globals_by_args(args):
    print(">>>>> updating globals according to the following args:")
    print(args)
    # TODO
    pass
