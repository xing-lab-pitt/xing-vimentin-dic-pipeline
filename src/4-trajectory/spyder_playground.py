# In[0]:load
import sys

sys.path.insert(0, "/home/thomas/research/projects/a549_40x/src/4_wk_trajectory/")
sys.path.insert(1, "/home/thomas/research/projects/a549_40x/src/3-compute/tools/")

import pickle
import sys
from os import listdir

import numpy as np
import numpy.ma
import pandas as pd
import pipe_util2
import scipy.ndimage
import scipy.sparse
from scipy.ndimage import distance_transform_edt
from scipy.optimize import linear_sum_assignment
from skimage.io import imread
from track_module import (am_obj_info, break_link, cal_cell_fusion,
                          cal_cell_split, cal_size_correlation,
                          calculate_area_penalty, compute_cost,
                          compute_overlap_matrix, compute_overlap_pair,
                          compute_overlap_single, compute_specific_overlap,
                          connect_link, false_seg_mark, find_am_sisters,
                          find_border_obj, find_fuse_pairs_to_break,
                          find_mitosis_pairs_to_break,
                          find_split_pairs_to_break, find_uni,
                          generate_traj_seri, get_mitotic_triple_scores,
                          judge_apoptosis_tracklet, judge_border,
                          judge_fuse_type, judge_mol_type, judge_split_type,
                          judge_traj_am, record_traj_start_end, relabel_traj,
                          search_false_link, search_wrong_mitosis,
                          traj_start_end_info)

init_img_num = 31
init_obj_num = 78

# In[2]:main

main_path = "/home/thomas/research/projects/a549_40x/data/"
output_path = main_path + "out/pcna/01-13-22_72hr_no-treat/XY1/"


seg_path = dir_path + "seg/"
seg_img_list = sorted(listdir(seg_path))
df = pd.read_csv(dir_path + "Per_Object_modify.csv")
am_record = pd.read_csv(dir_path + "am_record.csv")

mother_cells = []
daughter_cells = []
mitosis_labels = []

traj_start, traj_end, traj_start_xy, traj_end_xy, traj_start_area, traj_end_area = traj_start_end_info(df)
F = np.column_stack((traj_start_xy, traj_start, np.expand_dims(traj_start_area[:], axis=1)))
L = np.column_stack((traj_end_xy, traj_end, np.expand_dims(traj_end_area[:], axis=1)))

# In[3]:main II

mitoses, m_dist, size_simi = get_mitotic_triple_scores(
    F, L, mitosis_max_distance=mitosis_max_dist, size_simi_thres=simi_thres
)
n_mitoses = len(m_dist)


# %%
