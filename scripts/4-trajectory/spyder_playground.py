# In[0]:load
import sys

sys.path.insert(0, "/home/thomas/research/projects/a549_40x/scripts/4_wk_trajectory/")
sys.path.insert(1, "/home/thomas/research/projects/a549_40x/scripts/memes/")

import numpy as np
import numpy.ma
from scipy.ndimage import distance_transform_edt
import scipy.ndimage
import scipy.sparse
import pandas as pd
from os import listdir
from skimage.io import imread
from scipy.optimize import linear_sum_assignment
import pickle
import sys

from track_module import (
    compute_overlap_matrix,
    compute_overlap_pair,
    compute_overlap_single,
    generate_traj_seri,
    relabel_traj,
    record_traj_start_end,
    judge_mol_type,
    search_false_link,
    judge_border,
    find_border_obj,
    break_link,
    connect_link,
    false_seg_mark,
    judge_traj_am,
    judge_apoptosis_tracklet,
    traj_start_end_info,
    am_obj_info,
    compute_specific_overlap,
    compute_cost,
    calculate_area_penalty,
    find_am_sisters,
    cal_cell_fusion,
    cal_cell_split,
    find_mitosis_pairs_to_break,
    find_fuse_pairs_to_break,
    find_split_pairs_to_break,
    judge_fuse_type,
    judge_split_type,
    find_uni,
    get_mitotic_triple_scores,
    cal_size_correlation,
    search_wrong_mitosis,
)


import pipe_util2

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
