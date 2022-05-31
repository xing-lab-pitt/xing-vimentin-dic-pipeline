# In[1]: import
import sys

sys.path.insert(0, "C:/Users/14432/OneDrive/Research/Projects/A549_HMGA1/src/4_wk_trajectory/")
sys.path.insert(1, "C:/Users/14432/OneDrive/Research/Projects/A549_HMGA1/src/memes/")

import copy
import glob
import os
import pickle
from collections import OrderedDict
from math import exp, log, pi, sqrt
from os import listdir

import contour_class
import cv2
import image_warp
import numpy as np
import pandas as pd
import pipe_util2
import scipy.interpolate.fitpack as fitpack
import scipy.io as sio
import scipy.misc
import scipy.ndimage as ndimage
import seaborn as sns
import utils
from cell_class import fluor_single_cell, single_cell
from contour_tool import (align_contour_to, align_contours,
                          df_find_contour_points, find_contour_points,
                          generate_contours)
from hmmlearn import hmm
from matplotlib import animation, cm
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from persim import PersImage
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import find_peaks, medfilt
from scipy.stats import kde
from skimage import measure
from skimage.color import label2rgb
from skimage.feature import peak_local_max
from skimage.io import imread
from skimage.measure import label, regionprops
from skimage.morphology import closing, opening
from skimage.segmentation import find_boundaries
from sklearn import (cluster, decomposition, manifold, metrics, mixture,
                     model_selection, preprocessing, random_projection)
from sklearn.neighbors import BallTree, KernelDensity, kneighbors_graph
from sklearn.preprocessing import StandardScaler
from traj_class import fluor_single_cell_traj, single_cell_traj


# In[2]: functions
def generate_single_cell_img(img, seg, img_num, obj_num):
    # single_obj_mask=morphology.binary_dilation(seg==obj_num,morphology.disk(6))
    single_obj_mask = seg == obj_num
    single_obj_mask = label(single_obj_mask)
    rps = regionprops(single_obj_mask)
    candi_r = [r for r in rps if r.label == 1][0]
    candi_box = candi_r.bbox
    single_cell_img = single_obj_mask * img

    crop_cell_img = single_cell_img[candi_box[0] : candi_box[2], candi_box[1] : candi_box[3]]
    crop_cell_img_env = img[candi_box[0] : candi_box[2], candi_box[1] : candi_box[3]]

    crop_single_obj_mask = single_obj_mask[candi_box[0] : candi_box[2], candi_box[1] : candi_box[3]]
    rps = regionprops(crop_single_obj_mask)
    candi_r = [r for r in rps if r.label == 1][0]
    center = candi_r.centroid

    return crop_cell_img, crop_cell_img_env


def find_mother(df, mitosis_df, traj_label):
    daughter_flag = 0
    if traj_label != -1:
        if (mitosis_df["sis1_traj_label"] == traj_label).any() or (mitosis_df["sis2_traj_label"] == traj_label).any():
            daughter_flag = 1
    return daughter_flag


def find_offspring(df, mitosis_df, family_tree, traj_label):
    mother_label = traj_label
    if mother_label != -1 and (mitosis_df["mother_traj_label"] == mother_label).any():
        family_tree[int(mother_label)] = []
        sis1_label = mitosis_df.loc[mitosis_df["mother_traj_label"] == mother_label, "sis1_traj_label"].values[0]
        sis2_label = mitosis_df.loc[mitosis_df["mother_traj_label"] == mother_label, "sis2_traj_label"].values[0]
        if sis1_label != -1:
            family_tree[int(mother_label)].append(int(sis1_label))
        if sis2_label != -1:
            family_tree[int(mother_label)].append(int(sis2_label))
        family_tree = find_offspring(df, mitosis_df, family_tree, sis1_label)
        family_tree = find_offspring(df, mitosis_df, family_tree, sis2_label)
        return family_tree
    else:
        return family_tree


def parse(node, tree):
    if node not in tree:
        yield [node]
    else:
        for next_node in tree[node]:
            for r in parse(next_node, tree):
                yield [node] + r


def find_abnormal_fluor(traj_fluor, traj_t, peak_h=5):
    mask = traj_fluor != 0
    #     inds=np.where(traj_fluor!=0)[0]
    non0_traj_t = traj_t[mask]
    non0_traj_fluor = traj_fluor[mask]
    mean_fluct = np.mean(abs(np.diff(non0_traj_fluor)))

    ind1 = find_peaks(np.diff(non0_traj_fluor) / mean_fluct, height=peak_h)[0] + 1
    ind2 = (
        non0_traj_fluor.shape[0] - (find_peaks(np.diff(np.flip(non0_traj_fluor, 0)) / mean_fluct, height=peak_h)[0]) - 2
    )
    inds = np.unique(np.concatenate((ind1, ind2)))

    abn_t = non0_traj_t[inds]
    abn_inds = np.where(np.in1d(traj_t, abn_t))[0]  # find index of abn_t in traj_t
    return abn_inds


def generate_fluor_long_traj(df, cells, am_record, traj_df, traj, fluor_name, feature_list, fluor_feature_name):

    haralick_labels = [
        "Angular Second Moment",
        "Contrast",
        "Correlation",
        "Sum of Squares: Variance",
        "Inverse Difference Moment",
        "Sum Average",
        "Sum Variance",
        "Sum Entropy",
        "Entropy",
        "Difference Variance",
        "Difference Entropy",
        "Information Measure of Correlation 1",
        "Information Measure of Correlation 2",
        "Maximal Correlation Coefficient",
    ]

    traj_record = pd.DataFrame(traj_df.loc[:, "1" : str(time_span)])
    traj_record = traj_record.values
    traj_quan, traj_len = traj_record.shape[0], traj_record.shape[1]

    traj_xy = []
    traj_feature = []
    traj_contour = []
    traj_cord = []
    traj_seri = []
    traj_am_flag = []

    traj_fluor_feature_values = []
    traj_haralick = []
    traj_norm_haralick = []
    traj_fluor_pca_cord = []
    #     traj_norm_fluor_pca_cord=[]
    for img_num in range(1, traj_len + 1):
        obj_num = traj[img_num - 1]
        if obj_num != -1:
            ind = df.loc[(df["ImageNumber"] == img_num) & (df["ObjectNumber"] == obj_num)].index.tolist()[0]

            if hasattr(cells[ind], "cell_contour") and hasattr(cells[ind], "pca_cord"):
                traj_contour.append(cells[ind].cell_contour.points.flatten())
                traj_cord.append(cells[ind].pca_cord)
                traj_seri.append([img_num, obj_num])
                traj_xy.append([df.loc[ind, "Cell_AreaShape_Center_X"], df.loc[ind, "Cell_AreaShape_Center_Y"]])
                traj_feature.append(df.loc[ind, "Cell_AreaShape_Area":"Cell_AreaShape_Solidity"].values.tolist())

                if ((am_record["ImageNumber"] == img_num) & (am_record["ObjectNumber"] == obj_num)).any():
                    am_flag = np.asscalar(
                        am_record.loc[
                            (am_record["ImageNumber"] == img_num) & (am_record["ObjectNumber"] == obj_num), "am_flag"
                        ].values
                    )
                    traj_am_flag.append(am_flag)
                else:
                    traj_am_flag.append(0)

                if hasattr(cells[ind], fluor_name + "_feature_values"):
                    exec("traj_fluor_feature_values.append(np.array(cells[ind]." + fluor_name + "_feature_values[:3]))")
                    exec("traj_haralick.append(np.array(cells[ind]." + fluor_name + "_feature_values[3]))")
                    exec("traj_norm_haralick.append(np.array(cells[ind]." + fluor_name + "_feature_values[4]))")
                    exec("traj_fluor_pca_cord.append(cells[ind]." + fluor_feature_name[0] + "_pca_cord)")
                else:
                    traj_fluor_feature_values.append(np.zeros((3,)))
                    traj_haralick.append(
                        (
                            np.zeros(
                                13,
                            )
                        )
                    )
                    traj_norm_haralick.append(
                        (
                            np.zeros(
                                13,
                            )
                        )
                    )
                    traj_fluor_pca_cord.append(np.zeros((5,)))

    traj_xy = np.asarray(traj_xy)
    traj_feature = np.asarray(traj_feature)
    traj_contour = np.asarray(traj_contour)
    traj_cord = np.asarray(traj_cord)
    traj_seri = np.asarray(traj_seri)

    traj_am_flag = np.asarray(traj_am_flag)

    traj_fluor_feature_values = np.asarray(traj_fluor_feature_values)

    traj_haralick = np.asarray(traj_haralick)
    traj_norm_haralick = np.asarray(traj_norm_haralick)
    traj_fluor_pca_cord = np.asarray(traj_fluor_pca_cord)

    mask = traj_fluor_feature_values[:, 0] != 0

    abn_inds = find_abnormal_fluor(traj_fluor_feature_values[:, 0], traj_seri[:, 0])
    if len(abn_inds) > 0:
        traj_fluor_feature_values[abn_inds, :] = 0
        traj_haralick[abn_inds, :] = 0
        traj_norm_haralick[abn_inds, :] = 0
        print(traj_fluor_pca_cord.shape)
        traj_fluor_pca_cord[abn_inds, :] = 0

    return (
        traj_feature,
        traj_contour,
        traj_cord,
        traj_seri,
        traj_am_flag,
        [
            traj_fluor_feature_values[:, 0],
            traj_fluor_feature_values[:, 1],
            traj_fluor_feature_values[:, 2],
            traj_haralick,
            traj_norm_haralick,
        ],
        traj_fluor_pca_cord,
    )


# In[3]: define

main_path = "C:/Users/14432/OneDrive/Research/Projects/A549_HMGA1/data/livecell_24hr/replica_1-09-03-21/"
output_path = main_path + "out/"
sct_path = output_path + "sct_10fr_buffer/"
if not os.path.exists(sct_path):
    os.makedirs(sct_path)

# 5 fr buffer
# posi_end=21
# time_span=225
# traj_len_thres=67

# begin_range=151
# end_range=219

# 10 fr buffer
posi_end = 20
buffer_frames = 10
time_span = 361
traj_len_thres = 72
frag_len_thres = traj_len_thres - buffer_frames

begin_range = time_span - traj_len_thres
end_range = time_span - buffer_frames

loss_ratio_thres = 0.5

# In[4]: main

feature_list = ["mean_intensity", "std_intensity", "intensity_range", "haralick", "norm_haralick"]
fluor_feature_name = ["vimentin_haralick", "norm_vimentin_haralick"]
for posi in range(1, posi_end + 1):

    if posi <= 9:
        posi_label = "XY0" + str(posi)
    else:
        posi_label = "XY" + str(posi)
    print(posi_label)

    dir_path = output_path + posi_label + "/"

    df = pd.read_csv(dir_path + "Per_Object_relink.csv")
    am_record = pd.read_csv(dir_path + "am_record.csv")
    traj_df = pd.read_csv(dir_path + "traj_object_num.csv")
    mitosis_df = pd.read_csv(dir_path + "mitosis_record.csv")

    with open(dir_path + "cells/" + "fluor_cells", "rb") as fp:
        cells = pickle.load(fp)

    traj_record = pd.DataFrame(traj_df.loc[:, "1" : str(time_span)])
    traj_record = traj_record.values
    traj_quan, traj_len = traj_record.shape[0], traj_record.shape[1]

    for traj_label in range(1, traj_quan + 1):
        print(traj_label)
        cur_traj = traj_record[traj_label - 1, :]
        traj_start_t = np.where(cur_traj != -1)[0][0] + 1
        daughter_flag = find_mother(df, mitosis_df, traj_label)
        if daughter_flag == 1:
            continue
        if traj_start_t <= begin_range:
            family_tree = {}
            family_tree = find_offspring(df, mitosis_df, family_tree, traj_label=traj_label)
            print("family", family_tree)
            all_branches = list(list(parse(traj_label, family_tree)))
            print("all branch", all_branches)

            for branch in all_branches:
                branch_end_label = branch[-1]
                branch_end_traj = traj_record[branch_end_label - 1, :]
                branch_end_t = np.where(branch_end_traj != -1)[0][-1] + 1
                if branch_end_t >= end_range:
                    long_traj = -1 * np.ones((time_span,))
                    divide_points = []
                    traj_name = ""
                    for sub_label in branch:
                        sub_traj = traj_record[sub_label - 1, :]
                        mask = sub_traj != -1
                        if (np.where(sub_traj != -1)[0][-1] + 1) != branch_end_t:
                            divide_points.append(np.where(sub_traj != -1)[0][-1] + 1)
                        long_traj[mask] = sub_traj[mask]
                        traj_name = traj_name + str(sub_label) + "_"

                    whole_branch = long_traj[traj_start_t - 1 : branch_end_t]
                    long_traj_loss_ratio = whole_branch[whole_branch == -1].shape[0] * 1.0 / whole_branch.shape[0]
                    print(whole_branch.shape[0])
                    if long_traj_loss_ratio < loss_ratio_thres:

                        (
                            traj_feature,
                            traj_contour,
                            traj_cord,
                            traj_seri,
                            traj_am_flag,
                            traj_fluor_feature_values,
                            traj_fluor_pca_cord,
                        ) = generate_fluor_long_traj(
                            df, cells, am_record, traj_df, long_traj, "vimentin", feature_list, fluor_feature_name
                        )
                        traj_sct = fluor_single_cell_traj(traj_seri, traj_contour)
                        traj_sct.set_traj_feature(traj_feature)
                        traj_sct.set_traj_cord(traj_cord)
                        traj_sct.set_traj_divide_points(np.array(divide_points))
                        traj_sct.set_traj_am_flag(traj_am_flag)
                        traj_sct.set_traj_fluor_features("vimentin", feature_list, traj_fluor_feature_values)
                        traj_sct.set_traj_fluor_pca_cord(fluor_feature_name[0], traj_fluor_pca_cord)

                        if branch_end_t == time_span:
                            if branch_end_label == traj_label:
                                with open(
                                    sct_path + "fluor_sct_" + posi_label + "_" + traj_name + "long_traj", "wb"
                                ) as fp:
                                    pickle.dump(traj_sct, fp)
                            else:
                                with open(
                                    sct_path + "indirect_fluor_sct_" + posi_label + "_" + traj_name + "long_traj", "wb"
                                ) as fp:
                                    pickle.dump(traj_sct, fp)

                        else:
                            if branch_end_label == traj_label:
                                with open(
                                    sct_path
                                    + "frag_fluor_sct_"
                                    + posi_label
                                    + "_"
                                    + traj_name
                                    + "long_traj_label_"
                                    + str(branch_end_label)
                                    + "_et_"
                                    + str(branch_end_t)
                                    + "_obj_"
                                    + str(int(traj_seri[-1][1])),
                                    "wb",
                                ) as fp:
                                    pickle.dump(traj_sct, fp)
                            else:
                                with open(
                                    sct_path
                                    + "frag_indirect_fluor_sct_"
                                    + posi_label
                                    + "_"
                                    + traj_name
                                    + "long_traj"
                                    + str(branch_end_label)
                                    + "_et_"
                                    + str(branch_end_t)
                                    + "_obj_"
                                    + str(int(traj_seri[-1][1])),
                                    "wb",
                                ) as fp:
                                    pickle.dump(traj_sct, fp)
