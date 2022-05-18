#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sys
from scipy.spatial.distance import euclidean, cosine
import math
from math import pi
import scipy
from skimage.io import imread
from skimage.measure import label, regionprops
from skimage.color import label2rgb
import os
from os import listdir
from PIL import Image, ImageDraw, ImageFont

from resnet50 import res_model
from skimage import morphology
from scipy import ndimage
import cv2
import time
from skimage.segmentation import clear_border
import pickle
from matplotlib import pyplot as plt
import itertools
from datetime import datetime

from scipy.optimize import linear_sum_assignment
import sqlite3
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
)
from cnn_prep_data import generate_single_cell_img_edt
import glob
import pipe_util2
import config

# ----parameter setting -----------
# depend on: cell type, time interval-------
# 1/2 max distance two traj end for selecting possible mitosis: cell type
# and time interval
mitosis_max_dist = config.mitosis_max_dist

# size_similarity=1-abs(size1-size2)/(size1+size2) size similarity between
# two possible sister cells
size_similarity_threshold = config.size_similarity_threshold

# for judging traj beginning or end is mitosis or apoptosi or not: time
# interval
t_search_range = 3
traj_len_thres = 6  # for judging single cell, if trajectory length is larger than this value, it is probably single cell:time  interval


def icnn_seg_load_weight(icnn_seg_weights):
    obj_h = 128
    obj_w = 128
    input_shape = (obj_h, obj_w, 1)
    nb_class = 3
    icnn_seg = res_model(input_shape, nb_class)
    icnn_seg.load_weights(icnn_seg_weights)
    return icnn_seg


def traj_reconganize1(img_path, output_path, icnn_seg_weights, DIC_channel_label, obj_h=128, obj_w=128):

    """
    :param img_path: the original image folder
    :param output_path: the _output folder.
    :return: generating several files under the _output folder
    """

    icnn = icnn_seg_load_weight(icnn_seg_weights)

    # preparing paths
    print("processing %s" % (img_path), flush=True)
    img_path = pipe_util2.correct_folder_str(img_path)
    img_list = sorted(glob.glob(img_path + "*" + DIC_channel_label + "*"))
    for presplit_cell_index in range(len(img_list)):
        img_list[presplit_cell_index] = os.path.basename(img_list[presplit_cell_index])

    seg_path = pipe_util2.correct_folder_str(output_path) + "seg/"
    seg_img_list = sorted(listdir(seg_path))

    dir_path = pipe_util2.correct_folder_str(output_path)

    # TODO: Per_Object is generated in the current module later, remove following code
    #     df = pd.read_csv(dir_path + 'Per_Object.csv')
    #     relation_df=pd.read_csv(dir_path + 'Per_Relationships.csv')

    # reading cell_track.db and am_record.csv
    conn = sqlite3.connect(os.path.join(dir_path, "cell_track.db"))

    cell_track_db_df = pd.read_sql_query("SELECT * FROM Per_Object", conn)
    relation_df = pd.read_sql_query("SELECT * FROM Per_Relationships", conn)
    am_record = pd.read_csv(os.path.join(dir_path, "am_record.csv"))
    t_span = max(cell_track_db_df["ImageNumber"])

    pipe_util2.print_time()

    # TODO: dangerous operation: semantics of cell_track_db_df is changed, may cause bugs and user confusion.
    # TODO: try using other variable name, e.g. relabel_cell_track_db
    cell_track_db_df = relabel_traj(cell_track_db_df)
    traj_start, traj_end = record_traj_start_end(cell_track_db_df)

    # find candidate mitosis pairs from apoptosis and mitosis cells
    am_arr, am_xy, am_area = am_obj_info(am_record, cell_track_db_df)

    # F is a n by 5 matrix
    # F columns: obj_x, obj_y, img_num, obj_num, area
    F_np_arr = np.column_stack((am_xy, am_arr, np.expand_dims(am_area[:], axis=1)))

    candidate_am_sisters = find_am_sisters(
        F_np_arr, mitosis_max_distance=mitosis_max_dist, size_simi_thres=size_similarity_threshold
    ).tolist()
    # ---------------------------------------
    # utils.print_time()

    # find fuse and split segmentation, border obj and false link
    prefuse_cells = []
    prefuse_group = (
        []
    )  # each element is a list include all prefuse cells in a fuse event, corresponding to postfuse_cells
    postfuse_cells = []  # include: img_num,obj_num

    presplit_cells = []  # include: img_num,obj_num
    postsplit_group = []  # each element is a list include all postsplit cells in a split event
    postsplit_cells = []

    false_link = []
    border_obj = []

    for img_num in range(1, len(img_list) + 1):
        print("processing %d/%d images" % (img_num, len(img_list)), flush=True)
        # -------------find obj on border------------------
        border_obj.extend(find_border_obj(seg_path, seg_img_list, img_num))

        if img_num == t_span:
            break
        img_num_cur = img_num
        img_num_next = img_num + 1

        # TODO: why do we need segpath and rely on path again here?
        # TODO: modularize: read/use image input directly here, compute* functions should just COMPUTE
        frame_overlap = compute_overlap_matrix(seg_path, seg_img_list, img_num_cur, img_num_next)

        # TODO: too many indentations and logics. modularize the following function: write a wrapper
        # find false link with max_overlap relation
        target_idx_list = cell_track_db_df[cell_track_db_df["ImageNumber"] == img_num_next].index.tolist()
        for target_idx in target_idx_list:
            if cell_track_db_df.iloc[target_idx]["Cell_TrackObjects_ParentImageNumber"] == img_num_cur:
                target_o_n = int(cell_track_db_df.iloc[target_idx]["ObjectNumber"])
                source_o_n = int(cell_track_db_df.iloc[target_idx]["Cell_TrackObjects_ParentObjectNumber"])
                # print(source_o_n)
                rel_flag = judge_mol_type(frame_overlap, source_o_n, target_o_n)
                false_pair = search_false_link(
                    cell_track_db_df,
                    relation_df,
                    frame_overlap,
                    img_num_cur,
                    source_o_n,
                    img_num_next,
                    target_o_n,
                    rel_flag,
                )

                if len(false_pair) > 0:
                    false_link.append(false_pair)

        # ----------------find split and merge------------------------------------
        area_arr = cell_track_db_df.loc[(cell_track_db_df["ImageNumber"] == img_num_cur), "Cell_AreaShape_Area"].values
        area_arr_R = cell_track_db_df.loc[
            (cell_track_db_df["ImageNumber"] == img_num_next), "Cell_AreaShape_Area"
        ].values  # area array in img2

        nb_cell_1 = frame_overlap.shape[0]
        nb_cell_2 = frame_overlap.shape[1]

        postf_cells, pref_group = cal_cell_fusion(frame_overlap, img_num_cur, img_num_next, nb_cell_1, nb_cell_2)

        pres_cells, posts_group = cal_cell_split(frame_overlap, img_num_cur, img_num_next, nb_cell_1, nb_cell_2)

        postfuse_cells.extend(postf_cells)
        prefuse_group.extend(pref_group)
        presplit_cells.extend(pres_cells)
        postsplit_group.extend(posts_group)

    np.save(os.path.join(dir_path, "border_obj.npy"), np.array(border_obj))
    np.save(os.path.join(dir_path, "false_link.npy"), np.array(false_link))

    # break false link and relabel traj
    cell_track_db_df, relation_df = break_link(cell_track_db_df, relation_df, false_link)

    cell_track_db_df = relabel_traj(cell_track_db_df)
    traj_start, traj_end = record_traj_start_end(cell_track_db_df)

    # find mitosis_pairs_to_break,fuse_pairs_to_break,split_pairs_to break
    candidate_am_sisters, mitosis_pairs_to_break = find_mitosis_pairs_to_break(
        relation_df, candidate_am_sisters, false_link
    )

    postfuse_cells, prefuse_group, fuse_pairs, fuse_pairs_to_break = find_fuse_pairs_to_break(
        relation_df, postfuse_cells, prefuse_group, false_link, border_obj
    )

    presplit_cells, postsplit_group, split_pairs, split_pairs_to_break = find_split_pairs_to_break(
        relation_df, presplit_cells, postsplit_group, false_link, border_obj
    )

    for f_g in prefuse_group:
        prefuse_cells.extend(f_g)
    for s_g in postsplit_group:
        postsplit_cells.extend(s_g)

    # TODO: use os.path.join or Pathlib instead of +
    np.save(dir_path + "prefuse_cells.npy", np.array(prefuse_cells))
    np.save(dir_path + "postfuse_cells.npy", np.array(postfuse_cells))
    np.save(dir_path + "fuse_pairs.npy", np.array(fuse_pairs))
    np.save(dir_path + "fuse_pairs_to_break.npy", np.array(fuse_pairs_to_break))

    np.save(dir_path + "presplit_cells.npy", np.array(presplit_cells))
    np.save(dir_path + "postsplit_cells.npy", np.array(postsplit_cells))
    np.save(dir_path + "split_pairs.npy", np.array(split_pairs))
    np.save(dir_path + "split_pairs_to_break.npy", np.array(split_pairs_to_break))

    with open(dir_path + "prefuse_group", "wb") as fp:
        pickle.dump(prefuse_group, fp)
    with open(dir_path + "postsplit_group", "wb") as fp:
        pickle.dump(postsplit_group, fp)

    # break pairs to break
    pairs_to_break = []
    pairs_to_break.extend(fuse_pairs_to_break)
    pairs_to_break.extend(split_pairs_to_break)
    pairs_to_break.extend(mitosis_pairs_to_break)
    if len(pairs_to_break) > 0:
        # there are same pairs in split_pairs_to_break and
        # mitosis_pairs_to_break
        pairs_to_break = np.unique(np.asarray(pairs_to_break), axis=0).tolist()
    print(len(pairs_to_break))

    cell_track_db_df, relation_df = break_link(cell_track_db_df, relation_df, pairs_to_break)
    cell_track_db_df = relabel_traj(cell_track_db_df)
    cell_track_db_df.to_csv(dir_path + "Per_Object_break.csv", index=False, encoding="utf-8")
    relation_df.to_csv(dir_path + "Per_Relationships_break.csv", index=False, encoding="utf-8")
    traj_start, traj_end = record_traj_start_end(cell_track_db_df)

    # judge fuse,split type and find candidate_mitosis
    false_traj_label = []
    candidate_mitosis_label = []
    false_mitosis_obj = []

    # TODO: add documentation: mitosis and fuse cell cases, notebook or image
    # dealing with mitosis and then fuse cells
    candidate_mitosis_fc_label = []
    candidate_mitosis_fp_label = []
    candidate_mitosis_fp_group = []
    candidate_mitosis_fp_group_xy = []

    mitosis_fuse_fp_label = []
    mitosis_fuse_sc_label = []
    mitosis_fuse_link_pairs = []

    for presplit_cell_index in range(len(postfuse_cells)):
        fc_cell = postfuse_cells[presplit_cell_index]
        fc_i_n, fc_o_n = fc_cell[0], fc_cell[1]
        fc_img = generate_single_cell_img_edt(img_path, seg_path, img_list, seg_img_list, obj_h, obj_w, fc_i_n, fc_o_n)
        fc_prob = icnn.predict(fc_img)[0]
        fc_am_flag = judge_traj_am(
            cell_track_db_df, am_record, fc_i_n, fc_o_n, judge_later=True, t_range=t_search_range
        )

        fp_group = prefuse_group[presplit_cell_index]
        fp_group_prob = []
        fp_group_am_flag = []
        for [fp_i_n, fp_o_n] in fp_group:
            fp_img = generate_single_cell_img_edt(
                img_path, seg_path, img_list, seg_img_list, obj_h, obj_w, fp_i_n, fp_o_n
            )
            fp_prob = icnn.predict(fp_img)[0]
            fp_group_prob.append(fp_prob.tolist())
            fp_am_flag = judge_traj_am(
                cell_track_db_df, am_record, fp_i_n, fp_o_n, judge_later=False, t_range=t_search_range
            )
            fp_group_am_flag.append(fp_am_flag)

        f_label, m_fc_label, m_fp_group_label, m_fp_group, m_fp_group_xy, fc_type, fp_group_type = judge_fuse_type(
            cell_track_db_df, am_record, fc_cell, fp_group, fc_prob, fp_group_prob, tracklet_len_thres=traj_len_thres
        )
        false_traj_label.extend(f_label)

        if len(m_fc_label) > 0:
            candidate_mitosis_fc_label.extend(m_fc_label)
            candidate_mitosis_fp_label.append(m_fp_group_label)
            candidate_mitosis_fp_group.append(m_fp_group)
            candidate_mitosis_fp_group_xy.append(m_fp_group_xy)

    # TODO: understand the following code.
    for presplit_cell_index in range(len(presplit_cells)):
        sp_cell = presplit_cells[presplit_cell_index]
        sp_image_index, sp_object_index = sp_cell[0], sp_cell[1]
        sp_label = np.asscalar(
            cell_track_db_df.loc[
                (cell_track_db_df["ImageNumber"] == sp_image_index)
                & (cell_track_db_df["ObjectNumber"] == sp_object_index),
                "Cell_TrackObjects_Label",
            ].values
        )
        sp_img = generate_single_cell_img_edt(
            img_path, seg_path, img_list, seg_img_list, obj_h, obj_w, sp_image_index, sp_object_index
        )
        sp_prob = icnn.predict(sp_img)[0]

        mitosis_fuse_flag = 0

        sc_group = postsplit_group[presplit_cell_index]
        if sp_label in candidate_mitosis_fc_label and len(sc_group) == 2:
            mitosis_fuse_flag = 1
            ind = candidate_mitosis_fc_label.index(sp_label)
            mitosis_fuse_fp_label.extend(candidate_mitosis_fp_label[ind])

        # TODO: modularize: BIG IF, split based on case
        # TODO: describe what it does
        if mitosis_fuse_flag == 1:
            if sp_label not in false_traj_label:
                false_traj_label.append(sp_label)

            sc_group_xy = []
            for [sc_image_index, sc_object_index] in sc_group:
                sc_label = np.asscalar(
                    cell_track_db_df.loc[
                        (cell_track_db_df["ImageNumber"] == sc_image_index)
                        & (cell_track_db_df["ObjectNumber"] == sc_object_index),
                        "Cell_TrackObjects_Label",
                    ].values
                )
                sc_group_xy.append(
                    cell_track_db_df.loc[
                        (cell_track_db_df["ImageNumber"] == sc_image_index)
                        & (cell_track_db_df["ObjectNumber"] == sc_object_index),
                        ["Cell_AreaShape_Center_X", "Cell_AreaShape_Center_Y"],
                    ]
                    .values[0]
                    .tolist()
                )
                mitosis_fuse_sc_label.append(sc_label)

            d_matrix = np.zeros((2, 2))
            for m in range(2):
                for n in range(2):
                    x1, y1 = candidate_mitosis_fp_group_xy[ind][m][0], candidate_mitosis_fp_group_xy[ind][m][1]
                    x2, y2 = sc_group_xy[n][0], sc_group_xy[n][1]
                    d_matrix[m, n] = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            r_ind, c_ind = linear_sum_assignment(d_matrix)
            for r, c in zip(r_ind, c_ind):
                mitosis_fuse_link_pairs.append(
                    [
                        candidate_mitosis_fp_group[ind][r][0],
                        candidate_mitosis_fp_group[ind][r][1],
                        sc_group[c][0],
                        sc_group[c][1],
                    ]
                )

        else:
            sc_group_prob = []
            for [sc_image_index, sc_object_index] in sc_group:
                sc_img = generate_single_cell_img_edt(
                    img_path, seg_path, img_list, seg_img_list, obj_h, obj_w, sc_image_index, sc_object_index
                )
                sc_prob = icnn.predict(sc_img)[0]
                sc_group_prob.append(sc_prob.tolist())

            candidate_mitosis_flag, f_label, cm_label, fm_obj, sp_type, sc_group_type = judge_split_type(
                cell_track_db_df,
                am_record,
                sp_cell,
                sc_group,
                sp_prob,
                sc_group_prob,
                tracklet_len_thres=traj_len_thres,
            )

            if candidate_mitosis_flag == 0:
                if len(f_label) > 0:
                    false_traj_label.extend(f_label)

            else:
                candidate_mitosis_label.extend(cm_label)
                if len(fm_obj) > 0:
                    false_mitosis_obj.extend(fm_obj)

    # TODO: avoid pickle
    # save data
    # false_traj_label: list[int]
    # candidate_mitosis_label: list[int]
    # false_mitosis_obj: list of object
    with open(dir_path + "false_traj_label", "wb") as fp:
        pickle.dump(false_traj_label, fp)
    with open(dir_path + "candidate_mitosis_label", "wb") as fp:
        pickle.dump(candidate_mitosis_label, fp)
    with open(dir_path + "false_mitosis_obj", "wb") as fp:
        pickle.dump(false_mitosis_obj, fp)

    for tlabel in false_traj_label:
        # TODO: what is the meaning of the following logics?
        # TODO: e.g. tlabel_is_not_mitosis_or_fuse = boolean
        if (
            tlabel not in candidate_mitosis_label
            and tlabel not in mitosis_fuse_fp_label
            and tlabel not in mitosis_fuse_sc_label
        ):
            cell_track_db_df.loc[cell_track_db_df["Cell_TrackObjects_Label"] == tlabel, "Cell_TrackObjects_Label"] = -1

    cell_track_db_df, relation_df = connect_link(cell_track_db_df, relation_df, mitosis_fuse_link_pairs)
    cell_track_db_df = relabel_traj(cell_track_db_df)
    cell_track_db_df.to_csv(dir_path + "Per_Object_modify.csv", index=False, encoding="utf-8")
    traj_start, traj_end = record_traj_start_end(cell_track_db_df)


if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    img_path = sys.argv[1]
    output_path = sys.argv[2]
    icnn_seg_wts = sys.argv[3]
    DIC_channel_label = sys.argv[4]
    traj_reconganize1(img_path, output_path, icnn_seg_wts, DIC_channel_label)
