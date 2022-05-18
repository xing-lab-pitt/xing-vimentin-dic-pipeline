#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import multiprocessing
import os
import pickle
import sys
from math import ceil, pi, sqrt
from multiprocessing import Process
from os import listdir

import config
import cv2
import mahotas.features.texture as mht
import numpy as np
import pandas as pd
import pipe_util2
import scipy.misc
import scipy.ndimage as ndi
from cell_class import fluor_single_cell, single_cell
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import bisplev, bisplrep
from scipy.signal import medfilt
from scipy.stats import kurtosis, skew
from skimage.color import label2rgb
from skimage.feature import peak_local_max
from skimage.filters import frangi, gabor
from skimage.filters.rank import otsu
from skimage.io import imread
from skimage.measure import label, regionprops
from skimage.morphology import (closing, dilation, disk, h_maxima,
                                local_maxima, opening, remove_small_holes,
                                remove_small_objects)
from skimage.segmentation import clear_border, watershed
from sklearn.preprocessing import RobustScaler, StandardScaler

# In[2]:


# main_path='/home/zoro/Desktop/experiment_data/2019-03-22_a549_tgf4ng_2d/'
# main_path = '/mnt/data0/Ke/weikang_exp_process/2019-05-05_A549_vim/'

# cell_output_path=main_path+'output/'
# # vim_input_path=main_path+'vimentin/'
# vim_input_path=main_path+'a549_tif/vimentin/'

# cells_path=main_path+'cells/'
# fluor_cells_path=main_path+'fluor_cells/'
# # fluor_cells_path=main_path+'a549_tif/fluor/'

fluor_interval = config.fluor_interval


def compute_nuc_fluor_info(seg, fluor_img):
    rps = regionprops(seg)
    histogram_features = []
    fluor_haralick = []
    norm_fluor_haralick = []
    scale_fluor_haralick = []
    for i in range(len(rps)):
        cell_num = int(i + 1)
        r = rps[i]
        cell_mask = seg == cell_num
        region_cell_mask = cell_mask[r.bbox[0] : r.bbox[2], r.bbox[1] : r.bbox[3]]

        crop_img = fluor_img[r.bbox[0] : r.bbox[2], r.bbox[1] : r.bbox[3]]
        cell_img = (fluor_img * cell_mask)[r.bbox[0] : r.bbox[2], r.bbox[1] : r.bbox[3]]

        nuc_mask = cell_img > otsu(cell_img, selem=disk(max(cell_img.shape) / 2), mask=cell_img > 0)
        nuc_mask = remove_small_objects(opening(nuc_mask), 100)
        nuc_mask = remove_small_holes(nuc_mask, 100)

        nuc_label = label(nuc_mask)

        # the haralick features have four directions, to meet rotation
        # invariance,use average for each feature

        if np.sum(nuc_mask) > ceil(0.1 * np.sum(cell_mask)) and np.amax(nuc_label) == 1:
            # original nucleus stats
            ori_nuc_area = np.sum(nuc_mask)
            nuc_radius = round(sqrt(np.sum(nuc_mask) / pi))

            # smoothening + dilation
            nuc_mask = closing(nuc_mask, disk(round(nuc_radius / 3)))
            nuc_mask = opening(nuc_mask, disk(round(nuc_radius / 3)))
            curr_nuc_area = np.sum(nuc_mask)
            if curr_nuc_area < ori_nuc_area:
                while curr_nuc_area < ori_nuc_area:
                    nuc_mask = dilation(nuc_mask)
                    curr_nuc_area = np.sum(nuc_mask)
            nuc_mask = nuc_mask * cell_mask[r.bbox[0] : r.bbox[2], r.bbox[1] : r.bbox[3]]

            nuc_img = nuc_mask * cell_img
            int_prof = nuc_img[np.nonzero(nuc_img)]

            int_mean = np.sum(int_prof) / np.sum(nuc_mask)
            int_min, int_max = np.quantile(int_prof, 0.02), np.quantile(int_prof, 0.98)
            int_std = np.std(int_prof)
            int_skew = skew(int_prof)
            int_kurt = kurtosis(int_prof, fisher=True)
            hist_fea = np.array([int_mean, int_min, int_max, int_std, int_skew, int_kurt])  # histogram features
            histogram_features.append(hist_fea)

            norm_nuc_img = (nuc_img - np.amin(nuc_img[nuc_mask]) + 1) * nuc_mask
            fl_hara = mht.haralick(nuc_img, ignore_zeros=True, return_mean=True)
            norm_fl_hara = mht.haralick(norm_nuc_img, ignore_zeros=True, return_mean=True)

            fluor_haralick.append(fl_hara)
            norm_fluor_haralick.append(norm_fl_hara)
        else:
            histogram_features.append(np.zeros((6,)))
            fluor_haralick.append(np.zeros((13,)))
            norm_fluor_haralick.append(np.zeros((13,)))

    histogram_features = np.array(histogram_features)
    fluor_haralick = np.array(fluor_haralick)
    norm_fluor_haralick = np.array(norm_fluor_haralick)

    return histogram_features, fluor_haralick, norm_fluor_haralick


# In[7]:


feature_list = ["histogram_features", "haralick", "norm_haralick"]


def single_folder_run(img_folder, output_path, pcna_chan_label):

    output_path = pipe_util2.correct_folder_str(output_path)
    cells_path = output_path + "cells/"
    fluor_cells_path = cells_path
    cell_seg_path = output_path + "seg/"
    pcna_img_path = pipe_util2.correct_folder_str(img_folder)

    cell_seg_list = sorted(listdir(cell_seg_path))
    print(pcna_img_path)
    pcna_img_list = sorted(glob.glob(pcna_img_path + "*" + pcna_chan_label + "*"))

    print(len(pcna_img_list))

    for i in range(len(pcna_img_list)):
        pcna_img_list[i] = os.path.basename(pcna_img_list[i])

    t_span = len(cell_seg_list)

    df = pd.read_csv(output_path + "/Per_Object_relink" + ".csv")

    with open(cells_path + "fluor_cells", "rb") as fp:
        cells = pickle.load(fp)

    for ti in np.arange(0, t_span, fluor_interval):
        img_num = ti + 1
        print("position:%s, progress:%d/%d" % (output_path, img_num, t_span), flush=True)
        cell_seg = imread(cell_seg_path + cell_seg_list[ti])

        pcna_img = imread(pcna_img_path + pcna_img_list[ti])
        histogram_features, fluor_haralick, norm_fluor_haralick = compute_nuc_fluor_info(cell_seg, pcna_img)

        print(histogram_features.shape)
        print(fluor_haralick.shape)
        for obj_num in np.arange(1, np.amax(cell_seg) + 1):
            ind = df.loc[(df["ImageNumber"] == img_num) & (df["ObjectNumber"] == obj_num)].index.tolist()[0]
            pcna_features = [
                histogram_features[obj_num - 1, :],
                fluor_haralick[obj_num - 1, :],
                norm_fluor_haralick[obj_num - 1, :],
            ]

            cells[ind].set_fluor_features("pcna", feature_list, pcna_features)

    with open(fluor_cells_path + "pcna_cells-02", "wb") as fp:
        pickle.dump(cells, fp)


if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    img_path = sys.argv[1]
    output_path = sys.argv[2]
    pcna_chan_label = sys.argv[3]
    single_folder_run(img_path, output_path, pcna_chan_label)
