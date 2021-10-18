#!/usr/bin/env python
# coding: utf-8

import numpy as np
from skimage import measure
from skimage.segmentation import find_boundaries
from skimage.morphology import opening, closing
from skimage.io import imread
from matplotlib import pyplot as plt
from os import listdir
import pandas as pd
from scipy.stats import kde
import seaborn as sns
import copy
import pickle

import contour_class
import utility_tools as utility_tools
import scipy.ndimage as ndimage
import scipy.interpolate.fitpack as fitpack
import numpy
import image_warp as image_warp
from contour_tool import (
    df_find_contour_points,
    find_contour_points,
    generate_contours,
    align_contour_to,
    align_contours,
)
import pipe_util2

# In[2]:


# main_path = '/mnt/data0/Ke/weikang_exp_process/2019-05-05_A549_vim/'
# input_path=main_path+'a549_tif/vimentin/' # 'img/'
# output_path=main_path+'output/'


def cell_contours_calculation(output_path, mean_contour_path):
    """

    :param output_path: string, output folder
    :param mean_contour_path: path to the calculated mean_contour.
    :return:
    """
    output_path = pipe_util2.folder_verify(output_path)
    dir_path = output_path
    seg_path = pipe_util2.folder_verify(output_path + "seg")
    seg_img_list = sorted(listdir(seg_path))
    df = pd.read_csv(dir_path + "Per_Object_relink" + ".csv")

    with open(output_path + "cells/" + "cells", "rb") as fp:
        cells = pickle.load(fp)

    with open(mean_contour_path, "rb") as fp:
        mean_contour = pickle.load(fp)

    cell_contour_points_and_cell = df_find_contour_points(df, seg_path, seg_img_list, contour_value=0.5)
    cell_contours, sort_cell_arr = generate_contours(
        cell_contour_points_and_cell, closed_only=True, min_area=None, max_area=None, axis_align=False
    )
    for i in range(sort_cell_arr.shape[0]):
        img_num, obj_num = sort_cell_arr[i, 0], sort_cell_arr[i, 1]
        ind = df.loc[(df["ImageNumber"] == img_num) & (df["ObjectNumber"] == obj_num)].index.tolist()[0]

        cell_contours[i].resample(num_points=150)
        cell_contours[i].axis_align()
        align_contour_to(cell_contours[i], mean_contour, allow_reflection=True, allow_scaling=True)
        scale_back = utility_tools.decompose_homogenous_transform(cell_contours[i].to_world_transform)[1]
        cell_contours[i].scale(scale_back)

        cells[ind].set_cell_contour(cell_contours[i])

    with open(output_path + "cells/" + "cells", "wb") as fp:
        pickle.dump(cells, fp)


# In[ ]:
