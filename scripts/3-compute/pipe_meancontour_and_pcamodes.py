import numpy as np
from skimage import measure
from skimage.segmentation import find_boundaries
from skimage.morphology import opening, closing
from skimage.io import imread
from matplotlib import pyplot as plt
import os
from os import listdir
import pandas as pd
from scipy.stats import kde
import seaborn as sns
import copy
import pickle
import scipy.ndimage as ndimage
import scipy.interpolate.fitpack as fitpack

import contour_class
import utility_tools

import image_warp
from contour_tool import (
    df_find_contour_points,
    find_contour_points,
    generate_contours,
    align_contour_to,
    align_contours,
)

import hj_util
import sys

output_path = sys.argv[1]

output_path = hj_util.folder_verify(output_path)

seg_path = output_path + "seg/"
seg_img_list = sorted(listdir(seg_path))

contour_path = output_path + "contour/"
hj_util.create_folder(contour_path)


def list_sub_group(list_len, num):
    inxs = []
    i = 0
    while i < list_len:
        sub_inx = []
        j = 0
        while j < num:
            sub_inx.append(i)
            j = j + 1
            i = i + 1

        inxs.append(sub_inx)

    return inxs


inx_list = list_sub_group(len(seg_img_list), 3)

contour_points_and_obj = find_contour_points(seg_path, seg_img_list, contour_value=0.5)
cell_contours, sort_obj_arr = generate_contours(
    contour_points_and_obj, closed_only=True, min_area=None, max_area=None, axis_align=True
)

print("check point 1")

for i in range(len(cell_contours)):
    cell_contours[i].resample(num_points=150)
    cell_contours[i].axis_align()
    points = cell_contours[i].points

mean_contour, iters = align_contours(cell_contours, allow_reflection=True, allow_scaling=False, max_iters=20)
with open("mean_cell_contour", "wb") as fp:
    pickle.dump(mean_contour, fp)

for i in range(len(cell_contours)):
    scale_back = utility_tools.decompose_homogenous_transform(cell_contours[i].to_world_transform)[1]
    cell_contours[i].scale(scale_back)
    points = cell_contours[i].points

pca_contours = contour_class.PCAContour.from_contours(
    contours=cell_contours, required_variance_explained=0.98, return_positions=False
)

with open("pca_contours", "wb") as fp:
    pickle.dump(pca_contours, fp)
