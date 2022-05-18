import copy
import os
import pickle
import sys
from os import listdir

import contour_class
import image_warp
import numpy as np
import pandas as pd
import scipy.interpolate.fitpack as fitpack
import scipy.ndimage as ndimage
import seaborn as sns
import utils
from contour_tool import (align_contour_to, align_contours,
                          df_find_contour_points, find_contour_points,
                          generate_contours)
from matplotlib import pyplot as plt
from scipy.stats import kde
from skimage import measure
from skimage.io import imread
from skimage.morphology import closing, opening
from skimage.segmentation import find_boundaries

output_path = sys.argv[1]

output_path = utils.correct_folder_str(output_path)

seg_path = output_path + "seg/"
seg_img_list = sorted(listdir(seg_path))

contour_path = output_path + "contour/"
utils.create_folder(contour_path)


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
    scale_back = utils.decompose_homogenous_transform(cell_contours[i].to_world_transform)[1]
    cell_contours[i].scale(scale_back)
    points = cell_contours[i].points

pca_contours = contour_class.PCAContour.from_contours(
    contours=cell_contours, required_variance_explained=0.98, return_positions=False
)

with open("pca_contours", "wb") as fp:
    pickle.dump(pca_contours, fp)
