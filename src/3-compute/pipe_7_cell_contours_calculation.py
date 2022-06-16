#!/usr/bin/env python
# coding: utf-8

import copy
import pickle
import sys
from os import listdir

import legacy_utils.contour_class
import legacy_utils.contour_class as contour_class
import legacy_utils.image_warp as image_warp
import numpy
import numpy as np
import pandas as pd
import legacy_utils.utils as utils
import scipy.interpolate.fitpack as fitpack
import scipy.ndimage as ndimage
import seaborn as sns
from legacy_utils.contour_tool import (align_contour_to, align_contours,
                          df_find_contour_points, find_contour_points,
                          generate_contours)
from matplotlib import pyplot as plt
from scipy.stats import kde
from skimage import measure
from skimage.io import imread
from skimage.morphology import closing, opening
from skimage.segmentation import find_boundaries


def cell_contours_calculation(output_path, mean_contour_path):
    """

    :param output_path: string, output folder
    :param mean_contour_path: path to the calculated mean_contour.
    :return:
    """
    output_path = utils.correct_folder_str(output_path)
    dir_path = output_path
    seg_path = utils.correct_folder_str(output_path + "seg")
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
        scale_back = utils.decompose_homogenous_transform(cell_contours[i].to_world_transform)[1]
        cell_contours[i].scale(scale_back)
        cells[ind].set_cell_contour(cell_contours[i])

    with open(output_path + "cells/" + "cells", "wb") as fp:
        pickle.dump(cells, fp)


if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    output_path = sys.argv[1]
    mean_contour_path = sys.argv[2]
    cell_contours_calculation(output_path, mean_contour_path)
