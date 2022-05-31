#!/usr/bin/env python
# coding: utf-8
# In[1]: import
import sys

sys.path.insert(1, "/home/thomas/research/projects/emt/scripts/memes/")
import warnings

warnings.filterwarnings("ignore")

import copy
import glob
import os
import pickle
from math import pi, sqrt
from os import listdir

import legacy_utils.contour_class as contour_class
import cv2
import legacy_utils.image_warp as image_waro
import numpy as np
import pandas as pd
import legacy_utils.utils as utils
import scipy.misc
import scipy.ndimage as ndi
from legacy_utils.cell_class import single_cell
from legacy_utils.contour_tool import (align_contour_to, align_contours,
                          df_find_contour_points, find_contour_points,
                          generate_contours)
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from skimage import measure
from skimage.color import label2rgb
from skimage.feature import peak_local_max
from skimage.io import imread
from skimage.measure import label, regionprops
from skimage.morphology import (h_maxima, local_maxima, opening,
                                remove_small_objects)
from skimage.segmentation import clear_border, watershed
from sklearn import cluster, decomposition, manifold
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler

# In[2]:


# main_path='/home/zoro/Desktop/experiment_data/2019-03-22_a549_tgf4ng_2d/'
# main_path = '/mnt/data0/Ke/weikang_exp_process/2019-05-05_A549_vim/'
# cells_path=main_path+'fluor_cells/'
# posi_start=1
# posi_end=1


# do not use StandarScaler on cell contour points
# ----------cal cell_contour pca coordinates-------------------
def morph_pca(all_dataset_path, all_dataset_names, pattern="XY"):
    """

    :param all_datset_path: string, including several output folders
    :param pattern: string, for indexing output folders
    :return:
    """
    for dataset_idx in range(len(all_dataset_names)):
        all_data = None
        cur_dataset_path = utils.correct_folder_str(all_dataset_path + all_dataset_names[dataset_idx])
        output_path_list = utils.count_pattern_in_folder(cur_dataset_path, pattern)
        for i in range(len(output_path_list)):
            output_path = output_path_list[i]
            output_path = utils.correct_folder_str(output_path)
            cells_path = output_path + "cells/"

            with open(cells_path + "cells", "rb") as fp:
                cells = pickle.load(fp)

            data = np.array(
                [single_cell.cell_contour.points for single_cell in cells if hasattr(single_cell, "cell_contour")]
            )

            if all_data is None:
                # first time initialization with data
                all_data = data
            else:
                all_data = np.vstack((all_data, data))

        X = all_data
        X, data_point_shape = utils.flatten_data(X)
        X = X.astype(np.float)
        print(X.shape)

        pca = decomposition.PCA(n_components=0.98, svd_solver="full")
        Y = pca.fit_transform(X)
        print(pca.explained_variance_ratio_, sum(pca.explained_variance_ratio_))

        # visualize  PC1 and PC2

        # plt.scatter(Y[:,0],Y[:,1],s=0.1)
        # plt.xlabel('PC1')
        # plt.ylabel('PC2')
        # plt.show()

        with open(cur_dataset_path + "morph_pca", "wb") as fp:
            pickle.dump(pca, fp)

        # do not use StandarScaler on cell contour points
        # ----------cal cell_contour pca coordinates-------------------
        for output_path in output_path_list:
            output_path = utils.correct_folder_str(output_path)
            cells_path = output_path + "cells/"

            with open(cells_path + "pcna_cells-02", "rb") as fp:
                cells = pickle.load(fp)
            for i in range(len(cells)):
                if hasattr(cells[i], "cell_contour"):
                    data = np.expand_dims(cells[i].cell_contour.points, axis=0)
                    X, X_shape = utils.flatten_data(data)
                    Y = pca.transform(X)[0]
                    cells[i].set_pca_cord(Y)
            print("dumping " + cells_path + "pcna_cells-02")
            with open(cells_path + "pcna_cells-02", "wb") as fp:
                pickle.dump(cells, fp)


if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    all_datset_path = (
        "/home/thomas/research/projects/emt/data/out/pcna/"  # directory containing all outputs of all datasets
    )
    all_datsets = [
        "01-13-22_72hr_no-treat",
        "01-18-22_72hr_no-treat",
        "01-27-22_72hr_no-treat",
        "02-03-22_72hr_no-treat",
        "02-11-22_72hr_no-treat",
        "02-21-22_72hr_no-treat",
        "12-19-21_72hr_no-treat",
    ]  # directory names of all datasets
    morph_pca(all_datset_path, all_datsets)

# %%
