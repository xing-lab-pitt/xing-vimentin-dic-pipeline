# In[1]: import
import sys

sys.path.insert(1, "/home/thomas/research/projects/emt/src/memes/")
import warnings

warnings.filterwarnings("ignore")

import copy
import glob
import os
import pickle
from math import pi, sqrt
from os import listdir

import cv2
import numpy as np
import pandas as pd
import legacy_utils.utils as utils
import scipy.misc
import scipy.ndimage as ndi
from legacy_utils.cell_class import fluor_single_cell, single_cell
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from skimage import measure
from skimage.color import label2rgb
from skimage.feature import peak_local_max
from skimage.io import imread
from skimage.measure import label, regionprops
from skimage.morphology import h_maxima, local_maxima, opening, remove_small_objects
from skimage.segmentation import clear_border, watershed
from sklearn import cluster, decomposition, manifold
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import Normalizer, RobustScaler, StandardScaler


# In[2]: function
def vim_haralick_pca(all_datset_path, all_datsets, norm, pattern="XY"):
    """

    :param all_datset_path: folder including all output folders
    :param pattern: the pattern for matching the output folders
    :return:
    """
    for datset_idx in range(len(all_datsets)):
        all_data = np.array([])
        cur_datset_path = utils.correct_folder_str(all_datset_path + all_datsets[datset_idx])
        output_path_list = utils.count_pattern_in_folder(cur_datset_path, pattern)
        i = 0
        while i < len(output_path_list):
            output_path = output_path_list[i]
            output_path = utils.correct_folder_str(output_path)
            cells_path = output_path + "cells/"

            with open(cells_path + "pcna_cells-02", "rb") as fp:
                cells = pickle.load(fp)

            if norm == False:
                data = np.array(
                    [
                        single_cell.vimentin_feature_values[3]
                        for single_cell in cells
                        if hasattr(single_cell, "vimentin_feature_values")
                    ]
                )
            else:
                data = np.array(
                    [
                        single_cell.vimentin_feature_values[4]
                        for single_cell in cells
                        if hasattr(single_cell, "vimentin_feature_values")
                    ]
                )

            if all_data.size == 0:
                all_data = data
            else:
                all_data = np.vstack((all_data, data))
            i = i + 1

        scaler = StandardScaler()

        X = scaler.fit_transform(all_data)
        print(X.shape)

        pca = decomposition.PCA(n_components=0.98, svd_solver="full")
        Y = pca.fit_transform(X)
        print(pca.components_, pca.explained_variance_ratio_)

        # dot_color=np.arange(Y[:].shape[0])
        # cm=plt.cm.get_cmap('jet')

        # sc=plt.scatter(Y[:,0],Y[:,1],c=dot_color,cmap=cm)
        # plt.scatter(Y[:,0],Y[:,1],s=0.1)
        # #plt.axis([-100000,500000,-2000,2000])
        # plt.savefig("vim_haralic_1.png",dpi = 300)
        # plt.show()
        if norm == False:
            with open(cur_datset_path + "vimentin_haralick_pca", "wb") as fp:
                pickle.dump(pca, fp)
            fluor_feature_name = "vimentin_haralick"
        else:
            with open(cur_datset_path + "norm_vimentin_haralick_pca", "wb") as fp:
                pickle.dump(pca, fp)
            fluor_feature_name = "norm_vimentin_haralick"

        for output_path in output_path_list:
            output_path = utils.correct_folder_str(output_path)
            cells_path = output_path + "cells/"

            with open(cells_path + "pcna_cells-02", "rb") as fp:
                cells = pickle.load(fp)

            for i in range(len(cells)):
                if hasattr(cells[i], "vimentin_feature_values"):
                    if norm == False:
                        X = np.expand_dims(cells[i].vimentin_feature_values[3], axis=0)
                    else:
                        X = np.expand_dims(cells[i].vimentin_feature_values[4], axis=0)
                    X = scaler.transform(X)
                    Y = pca.transform(X)[0]
                    cells[i].set_fluor_pca_cord(fluor_feature_name, Y)
            print("dumping " + cells_path + "pcna_cells-02")
            with open(cells_path + "pcna_cells-02", "wb") as fp:
                pickle.dump(cells, fp)


# In[3]: run
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
    norm = False  # whether to use normalized haralick values
    vim_haralick_pca(all_datset_path, all_datsets, norm)

# %%