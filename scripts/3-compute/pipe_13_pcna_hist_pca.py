# In[1]: import
import sys

sys.path.insert(1, "/home/thomas/research/projects/emt/scripts/memes/")

import copy
import glob
import os
import pickle
from math import pi, sqrt
from os import listdir

import cv2
import numpy as np
import pandas as pd
import pipe_util2
import scipy.misc
import scipy.ndimage as ndi
from cell_class import fluor_single_cell, single_cell
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
from sklearn.preprocessing import Normalizer, RobustScaler, StandardScaler


# In[2]: function
def pcna_haralick_pca(all_datset_path, all_datsets, pattern="XY"):
    """

    :param all_datset_path: folder including all output folders
    :param pattern: the pattern for matching the output folders
    :return:
    """
    for datset_idx in range(len(all_datsets)):
        all_data = np.array([])
        curr_datset_path = pipe_util2.correct_folder_str(all_datset_path + all_datsets[datset_idx])
        all_datset_path = pipe_util2.correct_folder_str(all_datset_path)
        output_path_list = pipe_util2.count_pattern_in_folder(curr_datset_path, pattern)
        i = 0
        while i < len(output_path_list):
            output_path = output_path_list[i]
            output_path = pipe_util2.correct_folder_str(output_path)
            cells_path = output_path + "cells/"

            with open(cells_path + "pcna_cells-02", "rb") as fp:
                cells = pickle.load(fp)

            data = np.array(
                [
                    single_cell.pcna_feature_values[0]
                    for single_cell in cells
                    if hasattr(single_cell, "pcna_feature_values")
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
        print(X[2])

        pca = decomposition.PCA(n_components=0.98, svd_solver="full")
        Y = pca.fit_transform(X)
        print(pca.explained_variance_ratio_, sum(pca.explained_variance_ratio_))
        print(pca.components_)

        # dot_color=np.arange(Y[:].shape[0])
        # cm=plt.cm.get_cmap('jet')
        # sc=plt.scatter(Y[:,0],Y[:,1],c=dot_color,cmap=cm)
        # plt.scatter(Y[:,0],Y[:,1],s=0.1)
        # plt.savefig("pcna_histogram.png",dpi = 300)

        with open(curr_datset_path + "pcna_histogram_pca", "wb") as fp:
            pickle.dump(pca, fp)
        fluor_feature_name = "pcna_histogram"

        for output_path in output_path_list:
            output_path = pipe_util2.correct_folder_str(output_path)
            cells_path = output_path + "cells/"

            with open(cells_path + "pcna_cells-02", "rb") as fp:
                cells = pickle.load(fp)

            for i in range(len(cells)):
                if hasattr(cells[i], "pcna_feature_values"):
                    X = np.expand_dims(cells[i].pcna_feature_values[0], axis=0)
                    X = scaler.transform(X)
                    Y = pca.transform(X)[0]
                    cells[i].set_fluor_pca_cord(fluor_feature_name, Y)
            with open(cells_path + "pcna_cells-02", "wb") as fp:
                pickle.dump(cells, fp)


# In[1]: run
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
    pcna_haralick_pca(all_datset_path, all_datsets)


# %%
