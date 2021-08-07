import copy
import numpy as np
from skimage.segmentation import watershed, clear_border
import scipy.misc
from skimage.io import imread
from matplotlib import pyplot as plt
import scipy.ndimage as ndi
import pickle
import os
from os import listdir
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects, local_maxima, h_maxima, opening
from skimage import measure
from skimage.measure import regionprops, label
from skimage.color import label2rgb
from PIL import Image, ImageDraw, ImageFont
from math import pi, sqrt
import cv2
import glob
import pandas as pd
import seaborn as sns

from sklearn import decomposition, cluster, manifold
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler, RobustScaler, Normalizer
from cell_class import single_cell, fluor_single_cell
import pipe_util2


def haralick_pca(top_path, pattern="output"):
    """

    :param top_path: folder including all output folders
    :param pattern: the pattern for matching the output folders
    :return:
    """

    all_data = np.array([])
    top_path = pipe_util2.folder_verify(top_path)
    output_path_list = pipe_util2.folder_file_num(top_path, pattern)
    i = 0
    while i < len(output_path_list):
        output_path = output_path_list[i]
        output_path = pipe_util2.folder_verify(output_path)
        cells_path = output_path + "cells/"

        with open(cells_path + "fluor_cells", "rb") as fp:
            cells = pickle.load(fp)
        data = np.array(
            [
                single_cell.vimentin_feature_values[3]
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

    dot_color = np.arange(Y[:].shape[0])
    cm = plt.cm.get_cmap("jet")

    sc = plt.scatter(Y[:, 0], Y[:, 1], c=dot_color, cmap=cm)
    plt.scatter(Y[:, 0], Y[:, 1], s=0.1)
    # plt.axis([-100000,500000,-2000,2000])
    plt.savefig("hiralic_1.png", dpi=300)
    plt.show()

    sns.kdeplot(Y[:, 0], Y[:, 1], n_levels=100, shade=True)
    # plt.axis([-100000,500000,-2000,2000])
    plt.savefig("hiralic_2.png", dpi=300)
    plt.show()

    with open(top_path + "norm_haralick_pca", "wb") as fp:
        pickle.dump(pca, fp)

    # with open(main_path+'haralick_pca', 'rb') as fp:
    #     pca=pickle.load(fp)

    fluor_feature_name = "vimentin_haralick"
    for output_path in output_path_list:
        output_path = pipe_util2.folder_verify(output_path)
        cells_path = output_path + "cells/"

        with open(cells_path + "fluor_cells", "rb") as fp:
            cells = pickle.load(fp)

        for i in range(len(cells)):
            if hasattr(cells[i], "vimentin_feature_values"):
                X = np.expand_dims(cells[i].vimentin_feature_values[4], axis=0)
                X = scaler.transform(X)
                Y = pca.transform(X)[0]
                cells[i].set_fluor_pca_cord(fluor_feature_name, Y)
        with open(cells_path + "fluor_cells", "wb") as fp:
            pickle.dump(cells, fp)
