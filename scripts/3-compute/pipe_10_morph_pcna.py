#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from sklearn.preprocessing import StandardScaler
from cell_class import single_cell
import contour_class
import utility_tools

import image_warp
from contour_tool import df_find_contour_points, find_contour_points, generate_contours, align_contour_to, align_contours
import pipe_util2

# In[2]:


# main_path='/home/zoro/Desktop/experiment_data/2019-03-22_a549_tgf4ng_2d/'
# main_path = '/mnt/data0/Ke/weikang_exp_process/2019-05-05_A549_vim/'
# cells_path=main_path+'fluor_cells/'
# posi_start=1
# posi_end=1


# In[ ]:


# do not use StandarScaler on cell contour points
# ----------cal cell_contour pca coordinates-------------------
def morph_pca(top_path, pattern="output" ):
    """

    :param top_path: string, including several output folders
    :param pattern: string, for indexing output folders
    :return:
    """
    all_data = np.array([])
    top_path = pipe_util2.folder_verify(top_path)
    output_path_list = pipe_util2.folder_file_num(top_path, pattern)
    i = 0
    while i<len(output_path_list):
        output_path = output_path_list[i]
        output_path = pipe_util2.folder_verify(output_path)
        cells_path = output_path + "cells/"

        with open(cells_path + 'cells', 'rb') as fp:
            cells = pickle.load(fp)

        data = np.array([single_cell.cell_contour.points for single_cell in cells if hasattr(
            single_cell, 'cell_contour')])

        if all_data.size == 0:
            all_data = data
        else:
            all_data = np.vstack((all_data, data))
        i = i+1

    # print(all_data.shape)
    # mean = all_data.mean(axis = 0)
    # plt.plot(mean[:,0],mean[:,1],'.')
    # plt.show()
    X = all_data
    X, data_point_shape = utility_tools.flatten_data(X)
    X = X.astype(np.float)
    print(X.shape)

    pca = decomposition.PCA(n_components=0.98, svd_solver='full')
    Y = pca.fit_transform(X)
    print(pca.explained_variance_ratio_, sum(pca.explained_variance_ratio_))

    # plt.scatter(Y[:,0],Y[:,1],s=0.1)
    # plt.xlabel('PC1')
    # plt.ylabel('PC2')
    # plt.show()
    # sns.kdeplot(Y[:,0],Y[:,1],n_levels=100,shade=True)
    # plt.show()

    with open(top_path + 'morph_pca', 'wb') as fp:
        pickle.dump(pca, fp)


    # do not use StandarScaler on cell contour points
    # ----------cal cell_contour pca coordinates-------------------
    for output_path in output_path_list:
        output_path = pipe_util2.folder_verify(output_path)
        cells_path = output_path+"cells/"

        with open(cells_path + 'fluor_cells', 'rb') as fp:
            cells = pickle.load(fp)
        for i in range(len(cells)):
            if hasattr(cells[i], 'cell_contour'):
                data = np.expand_dims(cells[i].cell_contour.points, axis=0)
                X, X_shape = utility_tools.flatten_data(data)
                Y = pca.transform(X)[0]
                cells[i].set_pca_cord(Y)
        with open(cells_path + 'fluor_cells', 'wb') as fp:
            pickle.dump(cells, fp)

if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    top_path = sys.argv[1]
    morph_pca(top_path)
