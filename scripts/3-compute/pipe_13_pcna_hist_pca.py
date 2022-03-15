# In[1]: import
import sys
sys.path.insert(1,'/home/thomas/research/projects/emt/scripts/memes/')

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

from sklearn import decomposition, cluster, manifold
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler, RobustScaler, Normalizer
from cell_class import single_cell, fluor_single_cell
import pipe_util2

# In[2]: function
def pcna_haralick_pca(all_datset_path,all_datsets,pattern='XY'):
    """

    :param all_datset_path: folder including all output folders
    :param pattern: the pattern for matching the output folders
    :return:
    """
    for datset_idx in range(len(all_datsets)):
        all_data = np.array([])
        curr_datset_path = pipe_util2.folder_verify(all_datset_path+all_datsets[datset_idx])
        all_datset_path = pipe_util2.folder_verify(all_datset_path)
        output_path_list = pipe_util2.folder_file_num(curr_datset_path, pattern)
        i = 0
        while i < len(output_path_list):
            output_path = output_path_list[i]
            output_path = pipe_util2.folder_verify(output_path)
            cells_path = output_path + "cells/"

            with open(cells_path + 'pcna_cells-02', 'rb') as fp:
                cells = pickle.load(fp)
            
            data = np.array([single_cell.pcna_feature_values[0]
                for single_cell in cells if hasattr(single_cell, 'pcna_feature_values')])

            if all_data.size == 0:
                all_data = data
            else:
                all_data = np.vstack((all_data, data))
            i = i+1

        scaler = StandardScaler()
        print(all_data.shape)
        X = scaler.fit_transform(all_data)
        print(X.shape)

        pca = decomposition.PCA(n_components=0.98, svd_solver='full')
        Y = pca.fit_transform(X)
        print(pca.explained_variance_ratio_, sum(pca.explained_variance_ratio_))
        print(pca.components_)

        dot_color=np.arange(Y[:].shape[0])
        cm=plt.cm.get_cmap('jet')
        sc=plt.scatter(Y[:,0],Y[:,1],c=dot_color,cmap=cm)
        plt.scatter(Y[:,0],Y[:,1],s=0.1)

        plt.savefig("pcna_histogram.png",dpi = 300)

        with open(all_datset_path+'pcna_histogram_pca', 'wb') as fp:
            pickle.dump(pca, fp)
        fluor_feature_name = 'pcna_histogram'
        
        # for output_path in output_path_list:
        #     output_path = pipe_util2.folder_verify(output_path)
        #     cells_path = output_path + "cells/"

        #     with open(cells_path + 'pcna_cells-02', 'rb') as fp:
        #         cells = pickle.load(fp)

        #     for i in range(len(cells)):
        #         if hasattr(cells[i], 'pcna_feature_values'):
        #             if norm == False:
        #                 X = np.expand_dims(cells[i].pcna_feature_values[1], axis=0)
        #             else:
        #                 X = np.expand_dims(cells[i].pcna_feature_values[2], axis=0)
        #             X = scaler.transform(X)
        #             Y = pca.transform(X)[0]
        #             cells[i].set_fluor_pca_cord(fluor_feature_name, Y)
            # with open(cells_path + 'pcna_cells-02', 'wb') as fp:
            #     pickle.dump(cells, fp)

# In[1]: run
if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    all_datset_path = '/home/thomas/research/projects/emt/data/out/pcna/' # directory containing all outputs of all datasets
    all_datsets =     all_datsets = ['01-13-22_72hr_no-treat'] # directory names of all datasets
    pcna_haralick_pca(all_datset_path,all_datsets)

# %%
