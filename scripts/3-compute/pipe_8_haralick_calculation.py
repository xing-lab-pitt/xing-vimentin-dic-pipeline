#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from skimage.morphology import remove_small_objects, local_maxima, h_maxima, disk, dilation
from skimage.measure import regionprops, label
from skimage.color import label2rgb
from PIL import Image, ImageDraw, ImageFont
from math import pi
import cv2
import glob
import pandas as pd
from cell_class import single_cell, fluor_single_cell
from skimage.filters import frangi, gabor
import mahotas.features.texture as mht
from scipy.signal import medfilt
from scipy.interpolate import bisplrep, bisplev
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.preprocessing import RobustScaler, StandardScaler

import multiprocessing
from multiprocessing import Process
import pipe_util2
import sys
import config

fluor_interval = config.fluor_interval


def compute_fluor_info(seg, fluor_img):
    rps = regionprops(seg)
    mean_intensity = np.zeros((len(rps)))
    std_intensity = np.zeros((len(rps)))
    intensity_range = np.zeros((len(rps)))
    fluor_haralick = []
    norm_fluor_haralick = []
    scale_fluor_haralick = []
    for i in range(len(rps)):
        cell_num = int(i + 1)
        r = rps[i]
        cell_mask = seg == cell_num
        region_cell_mask = cell_mask[r.bbox[0] : r.bbox[2], r.bbox[1] : r.bbox[3]]

        crop_img = fluor_img[r.bbox[0] : r.bbox[2], r.bbox[1] : r.bbox[3]]
        cell_img = (fluor_img * cell_mask)[r.bbox[0] : r.bbox[2], r.bbox[1] : r.bbox[3]]

        mean_intensity[i] = np.sum(cell_img) * 1.0 / r.area
        std_intensity[i] = np.std(cell_img[region_cell_mask])

        min_value, max_value = np.amin(cell_img[region_cell_mask]), np.amax(cell_img[region_cell_mask])
        min_value = min_value - 1

        intensity_range[i] = max_value - min_value

        norm_cell_img = (cell_img - min_value) * region_cell_mask

        #         #-------normalize single cell to 0-1 and to 0-1024
        #         norm_crop_img=(crop_img-min_value)*1.0/(max_value-min_value)
        #         norm_cell_img=(np.ceil(norm_crop_img*norm_range)).astype(np.int)*region_cell_mask#norm_range=1024

        #         #-------use robust scaler to scale single cell
        #         transformer = RobustScaler().fit(np.expand_dims(cell_img[region_cell_mask],axis=1))
        #         scale_crop_img=transformer.transform(np.expand_dims(crop_img.flatten(),axis=1)).reshape((r.bbox[2]-r.bbox[0],r.bbox[3]-r.bbox[1]))
        #         min_scale_value,max_scale_value=np.amin(scale_crop_img[region_cell_mask]),np.amax(scale_crop_img[region_cell_mask])
        #         scale_crop_img=(scale_crop_img-min_scale_value)*1.0/(max_scale_value-min_scale_value)
        #         scale_cell_img=(np.ceil(scale_crop_img*norm_range)).astype(np.int)*region_cell_mask
        #         print(min_scale_value,max_scale_value)
        #         plt.imshow(scale_cell_img)
        #         plt.show()

        # the haralick features have four directions, to meet rotation
        # invariance,use average for each feature
        print(np.sum(cell_img))
        fl_hara = mht.haralick(cell_img, ignore_zeros=True, return_mean=True)
        norm_fl_hara = mht.haralick(norm_cell_img, ignore_zeros=True, return_mean=True)

        fluor_haralick.append(fl_hara)
        norm_fluor_haralick.append(norm_fl_hara)
    fluor_haralick = np.array(fluor_haralick)
    norm_fluor_haralick = np.array(norm_fluor_haralick)

    return mean_intensity, std_intensity, intensity_range, fluor_haralick, norm_fluor_haralick


# In[7]:


fluor_feature_list = config.fluor_feature_list


def single_folder_run(img_folder, output_path, vim_chan_label):

    output_path = pipe_util2.folder_verify(output_path)
    cells_path = output_path + "cells/"
    fluor_cells_path = cells_path
    cell_seg_path = output_path + "seg/"
    vim_img_path = pipe_util2.folder_verify(img_folder)

    cell_seg_list = sorted(listdir(cell_seg_path))
    print("vim_img_path: " + vim_img_path)
    vim_img_list = sorted(glob.glob(vim_img_path + "*" + vim_chan_label + "*"))

    print(len(vim_img_list))

    for i in range(len(vim_img_list)):
        vim_img_list[i] = os.path.basename(vim_img_list[i])

    t_span = len(cell_seg_list)

    df = pd.read_csv(output_path + "/Per_Object_relink" + ".csv")

    with open(cells_path + "cells", "rb") as fp:
        cells = pickle.load(fp)

    for k in range(len(cells)):
        fluor_single_cell.convert_to_class(cells[k])
    #     with open (fluor_cells_path+'fluor_cells_'+str(posi), 'rb') as fp:
    #         cells = pickle.load(fp)

    for ti in np.arange(0, t_span, fluor_interval):
        img_num = ti + 1
        print("position:%s, progress:%d/%d" % (output_path, img_num, t_span), flush=True)
        cell_seg = imread(cell_seg_path + cell_seg_list[ti])

        vim_img = imread(vim_img_path + vim_img_list[ti])
        #         plt.figure(figsize=(12,12))
        #         plt.imshow(vim_img)
        #         plt.show()
        #         plt.figure(figsize=(12,12))
        #         plt.imshow(frangi(vim_img,scale_range=(1, 3), scale_step=0.5))
        #         plt.show()

        mean_intensity, std_intensity, intensity_range, fluor_haralick, norm_fluor_haralick = compute_fluor_info(
            cell_seg, vim_img
        )

        #         print(img_num,mean_intensity,std_intensity)
        for obj_num in np.arange(1, np.amax(cell_seg) + 1):
            ind = df.loc[(df["ImageNumber"] == img_num) & (df["ObjectNumber"] == obj_num)].index.tolist()[0]
            vim_features = [
                mean_intensity[obj_num - 1],
                std_intensity[obj_num - 1],
                intensity_range[obj_num - 1],
                fluor_haralick[obj_num - 1, :],
                norm_fluor_haralick[obj_num - 1, :],
            ]

            cells[ind].set_fluor_features("vimentin", fluor_feature_list, vim_features)

    with open(fluor_cells_path + "fluor_cells", "wb") as fp:
        pickle.dump(cells, fp)


if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    img_path = sys.argv[1]
    output_path = sys.argv[2]
    vim_chan_label = sys.argv[3]
    single_folder_run(img_path, output_path, vim_chan_label)
