#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean, cosine
import math
from math import pi
import scipy
from skimage.io import imread
from skimage.measure import regionprops, find_contours
from skimage.color import label2rgb
from skimage import morphology, measure
from os import listdir
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt

from scipy import ndimage
from skimage.morphology import dilation, erosion, disk, remove_small_objects, binary_opening, opening
from skimage.feature import canny
import cv2
import numpy_indexed as npi
from skimage.filters import roberts, sobel, scharr, prewitt
import pickle

import pipe_util2


# In[16]:


posi_end = 1

# main_path='/home/zoro/Desktop/experiment_data/2019-03-22_a549_tgf4ng_2d/'
main_path = '/mnt/data0/Ke/weikang_exp_process/2019-05-05_A549_vim/'
output_path = main_path + 'output/'
# input_path=main_path+'img/'
input_path = main_path + 'a549_tif/'

cells_path = main_path + 'fluor_cells/'


# In[17]:


def judge_border(img):
    border_flag = []
    rps = regionprops(img)
    img_h = img.shape[0]
    img_w = img.shape[1]
    for r in rps:
        obj_bbox = r.bbox
        if obj_bbox[0] == 0 or obj_bbox[1] == 0 or obj_bbox[2] == img_h or obj_bbox[3] == img_w:
            border_flag.append(1)
        else:
            border_flag.append(0)
    return border_flag


def cal_border(img_num, seg_img, border_flag):
    neighbor_info = []
    x = seg_img
    neighbors = np.concatenate(
        [x[:, :-1].flatten(), x[:, +1:].flatten(), x[+1:, :].flatten(), x[:-1, :].flatten()])
    centers = np.concatenate(
        [x[:, +1:].flatten(), x[:, :-1].flatten(), x[:-1, :].flatten(), x[+1:, :].flatten()])
    valid = neighbors != centers
    regions, neighbors_per_regions = npi.group_by(
        centers[valid], neighbors[valid])
    for region, neighbors_per_region in zip(regions, neighbors_per_regions):
        if region != 0:

            unique_neighbors, neighbor_border_len = npi.count(
                neighbors_per_region)
            border_ratio = neighbor_border_len.astype(
                np.float) / neighbor_border_len.sum()
            inds = np.argsort(-border_ratio)
            sort_unique_neighbors = unique_neighbors[inds]
            sort_neighbor_border_len = neighbor_border_len[inds]
            sort_border_ratio = border_ratio[inds]

            non0_inds = [i for i, uni_nei in enumerate(
                sort_unique_neighbors) if uni_nei != 0]

            nonfree_border = sum(sort_border_ratio[non0_inds])

            neighbor_list = [
                (uni_nei,
                 nei_border_len) for uni_nei,
                nei_border_len in zip(
                    sort_unique_neighbors[non0_inds].tolist(),
                    sort_neighbor_border_len[non0_inds].tolist())]

            neighbor_info.append([img_num, region.astype(int), border_flag[(
                region - 1).astype(int)], nonfree_border, neighbor_list])

    return neighbor_info


# -----------calculate neighbor connected with pseudopodia----------
def local_neighbor_info_cal(img_path, output_path):

    img_path = pipe_util2.folder_verify(img_path)
    output_path = pipe_util2.folder_verify(output_path)

    ori_img_path = img_path
    mask_img_path = output_path + str(posi) + '/mask/'
    seg_img_path = output_path + str(posi) + '/seg/'
    ori_img_list = sorted(listdir(ori_img_path))
    mask_img_list = sorted(listdir(mask_img_path))
    seg_img_list = sorted(listdir(seg_img_path))
    df = pd.read_csv(
        output_path +
        str(posi) +
        '/Per_Object_relink_' +
        str(posi) +
        '.csv')

    with open(cells_path + 'fluor_cells_' + str(posi), 'rb') as fp:
        cells = pickle.load(fp)

    for i in range(len(ori_img_list)):
        img_num = i + 1

        ori_img = imread(ori_img_path + '/' + ori_img_list[img_num - 1])
        seg_img = imread(seg_img_path + '/' + seg_img_list[img_num - 1])
        mask_img = imread(mask_img_path + '/' + mask_img_list[img_num - 1])

        mask = seg_img > 0
        mask = (mask + mask_img) > 0

        m, n = ndimage.distance_transform_edt(
            seg_img == 0, return_distances=False, return_indices=True)
        dt_seg_img = (seg_img[m, n] * mask).astype(np.int)

#         plt.imshow(seg_img)
#         plt.show()
#         plt.imshow(mask_img)
#         plt.show()

#         plt.imshow(dt_seg_img)
#         plt.show()

        new_dt_seg_img = np.zeros(dt_seg_img.shape)
        for cell_label in range(1, np.amax(dt_seg_img) + 1):
            single_dt_obj_img = (dt_seg_img == cell_label)

            label_img = measure.label(single_dt_obj_img)
            if np.amax(label_img) > 1:
                single_obj_img = (seg_img == cell_label)
                single_dt_obj_img = ndimage.binary_propagation(
                    single_obj_img, mask=label_img)
                new_dt_seg_img += cell_label * single_dt_obj_img
            else:
                new_dt_seg_img += cell_label * single_dt_obj_img

#         plt.imshow(new_dt_seg_img)
#         plt.show()

        border_flag = judge_border(new_dt_seg_img.astype(int))

        nei_info = cal_border(img_num, new_dt_seg_img, border_flag)

        for j in range(len(nei_info)):
            ind = df.loc[(df['ImageNumber'] == img_num) & (
                df['ObjectNumber'] == nei_info[j][1])].index.tolist()[0]
            cells[ind].set_neighbor_info(
                nei_info[j][2], nei_info[j][3], nei_info[j][4])

    with open(cells_path + 'fluor_cells_' + str(posi), 'wb') as fp:
        pickle.dump(cells, fp)


# In[1]:


# #-----------calculate neighbor connected with cellbody----------
# for posi in range(1,posi_end+1):
#     print(posi)
#     seg_img_path=output_path+str(posi)+'/seg/'
#     seg_img_list=sorted(listdir(seg_img_path))
#     df=pd.read_csv(output_path+str(posi)+'/Per_Object_relink.csv')

#     with open (cells_path+'cells_'+str(posi), 'rb') as fp:
#         cells = pickle.load(fp)

#     for i in range(len(seg_img_list)):
#         img_num=i+1

#         seg_img=imread(seg_img_path+'/'+seg_img_list[img_num-1])


#         border_flag=judge_border(seg_img.astype(int))


#         nei_info=cal_border(img_num,seg_img,border_flag)
#         for j in range(len(nei_info)):
#             ind=df.loc[(df['ImageNumber']==img_num)&(df['ObjectNumber']==nei_info[j][1])].index.tolist()[0]
#             cells[ind].set_neighbor_info(nei_info[j][2],nei_info[j][3],nei_info[j][4])

#     with open(cells_path+'cells_'+str(posi), 'wb') as fp:
#         pickle.dump(cells, fp)
