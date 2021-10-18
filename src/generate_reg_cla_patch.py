#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.measure import label, regionprops
from skimage.color import label2rgb
import numpy as np
import os
from os import listdir
from PIL import Image as PImage
from pilutil import toimage
import scipy.misc

# from scipy.misc import imread
import glob
import random
from scipy.ndimage.morphology import distance_transform_edt
import skimage.morphology
import xml.etree.ElementTree as ET


# In[2]:


train_patch_count = 0
test_patch_count = 0


# In[3]:


main_path = "./A549_cellbody_emt_label/seg/"
label_path = main_path
# image_num=6# the specific image that you want to deal with specificly,
# use this with for ti in range(image_num,image_num+1)

test_start = 100
test_end = 100

# train_patch=30
# test_patch=10

# patch_size
train_patch_w = 320
train_patch_h = 320

test_patch_w = 320
test_patch_h = 320
# ---crop a image by several times to increase samples
crop_copy = 8

ori_path = label_path


# In[4]:


# ------weight map parameter----------
sigma = 30
w_thres = np.exp(-4)
gamma = 2

wv = 1 / np.exp(-2)


def cal_weights_map(Interior):
    Interior = np.asarray(Interior)
    if np.amax(Interior) > 0:
        seg = label(Interior)

        edt1 = distance_transform_edt(seg == 0)

        edt2 = np.zeros((int(np.amax(seg)), seg.shape[0], seg.shape[1]))
        for i in range(1, np.amax(seg) + 1):
            other_seg = seg.copy()
            other_seg[other_seg == i] = 0
            edt2[i - 1, :, :] = distance_transform_edt(other_seg == 0)

        edt = edt1 + np.amax(edt2, axis=0)

        bnd_weights = np.exp(-((edt / sigma) ** gamma)) * (seg == 0)
        bnd_weights[bnd_weights < w_thres] = 0

        norm_bnd_weights = bnd_weights / (max(np.amax(bnd_weights), 1 / wv) + 1e-7)
        bnd_weights = bnd_weights * wv  # np.median(bnd_weights[bnd_weights>0])

        gt = distance_transform_edt(Interior)
        if np.count_nonzero(gt) != 0:
            nonzero_gt = gt[gt > 0]
            median_edt = gt * 1.0 / np.median(nonzero_gt)
            norm_edt = gt * 1.0 / np.amax(nonzero_gt)

        median_weights_map = bnd_weights - median_edt
        norm_weights_map = norm_bnd_weights - norm_edt
    else:
        median_weights_map, norm_weights_map = np.zeros(Interior.shape), np.zeros(Interior.shape)

    median_wp = toimage(median_weights_map, high=np.max(median_weights_map), low=np.min(median_weights_map), mode="F")
    norm_wp = toimage(norm_weights_map, high=np.max(norm_weights_map), low=np.min(norm_weights_map), mode="F")
    return median_wp, norm_wp


# In[5]:


# generate regression and mask train_test data


train_img_path = main_path + "train/Img/"
if not os.path.exists(train_img_path):
    os.makedirs(train_img_path)
train_reg_path = main_path + "train/Bwdist/"
if not os.path.exists(train_reg_path):
    os.makedirs(train_reg_path)
train_mask_path = main_path + "train/Mask/"
if not os.path.exists(train_mask_path):
    os.makedirs(train_mask_path)
train_BIB_path = main_path + "train/BIB/"
if not os.path.exists(train_BIB_path):
    os.makedirs(train_BIB_path)
train_interior_path = main_path + "train/Interior/"
if not os.path.exists(train_interior_path):
    os.makedirs(train_interior_path)
train_boundary_path = main_path + "train/Boundary/"
if not os.path.exists(train_boundary_path):
    os.makedirs(train_boundary_path)

# train_mwp_path=main_path+'train/Median_wp/'
# if not os.path.exists(train_mwp_path):
#     os.makedirs(train_mwp_path)

train_nwp_path = main_path + "train/Norm_wp/"
if not os.path.exists(train_nwp_path):
    os.makedirs(train_nwp_path)


# random crop image into patches

Img_str = sorted(glob.glob(ori_path + "crop*.tif"))
Boundary_str = sorted(glob.glob(ori_path + "boundary*.png"))
Interior_str = sorted(glob.glob(ori_path + "interior*.png"))
BIB_str = sorted(glob.glob(ori_path + "BIB*.png"))
Norm_wp_str = sorted(glob.glob(ori_path + "norm_weights_map*.tif"))

for ti in range(len(Img_str)):
    print(ti)
    Img = PImage.open(Img_str[ti])
    Boundary = PImage.open(Boundary_str[ti])
    Interior = PImage.open(Interior_str[ti])
    BIB = PImage.open(BIB_str[ti])
    # Norm_wp=PImage.open(Norm_wp_str[ti])

    img_w = Img.size[0]
    img_h = Img.size[1]

    i = 0
    j = 0
    if ti < test_start or ti > test_end:
        # -------------------------------generate train patch

        train_patch = round((img_w * 1.0 / train_patch_w) * (img_h * 1.0 / train_patch_h) * crop_copy)

        print(train_patch)
        while i < train_patch:

            xmin = round(random.uniform(0, 1) * (img_w - train_patch_w))
            ymin = round(random.uniform(0, 1) * (img_h - train_patch_h))
            rect = [xmin, ymin, xmin + train_patch_w, ymin + train_patch_h]
            print(np.amax(np.asarray(Interior.crop(rect))))
            #             if np.amax(np.asarray(Interior.crop(rect)))>0:
            train_patch_count += 1

            crop_Img = Img.crop(rect)
            # crop_Img_str=Img_str[ti].replace(ori_path,'')
            crop_Img.save(train_img_path + "s" + str(train_patch_count) + ".tif")

            crop_Interior = Interior.crop(rect)
            crop_Interior.save(train_interior_path + "s" + str(train_patch_count) + ".png")

            #                 crop_Bwdist=Bwdist.crop(rect)
            crop_Bwdist = distance_transform_edt(crop_Interior)
            crop_Bwdist = toimage(crop_Bwdist, high=np.max(crop_Bwdist), low=np.min(crop_Bwdist), mode="F")
            crop_Bwdist.save(train_reg_path + "s" + str(train_patch_count) + ".tif")

            crop_Boundary = Boundary.crop(rect)
            crop_Boundary.save(train_boundary_path + "s" + str(train_patch_count) + ".png")

            crop_BIB = BIB.crop(rect)
            crop_BIB.save(train_BIB_path + "s" + str(train_patch_count) + ".png")

            #             crop_Median_wp,crop_Norm_wp=cal_weights_map(crop_Interior)
            #             crop_Median_wp.save(train_mwp_path+'s'+str(train_patch_count)+'.tif')
            # crop_Norm_wp=Norm_wp.crop(rect)
            # crop_Norm_wp.save(train_nwp_path+'s'+str(train_patch_count)+'.tif')

            i += 1
#     else:
#         #-----------------------------------generate test patch
#         test_patch=round((img_w*1.0/test_patch_w)*(img_h*1.0/test_patch_h)*crop_copy)
#         while (j<test_patch):
#             xmin=round(random.uniform(0, 1)*(img_w-test_patch_w))
#             ymin=round(random.uniform(0, 1)*(img_h-test_patch_h))
#             rect=[xmin,ymin,xmin+test_patch_w,ymin+test_patch_h]

#             crop_Img=Img.crop(rect)
#             crop_Img_str=Img_str[ti].replace(ori_path,'')
#             crop_Img.save(test_img_path+'s'+str(test_patch_count)+'.tif')

#             crop_Boundary=Boundary.crop(rect)
#             crop_Boundary_str=Boundary_str[ti].replace(ori_path,'')
#             crop_Boundary.save(test_boundary_path+'s'+str(test_patch_count)+'.png')

#             crop_Interior=Interior.crop(rect)
#             crop_Interior_str=Interior_str[ti].replace(ori_path,'')
#             crop_Interior.save(test_interior_path+'s'+str(test_patch_count)+'.png')


# #             crop_Bwdist=distance_transform_edt(crop_Interior)
# #             crop_Bwdist_str=Bwdist_str[ti].replace(ori_path,'')
# #             crop_Bwdist=toimage(crop_Bwdist,high=np.max(crop_Bwdist),low=np.min(crop_Bwdist),mode='F')
# #             crop_Bwdist.save(test_reg_path+'s'+str(test_patch_count)+'.tif')

#             crop_BIB=BIB.crop(rect)
#             crop_BIB_str=BIB_str[ti].replace(ori_path,'')
#             crop_BIB.save(test_BIB_path+'s'+str(test_patch_count)+'.png')

# #             crop_Colony=Colony.crop(rect);
# #             crop_Colony_str=Colony_str[ti].replace(ori_path,'')
# #             crop_Colony.save(test_mask_path+crop_Colony_str[0:len(crop_Colony_str)-4]+'s'+str(j)+'.png');

#             j+=1


# In[6]:


# test_img_path=main_path+'test/Img/'
# if not os.path.exists(test_img_path):
#     os.makedirs(test_img_path)
# test_reg_path=main_path+'test/Bwdist/'
# if not os.path.exists(test_reg_path):
#     os.makedirs(test_reg_path)
# test_mask_path=main_path+'test/Mask/'
# if not os.path.exists(test_mask_path):
#     os.makedirs(test_mask_path)
# test_BIB_path=main_path+'test/BIB/'
# if not os.path.exists(test_BIB_path):
#     os.makedirs(test_BIB_path)
# test_interior_path=main_path+'test/Interior/'
# if not os.path.exists(test_interior_path):
#     os.makedirs(test_interior_path)
# test_boundary_path=main_path+'test/Boundary/'
# if not os.path.exists(test_boundary_path):
#     os.makedirs(test_boundary_path)


# In[ ]:
