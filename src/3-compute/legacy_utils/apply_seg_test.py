#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import os
import sys
from os import listdir

import cv2
import numpy as np
import pandas as pd
from legacy_utils.cla_seg_model_loss import cla_seg
from keras import optimizers
from keras.layers import (Activation, BatchNormalization, Conv2D, Dropout,
                          Input, MaxPooling2D, UpSampling2D)
from keras.models import Model
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from legacy_utils.pilutil import toimage
from legacy_utils.reg_seg_lin_model import reg_seg_lin
from legacy_utils.reg_seg_model import reg_seg
from scipy import signal
from scipy.ndimage import distance_transform_edt, filters
from skimage import morphology
from skimage.color import label2rgb
from skimage.exposure import equalize_adapthist
from skimage.io import imread, imread_collection
from skimage.measure import find_contours, label, regionprops
from skimage.morphology import (h_maxima, h_minima, opening,
                                remove_small_objects)
from skimage.segmentation import clear_border, watershed

# In[8]:


weight_file = "A549_cellbody_mwp_ep100.hdf5"
n_labels = 2
autoencoder = reg_seg_lin()  # cla_seg(n_labels)
autoencoder.load_weights(weight_file)


# In[9]:


def color_num(labels):
    label_rgb = label2rgb(labels, bg_label=0)
    img_rgb = toimage(label_rgb)
    base = img_rgb.convert("RGBA")
    # make a blank image for the text, initialized to transparent text color
    txt = Image.new("RGBA", base.size, (255, 255, 255, 0))
    # get a font
    # fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 60)#specify
    # the path of font file
    fnt = ImageFont.truetype("arial.ttf", 40)
    # get a drawing context
    d = ImageDraw.Draw(txt)
    for region in regionprops(labels):
        cx = int(region.centroid[1])
        cy = int(region.centroid[0])
        d.text((cx, cy), str(labels[cy][cx]), font=fnt, fill=(255, 255, 255, 255))
    out = Image.alpha_composite(base, txt)
    return out


# In[10]:


def prep_data(img_path, img_list, img_num):

    img = imread(img_path + img_list[img_num - 1])
    #         plt.imshow(img)
    #         plt.show()
    img = (img - np.amin(img)) * 1.0 / (np.amax(img) - np.amin(img))  # img*1.0 transform array to double

    img = img * 1.0 / np.median(img)

    img = np.expand_dims(img, axis=2)
    data = np.expand_dims(img, axis=0)
    return data


# In[11]:


def hmax_watershed(img, h_thres, small_obj_thres, mask_thres=0):
    # h_thres should be high to avoid reg fragments
    ws_marker = np.zeros(img.shape)

    local_hmax = h_maxima(img, h_thres)
    local_hmax_label = label(local_hmax, connectivity=1)

    labels = watershed(-img, local_hmax_label, mask=img > mask_thres)
    labels = remove_small_objects(labels, small_obj_thres)

    labels = label(labels, connectivity=2)
    plt.title("labels")
    plt.imshow(labels)
    plt.show()
    print(np.amax(labels))
    return labels


# In[12]:


main_path = "/home/zoro/Desktop/experiment_data/2019-10-12_A549_vim_dtt10um_tgf4ng/"
img_path = main_path + "test_img/"
img_list = sorted(listdir(img_path))
output_path = main_path + "output/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

reg_path = output_path + "/reg/"
if not os.path.exists(reg_path):
    os.makedirs(reg_path)

for img_num in range(1, len(img_list) + 1):
    predict_data = prep_data(img_path, img_list, img_num)

    output = autoencoder.predict(predict_data, batch_size=1, verbose=0)
    im = output[0, :, :, 0]
    plt.imshow(im)
    plt.show()
    img = Image.fromarray(im)
    img.save(reg_path + "reg_" + img_list[img_num - 1])


# In[13]:


h_thres = 0.2
small_obj_thres = 1500
mask_thres = 0
reg_img_list = sorted(listdir(reg_path))


seg_path = output_path + "seg/"
if not os.path.exists(seg_path):
    os.makedirs(seg_path)

rgb_num_path = output_path + "rgb_num/"
if not os.path.exists(rgb_num_path):
    os.makedirs(rgb_num_path)

for i in range(len(reg_img_list)):
    img = imread(img_path + img_list[i])
    reg_img = imread(reg_path + reg_img_list[i])
    img_name = img_list[i][0 : len(img_list[i]) - 4]
    reg_img = -reg_img * (reg_img < -0.05)
    seg = hmax_watershed(reg_img, h_thres=h_thres, small_obj_thres=small_obj_thres, mask_thres=0)

    # should use np.uint32,could be save correctly
    img_seg = Image.fromarray(seg.astype(np.uint32), "I")
    img_seg.save(seg_path + "seg_" + img_name + ".png")

    rgb_num = color_num(seg)
    rgb_num.save(rgb_num_path + "rgb_" + img_name + ".png")


# In[14]:


def find_contours_labelimg(seg_img, contour_value):
    seg_img = opening(seg_img)
    contours = []

    rps = regionprops(seg_img)
    r_labels = [r.label for r in rps]
    # print(r_labels)
    contours = []
    for label in r_labels:
        single_obj_seg_img = seg_img == label
        single_contour = find_contours(
            single_obj_seg_img, level=contour_value, fully_connected="low", positive_orientation="low"
        )
        # print(len(single_contour))
        max_len = 0
        for i in range(len(single_contour)):
            if len(single_contour[i]) >= max_len:
                maj_i = i
                max_len = len(single_contour[i])
        # need append the element of in single_contour instead of the whole
        # array
        contours.append(single_contour[maj_i])
    return contours


# In[15]:


contours = find_contours_labelimg(seg, contour_value=0.5)
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(img)
plt.show()
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(img, interpolation="nearest", cmap=plt.cm.gray)

for n, contour in enumerate(contours):
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

ax.axis("image")
plt.show()


# In[57]:


# from keras import backend as K
# from keras.losses import kullback_leibler_divergence,mean_squared_error
# y_true=imread(main_path+'bwdist_D0 Dish1 F6 300ms Pcr1s0.tif')
# plt.imshow(y_true)
# plt.show()
# y_pred=reg_img
# plt.imshow(y_pred)
# plt.show()
# print((np.square(y_true - y_pred)).mean(axis=None))

# K.print_tensor(mean_squared_error(y_true,y_pred))
# K.print_tensor(kullback_leibler_divergence(y_true>0, y_pred>0))


# In[11]:


# fig, axes = plt.subplots(nrows=1,ncols=3, figsize=(20, 20), sharex=True, sharey=True,
#                          subplot_kw={'adjustable': 'box-forced'})
# ax = axes.ravel()


# ax[0].imshow(rgb_num, cmap=plt.cm.Spectral, interpolation='nearest')
# ax[0].set_title('Separated objects')

# ax[1].imshow(img, cmap=plt.cm.gray)
# ax[1].autoscale(False)
# ax[1].plot(local_hmy[:], local_hmx[:], 'r.')#reverse order of cx and cy, because they are from skimage
# ax[1].axis('off')
# ax[1].set_title('Peak local max')

# ax[2].imshow(labels, cmap=plt.cm.Spectral, interpolation='nearest')
# ax[2].set_title('Separated objects')

# for a in ax:
#     a.set_axis_off()
# fig.tight_layout()
# plt.show()


# In[2]:


# from skimage.filters import frangi,hessian

# img=imread('/home/zoro/Desktop/z_temp_test_data/reg_hk2_fucci_1dxy01t001c3.tif')
# plt.imshow(img)
# plt.show()
# plt.imshow(frangi(img))
# plt.show()

# plt.imshow(hessian(img))
# plt.show()


# In[3]:


# from scipy import ndimage
# from sklearn import mixture
# edt=ndimage.distance_transform_edt(img>0)
# ratio=np.amax(edt)/np.amax(img)
# print(np.amax(edt),np.amax(img),ratio)
# cnn_edt=np.round(ratio*img)
# plt.imshow(cnn_edt)
# plt.show()

# Dxy=[]
# for i in np.arange(0,img.shape[0],5):
#     for j in np.arange(0,img.shape[1],5):
#         if cnn_edt[i,j]>0:
#             for k in range(cnn_edt[i,j]):
#                 Dxy.append([i,j])
# Dxy=np.array(Dxy)
# dpgmm = mixture.BayesianGaussianMixture(n_components=20,covariance_type='full').fit(Dxy)
# print(dpgmm.weights_)


# In[4]:


# y_pred=dpgmm.predict(Dxy)
# plt.scatter(Dxy[:,0],Dxy[:,1],c=y_pred)
# plt.show()


# In[5]:


# from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
# from skimage.segmentation import mark_boundaries

# #img=imread('/home/zoro/Desktop/z_temp_test_data/reg_hk2_fucci_1dxy01t001c3.tif')
# # segments_fz = felzenszwalb(img, scale=1, sigma=1, min_size=100)
# # plt.imshow(mark_boundaries(img, segments_fz))
# # plt.show()

# segments_slic = slic(img.astype(np.float), n_segments=100,multichannel=False,enforce_connectivity=False, compactness=0.1, sigma=1)
# plt.imshow(mark_boundaries(img, segments_slic))
# plt.show()

# segments_quick = quickshift(img.astype(np.double), kernel_size=10, max_dist=6,convert2lab=False, ratio=0.5)
# plt.imshow(mark_boundaries(img, segments_quick))
# plt.show()


# In[6]:


# from skimage.filters import sobel,scharr
# from skimage.morphology import disk
# from skimage.filters.rank import gradient
# edge_img=scharr(img)
# plt.imshow(edge_img)
# plt.show()


# out = gradient(img, disk(5))
# plt.imshow(out)
# plt.show()


# In[7]:


# h = 0.01
# h_mini =h_minima(edge_img, h)
# plt.imshow(h_mini)
# plt.show()
# label_h_minima = label(h_mini)
# j=0
# local_hminx=[]
# local_hminy=[]
# for region in regionprops(label_h_minima):
#     cx=int(region.centroid[0])
#     local_hminx.append(cx)
#     cy=int(region.centroid[1])
#     local_hminy.append(cy)
# plt.imshow(edge_img, cmap=plt.cm.gray)
# plt.autoscale(False)
# plt.plot(local_hminy[:], local_hminx[:], 'r.')#reverse order of cx and cy, because they are from skimage
# plt.axis('off')
# plt.show()


# In[ ]:
