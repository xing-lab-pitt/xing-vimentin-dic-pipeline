#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from skimage.segmentation import watershed, clear_border
from skimage.io import imread
from matplotlib import pyplot as plt
import scipy.ndimage as ndi
import pickle
import os
from os import listdir
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects, local_maxima, h_maxima
from skimage.measure import regionprops, label
from skimage.color import label2rgb
from PIL import Image, ImageDraw, ImageFont
from math import pi
import cv2
import glob
from pilutil import toimage


# In[2]:


main_path = '/home/zoro/Desktop/experiment_data/2019-03-10_HK2_fucci/'
input_path = main_path + 'cdt1/'
output_path = main_path + 'cdt1_output/'

posi_end = 20


# In[12]:


h_thres_1st = 0.2
small_obj_thres = 200
mask_thres = 0


# In[4]:


def color_num(labels):
    label_rgb = label2rgb(labels, bg_label=0)
    img_rgb = toimage(label_rgb)
    base = img_rgb.convert('RGBA')
    # make a blank image for the text, initialized to transparent text color
    txt = Image.new('RGBA', base.size, (255, 255, 255, 0))
    # get a font
    fnt = ImageFont.truetype('arial.ttf', 40)
    # get a drawing context
    d = ImageDraw.Draw(txt)
    for region in regionprops(labels):
        cx = int(region.centroid[1])
        cy = int(region.centroid[0])
        d.text(
            (cx, cy), str(
                labels[cy][cx]), font=fnt, fill=(
                255, 255, 255, 255))
    out = Image.alpha_composite(base, txt)
    return out


# In[13]:


for posi in range(1, posi_end + 1):
    img_path = input_path + str(posi) + '/'
    ori_img_list = sorted(listdir(img_path))

    reg_path = output_path + str(posi) + '/reg/'
    reg_img_list = sorted(listdir(reg_path))

    seg_path = output_path + str(posi) + '/seg/'
    if not os.path.exists(seg_path):
        os.makedirs(seg_path)

    rgb_num_path = output_path + str(posi) + '/rgb_num/'
    if not os.path.exists(rgb_num_path):
        os.makedirs(rgb_num_path)

    for i in range(len(reg_img_list)):

        ori_img = imread(img_path + ori_img_list[i])

        reg_img = imread(reg_path + reg_img_list[i])

        img_name = ori_img_list[i][0:len(ori_img_list[i]) - 4]

#         img_h=reg_img.shape[0]
#         img_w=reg_img.shape[1]
#         reg_img=np.reshape(reg_img,(img_h,img_w))


#         ws_marker=np.zeros((img_h,img_w))
        local_hmax = h_maxima(reg_img, h_thres_1st)
        local_hmax_label = label(local_hmax, connectivity=1)
#         j=0
#         local_hmx=[]
#         local_hmy=[]
#         for region in regionprops(local_hmax_label):
#             cx=int(region.centroid[0])
#             local_hmx.append(cx)
#             cy=int(region.centroid[1])
#             local_hmy.append(cy)
#             j+=1
#             ws_marker[cx][cy]=int(j)

        # print np.amax(ws_marker)
        labels = watershed(-reg_img, local_hmax_label,
                           mask=reg_img > mask_thres)
        labels = remove_small_objects(labels, small_obj_thres)
        labels = clear_border(labels)

        labels = label(labels, connectivity=1)
        print(np.amax(labels))

        rgb_num = color_num(labels)
        rgb_num.save(rgb_num_path + 'rgb_' + img_name + '.png')

        # should use np.uint32,could be save correctly
        img_seg = Image.fromarray(labels.astype(np.uint32), 'I')
        img_seg.save(seg_path + 'seg_' + img_name + '.png')


# In[ ]:


# fig, axes = plt.subplots(nrows=2,ncols=2, figsize=(20, 20), sharex=True, sharey=True,
#                          subplot_kw={'adjustable': 'box-forced'})
# ax = axes.ravel()
# ax[0].imshow(ori_img, cmap=plt.cm.gray, interpolation='nearest')
# ax[0].set_title('regression')
# # ax[1].imshow(img>0, cmap=plt.cm.gray, interpolation='nearest')
# # ax[1].set_title('mask')

# ax[1].imshow(rgb_num, cmap=plt.cm.spectral, interpolation='nearest')
# ax[1].set_title('Separated objects')

# ax[2].imshow(reg_img, cmap=plt.cm.gray)
# ax[2].autoscale(False)
# ax[2].plot(local_hmy[:], local_hmx[:], 'r.')#reverse order of cx and cy, because they are from skimage
# ax[2].axis('off')
# ax[2].set_title('Peak local max')

# ax[3].imshow(labels, cmap=plt.cm.spectral, interpolation='nearest')
# ax[3].set_title('Separated objects')

# for a in ax:
#     a.set_axis_off()
# fig.tight_layout()
# plt.show()


# In[ ]:
