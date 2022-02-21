#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import os
from os import listdir
from PIL import Image as PImage
from pilutil import toimage
from skimage.io import imread
import glob
import random

from pilutil import toimage
import pandas as pd
# import opencv
from skimage.transform import resize
import sys
import hj_util as util

input_path=sys.argv[1]
resize_path=sys.argv[2]
position_label=sys.argv[3]

input_path = util.folder_verify(input_path)
resize_path = util.folder_verify(resize_path)

if not os.path.exists(resize_path):
    os.makedirs(resize_path)

img_list=sorted(glob.glob(input_path + "*" + position_label + "*"))

for i in range(len(img_list)):
    print(img_list[i])
    img=imread(img_list[i])
    re_img_arr=(resize(img, (img.shape[0]*2,img.shape[1]*2), preserve_range=True, anti_aliasing=False,order=0)/4).astype(np.int)
    re_Img=PImage.fromarray(re_img_arr.astype(np.uint32),'I')#should use np.uint32,could be save correctly
    img_name = os.path.basename(img_list[i])
    re_Img.save(resize_path + img_name)


