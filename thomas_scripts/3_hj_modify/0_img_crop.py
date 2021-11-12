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

input_path = sys.argv[1]
crop_path = sys.argv[2]
position_label = sys.argv[3]

input_path = util.folder_verify(input_path)
crop_path = util.folder_verify(crop_path)

if not os.path.exists(crop_path):
    os.makedirs(crop_path)

img_list = sorted(glob.glob(input_path + "*" + position_label + "*"))

for i in range(len(img_list)):
    print(img_list[i])
    img = imread(img_list[i])
    img_cropped = img[24 : img.shape[0] - 24, 24 : img.shape[0] - 24]
    crop_Img = PImage.fromarray(img_cropped.astype(np.uint32), "I")  # should use np.uint32,could be save correctly
    img_name = os.path.basename(img_list[i])
    crop_Img.save(crop_path + img_name)
