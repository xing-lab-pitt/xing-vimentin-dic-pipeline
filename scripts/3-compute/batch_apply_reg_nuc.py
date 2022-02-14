#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import sys
import cv2
from skimage.io import imread, imread_collection
from scipy import signal
from scipy.ndimage import filters, distance_transform_edt
from PIL import Image
from skimage.filters import gaussian
import os
from os import listdir

from keras.models import Model
from keras import optimizers
from keras.layers import Input, Activation, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Dropout
from reg_seg_model import reg_seg
import glob
from skimage.exposure import equalize_adapthist
from matplotlib import pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.interpolate import bisplrep, bisplev
from cnn_prep_data import prep_fluor_data


# In[2]:


main_path = "/home/zoro/Desktop/experiment_data/2019-03-10_HK2_fucci/"
input_path = main_path + "cdt1/"
output_path = main_path + "cdt1_output/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

posi_end = 20

weight_file = "HK2_nuc.hdf5"
autoencoder = reg_seg()
autoencoder.load_weights(weight_file)


# In[30]:


for posi in range(1, posi_end + 1):
    img_path = input_path + str(posi) + "/"
    img_list = sorted(listdir(img_path))
    posi_path = output_path + str(posi) + "/reg/"
    if not os.path.exists(posi_path):
        os.makedirs(posi_path)
    for i in range(len(img_list)):
        img_num = i + 1
        predict_data = prep_fluor_data(img_path, img_list, img_num)
        output = autoencoder.predict(predict_data, batch_size=1, verbose=0)
        # save image to the exat value
        img = Image.fromarray(output[0, :, :, 0])
        img.save(posi_path + "reg_" + img_list[i])


# In[ ]:
