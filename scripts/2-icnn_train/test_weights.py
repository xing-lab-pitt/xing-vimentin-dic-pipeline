# In[0]: import
import sys
sys.path.insert(0, 'C:/Users/14432/OneDrive/Research/Projects/A549_144hr/scripts/memes/')

import numpy as np
import os
from os import listdir

from skimage.io import imread

import matplotlib.pyplot as plt

from cnn_prep_data import keep_aspect_resize, obj_transform
from resnet50 import res_model

# In[1]: initiate

dat_dir='C:/Users/14432/OneDrive/Research/Projects/A549_144hr/data/'
wts_dir = dat_dir + 'wts/'
icnn_am_weights=wts_dir+'icnn_am_dat-gen_on_dat-comb_on_alph_15E-4_run_5-10-11-21.hdf5'
obj_h=128
obj_w=128
input_shape=(obj_h,obj_w,1)
nb_class=3
icnn_am=res_model(input_shape,nb_class)
icnn_am.load_weights(icnn_am_weights)

# other-0, mitosis-1, apoptosis-2
test_dir=dat_dir+'test/icnn_am/mitosis/'
test_files = listdir(test_dir)
am_record = [0,0,0]
for file in test_files:
    img = imread(test_dir+file)
    
    trans_img = obj_transform(keep_aspect_resize(img,obj_h,obj_w),random_eah=False)
    trans_img = np.expand_dims(trans_img,axis=0)
    
    output = icnn_am.predict(trans_img)
    am_flag = np.argmax(output)
    am_record[am_flag] = am_record[am_flag]+1