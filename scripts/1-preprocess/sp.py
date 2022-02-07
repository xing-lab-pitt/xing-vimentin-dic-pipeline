# In[0]: import
import sys
sys.path.insert(1, 'C:/Users/14432/OneDrive/Research/Projects/a549_pcna/scripts/memes/')

import warnings
warnings.filterwarnings("ignore")

import os
import matplotlib.pyplot as plt
import skimage.measure as skm
import cv2 as cv
from sklearn.model_selection import train_test_split
import argparse
from skimage.transform import resize
import matplotlib.patches as patches
import numpy as np
import glob 

import hj_util

# In[1]: define
# Create the parser
main_path = hj_util.folder_verify('C:/Users/14432/OneDrive/Research/Projects/a549_pcna/data/test/reg')
test_path = hj_util.folder_verify(main_path+'tile')
edt_dir =  hj_util.folder_verify(main_path+'edt_tile_pt07')
seg_dir = hj_util.folder_verify(main_path+'crop')

test_filenames = os.listdir(test_path)
#--weight_file, This is the weight file used to generate the edt predictions. This is only used for record keeping
#--test_dir, This is a directory with only the tiles used for testing
#--edt_dir, This is the directory with the results from the cell segmentation model
#--seg_dir, This is the directory where all the crops are, it does not need to only contain test crops, but all from the training set. This script will use globs to find the correct crops
# In[2]: main
tile = test_filenames[0]
tile_wt_ext=tile.split('.')[0]
seg_wt_ext_wt_channel='_'.join([x for x in tile_wt_ext.split('_') if x!= 'C1'])
seg_glob=seg_wt_ext_wt_channel+'*'

original_tile_bw=cv.imread(test_path+tile,0)
seg_zeros_matrix=np.full_like(original_tile_bw,0)
edt_tile=cv.imread(edt_dir+'edt_'+tile,-1)

for seg_num in range(len(glob.glob(seg_dir+'BIB_'+seg_glob))):

    #creates filenames and loads interior and original crop images
    ################
    interior_seg=seg_dir+'interior_'+seg_wt_ext_wt_channel+'_cr'+str(seg_num+1)+'.png'
    crop_seg=seg_dir+'crop_'+seg_wt_ext_wt_channel+'_cr'+str(seg_num+1)+'.png'

    interior_img=cv.imread(interior_seg,-1)
    crop_img_bw=cv.imread(crop_seg,0)
    ################

    # Apply template Matching

    ################

    w,h=crop_img_bw.shape[::-1]

    res = cv.matchTemplate(original_tile_bw,crop_img_bw,eval('cv.TM_CCOEFF'))
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    top_left = max_loc

    bottom_right = (top_left[0] + w, top_left[1] + h)
    ###############

    # I "paste" these values into the zeros matrix using the coordinates from the template matching. This will place all crops from one tile into the same image
    seg_zeros_matrix[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]]=interior_img
    
labeled_seg_zeros_matrix=skm.label(seg_zeros_matrix)

#keeps track of the cell accuracy's to be averaged later
cell_acc_list=[]
