#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from skimage import measure
from skimage.segmentation import find_boundaries
from skimage.morphology import opening, closing
from skimage.io import imread
from matplotlib import pyplot as plt
import os
from os import listdir
import pandas as pd
from scipy.stats import kde
import seaborn as sns
import copy
import pickle
from cell_class import single_cell

import contour_class
import utility_tools as utility_tools
import scipy.ndimage as ndimage
import scipy.interpolate.fitpack as fitpack
import image_warp as image_warp
from contour_tool import (
    df_find_contour_points,
    find_contour_points,
    generate_contours,
    align_contour_to,
    align_contours,
)
import pipe_util2

# In[2]:

# main_path='/home/zoro/Desktop/experiment_data/2019-02-22_HK2_3d/'
# main_path = '/mnt/data0/Ke/weikang_exp_process/2019-05-05_A549_vim/'
# # input_path=main_path+'/img/'
# input_path=main_path+'a549_tif/vimentin/' # 'img/'
# output_path=main_path+'output/'

# -------------initiate all cells--------------------
def build_single_cell(output_path):
    output_path = pipe_util2.folder_verify(output_path)
    cells_path = pipe_util2.create_folder(output_path + "cells")
    dir_path = output_path

    df = pd.read_csv(dir_path + "Per_Object_relink" + ".csv")
    am_record = pd.read_csv(dir_path + "am_record" + ".csv")
    traj_df = pd.read_csv(dir_path + "traj_object_num" + ".csv")
    # todo: add tritc#/fitc# nucleus info
    # DIC cell info: image# segmentation mask channel, object# cell id in a single mask, pixelCoord
    # Cell_TrackObjects_Label: consistent cell id in trajectory
    cells = [single_cell(img_num=df.loc[i, "ImageNumber"], obj_num=df.loc[i, "ObjectNumber"]) for i in range(len(df))]
    for i in range(len(cells)):
        img_num, obj_num = cells[i].img_num, cells[i].obj_num
        # set_cell_feaures
        cells[i].set_cell_features(
            df.columns[3:22], df.loc[i, "Cell_AreaShape_Area":"Cell_AreaShape_Solidity"].values.tolist()
        )
        # set_traj_label
        cells[i].set_traj_label(
            np.asscalar(
                df.loc[
                    (df["ImageNumber"] == img_num) & (df["ObjectNumber"] == obj_num), "Cell_TrackObjects_Label"
                ].values
            )
        )
        # set_apoptosis_mitosis_flag
        if ((am_record["ImageNumber"] == img_num) & (am_record["ObjectNumber"] == obj_num)).any():
            cells[i].set_am_flag(
                am_record.loc[
                    (am_record["ImageNumber"] == img_num) & (am_record["ObjectNumber"] == obj_num), "am_flag"
                ].values
            )
        else:
            cells[i].set_am_flag(0)

    with open(cells_path + "cells", "wb") as fp:
        pickle.dump(cells, fp)


# In[ ]:


# #----------reset traj label
# for posi in range(1,posi_end+1):
#     dir_path=output_path+str(posi)+'/'
#     df=pd.read_csv(dir_path+'Per_Object_relink_'+str(posi)+'.csv')
#     am_record=pd.read_csv(dir_path+'am_record_'+str(posi)+'.csv')
#     traj_df=pd.read_csv(dir_path+'traj_object_num_'+str(posi)+'.csv')
#     with open (cells_path+'cells_'+str(posi), 'rb') as fp:
#         cells = pickle.load(fp)
#     for i in range(len(cells)):
#         cells[i].set_traj_label(np.asscalar(df.loc[(df['ImageNumber']==img_num)&(df['ObjectNumber']==obj_num),'Cell_TrackObjects_Label'].values))
