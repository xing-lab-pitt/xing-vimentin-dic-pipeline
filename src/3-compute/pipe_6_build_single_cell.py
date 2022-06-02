#!/usr/bin/env python
# coding: utf-8

# In[1]:


import copy
import os
import pickle
import sys
from os import listdir

import legacy_utils.contour_class as contour_class
import legacy_utils.image_warp as image_warp
import numpy as np
import pandas as pd
# from . import utils
import legacy_utils.utils as utils
import scipy.interpolate.fitpack as fitpack
import scipy.ndimage as ndimage
import seaborn as sns

from legacy_utils.cell_class import single_cell
from legacy_utils.contour_tool import (align_contour_to, align_contours,
                          df_find_contour_points, find_contour_points,
                          generate_contours)
from matplotlib import pyplot as plt
from scipy.stats import kde
from skimage import measure
from skimage.io import imread
from skimage.morphology import closing, opening
from skimage.segmentation import find_boundaries


# -------------initiate all cells--------------------
def build_single_cell(output_path):
    output_path = utils.correct_folder_str(output_path)
    cells_path = utils.create_folder(output_path + "cells")
    dir_path = output_path

    df = pd.read_csv(dir_path + "Per_Object_relink" + ".csv")
    am_record = pd.read_csv(dir_path + "am_record" + ".csv")
    traj_df = pd.read_csv(dir_path + "traj_object_num" + ".csv")
    # TODO: add tritc#/fitc# nucleus info
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


if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    output_path = sys.argv[1]
    build_single_cell(output_path)
