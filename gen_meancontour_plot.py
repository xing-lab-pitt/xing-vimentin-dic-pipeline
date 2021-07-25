import pickle
import copy
import numpy as np
from skimage import measure
from skimage.segmentation import find_boundaries
from skimage.morphology import opening, closing
from skimage.io import imread
from matplotlib import pyplot as plt
from os import listdir
import pandas as pd
from scipy.stats import kde
import seaborn as sns
import copy
import pickle
import scipy.ndimage as ndimage
import scipy.interpolate.fitpack as fitpack

import contour_class
import utility_tools

import image_warp
from contour_tool import df_find_contour_points, find_contour_points, generate_contours, align_contour_to, align_contours

main_path = '/mnt/data0/Ke/weikang_exp_process/2019-05-05_A549_vim/'
# with open (main_path+'/pca_contours', 'rb') as fp:
#     pca_contours = pickle.load(fp)


with open(main_path + '/output/mean_cell_contour', 'rb') as fp:
    mean_contour = pickle.load(fp)
plt.plot(mean_contour.points[:, 0], mean_contour.points[:, 1], '.')

plt.savefig('mean_contour0.png')
