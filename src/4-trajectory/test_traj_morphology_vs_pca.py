# In[1]:

import copy
import glob
import os
import pickle
from itertools import product
from math import exp, log
from os import listdir

import contour_class
import image_warp
import numpy as np
import pandas as pd
import scipy.interpolate.fitpack as fitpack
import scipy.io as sio
import scipy.ndimage as ndimage
import seaborn as sns
import utils
from cell_class import fluor_single_cell, single_cell
from contour_tool import (align_contour_to, align_contours,
                          df_find_contour_points, find_contour_points,
                          generate_contours)
from hmmlearn import hmm
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Rectangle, Wedge
# import kmapper as km
# from kmapper import jupyter
from mpl_toolkits.mplot3d import Axes3D
# from pymc import MCMC,flib,Model,MAP
# from ripser import Rips,ripser,plot_dgms
from persim import PersImage
from scipy import signal, stats
from scipy.signal import medfilt, wiener
from scipy.stats import kde
from skimage import measure
from skimage.io import imread
from skimage.morphology import closing, opening
from skimage.segmentation import find_boundaries
from sklearn import (cluster, decomposition, manifold, metrics, mixture,
                     model_selection, preprocessing, random_projection)
from sklearn.cluster import DBSCAN, KMeans
from sklearn.manifold import MDS
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.neighbors import BallTree, kneighbors_graph
# import bnpy
from sklearn.preprocessing import (MinMaxScaler, QuantileTransformer,
                                   RobustScaler, StandardScaler)
from statsmodels import robust
# import sparse
from statsmodels.tsa.stattools import grangercausalitytests
from traj_class import fluor_single_cell_traj, single_cell_traj
from traj_scale import sp_traj_scaling, ssp_traj_scaling
from tslearn.barycenters import dtw_barycenter_averaging, softdtw_barycenter
from tslearn.clustering import (GlobalAlignmentKernelKMeans, KShape,
                                TimeSeriesKMeans, silhouette_score)
from tslearn.metrics import (cdist_dtw, cdist_gak, dtw_path,
                             dtw_subsequence_path)
# from pykalman import KalmanFilter
# from neupy import algorithms, utils
from tslearn.utils import to_time_series, to_time_series_dataset

# In[2]

main_path = "C:/Users/14432/OneDrive/Research/Projects/Nikon_A549/data/A549_vim_rfp_tgfb_livecell/"
output_path = main_path + "out/weik_08-02-21_altered_param/"
result_path = output_path + "result/"
sct_path = output_path + "single_cell_traj/"
# sct_list=sorted(listdir(sct_path))
sct_list = sorted(glob.glob(sct_path + "[!indirect_]*"))

with open(main_path + "morph_pc_scaler", "rb") as fp:
    morph_pc_scaler = pickle.load(fp)
with open(main_path + "vim_pc_scaler", "rb") as fp:
    vim_pc_scaler = pickle.load(fp)

all_morph_traj = np.array([])
all_hara_traj = np.array([])
all_scale_morph_traj = np.array([])
all_scale_hara_traj = np.array([])
all_ds_morph_traj = np.array([])
all_ds_hara_traj = np.array([])

all_reaction_traj = []
for i in range(len(sct_list)):
    with open(sct_list[i], "rb") as fp:
        sct = pickle.load(fp)
    if hasattr(sct, "traj_scale_cord") and hasattr(sct, "traj_vimentin_scale_haralick_pca_cord"):
        if sct.traj_scale_cord is not None:

            mask = sct.traj_vimentin_feature_values[0] != 0
            traj_t = sct.traj_seri[mask][:, 0]

            morph_traj = sct.traj_cord[mask]
            hara_traj = sct.traj_vimentin_haralick_pca_cord[mask]
            scale_morph_traj = sct.traj_scale_cord[mask]
            scale_hara_traj = sct.traj_vimentin_scale_haralick_pca_cord[mask]
            ds_morph_traj = (sct.traj_scale_cord[mask] - morph_pc_scaler.mean_) / np.sqrt(morph_pc_scaler.var_)
            ds_hara_traj = (sct.traj_vimentin_scale_haralick_pca_cord[mask] - vim_pc_scaler.mean_) / np.sqrt(
                vim_pc_scaler.var_
            )

            if len(all_morph_traj) > 0:
                all_morph_traj = np.vstack((all_morph_traj, morph_traj))
                all_hara_traj = np.vstack((all_hara_traj, hara_traj))
                all_scale_morph_traj = np.vstack((all_scale_morph_traj, scale_morph_traj))
                all_scale_hara_traj = np.vstack((all_scale_hara_traj, scale_hara_traj))
                all_ds_morph_traj = np.vstack((all_ds_morph_traj, ds_morph_traj))
                all_ds_hara_traj = np.vstack((all_ds_hara_traj, ds_hara_traj))
            else:
                all_morph_traj = morph_traj
                all_hara_traj = hara_traj
                all_scale_morph_traj = scale_morph_traj
                all_scale_hara_traj = scale_hara_traj
                all_ds_morph_traj = ds_morph_traj
                all_ds_hara_traj = ds_hara_traj

dot_color = np.arange(all_morph_traj[:].shape[0])
cm = plt.cm.get_cmap("jet")
plt.scatter(all_morph_traj[:, 0], all_hara_traj[:, 0], s=0.3, c=dot_color, cmap=cm)
plt.xlim(-300, 800)
plt.ylim(-10, 15)
plt.xlabel("morph PC1")
plt.ylabel("hara PC1")
plt.show()

dot_color = np.arange(all_scale_morph_traj[:].shape[0])
cm = plt.cm.get_cmap("jet")
plt.scatter(all_scale_morph_traj[:, 0], all_scale_hara_traj[:, 0], s=0.3, c=dot_color, cmap=cm)
plt.xlim(-6, 7)
plt.ylim(-10, 13)
plt.xlabel("scale morph PC1")
plt.ylabel("scale hara PC1")
plt.show()

dot_color = np.arange(all_scale_morph_traj[:].shape[0])
cm = plt.cm.get_cmap("jet")
plt.scatter(all_ds_morph_traj[:, 0], all_ds_hara_traj[:, 0], s=0.3, c=dot_color, cmap=cm)
plt.xlim(-6, 7)
plt.ylim(-10, 13)
plt.xlabel("ds morph PC1")
plt.ylabel("ds hara PC1")
plt.show()
