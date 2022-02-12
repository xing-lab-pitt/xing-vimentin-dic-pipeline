# In[1]:

import numpy as np
from skimage import measure
from skimage.segmentation import find_boundaries
from skimage.morphology import opening,closing
from skimage.io import imread
from matplotlib import pyplot as plt
import os
from os import listdir
import pandas as pd
from scipy.stats import kde
import seaborn as sns
import copy
from math import exp,log
import pickle
import scipy.ndimage as ndimage
import scipy.interpolate.fitpack as fitpack
from sklearn import manifold,decomposition,random_projection,cluster,metrics,preprocessing,mixture,model_selection
from sklearn.neighbors import kneighbors_graph,BallTree
from hmmlearn import hmm
# from pymc import MCMC,flib,Model,MAP
# from ripser import Rips,ripser,plot_dgms
from persim import PersImage
import scipy.io as sio
# import kmapper as km
# from kmapper import jupyter
from mpl_toolkits.mplot3d import Axes3D
from cell_class import single_cell,fluor_single_cell
import contour_class
import utility_tools
import image_warp
from contour_tool import df_find_contour_points,find_contour_points,generate_contours,align_contour_to,align_contours
from scipy.signal import medfilt,wiener
from traj_class import single_cell_traj,fluor_single_cell_traj
# import bnpy
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,QuantileTransformer
from itertools import product
from sklearn.cluster import DBSCAN,KMeans
from sklearn.metrics import silhouette_score,davies_bouldin_score
import glob
# from pykalman import KalmanFilter
# from neupy import algorithms, utils
from tslearn.utils import to_time_series,to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans,KShape,GlobalAlignmentKernelKMeans,silhouette_score
from tslearn.metrics import dtw_path,dtw_subsequence_path,cdist_gak,cdist_dtw
from tslearn.barycenters import dtw_barycenter_averaging, softdtw_barycenter
from sklearn.manifold import MDS
# import sparse
from statsmodels.tsa.stattools import grangercausalitytests
from scipy import signal,stats
from statsmodels import robust
from matplotlib.patches import Circle, Wedge, Rectangle
from traj_scale import sp_traj_scaling,ssp_traj_scaling

# In[2]

main_path='C:/Users/14432/OneDrive/Research/Projects/Nikon_A549/data/A549_vim_rfp_tgfb_livecell/'
output_path=main_path+'out/weik_08-02-21_altered_param/'
result_path=output_path+'result/'
sct_path=output_path+'single_cell_traj/'
#sct_list=sorted(listdir(sct_path))
sct_list=sorted(glob.glob(sct_path+'[!indirect_]*'))

with open (main_path+'morph_pc_scaler','rb') as fp:
    morph_pc_scaler=pickle.load(fp)
with open (main_path+'vim_pc_scaler','rb') as fp:
    vim_pc_scaler=pickle.load(fp)

all_morph_traj = np.array([])
all_hara_traj = np.array([])
all_scale_morph_traj = np.array([])
all_scale_hara_traj = np.array([])
all_ds_morph_traj = np.array([])
all_ds_hara_traj = np.array([])

all_reaction_traj=[]
for i in range(len(sct_list)):
    with open (sct_list[i], 'rb') as fp:
        sct = pickle.load(fp)
    if hasattr(sct,'traj_scale_cord') and hasattr(sct,'traj_vimentin_scale_haralick_pca_cord'):
        if sct.traj_scale_cord is not None:
            
            mask=sct.traj_vimentin_feature_values[0]!=0
            traj_t=sct.traj_seri[mask][:,0]
            
            morph_traj=sct.traj_cord[mask]
            hara_traj=sct.traj_vimentin_haralick_pca_cord[mask]
            scale_morph_traj=sct.traj_scale_cord[mask]
            scale_hara_traj=sct.traj_vimentin_scale_haralick_pca_cord[mask]
            ds_morph_traj=(sct.traj_scale_cord[mask]-morph_pc_scaler.mean_)/np.sqrt(morph_pc_scaler.var_)
            ds_hara_traj=(sct.traj_vimentin_scale_haralick_pca_cord[mask]-vim_pc_scaler.mean_)/np.sqrt(vim_pc_scaler.var_)
            
            if len(all_morph_traj) > 0:
                all_morph_traj = np.vstack((all_morph_traj,morph_traj))
                all_hara_traj = np.vstack((all_hara_traj,hara_traj))
                all_scale_morph_traj = np.vstack((all_scale_morph_traj,scale_morph_traj))
                all_scale_hara_traj = np.vstack((all_scale_hara_traj,scale_hara_traj))
                all_ds_morph_traj = np.vstack((all_ds_morph_traj,ds_morph_traj))
                all_ds_hara_traj = np.vstack((all_ds_hara_traj,ds_hara_traj))
            else:
                all_morph_traj = morph_traj
                all_hara_traj = hara_traj
                all_scale_morph_traj = scale_morph_traj
                all_scale_hara_traj = scale_hara_traj
                all_ds_morph_traj = ds_morph_traj
                all_ds_hara_traj = ds_hara_traj

dot_color=np.arange(all_morph_traj[:].shape[0])
cm=plt.cm.get_cmap('jet')
plt.scatter(all_morph_traj[:,0],all_hara_traj[:,0],s=0.3,c=dot_color,cmap=cm)
plt.xlim(-300, 800) 
plt.ylim(-10, 15) 
plt.xlabel('morph PC1')
plt.ylabel('hara PC1')
plt.show()

dot_color=np.arange(all_scale_morph_traj[:].shape[0])
cm=plt.cm.get_cmap('jet')
plt.scatter(all_scale_morph_traj[:,0],all_scale_hara_traj[:,0],s=0.3,c=dot_color,cmap=cm)
plt.xlim(-6, 7) 
plt.ylim(-10, 13) 
plt.xlabel('scale morph PC1')
plt.ylabel('scale hara PC1')
plt.show()

dot_color=np.arange(all_scale_morph_traj[:].shape[0])
cm=plt.cm.get_cmap('jet')
plt.scatter(all_ds_morph_traj[:,0],all_ds_hara_traj[:,0],s=0.3,c=dot_color,cmap=cm)
plt.xlim(-6, 7) 
plt.ylim(-10, 13) 
plt.xlabel('ds morph PC1')
plt.ylabel('ds hara PC1')
plt.show()

