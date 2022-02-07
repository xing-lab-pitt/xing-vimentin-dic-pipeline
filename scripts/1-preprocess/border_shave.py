# In[0]: import
import sys
sys.path.insert(1, 'C:/Users/14432/OneDrive/Research/Projects/a549_pcna/scripts/memes/')
import warnings
warnings.filterwarnings("ignore")

from skimage.io import imread
from skimage.measure import regionprops,label
from skimage.segmentation import flood

from matplotlib import pyplot as plt
import numpy as np
import os
import glob
from PIL import Image
from pilutil import toimage
from scipy.ndimage import binary_fill_holes as fill_holes

# In[1]: initialize
dat_dir = 'C:/Users/14432/OneDrive/Research/Projects/a549_pcna/data/'
train_path = dat_dir + 'train/reg/seg/'
img_path = train_path + 'a549_vim_rfp_pcna_gfp_g418_control_med_121321/'
img_files = sorted(glob.glob(img_path + '*XY5_tile2_cr1*'))

crop = imread(img_files[4])
bib = imread(img_files[0])
plt.imshow(crop)
# plt.imshow(bib)
# In[2]: main
for file in img_files:

    filename = os.path.basename(file)
    img = imread(file)
    
    img_shaved = img[-420:,:]
    img_shaved = toimage(img_shaved)
    img_shaved.save(file)