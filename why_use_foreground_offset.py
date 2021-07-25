#!/usr/bin/env python
# coding: utf-8

# In[2]:


from skimage.io import imread
import numpy as np
from matplotlib import pyplot as plt
img = imread(
    '/media/zoro/easystore/xing-lab-1/experiment_data/2019-03-10_HK2_fucci/cdt1/2/hk2_fuccixy02t552c1.tif')
seg = imread(
    '/home/zoro/Desktop/experiment_data/2019-03-10_HK2_fucci/cdt1_output/2/seg/seg_hk2_fuccixy02t552c1.png')
fg_mask = seg > 0
bg_mask = seg == 0
bg_offset = np.load(
    '/home/zoro/Desktop/experiment_data/2019-03-10_HK2_fucci/background_offset_cdt1.npy')
fg_offset = np.load(
    '/home/zoro/Desktop/experiment_data/2019-03-10_HK2_fucci/foreground_offset_cdt1.npy')
plt.figure(figsize=(8, 8))
plt.title('original_image')
plt.imshow(img)
plt.show()

plt.figure(figsize=(8, 8))
plt.title('background correction')
plt.imshow(img + bg_offset)
plt.show()

plt.figure(figsize=(8, 8))
plt.title('foreground correction')
plt.imshow(img + fg_offset)
plt.show()

plt.figure(figsize=(8, 8))
plt.title('foreground and background correction')
plt.imshow(img + fg_mask * fg_offset + bg_mask * bg_offset)
plt.show()


# In[ ]:
