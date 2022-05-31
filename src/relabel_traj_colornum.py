#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
# from track_module import color_label_LAP,color_label_overlap
import sqlite3
from os import listdir

import numpy as np
import pandas as pd
import scipy
from imageio import imread
from PIL import Image, ImageDraw, ImageFont
from pilutil import toimage
from scipy.spatial.distance import cosine, euclidean
from skimage.color import label2rgb
from skimage.measure import regionprops

# In[2]:


def color_label_LAP(labels, df, img_num, path_output, img_name):
    label_rgb = label2rgb(labels, bg_label=0)
    # print(label_rgb.shape)
    img_rgb = scipy.misc.toimage(label_rgb)
    base = img_rgb.convert("RGBA")
    # img_rgb=Image.fromarray(label_rgb,'RGB')
    # base = img_rgb.convert('RGBA')
    # make a blank image for the text, initialized to transparent text color
    txt = Image.new("RGBA", base.size, (255, 255, 255, 0))
    # get a font
    fnt = ImageFont.truetype("arial.ttf", 60)
    # get a drawing context
    d = ImageDraw.Draw(txt)
    # print(img_num)
    all_idx = df[df["ImageNumber"] == img_num].index.tolist()
    # print(all_idx)
    obj_num = 0
    for region in regionprops(labels):

        cx = int(region.centroid[1])
        cy = int(region.centroid[0])
        idx = all_idx[obj_num]

        num_label = int(df.loc[idx]["Cell_TrackObjects_Label"])
        # print(num_label)
        d.text((cx, cy), str(num_label), font=fnt, fill=(255, 255, 255, 255))  # str(labels[cy][cx])
        obj_num += 1
    out = Image.alpha_composite(base, txt)
    # out.show()
    out.save(path_output + "/" + img_name + ".png", "PNG")


# In[4]:


main_path = "/home/zoro/Desktop/experiment_data/2019-03-22_a549_tgf4ng_2d/"
input_path = main_path + "img/"
output_path = main_path + "output/"

posi_end = 2


# relation_df=pd.read_csv(dir_path + '/Per_Relationships.csv')
# conn = sqlite3.connect(dir_path + '/cell_track.db')
# df=pd.read_sql_query('SELECT * FROM Per_Object',conn)

for posi in range(2, posi_end + 1):

    dir_path = output_path + str(posi) + "/"

    seg_path = dir_path + "seg/"
    seg_img_list = sorted(listdir(seg_path))

    traj_label_path = dir_path + "traj_label/"
    if not os.path.exists(traj_label_path):
        os.makedirs(traj_label_path)

    df = pd.read_csv(dir_path + "/Per_Object_relink_" + str(posi) + ".csv")
    #     df=pd.read_csv(dir_path + '/Per_Object_mitosis.csv')
    #     df=pd.read_csv(dir_path + '/Per_Object_modify.csv')
    t_span = max(df["ImageNumber"])

    for i in range(len(seg_img_list)):
        img_num = i + 1
        seg = imread(seg_path + "/" + seg_img_list[i])
        img_name = seg_img_list[i][4:-4]
        #         print(img_name)
        color_label_LAP(seg, df, img_num, traj_label_path, img_name)


# In[ ]:
