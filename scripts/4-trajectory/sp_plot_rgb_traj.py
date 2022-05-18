# In[0] import
import os
# from track_module import color_label_LAP,color_label_overlap
import sqlite3
from os import listdir

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from imageio import imread
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.distance import cosine, euclidean
from skimage.color import label2rgb
from skimage.measure import regionprops

# In[1] color_label_LAP function


def color_label_LAP(labels, df, img_num, path_output, img_name):

    label_rgb = label2rgb(labels, bg_label=0) * 255
    label_rgb_int = np.uint8(label_rgb)
    img_rgb = Image.fromarray(label_rgb_int)
    base = img_rgb.convert("RGBA")

    txt = Image.new("RGBA", base.size, (255, 255, 255, 0))
    fnt = ImageFont.truetype("arial.ttf", 25)
    d = ImageDraw.Draw(txt)
    all_idx = df[df["ImageNumber"] == img_num].index.tolist()

    obj_num = 0
    for region in regionprops(labels):

        cx = int(region.centroid[1] - 45)
        cy = int(region.centroid[0] - 25)
        idx = all_idx[obj_num]

        num_label = int(df.loc[idx]["Cell_TrackObjects_Label"])  # gets trajectory label number
        d.text(
            (cx, cy), str(obj_num + 1) + ";" + str(num_label), font=fnt, fill=(255, 255, 255, 255)
        )  # str(labels[cy][cx])
        obj_num += 1

    out = Image.alpha_composite(base, txt)
    out.save(path_output + "/" + img_name + ".png", "PNG")


# In[2] define vars

main_path = "C:/Users/14432/OneDrive/Research/Projects/Nikon_A549/data/A549_vim_rfp_tgfb_livecell_12hr/"
output_path = main_path + "out/weik_08-02-21_altered_param/"

posi_end = 21

# In[3] plot

for posi in range(1, posi_end + 1):

    if posi <= 9:
        posi_label = "XY0" + str(posi)
    else:
        posi_label = "XY" + str(posi)
    dir_path = output_path + posi_label + "/"

    seg_path = dir_path + "seg/"
    seg_img_list = sorted(listdir(seg_path))

    traj_label_path = dir_path + "rbg_traj_label/"
    if not os.path.exists(traj_label_path):
        os.makedirs(traj_label_path)

    df = pd.read_csv(dir_path + "/Per_Object_relink.csv")
    t_span = max(df["ImageNumber"])

    for i in range(len(seg_img_list) - 25, len(seg_img_list)):
        img_num = i + 1
        seg = imread(seg_path + "/" + seg_img_list[i])
        img_name = seg_img_list[i][4:-8]
        color_label_LAP(seg, df, img_num, traj_label_path, img_name)
