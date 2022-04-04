# In[0]: import
import sys

sys.path.insert(1, "C:/Users/14432/OneDrive/Research/Projects/a549_pcna/scripts/memes/")

import numpy as np
from skimage.segmentation import watershed, clear_border
from skimage.io import imread
from matplotlib import pyplot as plt
import scipy.ndimage as ndi
import pickle
import os
from os import listdir
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects, local_maxima, h_maxima, disk, dilation
from skimage.measure import regionprops, label
from skimage.color import label2rgb
from PIL import Image, ImageDraw, ImageFont
from resnet50 import res_model
from math import pi
import cv2
import glob
from skimage.exposure import equalize_adapthist
from pilutil import toimage
from cnn_prep_data import keep_aspect_resize, obj_transform

import pandas as pd
from scipy import ndimage
import sys
import hj_util
from skimage.segmentation import clear_border

from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

# In[1]: define
size_thres1 = 10000
size_thres2 = 12000
small_obj_thres = 1500
roundness_thres = 0.8
asp_rat_thres = 0.1
mask_h = 0.02


def color_num(labels):
    label_rgb = label2rgb(labels, bg_label=0)
    img_rgb = toimage(label_rgb)
    base = img_rgb.convert("RGBA")
    # make a blank image for the text, initialized to transparent text color
    txt = Image.new("RGBA", base.size, (255, 255, 255, 0))
    # get a font
    fnt = ImageFont.truetype(
        "/net/capricorn/home/xing/huijing/Segmentation/scripts/vimentin_DIC_segmentation_pipeline/hj_modify_pipe/arial.ttf",
        40,
    )
    # get a drawing context
    d = ImageDraw.Draw(txt)
    for region in regionprops(labels):
        cx = int(region.centroid[1])
        cy = int(region.centroid[0])
        d.text((cx, cy), str(labels[cy][cx]), font=fnt, fill=(255, 255, 255, 255))
    out = Image.alpha_composite(base, txt)
    return out


def remove_thin_objects(labels, local_hmax_label, asp_rat_thres):

    rps = regionprops(labels)
    for r in rps:
        candi_r = [r][0]
        candi_box = candi_r.bbox

        crop_img = labels[candi_box[0] : candi_box[2], candi_box[1] : candi_box[3]]
        img_h = crop_img.shape[0]
        img_w = crop_img.shape[1]

        asp_rat = min(img_h / img_w, img_w / img_h)

        if asp_rat < asp_rat_thres:

            labels[labels == r.label] = 0
            local_hmax_label[local_hmax_label == r.label] = 0

    labels = label(labels, connectivity=2)

    return labels, local_hmax_label


# ---do not remove small objects before calculate overlap
def hmax_watershed(img, h_thres, asp_rat_thres, mask_thres=0):

    local_hmax = h_maxima(img, h_thres)
    local_hmax_label = label(local_hmax, connectivity=1)

    labels = watershed(-img, local_hmax_label, mask=img > mask_thres)
    labels, local_hmax_label = remove_thin_objects(labels, local_hmax_label, asp_rat_thres)

    return labels, local_hmax_label


def quick_edt_watershed(img_path, out_path, chan_label):

    img_path = hj_util.folder_verify(img_path)
    img_list = sorted(glob.glob(img_path + "*" + chan_label + "*"))

    out_path = hj_util.folder_verify(out_path)
    reg_path = hj_util.folder_verify(out_path + "edt_test")
    reg_list = sorted(os.listdir(reg_path))  # this is the edt folder
    seg_path = out_path + "seg_test/"
    hj_util.create_folder(seg_path)  # the folder will be re-created every time.

    rgb_num_path = out_path + "rgb_test/"
    hj_util.create_folder(rgb_num_path)  # the folder will be re-created every time.

    for i in range(len(img_list)):

        img = imread(img_list[i])
        # print(img.shape)
        # print(reg_path)
        # print(reg_list[0])
        reg = imread(reg_path + reg_list[i])
        img_name = os.path.basename(img_list[i])[0 : len(img_list[i]) - 4]

        reg_flat = reg[reg != 0.0].reshape(-1)
        thresholds = np.quantile(reg_flat, [0.35, 0.97])

        low_h = thresholds[0]
        high_h = thresholds[1]
        print("img " + img_list[i] + "seed thresholds: low h - " + str(low_h) + ", high h - " + str(high_h))

        low_h_seg, low_h_markers = hmax_watershed(reg, low_h, asp_rat_thres, mask_thres=mask_h)

        high_h_seg, high_h_markers = hmax_watershed(reg, high_h, asp_rat_thres, mask_thres=mask_h)

        labels = watershed(-reg, low_h_markers, mask=reg > mask_h)
        labels, low_h_markers = remove_thin_objects(labels, low_h_markers, asp_rat_thres)
        labels = remove_small_objects(labels, small_obj_thres)
        labels = clear_border(labels)
        labels = label(labels, connectivity=2)

        rps = regionprops(labels)
        candi_labels = [
            r.label
            for r in rps
            if r.area < size_thres1 or (4 * pi * r.area / r.perimeter ** 2 > roundness_thres and r.area < size_thres2)
        ]

        # should use np.uint32,could be save correctly
        img_seg = Image.fromarray(labels.astype(np.uint32), "I")
        img_seg.save(seg_path + "seg_" + img_name + ".png")

        rgb_num = color_num(labels)
        rgb_num.save(rgb_num_path + "rgb_" + img_name + ".png")


if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    main_path = "C:/Users/14432/OneDrive/Research/Projects/a549_pcna/data"
    img_path = main_path + "/ori/test_batch"
    output_path = main_path + "/out/01-13-22_72hr_no-treat/XY1_test"
    DIC_chan_label = "C1"
    quick_edt_watershed(img_path, output_path, DIC_chan_label)
