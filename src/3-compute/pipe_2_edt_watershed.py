"""
we could use a  technique that could segment all the possible cells in the prediction of distance transform.  
when small cells are  recognized, other small pieces and noise would also be identified.
This will lead to over-segmentation.
Here we utilize the over-segmentation to segment the mitosis and apoptosis cells. 
Then, we extract the small cells that below size threshold. And use an identification CNN  to identify them,
only the apoptosis and mitosis cells are kept. 
After correctly segment these cells, we remove these cells from the prediction of distance transform. 
"""

import glob
import os
import pickle
import sys
from math import pi
from os import listdir

import cv2
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import legacy_utils.utils as utils
from legacy_utils.cnn_prep_data import keep_aspect_resize, obj_transform
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from legacy_utils.pilutil import toimage
from legacy_utils.resnet50 import res_model
from scipy import ndimage
from skimage.color import label2rgb
from skimage.exposure import equalize_adapthist
from skimage.feature import peak_local_max
from skimage.io import imread
from skimage.measure import label, regionprops
from skimage.morphology import (dilation, disk, h_maxima, local_maxima,
                                remove_small_objects)
from skimage.segmentation import clear_border, watershed
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

size_thres1 = 10000
size_thres2 = 12000
roundness_thres = 0.8
asp_rat_thres = 0.10

icnn_am_weights = sys.argv[3]
obj_h = int(sys.argv[6])
obj_w = int(sys.argv[6])
input_shape = (obj_h, obj_w, 1)
nb_class = 3
icnn_am = res_model(input_shape, nb_class)
icnn_am.load_weights(icnn_am_weights)

icnn_seg_weights = sys.argv[4]
obj_h = int(sys.argv[6])
obj_w = int(sys.argv[6])
input_shape = (obj_h, obj_w, 1)
nb_class = 3
icnn_seg = res_model(input_shape, nb_class)
icnn_seg.load_weights(icnn_seg_weights)


def color_num(labels):
    label_rgb = label2rgb(labels, bg_label=0)
    img_rgb = toimage(label_rgb)
    base = img_rgb.convert("RGBA")
    # make a blank image for the text, initialized to transparent text color
    txt = Image.new("RGBA", base.size, (255, 255, 255, 0))
    # get a font
    fnt = ImageFont.truetype("arial.ttf", 40)
    # get a drawing context
    d = ImageDraw.Draw(txt)
    for region in regionprops(labels):
        cx = int(region.centroid[1])
        cy = int(region.centroid[0])
        d.text((cx, cy), str(labels[cy][cx]), font=fnt, fill=(255, 255, 255, 255))
    out = Image.alpha_composite(base, txt)
    return out


# this is faster than the one from lineage mapper,only with matrix calculation
def compute_overlap_matrix(seg1, seg2):

    nb_cell_1 = np.amax(seg1)
    nb_cell_2 = np.amax(seg2)

    seg_overlap = np.zeros((nb_cell_1, nb_cell_2))
    for obj_idx1 in range(nb_cell_1):
        obj_num1 = obj_idx1 + 1
        sc_img = seg1 == obj_num1
        ol_judge = np.logical_and(sc_img, seg2)
        ol_value = np.multiply(ol_judge, seg2)
        ol_obj2 = np.unique(ol_value).tolist()
        # ol_obj2=ol_obj2[ol_obj2!=0]
        ol_obj2.remove(0)
        if len(ol_obj2) > 0:
            for obj_num2 in ol_obj2:
                ol_area = np.sum(ol_value == obj_num2)
                obj_idx2 = obj_num2 - 1
                seg_overlap[obj_idx1][obj_idx2] = ol_area

    return seg_overlap


# In[30]:


# -------calculate the cell fusion -----------------------
def cal_cell_fusion(frame_overlap):
    nb_cell_1 = frame_overlap.shape[0]
    nb_cell_2 = frame_overlap.shape[1]

    prefuse_group = (
        []
    )  # each element is a list include all prefuse cells in a fuse event, corresponding to postfuse_cells
    postfuse_cells = []  # include: img_num,obj_num
    frame_fusion = np.zeros(frame_overlap.shape)
    for source_o_n in range(1, nb_cell_1 + 1):
        # find target whose max_overlap mother is source
        ol_target = frame_overlap[source_o_n - 1, :]
        if np.all(ol_target == 0):  # if source obj have no overlap target
            target_o_n = 0
        else:
            target_o_n = (
                np.argmax(frame_overlap, axis=1)[source_o_n - 1] + 1
            )  # axis=1,maximum of each row,return column index

        if target_o_n > 0:
            frame_fusion[source_o_n - 1, target_o_n - 1] = 1

        # Compute the sum vector S which is the sum of all the columns of frame_fusion matrix. The fusion target region
        # will have at least 2 cells tracked to it => S>1
    S = np.sum(frame_fusion, axis=0)
    frame_fusion[:, S == 1] = 0
    # Update the sum vector
    S = np.sum(frame_fusion, axis=0)

    for i in range(len(np.where(S >= 2)[0])):
        f_group = []
        postfuse_cells.append([np.where(S >= 2)[0][i] + 1])  # num of prefuse cells:S[np.where(S >= 2)[0][i]]
        frame_fusion_i = frame_fusion[:, np.where(S >= 2)[0][i]]

        for r in range(len(np.where(frame_fusion_i == 1)[0])):
            # fuse_pairs.append([img_num_1,np.where(frame_fusion_i==1)[0][r]+1,img_num_2,np.where(S >= 2)[0][i]+1])
            f_group.append(np.where(frame_fusion_i == 1)[0][r] + 1)
        prefuse_group.append(f_group)
    return postfuse_cells, prefuse_group


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


def generate_single_cell_img_edt(img, seg, obj_h, obj_w, obj_num):

    # single_obj_img=morphology.binary_dilation(seg_img==obj_num,morphology.diamond(16))
    single_obj_img = seg == obj_num
    single_obj_img = label(single_obj_img)
    rps = regionprops(single_obj_img)
    candi_r = [r for r in rps if r.label == 1][0]
    candi_box = candi_r.bbox
    single_cell_img = single_obj_img * img
    crop_img = single_cell_img[candi_box[0] : candi_box[2], candi_box[1] : candi_box[3]]

    inds = ndimage.distance_transform_edt(crop_img == 0, return_distances=False, return_indices=True)
    crop_img = crop_img[tuple(inds)]
    print(crop_img.shape)
    print(obj_w, obj_h)
    crop_img = obj_transform(keep_aspect_resize(crop_img, obj_h, obj_w), random_eah=False)
    crop_img = np.expand_dims(crop_img, axis=0)
    return crop_img


# In[33]:


# crop single segmented img by bounding box from the original image
def generate_single_cell_img_env(img, rps, obj_h, obj_w, obj_num):
    candi_r = [r for r in rps if r.label == obj_num][0]
    candi_box = candi_r.bbox
    crop_img = img[candi_box[0] : candi_box[2], candi_box[1] : candi_box[3]]

    crop_img = obj_transform(keep_aspect_resize(crop_img, obj_h, obj_w), random_eah=False)
    crop_img = np.expand_dims(crop_img, axis=0)

    # cell_img=toimage(crop_cell_img,high=np.max(crop_cell_img),low=np.min(crop_cell_img),mode='I')
    # cell_img.save(crop_path+'/i'+str(img_num)+'_o'+str(obj_num)+'.tif')
    return crop_img


def judge_fuse_type(low_h_markers, high_h_markers, fc_cell, fp_group, fc_prob, fp_group_prob):

    type_list = ["single", "fragment", "multi"]

    fc_prob[1] = 0
    fc_type = type_list[np.argmax(fc_prob)]

    nb_fp = len(fp_group)

    fp_group_type = []

    for fp_prob in fp_group_prob:
        fp_prob[2] = 0
        fp_group_type.append(type_list[np.argmax(np.array(fp_prob))])

    #     print(fc_type,fp_group_type)
    if fp_group_type.count("single") >= 1:
        for i in range(nb_fp):
            if fp_group_type[i] == "fragment":
                low_h_markers[np.where(low_h_markers == fp_group[i])] = 0
    else:
        for i in range(nb_fp):
            if not np.any(high_h_markers[np.where(low_h_markers == fp_group[i])]):
                low_h_markers[np.where(low_h_markers == fp_group[i])] = 0
    #     low_h_markers=dilation(low_h_markers,disk(10))
    #     plt.imshow(low_h_markers)
    #     plt.show()

    #             print('high_h_num',fc_cell,np.where(high_h_markers==fc_cell))
    #             print('low_h_num',fp_group[i],np.where(low_h_markers==fp_group[i]))
    #             print('high_h_low_h',high_h_markers[np.where(low_h_markers==fp_group[i])])

    #     if len(fp_group)==2:
    #         if fp_group_type.count('single')>=1:
    #             for i in range(nb_fp):
    #                 if fp_group_type[i]=='fragment':
    #                     low_h_markers[np.where(low_h_markers==fp_group[i])]=0

    #         if fc_type=='single' and fp_group_type.count('single')<=1:
    #             for i in range(nb_fp):

    #         if fc_type=='single' and fp_group_type.count('single')>1:
    #             for i in range(nb_fp):

    #         if fc_type=='multi' and fp_group_type.count('single')>1:
    #             for i in range(nb_fp):

    #         if fc_type=='multi' and fp_group_type.count('single')<=1:
    #     if len(fp_group)>2:
    #         if fp_group_type.count('single')>=1:
    #             for i in range(nb_fp):
    #                 if fp_group_type[i]=='fragment':
    #                     low_h_markers[np.where(low_h_markers==fp_group[i])]=0
    #         else:
    #             for i in range(nb_fp):
    #                 if high_h_markers[np.where(low_h_markers==fp_group[i])]==0:
    #                     low_h_markers[np.where(low_h_markers==fp_group[i])]=0

    return low_h_markers


def simple_edt_watershed(img_path, out_path, chan_label, small_obj_thres):

    """
    for a given folder:
        - create a corresponding seg/ folder
        - create a corresponding rgb_num/ folder
        - save img_seg into the seg/ folder
        - save rgb_num into the rgb_num/ folder

    :return:
    """

    img_path = utils.correct_folder_str(img_path)
    img_list = sorted(glob.glob(img_path + "*" + chan_label + "*"))

    out_path = utils.correct_folder_str(out_path)
    reg_path = utils.correct_folder_str(out_path + "edt")
    reg_list = sorted(
        os.listdir(reg_path)
    )  # this is the edt folder. # edt image path master_output_folder+#/reg/reg_xximgstring
    seg_path = out_path + "seg/"
    utils.create_folder(seg_path)  # the folder will be re-created every time.

    rgb_num_path = out_path + "rgb_num/"
    utils.create_folder(rgb_num_path)  # the folder will be re-created every time.

    # am_record = pd.DataFrame(columns=["ImageNumber", "ObjectNumber", "am_flag"])
    # all_am_count = 0

    for i in range(len(img_list)):

        img = imread(img_list[i])
        # print(img.shape)
        # print(reg_path)
        # print(reg_list[0])
        reg = imread(reg_path + reg_list[i])
        img_name = os.path.basename(img_list[i])[0 : len(img_list[i]) - 4]

        reg_flat = reg[reg != 0.0].reshape(-1)

        # TODO refactor: threshold percentage as hyperparam
        # print('Reg_flat shape', reg_flat.shape)
        THRESHOLDS = np.quantile(reg_flat, [0.25, 0.97])

        mask_h = 0.02
        low_h = THRESHOLDS[0]
        high_h = THRESHOLDS[1]
        print("low h: " + str(low_h))
        print("high h: " + str(high_h))

        low_h_seg, low_h_markers = hmax_watershed(reg, low_h, asp_rat_thres, mask_thres=mask_h)

        high_h_seg, high_h_markers = hmax_watershed(reg, high_h, asp_rat_thres, mask_thres=mask_h)

#         seg_overlap = compute_overlap_matrix(low_h_seg, high_h_seg)
#         fuse_cells, fuse_group = cal_cell_fusion(seg_overlap)

#         for m in range(len(fuse_cells)):

#             fc_obj = generate_single_cell_img_edt(img, high_h_seg, obj_h, obj_w, fuse_cells[m])
#             fc_prob = icnn_seg.predict(fc_obj)[0]
#             fp_group_prob = []

#             for n in range(len(fuse_group[m])):

#                 fp_obj = generate_single_cell_img_edt(img, low_h_seg, obj_h, obj_w, fuse_group[m][n])
#                 fp_prob = icnn_seg.predict(fp_obj)[0]
#                 fp_group_prob.append(fp_prob)

#             low_h_markers = judge_fuse_type(
#                 low_h_markers, high_h_markers, fuse_cells[m], fuse_group[m], fc_prob, fp_group_prob
#             )

#         labels = watershed(-reg, low_h_markers, mask=reg > mask_h)
#         labels, low_h_markers = remove_thin_objects(labels, low_h_markers, asp_rat_thres)
#         labels = remove_small_objects(labels, small_obj_thres)
#         labels = clear_border(labels)
#         labels = label(labels, connectivity=2)

#         rps = regionprops(labels)
#         candi_labels = [
#             r.label
#             for r in rps
#             if r.area < size_thres1 or (4 * pi * r.area / r.perimeter ** 2 > roundness_thres and r.area < size_thres2)
#         ]
# # MITOSIS STUFF
#         for candi_label in candi_labels:

#             candi_obj = generate_single_cell_img_env(img, rps, obj_h, obj_w, candi_label)
#             output = icnn_am.predict(candi_obj)
#             am_flag = np.argmax(output)

#             if am_flag > 0:
#                 all_am_count += 1
#                 am_record.loc[all_am_count] = [i + 1, candi_label, am_flag]

        # should use np.uint32,could be save correctly

        low_h_seg = remove_small_objects(low_h_seg, small_obj_thres)
        low_h_seg = clear_border(low_h_seg)
        low_h_seg = label(low_h_seg, connectivity=2)
        img_seg = Image.fromarray(low_h_seg.astype(np.uint32), "I")
        img_seg.save(seg_path + "seg_" + img_name + "h_val" + high_h + ".png")

        high_h_seg = remove_small_objects(high_h_seg, small_obj_thres)
        high_h_seg = clear_border(high_h_seg)
        high_h_seg = label(high_h_seg, connectivity=2)
        img_seg = Image.fromarray(high_h_seg.astype(np.uint32), "I")
        img_seg.save(seg_path + "seg_" + img_name + "h_val" + high_h + ".png")

        # rgb_num = color_num(labels)
        # rgb_num.save(rgb_num_path + "rgb_" + img_name + ".png")

    # am_record.to_csv(out_path + "am_record.csv", index=False, encoding="utf-8")


if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    img_path = sys.argv[1]
    output_path = sys.argv[2]
    dic_channel_label = sys.argv[5]
    small_obj_thres = int(sys.argv[7])
    simple_edt_watershed(img_path, output_path, dic_channel_label, small_obj_thres)
