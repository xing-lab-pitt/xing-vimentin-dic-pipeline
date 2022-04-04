import numpy as np
from skimage import measure
from skimage.segmentation import find_boundaries
from skimage.morphology import opening, closing
from skimage.io import imread
from matplotlib import pyplot as plt
import os
from os import listdir
import pandas as pd
from scipy.stats import kde
import seaborn as sns
import copy
from math import exp, log
import pickle
import scipy.ndimage as ndimage
import scipy.interpolate.fitpack as fitpack
from sklearn import (
    manifold,
    decomposition,
    random_projection,
    cluster,
    metrics,
    preprocessing,
    mixture,
    model_selection,
)
from sklearn.neighbors import kneighbors_graph, BallTree
from hmmlearn import hmm
from persim import PersImage
import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D
from cell_class import single_cell, fluor_single_cell
import contour_class
import utility_tools
import image_warp
from contour_tool import (
    df_find_contour_points,
    find_contour_points,
    generate_contours,
    align_contour_to,
    align_contours,
)
from scipy.signal import medfilt
from traj_class import single_cell_traj, fluor_single_cell_traj
from skimage.measure import label, regionprops
from matplotlib import animation
from collections import OrderedDict


def generate_single_cell_img(img, seg, img_num, obj_num):
    # single_obj_mask=morphology.binary_dilation(seg==obj_num,morphology.disk(6))
    single_obj_mask = seg == obj_num
    single_obj_mask = label(single_obj_mask)
    rps = regionprops(single_obj_mask)
    candi_r = [r for r in rps if r.label == 1][0]
    candi_box = candi_r.bbox
    single_cell_img = single_obj_mask * img

    crop_cell_img = single_cell_img[candi_box[0] : candi_box[2], candi_box[1] : candi_box[3]]
    crop_cell_img_env = img[candi_box[0] : candi_box[2], candi_box[1] : candi_box[3]]

    crop_single_obj_mask = single_obj_mask[candi_box[0] : candi_box[2], candi_box[1] : candi_box[3]]
    rps = regionprops(crop_single_obj_mask)
    candi_r = [r for r in rps if r.label == 1][0]
    center = candi_r.centroid

    return crop_cell_img, crop_cell_img_env


def find_mother(df, mitosis_df, traj_label):
    daughter_flag = 0
    if traj_label != -1:
        if (mitosis_df["sis1_traj_label"] == traj_label).any() or (mitosis_df["sis2_traj_label"] == traj_label).any():
            daughter_flag = 1
    return daughter_flag


def find_offspring(df, mitosis_df, family_tree, traj_label):
    mother_label = traj_label
    if mother_label != -1 and (mitosis_df["mother_traj_label"] == mother_label).any():
        family_tree[int(mother_label)] = []
        sis1_label = mitosis_df.loc[mitosis_df["mother_traj_label"] == mother_label, "sis1_traj_label"].values[0]
        sis2_label = mitosis_df.loc[mitosis_df["mother_traj_label"] == mother_label, "sis2_traj_label"].values[0]
        if sis1_label != -1:
            family_tree[int(mother_label)].append(int(sis1_label))
        if sis2_label != -1:
            family_tree[int(mother_label)].append(int(sis2_label))
        family_tree = find_offspring(df, mitosis_df, family_tree, sis1_label)
        family_tree = find_offspring(df, mitosis_df, family_tree, sis2_label)
        return family_tree
    else:
        return family_tree


def parse(node, tree):
    if node not in tree:
        yield [node]
    else:
        for next_node in tree[node]:
            for r in parse(next_node, tree):
                yield [node] + r


from scipy.signal import find_peaks


def find_abnormal_fluor(traj_fluor, traj_t, peak_h=5):
    mask = traj_fluor != 0
    #     inds=np.where(traj_fluor!=0)[0]
    non0_traj_t = traj_t[mask]
    non0_traj_fluor = traj_fluor[mask]
    mean_fluct = np.mean(abs(np.diff(non0_traj_fluor)))
    print(mean_fluct)

    ind1 = find_peaks(np.diff(non0_traj_fluor) / mean_fluct, height=peak_h)[0] + 1
    ind2 = (
        non0_traj_fluor.shape[0] - (find_peaks(np.diff(np.flip(non0_traj_fluor, 0)) / mean_fluct, height=peak_h)[0]) - 2
    )
    print(ind1, ind2)
    inds = np.unique(np.concatenate((ind1, ind2)))

    abn_t = non0_traj_t[inds]
    abn_inds = np.where(np.in1d(traj_t, abn_t))[0]  # find index of abn_t in traj_t
    print(abn_inds)
    return abn_inds


def generate_fluor_long_traj(df, cells, am_record, traj_df, traj, fluor_name, feature_list, fluor_feature_name):

    haralick_labels = [
        "Angular Second Moment",
        "Contrast",
        "Correlation",
        "Sum of Squares: Variance",
        "Inverse Difference Moment",
        "Sum Average",
        "Sum Variance",
        "Sum Entropy",
        "Entropy",
        "Difference Variance",
        "Difference Entropy",
        "Information Measure of Correlation 1",
        "Information Measure of Correlation 2",
        "Maximal Correlation Coefficient",
    ]

    traj_record = pd.DataFrame(traj_df.loc[:, "1" : str(time_span)])
    traj_record = traj_record.values
    traj_quan, traj_len = traj_record.shape[0], traj_record.shape[1]

    traj_xy = []
    traj_feature = []
    traj_contour = []
    traj_cord = []
    traj_seri = []
    traj_am_flag = []

    traj_fluor_feature_values = []
    traj_haralick = []
    traj_norm_haralick = []
    traj_fluor_pca_cord = []
    #     traj_norm_fluor_pca_cord=[]
    for img_num in range(1, traj_len + 1):
        obj_num = traj[img_num - 1]
        if obj_num != -1:
            ind = df.loc[(df["ImageNumber"] == img_num) & (df["ObjectNumber"] == obj_num)].index.tolist()[0]

            if hasattr(cells[ind], "cell_contour") and hasattr(cells[ind], "pca_cord"):
                traj_contour.append(cells[ind].cell_contour.points.flatten())
                traj_cord.append(cells[ind].pca_cord)
                traj_seri.append([img_num, obj_num])
                traj_xy.append([df.loc[ind, "Cell_AreaShape_Center_X"], df.loc[ind, "Cell_AreaShape_Center_Y"]])
                traj_feature.append(df.loc[ind, "Cell_AreaShape_Area":"Cell_AreaShape_Solidity"].values.tolist())

                if ((am_record["ImageNumber"] == img_num) & (am_record["ObjectNumber"] == obj_num)).any():
                    am_flag = np.asscalar(
                        am_record.loc[
                            (am_record["ImageNumber"] == img_num) & (am_record["ObjectNumber"] == obj_num), "am_flag"
                        ].values
                    )
                    traj_am_flag.append(am_flag)
                else:
                    traj_am_flag.append(0)

                if hasattr(cells[ind], fluor_name + "_feature_values"):
                    #                     print(cells[ind].vimentin_haralick_pca_cord)
                    exec("traj_fluor_feature_values.append(np.array(cells[ind]." + fluor_name + "_feature_values[:3]))")
                    exec("traj_haralick.append(np.array(cells[ind]." + fluor_name + "_feature_values[3]))")
                    exec("traj_norm_haralick.append(np.array(cells[ind]." + fluor_name + "_feature_values[4]))")
                    exec("traj_fluor_pca_cord.append(cells[ind]." + fluor_feature_name[0] + "_pca_cord)")
                #                     exec('traj_norm_fluor_pca_cord.append(cells[ind].'+fluor_feature_name[1]+'_pca_cord)')
                else:
                    traj_fluor_feature_values.append(np.zeros((3,)))
                    traj_haralick.append(
                        (
                            np.zeros(
                                13,
                            )
                        )
                    )
                    traj_norm_haralick.append(
                        (
                            np.zeros(
                                13,
                            )
                        )
                    )
                    traj_fluor_pca_cord.append(np.zeros((6,)))
    #                     traj_norm_fluor_pca_cord.append(np.zeros((6,)))
    #

    traj_xy = np.asarray(traj_xy)
    traj_feature = np.asarray(traj_feature)
    traj_contour = np.asarray(traj_contour)
    print(traj_xy.shape)

    traj_cord = np.asarray(traj_cord)
    traj_seri = np.asarray(traj_seri)

    traj_am_flag = np.asarray(traj_am_flag)

    traj_fluor_feature_values = np.asarray(traj_fluor_feature_values)
    print(traj_fluor_feature_values.shape)

    traj_haralick = np.asarray(traj_haralick)
    traj_norm_haralick = np.asarray(traj_norm_haralick)
    #     print(traj_fluor_pca_cord)
    traj_fluor_pca_cord = np.asarray(traj_fluor_pca_cord)

    #     traj_norm_fluor_pca_cord=np.asarray(traj_norm_fluor_pca_cord)

    #     plt.figure()
    #     dot_color=np.arange(traj_cord.shape[0])
    #     cm=plt.cm.get_cmap('jet')
    #     plt.scatter(np.arange(traj_seri.shape[0]),traj_am_flag,s=8,c=dot_color,cmap=cm)
    #     plt.ylabel('am_flag',fontsize=16)
    #     plt.show()

    mask = traj_fluor_feature_values[:, 0] != 0

    #     dot_color=np.arange(traj_cord[mask].shape[0])
    #     cm=plt.cm.get_cmap('jet')
    #     print(traj_fluor_feature_values[mask].shape[0])
    #     plt.scatter(np.arange(traj_fluor_feature_values[mask].shape[0]),traj_fluor_feature_values[mask][:,0],s=5,c=dot_color,cmap=cm)
    #     plt.ylabel('Mean Vimentin Intensity',fontsize=16)
    #     plt.show()

    abn_inds = find_abnormal_fluor(traj_fluor_feature_values[:, 0], traj_seri[:, 0])
    if len(abn_inds) > 0:
        traj_fluor_feature_values[abn_inds, :] = 0
        traj_haralick[abn_inds, :] = 0
        traj_norm_haralick[abn_inds, :] = 0
        traj_fluor_pca_cord[abn_inds, :] = 0
    #         traj_norm_fluor_pca_cord[abn_inds,:]=0

    #     mask=traj_fluor_feature_values[:,0]!=0
    #     dot_color=np.arange(traj_cord[mask].shape[0])
    #     cm=plt.cm.get_cmap('jet')
    #     plt.scatter(np.arange(traj_fluor_feature_values[mask].shape[0]),traj_fluor_feature_values[mask][:,0],s=5,c=dot_color,cmap=cm)
    #     plt.ylabel('Mean Vimentin Intensity',fontsize=16)
    #     plt.show()

    #     plt.scatter(np.arange(traj_fluor_feature_values.shape[0]),traj_fluor_feature_values[:,0],s=5)
    #     plt.ylabel('Mean Vimentin Intensity',fontsize=16)
    #     plt.show()

    #     sc=plt.scatter(traj_cord[mask][:,0],traj_cord[mask][:,1],c=dot_color,cmap=cm)
    #     plt.xlabel('PC1',fontsize=16)
    #     plt.ylabel('PC2',fontsize=16)
    #     #     plt.axis([-450,800,-400,400])
    #     plt.colorbar(sc)
    #     plt.show()

    #     print('pc1_norm_haralick_pc1_corr',np.corrcoef(traj_cord[mask][:,0],traj_norm_fluor_pca_cord[mask][:,0])[0,1])
    #     print('pc1_norm_haralick_pc2_corr',np.corrcoef(traj_cord[mask][:,0],traj_norm_fluor_pca_cord[mask][:,1])[0,1])

    #     print('pc2_norm_haralick_pc1_corr',np.corrcoef(traj_cord[mask][:,1],traj_norm_fluor_pca_cord[mask][:,0])[0,1])
    #     print('pc2_norm_haralick_pc2_corr',np.corrcoef(traj_cord[mask][:,1],traj_norm_fluor_pca_cord[mask][:,1])[0,1])
    #     sc=plt.scatter(traj_xy[mask][:,0],traj_xy[mask][:,1],c=dot_color,cmap=cm)
    #     plt.xlabel('X',fontsize=16)
    #     plt.ylabel('Y',fontsize=16)
    # #     plt.axis([0,1952,0,1952])
    #     plt.colorbar(sc)
    #     plt.show()

    #     print('mean_std_corr',np.corrcoef(traj_fluor_feature_values[mask][:,0],traj_fluor_feature_values[mask][:,1]))
    #     plt.scatter(np.arange(traj_fluor_feature_values[mask].shape[0]),traj_fluor_feature_values[mask][:,1],c=dot_color,cmap=cm)
    #     plt.ylabel('Std Vimentin Intensity',fontsize=16)
    #     plt.show()

    #     dot_color=np.arange(traj_fluor_feature_values[mask].shape[0])
    #     cm=plt.cm.get_cmap('jet')

    # print('mean_haralick_pc1_corr',np.corrcoef(traj_fluor_feature_values[mask][:,0],traj_fluor_pca_cord[mask][:,0])[0,1])
    # print('mean_haralick_pc2_corr',np.corrcoef(traj_fluor_feature_values[mask][:,0],traj_fluor_pca_cord[mask][:,1])[0,1])

    # sc=plt.scatter(traj_fluor_pca_cord[mask][:,0],traj_fluor_pca_cord[mask][:,1],s=8,c=dot_color,cmap=cm)
    # plt.xlabel('haralickPC1',fontsize=16)
    # plt.ylabel('haralickPC2',fontsize=16)
    # #     plt.axis([-450,800,-400,400])
    # plt.colorbar(sc)
    # plt.show()

    # #     print('mean_norm_haralick_pc1_corr',np.corrcoef(traj_fluor_feature_values[mask][:,0],traj_norm_fluor_pca_cord[mask][:,0])[0,1])
    # #     print('mean_norm_haralick_pc2_corr',np.corrcoef(traj_fluor_feature_values[mask][:,0],traj_norm_fluor_pca_cord[mask][:,1])[0,1])
    # #     print('mean_norm_haralick_pc3_corr',np.corrcoef(traj_fluor_feature_values[mask][:,0],traj_norm_fluor_pca_cord[mask][:,2])[0,1])

    #     sc=plt.scatter(np.arange(traj_fluor_feature_values[mask].shape[0]),traj_norm_fluor_pca_cord[mask][:,0],s=8,c=dot_color,cmap=cm)
    #     plt.xlabel('t',fontsize=16)
    #     plt.ylabel('normharalickPC1',fontsize=16)
    # #     plt.axis([-5,10,-3,4])
    #     plt.colorbar(sc)
    #     plt.show()

    #     sc=plt.scatter(np.arange(traj_fluor_feature_values[mask].shape[0]),traj_norm_fluor_pca_cord[mask][:,1],s=8,c=dot_color,cmap=cm)
    #     plt.xlabel('t',fontsize=16)
    #     plt.ylabel('normharalickPC2',fontsize=16)
    # #     plt.axis([-5,10,-3,4])
    #     plt.colorbar(sc)
    #     plt.show()

    # #     sc=plt.scatter(np.arange(traj_fluor_feature_values[mask].shape[0]),traj_norm_fluor_pca_cord[mask][:,2],s=8,c=dot_color,cmap=cm)
    # #     plt.xlabel('t',fontsize=16)
    # #     plt.ylabel('normharalickPC3',fontsize=16)
    # # #     plt.axis([-5,10,-3,4])
    # #     plt.colorbar(sc)
    # #     plt.show()

    #     ax = plt.axes(projection='3d')
    #     ax.scatter3D(traj_norm_fluor_pca_cord[mask][:,0],traj_norm_fluor_pca_cord[mask][:,1],traj_norm_fluor_pca_cord[mask][:,2],c=dot_color,cmap=cm)
    #     #     ax.view_init(50, 25)
    #     plt.xlabel('normharalickPC1',fontsize=16)
    #     plt.ylabel('normharalickPC2',fontsize=16)
    #     plt.ylabel('normharalickPC3',fontsize=16)
    #     plt.show()

    #     sc=plt.scatter(traj_norm_fluor_pca_cord[mask][:,0],traj_norm_fluor_pca_cord[mask][:,1],s=8,c=dot_color,cmap=cm)
    #     plt.xlabel('normharalickPC1',fontsize=16)
    #     plt.ylabel('normharalickPC2',fontsize=16)
    # #     plt.axis([-5,10,-3,4])
    #     plt.colorbar(sc)
    #     plt.show()

    # sc=plt.scatter(traj_cord[mask][:,0],traj_fluor_pca_cord[mask][:,0],s=8,c=dot_color,cmap=cm)
    # plt.xlabel('PC1',fontsize=16)
    # plt.ylabel('haralickPC1',fontsize=16)
    # #     plt.axis([-450,800,-400,400])
    # plt.colorbar(sc)
    # plt.show()

    #     sc=plt.scatter(traj_fluor_pca_cord[mask][:,0],traj_norm_fluor_pca_cord[mask][:,0],s=8,c=dot_color,cmap=cm)
    #     plt.xlabel('PC1',fontsize=16)
    #     plt.ylabel('normharalickPC2',fontsize=16)
    #     #     plt.axis([-450,800,-400,400])
    #     plt.colorbar(sc)
    #     plt.show()

    #     ax = plt.axes(projection='3d')
    #     ax.scatter3D(traj_cord[mask][:,0],traj_cord[mask][:,1],traj_norm_fluor_pca_cord[mask][:,0],c=dot_color,cmap=cm)
    #     #     ax.view_init(50, 25)
    #     plt.show()

    # #     ax = plt.axes(projection='3d')
    # #     ax.scatter3D(traj_norm_fluor_pca_cord[mask][:,0],traj_norm_fluor_pca_cord[mask][:,1],traj_cord[mask][:,0],c=dot_color,cmap=cm)
    # #     #     ax.view_init(50, 25)
    # #     plt.show()

    # #     for i in range(traj_haralick.shape[1]):
    # #         print('mean_norm_'+haralick_labels[i]+'_corr',np.corrcoef(traj_fluor_feature_values[mask][:,0],traj_norm_haralick[mask][:,i])[0,1])
    # #         print('std_norm_'+haralick_labels[i]+'_corr',np.corrcoef(traj_fluor_feature_values[mask][:,1],traj_norm_haralick[mask][:,i])[0,1])
    # # #         plt.plot(traj_haralick[mask][:,i],'b.')
    # #         plt.scatter(np.arange(traj_fluor_feature_values[mask].shape[0]),traj_norm_haralick[mask][:,i],c=dot_color,cmap=cm)
    # #         plt.ylabel(haralick_labels[i],fonbuiltins__)tsize=16)
    # #         plt.show()

    #     return traj_feature,traj_contour,traj_cord,traj_seri,traj_am_flag,\
    #            [traj_fluor_feature_values[:,0],traj_fluor_feature_values[:,1],traj_fluor_feature_values[:,2],\
    #            traj_haralick,traj_norm_haralick],traj_fluor_pca_cord,traj_norm_fluor_pca_cord
    return (
        traj_feature,
        traj_contour,
        traj_cord,
        traj_seri,
        traj_am_flag,
        [
            traj_fluor_feature_values[:, 0],
            traj_fluor_feature_values[:, 1],
            traj_fluor_feature_values[:, 2],
            traj_haralick,
            traj_norm_haralick,
        ],
        traj_fluor_pca_cord,
    )


main_path = "C:/Users/14432/OneDrive/Research/Projects/Nikon_A549/data/A549_vim_rfp_tgfb_livecell/"
input_path = main_path + "ori_resized/"
output_path = main_path + "out/weik_08-02-21_altered_param/"

sct_path = output_path + "sc_traj/"
if not os.path.exists(sct_path):
    os.makedirs(sct_path)


posi_end = 21
time_span = 225
traj_len_thres = 71

begin_range = 154
end_range = 224

loss_ratio_thres = 0.5  # if the ratio of obj_num is -1

feature_list = ["mean_intensity", "std_intensity", "intensity_range", "haralick", "norm_haralick"]
fluor_feature_name = ["vimentin_haralick", "norm_vimentin_haralick"]

posi = 1

img_path = input_path + "XY0" + str(posi) + "_C1/"
img_list = sorted(listdir(img_path))
fluor_img_path = input_path + "XY0" + str(posi) + "_C2/"
fluor_img_list = sorted(listdir(fluor_img_path))

dir_path = output_path + "XY0" + str(posi) + "/"

seg_path = dir_path + "seg/"
seg_list = sorted(listdir(seg_path))


df = pd.read_csv(dir_path + "Per_Object_relink.csv")
am_record = pd.read_csv(dir_path + "am_record.csv")
traj_df = pd.read_csv(dir_path + "traj_object_num.csv")
mitosis_df = pd.read_csv(dir_path + "mitosis_record.csv")


with open(dir_path + "cells/" + "fluor_cells", "rb") as fp:
    cells = pickle.load(fp)

traj_record = pd.DataFrame(traj_df.loc[:, "1" : str(time_span)])
traj_record = traj_record.values
traj_quan, traj_len = traj_record.shape[0], traj_record.shape[1]

for traj_label in range(1, traj_quan + 1):

    cur_traj = traj_record[traj_label - 1, :]
    traj_start_t = np.where(cur_traj != -1)[0][0] + 1
    daughter_flag = find_mother(df, mitosis_df, traj_label)

    if traj_start_t < begin_range:
        family_tree = {}
        family_tree = find_offspring(df, mitosis_df, family_tree, traj_label=traj_label)
        # print('family',family_tree)
        all_branches = list(list(parse(traj_label, family_tree)))
        # print('all branch',all_branches)

        for branch in all_branches:
            branch_end_label = branch[-1]
            branch_end_traj = traj_record[branch_end_label - 1, :]
            branch_end_t = np.where(branch_end_traj != -1)[0][-1] + 1

            if branch_end_t > end_range:
                print("trajectory " + str(traj_label))
                print("branch end t: " + str(branch_end_t))
                long_traj = -1 * np.ones((time_span,))
                divide_points = []
                traj_name = ""
                for sub_label in branch:
                    sub_traj = traj_record[sub_label - 1, :]
                    mask = sub_traj != -1
                    if (np.where(sub_traj != -1)[0][-1] + 1) != branch_end_t:
                        divide_points.append(np.where(sub_traj != -1)[0][-1] + 1)
                    long_traj[mask] = sub_traj[mask]
                    traj_name = traj_name + str(sub_label) + "_"
                whole_branch = long_traj[traj_start_t - 1 : branch_end_t]
                long_traj_loss_ratio = whole_branch[whole_branch == -1].shape[0] * 1.0 / whole_branch.shape[0]

                if long_traj_loss_ratio < loss_ratio_thres:

                    (
                        traj_feature,
                        traj_contour,
                        traj_cord,
                        traj_seri,
                        traj_am_flag,
                        traj_fluor_feature_values,
                        traj_fluor_pca_cord,
                    ) = generate_fluor_long_traj(
                        df, cells, am_record, traj_df, long_traj, "vimentin", feature_list, fluor_feature_name
                    )
                    traj_sct = fluor_single_cell_traj(traj_seri, traj_contour)
                    traj_sct.set_traj_feature(traj_feature)
                    traj_sct.set_traj_cord(traj_cord)
                    traj_sct.set_traj_divide_points(np.array(divide_points))
                    traj_sct.set_traj_am_flag(traj_am_flag)
                    traj_sct.set_traj_fluor_features("vimentin", feature_list, traj_fluor_feature_values)
                    traj_sct.set_traj_fluor_pca_cord(fluor_feature_name[0], traj_fluor_pca_cord)

                if branch_end_label == traj_label:
                    with open(sct_path + "fluor_sct_XY0" + str(posi) + "_" + traj_name + "long_traj", "wb") as fp:
                        pickle.dump(traj_sct, fp)
                else:
                    with open(
                        sct_path + "indirect_fluor_sct_XY0" + str(posi) + "_" + traj_name + "long_traj", "wb"
                    ) as fp:
                        pickle.dump(traj_sct, fp)

                fig = plt.figure()
                ims = []
                for img_num, obj_num in traj_seri[traj_fluor_feature_values[0] > 0]:
                    img_num = img_num.astype(np.int)
                    obj_num = obj_num.astype(np.int)
                    if obj_num != -1:
                        img = imread(img_path + img_list[img_num - 1])
                        seg_img = imread(seg_path + seg_list[img_num - 1])
                        crop_cell, crop_cell_env = generate_single_cell_img(img, seg_img, img_num, obj_num)
                        im = plt.imshow(crop_cell, animated=True)
                        ims.append([im])
                ani = animation.ArtistAnimation(fig, ims, interval=150)
                if branch_end_label == traj_label:
                    ani.save(sct_path + "morph_sct_XY0" + str(posi) + "_" + traj_name + "long_traj.gif")
                else:
                    ani.save(sct_path + "indirect_morph_sct_XY0" + str(posi) + "_" + traj_name + "long_traj.gif")

                board_size = 250
                fig = plt.figure()
                ims = []
                for img_num, obj_num in traj_seri[traj_fluor_feature_values[0] > 0]:
                    img_num = img_num.astype(np.int)
                    obj_num = obj_num.astype(np.int)
                    if obj_num != -1:
                        img = imread(fluor_img_path + fluor_img_list[img_num - 1])
                        seg_img = imread(seg_path + seg_list[img_num - 1])
                        crop_cell, crop_cell_env = generate_single_cell_img(img, seg_img, img_num, obj_num)
                        x_c, y_c = crop_cell.shape[0] // 2, crop_cell.shape[1] // 2
                        x_l, y_l = board_size / 2 - x_c, board_size / 2 - y_c
                        cell_on_board = np.zeros((board_size, board_size))
                        cell_on_board[x_l : x_l + crop_cell.shape[0], y_l : y_l + crop_cell.shape[1]] = crop_cell

                        im = plt.imshow(crop_cell, animated=True)
                        plt.axis("off")
                        ims.append([im])

                ani = animation.ArtistAnimation(fig, ims, interval=100)
                if branch_end_label == traj_label:
                    ani.save(sct_path + "fluor_sct_XY0" + str(posi) + "_" + traj_name + "long_traj.gif")
                else:
                    ani.save(sct_path + "indirect_fluor_sct_XY0" + str(posi) + "_" + traj_name + "long_traj.gif")
