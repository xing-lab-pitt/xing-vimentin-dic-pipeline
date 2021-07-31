
'''
we could use a  technique that could segment all the possible cells in the prediction of distance transform.  
when small cells are  recognized, other small pieces and noise would also be identified.
This will lead to over-segmentation.
Here we utilize the over-segmentation to segment the mitosis and apoptosis cells. 
Then, we extract the small cells that below size threshold. And use an identification CNN  to identify them,
only the apoptosis and mitosis cells are kept. 
After correctly segment these cells, we remove these cells from the prediction of distance transform. 
'''

import numpy as np
from skimage.segmentation import watershed,clear_border
from skimage.io import imread
from matplotlib import pyplot as plt
import scipy.ndimage as ndi
import pickle
import os
from os import listdir
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects,local_maxima,h_maxima,disk,dilation
from skimage.measure import regionprops,label
from skimage.color import label2rgb
from PIL import Image, ImageDraw, ImageFont
from resnet50 import res_model
from math import pi
import cv2
import glob
from skimage.exposure import equalize_adapthist
from pilutil import toimage
from cnn_prep_data import keep_aspect_resize,obj_transform

import pandas as pd
from scipy import ndimage
import sys
import pipe_util2

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# low_h=0.05
# high_h=0.4
# small_obj_thres=1500
# mask_thres=0
# weight_map_thres=-0.01

size_thres1=10000
size_thres2=12000
roundness_thres=0.8


def icnn_am_load_weight(icnn_am_weights):
    obj_h=128
    obj_w=128
    input_shape=(obj_h,obj_w,1)
    nb_class=3
    icnn_am=res_model(input_shape,nb_class)
    icnn_am.load_weights(icnn_am_weights)
    return icnn_am


def icnn_seg_load_weight(icnn_seg_weights):
    obj_h=128
    obj_w=128
    input_shape=(obj_h,obj_w,1)
    nb_class=3
    icnn_seg=res_model(input_shape,nb_class)
    icnn_seg.load_weights(icnn_seg_weights)
    return icnn_seg


def color_num(labels):
    label_rgb=label2rgb(labels,bg_label=0)
    img_rgb= toimage(label_rgb)
    base = img_rgb.convert('RGBA')
    # make a blank image for the text, initialized to transparent text color
    txt = Image.new('RGBA', base.size, (255,255,255,0))
    # get a font
    fnt = ImageFont.truetype('/net/capricorn/home/xing/huijing/Segmentation/scripts/vimentin_DIC_segmentation_pipeline/hj_modify_pipe/arial.ttf', 40)
    # get a drawing context
    d = ImageDraw.Draw(txt)
    for region in regionprops(labels):
        cx=int(region.centroid[1])     
        cy=int(region.centroid[0])
        d.text((cx,cy),str(labels[cy][cx]),font=fnt,fill=(255,255,255,255)) 
    out = Image.alpha_composite(base, txt)       
    return out


#this is faster than the one from lineage mapper,only with matrix calculation
def compute_overlap_matrix(seg1,seg2):

    nb_cell_1=np.amax(seg1)
    nb_cell_2=np.amax(seg2)

    seg_overlap=np.zeros((nb_cell_1,nb_cell_2))
    for obj_idx1 in range(nb_cell_1):
        obj_num1=obj_idx1+1
        sc_img=(seg1==obj_num1)
        ol_judge=np.logical_and(sc_img,seg2)
        ol_value=np.multiply(ol_judge,seg2)
        ol_obj2=np.unique(ol_value).tolist()
        #ol_obj2=ol_obj2[ol_obj2!=0]
        ol_obj2.remove(0)
        if len(ol_obj2)>0:
            for obj_num2 in ol_obj2:
                ol_area=np.sum(ol_value==obj_num2)
                obj_idx2=obj_num2-1
                seg_overlap[obj_idx1][obj_idx2]=ol_area  

    return seg_overlap


# In[30]:


#-------calculate the cell fusion -----------------------
def cal_cell_fusion(frame_overlap):
    nb_cell_1=frame_overlap.shape[0]
    nb_cell_2=frame_overlap.shape[1]
    
    prefuse_group=[]#each element is a list include all prefuse cells in a fuse event, corresponding to postfuse_cells
    postfuse_cells=[]#include: img_num,obj_num
    frame_fusion = np.zeros(frame_overlap.shape)
    for source_o_n in range(1,nb_cell_1+1):
        #find target whose max_overlap mother is source
        ol_target=frame_overlap[source_o_n-1,:]
        if np.all(ol_target==0):#if source obj have no overlap target
            target_o_n=0    
        else:      
            target_o_n=np.argmax(frame_overlap,axis=1)[source_o_n-1]+1#axis=1,maximum of each row,return column index
        
       
        if target_o_n> 0:
            frame_fusion[source_o_n-1, target_o_n-1] = 1
    
        #Compute the sum vector S which is the sum of all the columns of frame_fusion matrix. The fusion target region
        #will have at least 2 cells tracked to it => S>1
    S = np.sum(frame_fusion, axis=0)
    frame_fusion[:, S==1] = 0          
    # Update the sum vector
    S = np.sum(frame_fusion, axis=0)

    for i in range(len(np.where(S >= 2)[0])):
        f_group=[]
        postfuse_cells.append([np.where(S >= 2)[0][i]+1])#num of prefuse cells:S[np.where(S >= 2)[0][i]]
        frame_fusion_i=frame_fusion[:,np.where(S >= 2)[0][i]]

        for r in range(len(np.where(frame_fusion_i==1)[0])):
            #fuse_pairs.append([img_num_1,np.where(frame_fusion_i==1)[0][r]+1,img_num_2,np.where(S >= 2)[0][i]+1])
            f_group.append(np.where(frame_fusion_i==1)[0][r]+1)
        prefuse_group.append(f_group)
    return postfuse_cells,prefuse_group


#---do not remove small objects before calculate overlap
def hmax_watershed(img,h_thres, small_obj_thres, mask_thres=0):


   
    local_hmax=h_maxima(img,h_thres)
    local_hmax_label=label(local_hmax,connectivity=1)
    
#     ws_marker=np.zeros(img.shape)
#     j=0
#     local_hmx=[]
#     local_hmy=[]
#     for region in regionprops(local_hmax_label):        
#         cx=int(region.centroid[0])        
#         local_hmx.append(cx)
#         cy=int(region.centroid[1])
#         local_hmy.append(cy)
#         j+=1
#         ws_marker[cx][cy]=int(j) 
#     print(np.amax(local_hmax_label))
#     print(np.any(local_hmax_label-ws_marker))
#     ws_marker=local_hmax_label

    labels = watershed(-img, local_hmax_label, mask=img>mask_thres)
#     labels = remove_small_objects(labels, small_obj_thres)    
#     labels=label(labels,connectivity =2)
    print(np.amax(labels))
    return labels,local_hmax_label


def generate_single_cell_img_edt(img,seg,obj_h,obj_w,obj_num):
    
    #single_obj_img=morphology.binary_dilation(seg_img==obj_num,morphology.diamond(16))
    single_obj_img=seg==obj_num
    single_obj_img=label(single_obj_img)
    rps=regionprops(single_obj_img)
    candi_r=[r for r in rps if r.label==1][0]
    candi_box=candi_r.bbox        
    single_cell_img=single_obj_img*img
    crop_img=single_cell_img[candi_box[0]:candi_box[2],candi_box[1]:candi_box[3]]

    inds=ndimage.distance_transform_edt(crop_img==0, return_distances=False, return_indices=True)
    crop_img=crop_img[tuple(inds)]
    crop_img=obj_transform(keep_aspect_resize(crop_img,obj_h,obj_w),random_eah=False)
#     plt.imshow(crop_img[:,:,0])
#     plt.show()
    crop_img=np.expand_dims(crop_img,axis=0)
    return crop_img


# In[33]:


#crop single segmented img by bounding box from the original image 
def generate_single_cell_img_env(img,rps,obj_h,obj_w,obj_num):   
    
    """
    img is the original image, as np.array.
    rps is the label properties by skimage.
    obj_h and obj_w is predefined. Here is 128*128.
    obj_num is the label number. 
    """

    candi_r=[r for r in rps if r.label==obj_num][0]
    candi_box=candi_r.bbox        
    crop_img=img[candi_box[0]:candi_box[2],candi_box[1]:candi_box[3]]
    
    crop_img=obj_transform(keep_aspect_resize(crop_img,obj_h,obj_w),random_eah=False)    # obj_transform, normalize picture, and then add an axis at the end.
    crop_img=np.expand_dims(crop_img,axis=0)

#     cell_img=toimage(crop_cell_img,high=np.max(crop_cell_img),low=np.min(crop_cell_img),mode='I')
#     cell_img.save(crop_path+'/i'+str(img_num)+'_o'+str(obj_num)+'.tif')
    return crop_img


def judge_fuse_type(low_h_markers,high_h_markers,fc_cell,fp_group,fc_prob,fp_group_prob):
   

    type_list=['single','fragment','multi']
    
    fc_prob[1]=0
    fc_type=type_list[np.argmax(fc_prob)]
    
    nb_fp=len(fp_group)

    fp_group_type=[]
    
    
    for fp_prob in fp_group_prob:
        fp_prob[2]=0
        fp_group_type.append(type_list[np.argmax(np.array(fp_prob))])
    
#     print(fc_type,fp_group_type)
    if fp_group_type.count('single')>=1:
        for i in range(nb_fp):
            if fp_group_type[i]=='fragment':
                low_h_markers[np.where(low_h_markers==fp_group[i])]=0
    else:
        for i in range(nb_fp):
            if not np.any(high_h_markers[np.where(low_h_markers==fp_group[i])]):
                low_h_markers[np.where(low_h_markers==fp_group[i])]=0
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


def simple_edt_watershed(img_path, output_path, 
                         icnn_am_weights, icnn_seg_weights, 
                         small_obj_thres = 20, obj_h=128, obj_w=128):

    """
    for a given folder:
        - create a corresponding seg/ folder
        - create a corresponding rgb_num/ folder
        - save img_seg into the seg/ folder
        - save rgb_num into the rgb_num/ folder

    :return:
    """
    
    icnn_am = icnn_am_load_weight(icnn_am_weights)
    icnn_seg = icnn_seg_load_weight(icnn_seg_weights)
    
    img_path = pipe_util2.folder_verify(img_path)
    output_path = pipe_util2.folder_verify(output_path)
    reg_path = pipe_util2.folder_verify(output_path + 'edt')
    dir_path = output_path

    img_list = sorted(listdir(img_path))
    reg_list = sorted(listdir(reg_path)) # this is the edt folder. # edt image path master_output_folder+#/reg/reg_xximgstring

    seg_path = dir_path + 'seg/'
    pipe_util2.create_folder(seg_path) # the folder will be re-created every time.

    rgb_num_path = dir_path + 'rgb_num/'
    pipe_util2.create_folder(rgb_num_path) # the folder will be re-created every time.

    am_record = pd.DataFrame(
        columns=[
            'ImageNumber',
            'ObjectNumber',
            'am_flag'])
    all_am_count = 0

    for i in range(len(img_list)):
        print('image: %d/%d' % (i, len(img_list)))
        
        print(img_path + img_list[i])

        img = imread(img_path + img_list[i])

#         pil_img=Image.open(img_path+img_list[i])

        reg = imread(reg_path + reg_list[i])
        # -------------for weights map
#         reg=-reg*(reg<weight_map_thres)

        img_name = img_list[i][0:len(img_list[i]) - 4]

#         plt.figure(figsize=[12,12])
#         plt.title('ori_reg')
#         plt.imshow(reg)
#         plt.show()
        
        # addaptive range for watershed.
        # Des: use 
        reg_flat = reg[reg!=0.0].reshape(-1)
        thresholds = np.quantile(reg_flat,[0.15, 0.25, 0.5])
        mask_h = thresholds[0]
        low_h = thresholds[1]
        high_h = thresholds[2]
    
        low_h_seg, low_h_markers = hmax_watershed(
            reg, h_thres=low_h, small_obj_thres=small_obj_thres, mask_thres = mask_h)

#         rgb_num1=color_num(low_h_seg)
#         plt.figure(figsize=[12,12])
#         plt.imshow(rgb_num1)
#         plt.show()

        high_h_seg, high_h_markers = hmax_watershed(
            reg, h_thres=high_h, small_obj_thres=small_obj_thres, mask_thres = mask_h)

#         rgb_num2=color_num(high_h_seg)
#         plt.figure(figsize=[12,12])
#         plt.imshow(rgb_num2)
#         plt.show()

        seg_overlap = compute_overlap_matrix(low_h_seg, high_h_seg)
        fuse_cells, fuse_group = cal_cell_fusion(seg_overlap)

        for m in range(len(fuse_cells)):
            # print(fuse_cells[m])
            # print(fuse_group[m])

            fc_obj = generate_single_cell_img_edt(
                img, high_h_seg, obj_h, obj_w, fuse_cells[m])

            fc_prob = icnn_seg.predict(fc_obj)[0]
            fp_group_prob = []
            for n in range(len(fuse_group[m])):
                fp_obj = generate_single_cell_img_edt(
                    img, low_h_seg, obj_h, obj_w, fuse_group[m][n])
                fp_prob = icnn_seg.predict(fp_obj)[0]
                fp_group_prob.append(fp_prob)

#             print(fc_prob)
#             print(fp_group_prob)
            low_h_markers = judge_fuse_type(
                low_h_markers,
                high_h_markers,
                fuse_cells[m],
                fuse_group[m],
                fc_prob,
                fp_group_prob)

        labels = watershed(-reg, low_h_markers, mask=reg > mask_h)
        labels = remove_small_objects(labels, small_obj_thres)

        labels = label(labels, connectivity=2)

        rps = regionprops(labels)    # obtain lable properties
        
        # lable size smaller than the threshold 1
        # or lable size lable size smaller than threshold 2, but is pretty round. 
        candi_labels = [r.label for r in rps if r.area < size_thres1 or (
            4 * pi * r.area / r.perimeter**2 > roundness_thres and r.area < size_thres2)]

        for candi_label in candi_labels:

            candi_obj = generate_single_cell_img_env(
                img, rps, obj_h, obj_w, candi_label)
            #plt.imshow(candi_obj[0][:,:,0])
            #plt.show()
            output = icnn_am.predict(candi_obj)
            am_flag = np.argmax(output)
#             print(candi_label,am_flag,output)

            if am_flag > 0:
                all_am_count += 1
                am_record.loc[all_am_count] = [i + 1, candi_label, am_flag]

        # should use np.uint32,could be save correctly
        img_seg = Image.fromarray(labels.astype(np.uint32), 'I')
        img_seg.save(seg_path + 'seg_' + img_name + '.png')

        rgb_num = color_num(labels)
#         plt.figure(figsize=[12,12])
#         plt.imshow(rgb_num)
#         plt.show()

        rgb_num.save(rgb_num_path + 'rgb_' + img_name + '.png')

    am_record.to_csv(
        dir_path +'am_record.csv',
        index=False,
        encoding='utf-8')
    
if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    # arg1 is img_path
    # arg2 is output_path
    # arg3 is small_obj_thres
    arg_len = len(sys.argv)
    if arg_len == 5:
        simple_edt_watershed(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    elif arg_len == 6:
        small_obj_thres = int(sys.argv[5])
        simple_edt_watershed(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], small_obj_thres)