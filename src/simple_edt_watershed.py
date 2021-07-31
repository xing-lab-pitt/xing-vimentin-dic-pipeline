
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
from os import listdir
from skimage.morphology import remove_small_objects,local_maxima,h_maxima,disk,dilation
from skimage.measure import regionprops,label
from skimage.color import label2rgb
from PIL import Image, ImageDraw, ImageFont
from math import pi
from pilutil import toimage
from cnn_prep_data import keep_aspect_resize,obj_transform

import pipe_util2

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

size_thres1=1400
size_thres2=2500
roundness_thres=0.8

def color_num(labels):
    """Creating a image that have color and number for each cell"""
    label_rgb=label2rgb(labels,bg_label=0)
    img_rgb= toimage(label_rgb)
    base = img_rgb.convert('RGBA')
    # make a blank image for the text, initialized to transparent text color
    txt = Image.new('RGBA', base.size, (255,255,255,0))
    # get a font
    fnt = ImageFont.truetype('/net/capricorn/home/xing/huijing/Segmentation/scripts/vimentin_DIC_segmentation_pipeline/hj_modify_pipe/arial.ttf', 20)
    # get a drawing context
    d = ImageDraw.Draw(txt)
    for region in regionprops(labels):
        cx=int(region.centroid[1])     
        cy=int(region.centroid[0])
        d.text((cx,cy),str(labels[cy][cx]),font=fnt,fill=(255,255,255,255)) 
    out = Image.alpha_composite(base, txt)       
    return out


#---do not remove small objects before calculate overlap
def hmax_watershed(img, h_thres, mask_thres=0):

    local_hmax=h_maxima(img,h_thres)    # Determine all maxima of the image with height >= h.
    local_hmax_label=label(local_hmax,connectivity=1)    # get label

    # watershed img can be multi-dim
    # local_hmax_label is the desired number of markers. The "seed".
    # Note:
    # if the h_thres too low, label will "stick together", then the seed is less
    # If the h_thres too high, then some cells might not be able to reach, then less seed.
    # mask_thres, only values above this threshold will be considered.
    labels = watershed(-img, local_hmax_label, mask=img>mask_thres)
#     labels = remove_small_objects(labels, small_obj_thres)    
#     labels=label(labels,connectivity =2)
    return labels,local_hmax_label # return label and seeds

#crop single segmented img by bounding box from the original image 
def generate_single_cell_img_env(img,rps,obj_h,obj_w,obj_num):   
    
    """
    This function is used for generating single cells for prediction.
    The labels are selected based on the roundness and size.

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


def direct_edt_watershed(output_path, q1=0.25, q2=0.5, q3=0.92, small_obj_thres = 20):
    """
    This function does not select watershed masks.
    q1, q2, q3 are intensity quantile to use as threshold for mask, low_h, high_h
    Intensity below mask_h is ignored.
    Intensity above low_h and high_h will be considered for starting a seed.
    """
    
    """
    for a given folder:
        - create a corresponding seg/ folder
        - create a corresponding rgb_num/ folder
        - save img_seg into the seg/ folder
        - save rgb_num into the rgb_num/ folder

    :return:
    Save segmented cells 
    """

    output_path = pipe_util2.folder_verify(output_path)
    reg_path = pipe_util2.folder_verify(output_path+'edt')
    dir_path = output_path

    reg_list = sorted(listdir(reg_path)) # this is the edt folder. # edt image path master_output_folder+#/reg/reg_xximgstring

    seg_path = dir_path + 'seg/'
    pipe_util2.create_folder(seg_path) # the folder will be re-created every time.

    rgb_num_path = dir_path + 'rgb_num/'
    pipe_util2.create_folder(rgb_num_path) # the folder will be re-created every time.

    for i in range(len(reg_list)):
        #print('image: %d/%d' % (i, len(reg_list)))
        #print(reg_path + reg_list[i])

        reg = imread(reg_path + reg_list[i])
        # -------------for weights map
#         reg=-reg*(reg<weight_map_thres)

        img_name = reg_list[i][:-4]

        # addaptive range for watershed.
        reg_flat = reg[reg.nonzero()].flatten()
        thresholds = np.quantile(reg_flat,[q1, q2, q3])
        mask_h = thresholds[0]
        low_h = thresholds[1]
        high_h = thresholds[2]
        #print(reg_flat)

        high_h_seg, high_h_markers = hmax_watershed(
            reg, h_thres=high_h, mask_thres = mask_h)

        labels = watershed(-reg, high_h_markers, mask= reg >= mask_h)
        labels = remove_small_objects(labels, small_obj_thres)
        labels = label(labels, connectivity=2)

        # should use np.uint32,could be save correctly
        img_seg = Image.fromarray(labels.astype(np.uint32), 'I')
        img_seg.save(seg_path + 'seg_' + img_name + '.png')

        rgb_num = color_num(labels)
        rgb_num.save(rgb_num_path + 'rgb_' + img_name + '.png')


def simple_edt_watershed(output_path, small_obj_thres = 20):

    """
    for a given folder:
        - create a corresponding seg/ folder
        - create a corresponding rgb_num/ folder
        - save img_seg into the seg/ folder
        - save rgb_num into the rgb_num/ folder

    :return:
    Save segmented cells 
    """

    output_path = pipe_util2.folder_verify(output_path)
    reg_path = pipe_util2.folder_verify(output_path+'edt')
    dir_path = output_path

    reg_list = sorted(listdir(reg_path)) # this is the edt folder. # edt image path master_output_folder+#/reg/reg_xximgstring

    seg_path = dir_path + 'seg/'
    pipe_util2.create_folder(seg_path) # the folder will be re-created every time.

    rgb_num_path = dir_path + 'rgb_num/'
    pipe_util2.create_folder(rgb_num_path) # the folder will be re-created every time.

    for i in range(len(reg_list)):
        #print('image: %d/%d' % (i, len(reg_list)))
        #print(reg_path + reg_list[i])

        reg = imread(reg_path + reg_list[i])
        # -------------for weights map
#         reg=-reg*(reg<weight_map_thres)

        img_name = reg_list[i][:-4]

        # addaptive range for watershed.
        reg_flat = reg[reg.nonzero()].flatten()
        thresholds = np.quantile(reg_flat,[0.15, 0.25, 0.7])
        mask_h = thresholds[0]
        low_h = thresholds[1]
        high_h = thresholds[2]
        #print(reg_flat)

        high_h_seg, high_h_markers = hmax_watershed(
            reg, h_thres=high_h, mask_thres = mask_h)

        labels = watershed(-reg, high_h_markers, mask= reg >= mask_h)
        labels = remove_small_objects(labels, small_obj_thres)
        labels = label(labels, connectivity=2)

        rps = regionprops(labels)    # obtain lable properties

        # selecting labels
        # lable size smaller than the threshold 1
        # or lable size lable size smaller than threshold 2, but is pretty round. 
        #candi_labels = [r.label for r in rps if r.area < size_thres1 or (4 * pi * r.area / r.perimeter**2 > roundness_thres and r.area < size_thres2)]

        candi_labels = []
        j = 0
        while j<len(rps):
            #print(rps[i].area)
            if rps[j].area<size_thres1:
                candi_labels.append(rps[j].label)
            j = j+1
        
        all_labels = np.arange(1, labels.max())
        if len(candi_labels)==0:
            labels[:,:] = 0
        else: 
            rm_labels = list(set(all_labels)-set(candi_labels))
            #print(rm_labels)
            for x in rm_labels:
                labels[labels==x] =0
        

        # should use np.uint32,could be save correctly
        img_seg = Image.fromarray(labels.astype(np.uint32), 'I')
        img_seg.save(seg_path + 'seg_' + img_name + '.png')

        rgb_num = color_num(labels)
        rgb_num.save(rgb_num_path + 'rgb_' + img_name + '.png')

    
if __name__ == "__main__":
    pass