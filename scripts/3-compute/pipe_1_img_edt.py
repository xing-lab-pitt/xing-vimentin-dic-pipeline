from PIL import Image
from os import listdir
import glob
import os
import numpy as np
import keras.backend as K

from models import reg_seg, unet
from cnn_prep_data import prep_dic_data, img_transform, dic_bg_correction
from train_rotation_ver import predict_image
from skimage.io import imread
from skimage.transform import resize
import hj_util as util
import sys

def model(weight_file, mode = "reg_seg"):
    
    """Selecting the model to be used."""
    if mode == "reg_seg":
        model = reg_seg()
        print('loading wts file ' + weight_file)
        model.load_weights(weight_file)
        print('weight file loaded')
        return model
    elif mode =="unet":
        model = unet()
        model.load_weights(weight_file)
        return model

def patch_predict(input_folder, output_folder, weight_file):
    
    """
    This method has 4 different offsets. 100, 100, 77, 33.
    Then summing up the prediction.
    """
    
    autoencoder = model(weight_file)
    
    input_folder = util.folder_verify(input_folder)
    output_folder = util.folder_verify(output_folder)
    util.create_folder(output_folder)
    
    img_list = sorted(glob.glob(input_folder +"*"))
    for i in range(len(img_list)):
        print(img_list[i])
        img_name = os.path.basename(img_list[i])
        img = imread(img_list[i])
        img = img_transform(img) # This step normalize and change dimention of images
        output = predict_image(img, autoencoder)
        print('output dim: ', output.shape, output.sum())
        img = Image.fromarray(output[:, :, 0])
        img.save(output_folder + 'reg_' + img_name)

def single_predict(img_file, weight_file, model_mode = "reg_seg"):
    
    """
    This is a single picture prediction function. 
    model_model is a string, can be "reg_seg" or "unet". Default is reg_seg.
    """
    
    folder = os.path.dirname(img_file)
    name = os.path.basename(img_file)
    
    autoencoder = model(weight_file, model_mode)
    img = imread(img_file)
    
    x_adjust = int(round(img.shape[0]/32.0)*32)
    y_adjust = int(round(img.shape[1]/32.0)*32)       
    x_ori = img.shape[0]
    y_ori = img.shape[1]
    
    img = resize(img, (x_adjust, y_adjust), 
                 preserve_range = True, anti_aliasing = True)
    
    img = img_transform(img)
    output = autoencoder.predict(img[np.newaxis,:], batch_size=1, verbose=0)
    output = resize(output[0][:, :, 0], (x_ori, y_ori), 
                    preserve_range = True, anti_aliasing = True)
    
    print('output dim: ', output.shape)
    pred = Image.fromarray(output)
    pred.save(folder + "/" + 'reg_' + name)
    return output

def folder_edt_predict(input_folder, output_folder, weight_file, DIC_chan_label, model_mode):

    """
    Predict edt directly using the trained network (without a stacking prob. for patches).
    Resize the image to nearest 32x.
    After prediction, restore the picture size to original. 
    """
    
    autoencoder = model(weight_file, model_mode)
    
    input_folder = util.folder_verify(input_folder)
    output_folder = util.folder_verify(output_folder)

    edt_folder = output_folder + 'edt/'
    util.create_folder(edt_folder)
    
    img_list = sorted(glob.glob(input_folder + "*" + DIC_chan_label + "*"))

    for i in range(len(img_list)):
        img_name = os.path.basename(img_list[i])
        print(img_name)
        img = imread(img_list[i])

#        img0 = np.array(imread(img_list[0]))
#        bg0 = dic_bg_correction(img0, ordr=1)
#        bg = bg0 - np.mean(bg0)
#        img = img - bg

        x_adjust = int(round(img.shape[0]/32.0)*32)
        y_adjust = int(round(img.shape[1]/32.0)*32)
        x_ori = img.shape[0]
        y_ori = img.shape[1]
        img = resize(img, (x_adjust, y_adjust), 
                     preserve_range = True, anti_aliasing = True)

        img = img_transform(img) # This step normalize and change dimention of images
        predict_data = img[np.newaxis,:]
        output = autoencoder.predict(predict_data, batch_size=1, verbose=0) # the img adding a axis 
        output = resize(output[0][:, :, 0], (x_ori, y_ori), preserve_range = True, anti_aliasing = True) # restore to original size
        pred = Image.fromarray(output)
        pred.save(edt_folder + 'reg_' + img_name)

#    bg0 = Image.fromarray(bg0)
#    bg = Image.fromarray(bg)
#
#    bg0.save(edt_folder + 'bg0.tif')
#    bg.save(edt_folder + 'bg.tif')
    K.clear_session()

if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    img_path = sys.argv[1]
    print(img_path)
    output_path = sys.argv[2]
    print(output_path)  
    reg_seg_wts_file = sys.argv[3]
    print(reg_seg_wts_file) 
    DIC_chan_label = sys.argv[4]
    print(DIC_chan_label) 
    model_mode = sys.argv[5]
    print(model_mode) 
    folder_edt_predict(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])




# input folder
# output folder
# weight file
# multi_chan
# targ_chan
# model_mode
       
