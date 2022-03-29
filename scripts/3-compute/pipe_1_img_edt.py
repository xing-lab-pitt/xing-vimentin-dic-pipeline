'''Predicts the Euclidean Distance Transform(EDT) of the DIC channel images using trained CNN weights.

Load the same CNN model used to train the weights in 1-preprocessing(usually reg_seg is used). Reads all images in
a single folder sequentially and predict their EDT one-by-one using the CNN model. Each image is background-corrected
(optional) and resized to suit the dimensions of the CNN filters(32x) - it was observed this gives more accurate
results.
The predicted results are then resized back to their original shapes.

    Called by run_pipe_0-2_receiver.sh, example usage:

	img_path=~/data/ori/XY1
	output_path=~/data/out/XY1
	reg_wts_file=~/wts/reg/edt_weights.hdf5 
    DIC_chan_label=C1
    model_mode='reg_seg'

    python pipe_1_img_edt.py $img_path $output_path $reg_seg_wts_file $DIC_chan_label $model_mode 
'''

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

def load_model(weight_file, mode = "reg_seg"):
    """Fetches a model from models.py.
    
    Loads the CNN model along with the trained weights. Network architecture
    depends on the mode.
    
    Args:
        weight_file: File-path string to the trained CNN weights.
        mode: String that denotes the specific network architecture to be
        loaded, described in models.py.
        
    Returns:
        model: Loaded CNN model. 
    """

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

def folder_edt_predict(input_folder, output_folder, weight_file, DIC_chan_label, model_mode):
    """Predicts edt for all DIC images in a folder directly using the trained
    network.

    Resize the image to nearest 32x with optional background correction. After
    prediction, restore the predicted EDT images to original size. 
    
    Args:
        input_folder: Path string to the folder that contains the DIC images.
        output_folder: Path string to the folder where an 'edt' sulfolder is
        created. Predicted images are saved in the folder 'output_folder/edt/'
        weight_file: File-path string to the trained CNN weights.
        DIC_chan_label: String that denotes the DIC channel's labeling
        model_mode: String that denotes the specific network architecture to be
        loaded, described in models.py.
    """

    autoencoder = load_model(weight_file, model_mode)
    input_folder = util.folder_verify(input_folder)
    output_folder = util.folder_verify(output_folder)
    edt_folder = output_folder + 'edt/'
    util.create_folder(edt_folder)
    
    img_list = sorted(glob.glob(input_folder + "*" + DIC_chan_label + "*"))

    for i in range(len(img_list)):
        img_name = os.path.basename(img_list[i])
        print(img_name)
        img = imread(img_list[i])

        # The following block is an optional background-correction on the DIC
        # image. Originally, the correction was done on single tile images to
        # improve accuracy. However, recent experiments have been returning
        # larger images that may not be suitable with this method of background
        # correction. Consider commenting out this part or find a different 
        # method of background-correction.
    #    img0 = np.array(imread(img_list[0])) 
    #    bg0 = dic_bg_correction(img0, ordr=1)
    #    bg = bg0 - np.mean(bg0)
    #    img = img - bg

        # resizes the image dimensions to the nearest 32x. It was found that 
        # non-resized images returned erroneous results. This could be due to
        # the dimensions of the CNN's filters being 64x.
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

    # Saves the previously calculated backgrounds.
#    bg0 = Image.fromarray(bg0)
#    bg = Image.fromarray(bg)
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