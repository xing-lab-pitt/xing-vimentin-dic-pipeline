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
import utils as util
import sys
from tqdm import tqdm


def assemble_model(weight_file, mode="reg_seg"):
    """Selecting the model to be used."""
    if mode == "reg_seg":
        model = reg_seg()
        print("loading wts file " + weight_file)
        model.load_weights(weight_file)
        print("weight file loaded")
        return model
    elif mode == "unet":
        model = unet()
        model.load_weights(weight_file)
        return model


def patch_predict(input_folder, output_folder, weight_file):

    """
    This method has 4 different offsets. 100, 100, 77, 33.
    Then summing up the prediction.
    """

    autoencoder = assemble_model(weight_file)

    input_folder = util.correct_folder_str(input_folder)
    output_folder = util.correct_folder_str(output_folder)
    util.create_folder(output_folder)

    img_list = sorted(glob.glob(input_folder + "*"))
    for i in range(len(img_list)):
        print(img_list[i])
        img_name = os.path.basename(img_list[i])
        img = imread(img_list[i])
        img = img_transform(img)  # This step normalize and change dimention of images
        output = predict_image(img, autoencoder)
        print("output dim: ", output.shape)

        img = Image.fromarray(output[:, :, 0])
        img.save(output_folder + "reg_" + img_name)


def predict_single_image(img_file, weight_file, model_mode="reg_seg"):

    """
    This is a single picture prediction function.
    model_model is a string, can be "reg_seg" or "unet". Default is reg_seg.
    """

    folder = os.path.dirname(img_file)
    name = os.path.basename(img_file)

    autoencoder = assemble_model(weight_file, model_mode)
    img = imread(img_file)

    x_adjust = int(round(img.shape[0] / 32.0) * 32)
    y_adjust = int(round(img.shape[1] / 32.0) * 32)
    x_ori = img.shape[0]
    y_ori = img.shape[1]

    img = resize(img, (x_adjust, y_adjust), preserve_range=True, anti_aliasing=True)

    img = img_transform(img)
    output = autoencoder.predict(img[np.newaxis, :], batch_size=1, verbose=0)
    output = resize(output[0][:, :, 0], (x_ori, y_ori), preserve_range=True, anti_aliasing=True)

    print("output dim: ", output.shape)
    pred = Image.fromarray(output)
    pred.save(folder + "/" + "reg_" + name)
    return output


def folder_edt_predict(img_path, output_path, reg_seg_wts_path, dic_channel_label, model_mode):
    """Predicts edt for all DIC images in a folder directly using the trained
    network.

    Resize the image to nearest 32x with optional background correction. After
    prediction, restore the predicted EDT images to original size.

    Args:
        img_path: Path string to the folder that contains the DIC images.
        output_path: Path string to the folder where an 'edt' sulfolder is
        created. Predicted images are saved in the folder 'output_folder/edt/'
        reg_seg_wts_path:filepath string to the trained CNN weights.
        dic_channel_label: String that denotes the DIC channel's labeling
        model_mode: String that denotes the specific network architecture to be
        loaded, described in models.py.
    """

    autoencoder = assemble_model(reg_seg_wts_path, model_mode)

    img_path = util.correct_folder_str(img_path)
    output_path = util.correct_folder_str(output_path)

    edt_folder = output_path + "edt/"
    util.create_folder(edt_folder)

    img_list = sorted(glob.glob(img_path + "*" + dic_channel_label + "*"))
    print("img path:", img_path)
    print("image list:", img_list)
    for i in tqdm(range(len(img_list))):
        img_name = os.path.basename(img_list[i])
        print(img_name)
        img = imread(img_list[i])

        #        img0 = np.array(imread(img_list[0]))
        #        bg0 = dic_bg_correction(img0, ordr=1)
        #        bg = bg0 - np.mean(bg0)
        #        img = img - bg

        x_adjust = int(round(img.shape[0] / 32.0) * 32)
        y_adjust = int(round(img.shape[1] / 32.0) * 32)
        x_ori = img.shape[0]
        y_ori = img.shape[1]
        img = resize(img, (x_adjust, y_adjust), preserve_range=True, anti_aliasing=True)

        img = img_transform(img)  # This step normalize and change dimention of images
        predict_data = img[np.newaxis, :]
        output = autoencoder.predict(predict_data, batch_size=1, verbose=0)  # the img adding a axis
        output = resize(
            output[0][:, :, 0], (x_ori, y_ori), preserve_range=True, anti_aliasing=True
        )  # restore to original size
        pred = Image.fromarray(output)
        pred.save(edt_folder + "reg_" + img_name)

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
    reg_seg_wts_path = sys.argv[3]
    print(reg_seg_wts_path)
    dic_channel_label = sys.argv[4]
    print(dic_channel_label)
    model_mode = sys.argv[5]
    print(model_mode)
    folder_edt_predict(img_path, output_path, reg_seg_wts_path, dic_channel_label, model_mode)
