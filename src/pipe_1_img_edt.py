from PIL import Image
from os import listdir
import glob
import os
import numpy as np
import keras.backend as K

from models import reg_seg, unet
from cnn_prep_data import prep_dic_data, img_transform
from train_rotation_ver import predict_image
from skimage.io import imread
from skimage.transform import resize
import pipe_util2 as util
import sys


def model(weight_file, mode="reg_seg"):

    """Selecting the model to be used."""
    if mode == "reg_seg":
        model = reg_seg()
        model.load_weights(weight_file)
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

    autoencoder = model(weight_file)

    input_folder = util.folder_verify(input_folder)
    output_folder = util.folder_verify(output_folder)
    util.create_folder(output_folder)

    img_list = sorted(glob.glob(input_folder + "*"))
    for i in range(len(img_list)):
        print(img_list[i])
        img_name = os.path.basename(img_list[i])
        img = imread(img_list[i])
        img = img_transform(img)  # This step normalize and change dimention of images
        output = predict_image(img, autoencoder)
        print("output dim: ", output.shape, output.sum())
        img = Image.fromarray(output[:, :, 0])
        img.save(output_folder + "reg_" + img_name)


def single_predict(img_file, weight_file, model_mode="reg_seg"):

    """
    This is a single picture prediction function.
    model_model is a string, can be "reg_seg" or "unet". Default is reg_seg.
    """

    folder = os.path.dirname(img_file)
    name = os.path.basename(img_file)

    autoencoder = model(weight_file, model_mode)
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


def folder_edt_predict(input_folder, output_folder, weight_file, model_mode):
    """
    Predict edt directly using the trained network (without a stacking prob. for patches).
    Resize the image to nearest 32x.
    After prediction, restore the picture size to original.
    """

    autoencoder = model(weight_file, model_mode)

    input_folder = util.folder_verify(input_folder)
    output_folder = util.folder_verify(output_folder) + "edt/"
    util.create_folder(output_folder)

    img_list = sorted(glob.glob(input_folder + "*"))
    for i in range(len(img_list)):
        print(img_list[i])
        img_name = os.path.basename(img_list[i])
        img = imread(img_list[i])
        # print(img.max(), img.min())

        x_adjust = int(round(img.shape[0] / 32.0) * 32)
        y_adjust = int(round(img.shape[1] / 32.0) * 32)
        x_ori = img.shape[0]
        y_ori = img.shape[1]

        img = resize(img, (x_adjust, y_adjust), preserve_range=True, anti_aliasing=True)

        img = img_transform(img)  # This step normalize and change dimention of images
        # print(img.shape, img.max(), img.min())

        # if image is (512, 512
        # After img_transform, it will be (512, 512, 1)
        # My training data set might has the shape (17694, 32, 32, 1)
        # Thus adding one more dimention at the front is needed.

        # print(img.shape, img[np.newaxis,:])
        # return
        output = autoencoder.predict(img[np.newaxis, :], batch_size=1, verbose=0)  # the img adding a axis
        print(output.max(), output.min())

        output = resize(output[0][:, :, 0], (x_ori, y_ori), preserve_range=True, anti_aliasing=True)
        print(output.max(), output.min())
        print("output dim: ", output.shape)
        pred = Image.fromarray(output)
        pred.save(output_folder + "reg_" + img_name)
    K.clear_session()


# input folder
# output folder
# weight file
# model_mode
if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    folder_edt_predict(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
