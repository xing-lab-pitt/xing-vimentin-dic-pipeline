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
import hj_util as util
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
