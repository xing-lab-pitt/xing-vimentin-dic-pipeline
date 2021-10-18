import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from os import listdir
import cv2
from skimage.io import imread
import numpy as np
from resnet50 import res_model
import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.filters import gaussian
from unsharp_mask import unsharp_mask
from skimage.exposure import equalize_adapthist, rescale_intensity
from matplotlib import pyplot as plt
from sklearn.utils import class_weight
import tensorflow as tf
from cnn_prep_data import prep_icnn_am_train_data
import pickle


from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())


def class_weighted_crossentropy(class_weight):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 10e-8, 1.0 - 10e-8)
        return -tf.reduce_sum(y_true * class_weight * tf.log(y_pred))

    return loss


def icnn_am_train(train_folder, weight_file, img_r=128, img_c=128, batch_size=64, n_epoch=100):

    """
    train_folder - string, parent folder for trainign data
    weight_file - string, file path
    img_r - int
    img_c - int

    Note, the class is curresponding to that defined in cnn_prep.py"""
    input_shape = (img_r, img_c, 1)
    nb_class = 3
    model = res_model(input_shape, nb_class)
    model.summary()

    data_gen = ImageDataGenerator(rotation_range=180, zoom_range=(1, 1.25), horizontal_flip=True, vertical_flip=True)
    train_data, train_label = prep_icnn_am_train_data(train_folder, img_r, img_c)
    # print(train_data.shape,train_label.shape)
    # print(train_label)
    ylabel = np.argmax(train_label, axis=1)
    label_weight = class_weight.compute_class_weight("balanced", np.unique(ylabel), ylabel.flatten())
    # print(label_weight)

    model.compile(loss=class_weighted_crossentropy(label_weight), optimizer="adam", metrics=["accuracy"])

    history = model.fit_generator(
        data_gen.flow(train_data, train_label, batch_size=batch_size),
        steps_per_epoch=len(train_data) / batch_size,
        epochs=n_epoch,
        verbose=1,
    )

    # save the training weight
    model.save_weights(weight_file)
    # save the training history
    with open(weight_file[:-5] + ".pkl", "wb") as file_pi:
        pickle.dump(history.history, file_pi)


if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    # arg1 is train_folder
    # arg2 is weight_file

    icnn_am_train(sys.argv[1], sys.argv[2])
