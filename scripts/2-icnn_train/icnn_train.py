# In[0]: import
import sys

sys.path.insert(0, "C:/Users/14432/OneDrive/Research/Projects/A549_144hr/scripts/memes/")

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cv2
from matplotlib import pyplot as plt
import numpy as np
from os import listdir

import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# from keras.optimizers import adam
from skimage.exposure import equalize_adapthist, rescale_intensity
from skimage.filters import gaussian
from skimage.io import imread
from sklearn.utils import class_weight
import tensorflow as tf
from keras.optimizers import adam
from tensorflow.python.client import device_lib

from cnn_prep_data import prep_icnn_am_train_data, prep_icnn_seg_train_data
import utils
from resnet50 import res_model
from unsharp_mask import unsharp_mask

# In[1]: initialize
train_path = sys.argv[1]
wts_path = sys.argv[2]
wts_file = sys.argv[3]

obj_h = int(sys.argv[4])
obj_w = int(sys.argv[5])
nb_class = int(sys.argv[6])

train_mode = sys.argv[7]

train_path = utils.correct_folder_str(train_path)
wts_path = utils.correct_folder_str(wts_path)

print(device_lib.list_local_devices())

input_shape = (obj_h, obj_w, 1)
model = res_model(input_shape, nb_class)
model.summary()

# data_gen = ImageDataGenerator()


if train_mode == "am":
    train_data, train_label = prep_icnn_am_train_data(train_path, obj_h, obj_w)
    data_gen = ImageDataGenerator(rotation_range=180, zoom_range=(1, 0.95), horizontal_flip=True, vertical_flip=True)
elif train_mode == "seg":
    train_data, train_label = prep_icnn_seg_train_data(train_path, obj_h, obj_w)
    data_gen = ImageDataGenerator(rotation_range=180, zoom_range=(1, 1.25), horizontal_flip=True, vertical_flip=True)
print(train_data.shape, train_label.shape)
print(train_label)

ylabel = np.argmax(train_label, axis=1)
label_weight = class_weight.compute_class_weight("balanced", np.unique(ylabel), ylabel.flatten())
print(label_weight)


def class_weighted_crossentropy(class_weight):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 10e-8, 1.0 - 10e-8)
        return -tf.reduce_sum(y_true * class_weight * tf.log(y_pred))

    return loss


# In[2]: main
opt = adam(lr=0.0015)
model.compile(loss=class_weighted_crossentropy(label_weight), optimizer=opt, metrics=["accuracy"])

batch_size = 50
nb_epoch = 1000
history = model.fit_generator(
    data_gen.flow(train_data, train_label, batch_size=batch_size),
    steps_per_epoch=len(train_data) / batch_size,
    epochs=nb_epoch,
    verbose=1,
)
model.save_weights(wts_path + wts_file)
