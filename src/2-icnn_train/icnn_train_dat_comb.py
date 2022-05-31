# In[0]: import
import sys

sys.path.insert(0, "C:/Users/14432/OneDrive/Research/Projects/A549_144hr/src/memes/")

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from os import listdir

import cv2
import keras
import numpy as np
import tensorflow as tf
import utils
from cnn_prep_data import (keep_aspect_resize, obj_transform,
                           prep_icnn_am_train_data, prep_icnn_seg_train_data)
from keras.optimizers import adam
from keras.preprocessing.image import (ImageDataGenerator, array_to_img,
                                       img_to_array, load_img)
from matplotlib import pyplot as plt
from resnet50 import res_model
from skimage.exposure import equalize_adapthist, rescale_intensity
from skimage.filters import gaussian
from skimage.io import imread
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.client import device_lib
from unsharp_mask import unsharp_mask

# In[1]: define

print(device_lib.list_local_devices())

train_path = sys.argv[1]
wts_path = sys.argv[2]
wts_file = sys.argv[3]

obj_h = int(sys.argv[4])
obj_w = int(sys.argv[5])
nb_class = int(sys.argv[6])

train_mode = sys.argv[7]

train_path = utils.correct_folder_str(train_path)
wts_path = utils.correct_folder_str(wts_path)

input_shape = (obj_h, obj_w, 1)
model = res_model(input_shape, nb_class)
model.summary()

if train_mode == "am":
    train_data, train_label = prep_icnn_am_train_data(train_path, obj_h, obj_w)
    train_datagen = ImageDataGenerator(
        rotation_range=180, zoom_range=(1, 0.95), horizontal_flip=True, vertical_flip=True, validation_split=0.2
    )
elif train_mode == "seg":
    train_data, train_label = prep_icnn_seg_train_data(train_path, obj_h, obj_w)
    train_datagen = ImageDataGenerator(
        rotation_range=180, zoom_range=(1, 1.25), horizontal_flip=True, vertical_flip=True, validation_split=0.2
    )
print(train_data.shape, train_label.shape)
print(train_label)


def class_weighted_crossentropy(class_weight):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 10e-8, 1.0 - 10e-8)
        return -tf.reduce_sum(y_true * class_weight * tf.log(y_pred))

    return loss


ylabel = np.argmax(train_label, axis=1)
label_weight = class_weight.compute_class_weight("balanced", np.unique(ylabel), ylabel.flatten())

# In[2]: main
opt = adam(lr=0.0015)
model.compile(loss=class_weighted_crossentropy(label_weight), optimizer=opt, metrics=["accuracy"])

nb_epoch = 120
batch_size = 150
patience = 45
early_stopping = EarlyStopping(
    monitor="val_loss", min_delta=0.001, patience=patience, mode="auto", restore_best_weights=True
)

train_generator = train_datagen.flow(
    train_data, train_label, batch_size=batch_size, subset="training"
)  # set as training data

validation_generator = train_datagen.flow(
    train_data, train_label, batch_size=batch_size, subset="validation"  # same directory as training data
)  # set as validation data

model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_data) / batch_size,
    validation_data=validation_generator,
    validation_steps=len(train_data) / batch_size,
    epochs=nb_epoch,
    verbose=2,
    callbacks=[early_stopping],
)

model.save_weights(wts_path + wts_file)
