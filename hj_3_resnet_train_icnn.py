import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from os import listdir
import cv2
from skimage.io import imread
import numpy as np
from resnet50 import res_model
import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.filters import gaussian
from unsharp_mask import unsharp_mask
from matplotlib import pyplot as plt
from skimage import morphology
from skimage.measure import label,regionprops
from scipy import ndimage
from sklearn.utils import class_weight
import tensorflow as tf
from skimage.exposure import equalize_adapthist,rescale_intensity
from cnn_prep_data import prep_icnn_seg_train_data


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


restrain_folder = "/net/capricorn/home/xing/huijing/Segmentation/data/1-13-Incucyte/ch00_B4_3_restrain"
weight_file='DLD1_icnn_seg.hdf5'


obj_h=128
obj_w=128
input_shape=(obj_h,obj_w,1)
nb_class=3
model=res_model(input_shape,nb_class)
model.summary()


data_gen = ImageDataGenerator(zoom_range=(1,1.25),
    rotation_range=180,
    horizontal_flip=True,
    vertical_flip=True)


classes = ["one_cell_", "wrong_seg_", "multi_"]
train_data,train_label=prep_icnn_seg_train_data(restrain_folder,obj_h,obj_w, classes)
print(train_data.shape,train_label.shape)


def class_weighted_crossentropy(class_weight):  
    def loss(y_true, y_pred):
        y_pred  = tf.clip_by_value(y_pred, 10e-8, 1.-10e-8)
        return -tf.reduce_sum(y_true * class_weight * tf.log(y_pred))
    return loss



ylabel=np.argmax(train_label,axis=1)
label_weight=class_weight.compute_class_weight('balanced', np.unique(ylabel), ylabel.flatten())
print(label_weight)



model.compile(loss=class_weighted_crossentropy(label_weight), optimizer='adam', metrics=["accuracy"])



batch_size = 50
nb_epoch=500
history =model.fit_generator(data_gen.flow(train_data,train_label,batch_size=batch_size),
                                     steps_per_epoch=len(train_data)/batch_size, epochs=nb_epoch, verbose=1)

model.get_weights()
model.save_weights(weight_file)




