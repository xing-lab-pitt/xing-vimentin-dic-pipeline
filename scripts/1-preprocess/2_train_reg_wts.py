from models import reg_seg
from cnn_prep_data import prep_train_data
from tensorflow.keras.callbacks import EarlyStopping

import sys
import hj_util
from tensorflow.keras.callbacks import EarlyStopping

train_input_path = sys.argv[1]
train_gt_path = sys.argv[2]
wts_path = sys.argv[3]
wts_file = sys.argv[4]

train_input_path = hj_util.folder_verify(train_input_path)
train_gt_path = hj_util.folder_verify(train_gt_path)
wts_path = hj_util.folder_verify(wts_path)

n_epoch = 120
batch_size = 16 # number of pictures to include for each epoch
patience = 45
res_best_wts = True
early_stopping = EarlyStopping(monitor='val_loss', min_delta = 0.001, patience = patience, mode = 'auto', restore_best_weights = res_best_wts)

print('batch_size = ' + str(batch_size) + ', patience = ' + str(patience) + ', restore_best_weights = ' + str(res_best_wts))

autoencoder=reg_seg()

train_data, train_label = prep_train_data(train_input_path,train_gt_path)

train_label.shape

train_data.shape

history=autoencoder.fit(train_data, train_label, epochs=n_epoch, validation_split = 0.20, batch_size=batch_size, verbose=2,\
        callbacks=[early_stopping])
#history=autoencoder.fit(train_data, train_label, epochs=n_epoch, validation_split = 0.25, batch_size=batch_size, verbose=2)

autoencoder.save_weights(wts_path + wts_file)

