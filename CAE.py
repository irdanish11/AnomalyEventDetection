# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 16:02:53 2020

@author: junaid
"""

from keras.layers import Conv2D, Conv2DTranspose, ConvLSTM2D, TimeDistributed
from keras.callbacks.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from preprocessing import load_data
import pickle
import os




x_train, lst = load_data(orig_frames = 'ProcessedImages', canned_frames = 'CannyImages')


############# Model Definition ######################
autoencoder = Sequential()
autoencoder.add(TimeDistributed(Conv2D(filters = 64, kernel_size = (5, 5), strides =4 , padding = 'same', activation = 'relu', kernel_initializer = 'RandomNormal', data_format="channels_last"), input_shape = (8, 128, 128, 1)))
autoencoder.add(ConvLSTM2D(filters = 64, kernel_size = (3, 3), padding = 'same', activation = 'relu', kernel_initializer = 'RandomNormal', return_sequences = True))
autoencoder.add(ConvLSTM2D(filters = 32, kernel_size = (3, 3), padding = 'same', activation = 'relu', kernel_initializer = 'RandomNormal', return_sequences = True))

autoencoder.add(ConvLSTM2D(filters = 64, kernel_size = (3, 3), padding = 'same', activation = 'relu', kernel_initializer = 'RandomNormal', return_sequences = True))
autoencoder.add(TimeDistributed(Conv2DTranspose(filters = 1, kernel_size = (5, 5), strides = 4, padding = 'same', activation = 'relu', kernel_initializer = 'RandomNormal', name = 'Deconvolutional_Layer')))


autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy')

autoencoder.summary()
ckpt_path='./checkpoints'
os.makedirs(ckpt_path, exist_ok=True)
ckpt = ModelCheckpoint(ckpt_path+'/Autoencoder.h5', monitor='val_loss', save_best_only=True)
e_stop = EarlyStopping(monitor='val_accuracy', patience=3, min_delta=0.01)
History = autoencoder.fit(x_train, x_train, batch_size = 16, epochs = 50, validation_split = 0.15, callbacks=[ckpt, e_stop])
pickle.dump(History, 'ModelHistory')

