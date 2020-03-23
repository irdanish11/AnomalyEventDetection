# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 16:02:53 2020

@author: junaid
"""

from keras.layers import Conv2D, Conv2DTranspose, ConvLSTM2D, TimeDistributed
from keras.callbacks.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential
import cv2
from tqdm import tqdm
import numpy as np
import glob
import pickle
import os

def load_training_data(orig_frames, canned_frames, seq_size=8):
  path = orig_frames
  loc = canned_frames
  processed_imgs = glob.glob(path+'/*.tif')
  cany_imgs = glob.glob(loc+'/*.tif')
  lst = []
  count = 0
  seq_size //= 2
  for i in tqdm(range(len(processed_imgs)//seq_size)):
      seq = []
      for j in range(count, count+seq_size):
          seq.append(np.expand_dims(cv2.imread(processed_imgs[i], 2), axis = 2))
          seq.append(np.expand_dims(cv2.imread(cany_imgs[i], 2), axis = 2))
      count += seq_size
      lst.append(seq)
  x_train = np.array(lst)
  return x_train, lst


x_train, lst = load_training_data(orig_frames = 'ProcessedImages', canned_frames = 'CannyImages')


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

