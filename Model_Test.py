# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 17:27:23 2020

@author: junaid
"""

from keras.layers import Conv2D, Conv2DTranspose, ConvLSTM2D, TimeDistributed
from keras.models import Sequential, load_model
from keras.callbacks.callbacks import ModelCheckpoint, EarlyStopping
import cv2
from tqdm import tqdm
import numpy as np
import os
import pickle
import glob

def load_training_data(orig_frames, canned_frames, seq_size = 8):
  '''
  A function that will load the preprocessed images, first all the 
  processed images(orignal and edge detected) will be loaded to x_train,
  from there it will be batched(8 images...4orignal...4canned) into 'lst'. And
  this lst will be feed to the model in sequence for training.

  Parameters
  ----------
  orig_frames : string
    Name/path of the folder containing orignal processed images.
  canned_frames : string
    Name/path of the folder containing canny edged images.
  seq_size : integer, optional(fixed)
    This argument will decide the number of the images in the batchd,
    which in out case should be 8(4 orignal images, 4 canned images). The default is 8.

  Returns
  -------
  x_train : array of type 'floar32'
    Array of the all images combined, loaded from processed and canny edged images.
  lst : list
    A list that will contain 8 images per entery that will be feed to the model for 
    training.

  '''
  
  path = orig_frames
  loc = canned_frames
  
  processed_imgs = glob.glob(path+'/*.tif')
  cany_imgs = glob.glob(loc+'/*.tif')
  
  lst = []
  count = 0
  seq_size //= 2

  #Images will be read from the path and loaded into 'lst'
  for i in tqdm(range(len(processed_imgs)//seq_size)):
    seq = []
    for j in range(count, count+seq_size):
      seq.append(np.expand_dims(cv2.imread(processed_imgs[i], 2), axis = 2))
      seq.append(np.expand_dims((cv2.imread(cany_imgs[i], 2)/255), axis = 2))
      
    count += seq_size
    lst.append(seq)
  #A complete array of all the images combined
  x_test = np.array(lst)
  return x_test, lst

#Load the processed images and canny edged images
x_test, lst = load_training_data(orig_frames = 'Test_ProcessedImages', canned_frames = 'Test_CannyImages')

model = load_model('Autoencoder_final.h5')
model.summary()

result = []
for i in range (len(lst)):
  seq = np.array(lst[i]).reshape((1,8,128,128,1))
  result.append( model.predict(seq))

#is sy agy bta kia krna hy

