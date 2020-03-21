# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 16:02:53 2020

@author: junaid
"""

from keras.layers import Input, Reshape
from keras.layers import Conv2D, Conv2DTranspose, ConvLSTM2D
from keras.models import Model
import cv2
from PreProcessing import ReadFileNames
from tqdm import tqdm


def load_training_data(orig_frames, canned_frames):
  
  path = orig_frames
  loc = canned_frames
  
  onlyfiles, file_names, dirs = ReadFileNames(loc)
  x_train = []

  for i in tqdm(range(len(onlyfiles))):
    images = onlyfiles[i]
    for y in range(len(images)):
      img_name = path+'/'+dirs[i]+'/'+file_names[i][y]
      orig_frame = cv2.imread(img_name, 2)
      x_train.append(orig_frame)
      img_name = loc+'/'+dirs[i]+'/'+file_names[i][y]
      orig_name = cv2.imread(img_name, 0)
      x_train.append(orig_name)
    
  return x_train

x_train = load_training_data(orig_frames = 'ProcessedImages', canned_frames = 'CannyImages')


############# Model Definition ######################

image = Input(shape = (128, 128, 1), name ='Input_Images', batch_shape = (8, 128, 128, 1))

#-------------------Encoder--------------------#

enc_conv = Conv2D(filters = 64, kernel_size = (5, 5), strides = 4, padding = 'same', activation = 'relu', kernel_initializer = 'RandomNormal')(image)

enc_conv = Reshape(target_shape = (1, 32, 32, 64))(enc_conv)

enc_conv_lstm = ConvLSTM2D(filters = 64, kernel_size = (3, 3), padding = 'same', activation = 'relu', kernel_initializer = 'RandomNormal', return_sequences = True)(enc_conv)
encoder = ConvLSTM2D(filters = 32, kernel_size = (3, 3), padding = 'same', activation = 'relu', kernel_initializer = 'RandomNormal', return_sequences = True)(enc_conv_lstm)

#-------------------Decoder--------------------#

dec_conv_lstm = ConvLSTM2D(filters = 64, kernel_size = (3, 3), padding = 'same', activation = 'relu', kernel_initializer = 'RandomNormal', return_sequences = False)(encoder)
decoder = Conv2DTranspose(filters = 64, kernel_size = (5, 5), strides = 4, padding = 'same', activation = 'relu', kernel_initializer = 'RandomNormal', name = 'Deconvolutional_Layer')(dec_conv_lstm)

#-------------------Building the Model--------------------#

autoencoder = Model(inputs = image, outputs = decoder)
autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy')

autoencoder.summary()

autoencoder.fit(x_train, x_train, batch_size = 16, epochs = 50, validation_split = 0.15)
