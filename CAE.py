from keras.layers import Input
from keras.layers import Conv2D, Conv2DTranspose, ConvLSTM2D, Conv3D, Reshape
from keras.models import Model
import numpy as np
import os
import cv2


############# Model Definition ######################

image = Input(shape = (128, 128, 1), name = 'Input_Images', batch_shape=(8,128,128,1))
#-------------------Encoder--------------------#
enc_conv = Conv2D(filters = 64, kernel_size = 5, strides = 4, padding = 'same', activation = 'relu')(image)

enc_conv = Reshape(target_shape=(1,32,32,64))(enc_conv)

enc_conv_lstm = ConvLSTM2D(filters = 64, kernel_size = (3, 3), padding = 'same', activation = 'relu',  return_sequences = True)(enc_conv)
encoder = ConvLSTM2D(filters = 32, kernel_size = (3, 3), padding = 'same', activation = 'relu', return_sequences = True)(enc_conv_lstm)

#-------------------Decoder--------------------#
dec_conv_lstm = ConvLSTM2D(filters = 64, kernel_size = (3, 3), padding = 'same', activation = 'relu', return_sequences = False)(encoder)
decoder = Conv2DTranspose(filters = 64, kernel_size = (5, 5), strides = 4, padding = 'same', activation = 'relu', name = 'Convolutional_Layer')(dec_conv_lstm)

#-------------------Building the Model--------------------#
autoencoder = Model(inputs = image, outputs = decoder)
autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy')

autoencoder.summary()
