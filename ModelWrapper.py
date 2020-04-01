# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 18:14:28 2020

@author: danis
"""

from keras.layers import Conv3D,ConvLSTM2D,Conv3DTranspose, Input
from keras.models import Model
import numpy as np

def BuildModel(input_shape=(227,227,10,1)):
    if len(input_shape) != 4 or type(input_shape) != tuple:
        raise ValueError('Invalid value given to the argument `input_shape`, it must be a `tuple` containing 4 values in this manner: (height, width, frames_per_input, channels)')
    input = Input(shape=input_shape)
    
    spatial_enc = Conv3D(filters=128, kernel_size=(11,11,1), strides=(4,4,1), padding='valid', activation='tanh')(input)

    spatial_enc = Conv3D(filters=64, kernel_size=(5,5,1), strides=(2,2,1), padding='valid', activation='tanh')(spatial_enc)



    temporal_enc = ConvLSTM2D(filters=64,kernel_size=(3,3),strides=1,padding='same',dropout=0.4,recurrent_dropout=0.3,return_sequences=True)(spatial_enc)

	
    temporal_enc = ConvLSTM2D(filters=32,kernel_size=(3,3),strides=1,padding='same',dropout=0.3,return_sequences=True)(temporal_enc)


    temporal_dec = ConvLSTM2D(filters=64,kernel_size=(3,3),strides=1,return_sequences=True, padding='same',dropout=0.5)(temporal_enc)




    spatial_dec = Conv3DTranspose(filters=128,kernel_size=(5,5,1),strides=(2,2,1),padding='valid',activation='tanh')(temporal_dec)
    spatial_dec = Conv3DTranspose(filters=1,kernel_size=(11,11,1),strides=(4,4,1),padding='valid',activation='tanh')(spatial_dec)

    model = Model(inputs=input, outputs=spatial_dec)
    model.summary()
    model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
    return model

def GetTrainData(name, re_shape=(-1,227,227,10)):
    if type(name)!=str:
        raise TypeError('Provide a valid name of `string` datatype, to the `.npy` file.')
    if '.npy' not in name:
        name += '.npy'
        
    X_train = np.load(name)
    frames = X_train.shape[2]
    #Need to make number of frames divisible by 10
    frames -= frames%10
    
    X_train=X_train[:,:,:frames]
    X_train=X_train.reshape(re_shape)
    X_train=np.expand_dims(X_train,axis=4)
    return X_train