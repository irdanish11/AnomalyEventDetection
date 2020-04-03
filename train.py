# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 19:06:43 2020

@author: danish
"""

import ModelWrapper as mp
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf
import glob
import pickle
import os
import numpy as np


def TrainModel(X_train, model, ckpt_name, hist_name, ckpt_path, epochs, batch_size):
    os.makedirs(ckpt_path, exist_ok=True)
    ckpt = ModelCheckpoint(ckpt_path+'/'+ckpt_name, monitor='val_loss', save_best_only=True)
    e_stop = EarlyStopping(monitor='val_loss', patience=10)
    History = model.fit(X_train, X_train, batch_size=batch_size, epochs=epochs, validation_split = 0.1, callbacks = [ckpt, e_stop])
    with open(ckpt_path+'/'+hist_name, "wb") as f:
        pickle.dump(History.history, f)
    return History.history

def main(config_gpu=True):
    #Setting up the GPU to avoid VRAM and other conflicts.
    #For refernce visit: https://github.com/irdanish11/AnomalyEventDetection/issues/1
    if config_gpu:
        mp.TF_GPUsetup()
    data = glob.glob('Train_Data/*.npy')
    history = []
    i=1
    for name in data:
        file = name.split('\\')[1]
        ################## Loading Dataset ################
        print('\n\n------- Loading the Dataset: {0} -------'.format(file))
        
        ######################## Building Model ##################
        X_train = mp.GetTrainData(name=name, re_shape=(-1, 227, 227, 10))
        print('Shape of X_train: {0}'.format(np.shape(X_train)))
        print('\n---------------- Building the Model! ----------------')
        tf.reset_default_graph()
        model = mp.BuildModel(input_shape=(227, 227, 10, 1))   
        
        ################## Training Model ####################
        ckpt_name = file.split('.')[0]+'_Model.h5'
        hist_name = file.split('.')[0]+'_History'
        print('\n\n\t________________________________________________\n')
        print('\t\tModel Name: {0}'.format(ckpt_name))
        print('\t\tTraining Model Number {0}/{1}'.format(i, len(data)))
        print('\t________________________________________________\n\n')
        i += 1
        hist = TrainModel(X_train, model, ckpt_name, hist_name, ckpt_path='./checkpoints', epochs=50, batch_size=32)
        history.append(hist)
    print('\n---------- Model Trained on All Datasets Sucessfully! ----------')        

if __name__=='__main__':
    main(config_gpu=False)

