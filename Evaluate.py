# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 16:52:11 2020

@author: danish
"""

from matplotlib import pyplot as plt
import pickle

def PlotHistory(history, name, show=True, save=False, path=None):
    plt.clf()
    plt.ioff()
    # Plot training & validation accuracy values
    plt.figure(1)
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Model accuracy - '+name)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training Accuracy', 'Val Accuracy'], loc='upper left')
    if save:
        if path==None:
            raise ValueError('Path cannot be None when `save` is set to True, please provide valid path.')
        plt.savefig(path+'/'+name+'_Accuracy.png')
    if show:
        plt.show()
    # Plot training & validation loss values
    plt.figure(2)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss - '+name)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training Loss', 'Val Loss'], loc='upper right')
    if save:
        if path==None:
            raise ValueError('Path cannot be None when `save` is set to True, please provide valid path.')
        plt.savefig(path+'/'+name+'_Loss.png')
    if show:
        plt.show()


with open('TrainedSummary', 'rb') as f:
    hist = pickle.load(f)

PlotHistory(hist, name='UCSD PED1', save=True, path='./')
