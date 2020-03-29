# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 17:27:23 2020

@author: junaid
"""


from keras.models import load_model
from preprocessing import load_data
from sklearn.metrics import mean_squared_error as mse
from tqdm import tqdm

#Load the processed images and canny edged images
x_test, lst = load_data(orig_frames = 'ProcessedImages', canned_frames = 'CannyImages')

model = load_model('Autoencoder_final.h5')
#model.summary()

result = []
losses = []
classes = []
#Loss threshold was selected on the basis training data: The maximum loss in the training data was 0.021
#We made it .005 higher than the max value.
loss_th = 0.026
print('\nCalculating loss for the given inputs. \n')
for i in tqdm(range(len(x_test))):
    seq = x_test[i].reshape((1,8,128,128,1))
    pred = model.predict(seq)
    pred = pred.reshape((8,128,128))
    seq = seq.reshape((8,128,128))
    loss = []
    for i in range(len(seq)):
        l = mse(seq[i],pred[i])
        if l >= loss_th:
            classes.append('Anomly Frame')
        else:
            classes.append('Normal Frame')
        loss.append(l)
        losses.append(l)
    result.append(loss)
    
max(losses)
#is sy agy bta kia krna hy
