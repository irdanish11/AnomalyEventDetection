# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 20:56:50 2020

@author: danish
"""
import cv2
import ModelWrapper as mp
from tensorflow.keras.models import load_model
from preprocessing import Fit_Preprocessing, GlobalNormalization, ToJson
from preprocessing import ReadFileNames
import numpy as np
import tensorflow as tf


print('\nTensorflow GPU installed: '+str(tf.test.is_built_with_cuda()))
print('Is Tensorflow using GPU: '+str(tf.test.is_gpu_available()))


def WriteInfo(err, text, norm_count, anom_count):
    mp.PrintInline('{4}, Frame Status: {0}, Normal Frame Count: {1}/{2}, Anomaly Frame Count {3}/{2}'.format(text, norm_count, norm_count+anom_count, anom_count, err))

def get_model(model_path):
    print('\n\n------- Loading Model: {0} ! -------'.format(model_path.split('/')[-1]))
    print('\n---------------  This may take a while!  ---------------\n\n')
    model=load_model(model_path)
    print('\n\n------- Model Loaded! {0} ! -------\n\n'.format(model_path.split('/')[-1]))
    return model

def RealTimeDetection(model, threshold, serve_type='real-time', vid_path=None, verbose=True):
    if serve_type=='real-time':
        cap=cv2.VideoCapture(0)
    elif serve_type=='video':
        if vid_path==None:
            raise TypeError('Value of `vid_path` argument cannot be `None`, when `serve_type` value is `video`. Provide valid path of `str` datatype.')
        cap=cv2.VideoCapture(vid_path)
        
    _,frame=cap.read()
    shape = np.shape(frame)
    ret=True
    norm_count = 0
    anom_count = 0
    test_history = {'Serving Type':serve_type, 'Loss':[], 'Normal Frames': [], 
                    'Anomaly Frames':[], 'Total Frames':[]}
    print('\n\n------- Press q to exit the Real Time Detection! -------\n')
    while(cap.isOpened()):
        img_lst=[]
        v_frames = np.zeros(shape=(10, shape[0], shape[1], shape[2]), dtype=np.uint8)
        for i in range(10):
            ret,frame=cap.read()
            if (ret != True):
                cv2.destroyAllWindows()
                raise EOFError('The Video is Completed Succefully!')
            #copy the orignal frame for display.
            v_frames[i]=frame
            gray = mp.ImgProcess(frame, shape=(227,227))
            img_lst.append(gray)
        img_arr = mp.Img_LstArr(img_lst, re_shape=(227, 227, 10))
        #making prediction
        pred = model.predict(img_arr)
        #computing error
        loss = mp.MSE(img_arr, pred)
        err = 'Loss: {0:.5f}'.format(loss)
        if ret==True:
            test_history['Loss'].append(loss); test_history['Normal Frames'].append(norm_count)
            test_history['Anomaly Frames'].append(anom_count)
            test_history['Total Frames'].append(norm_count+anom_count)
            ToJson(test_history, name='Test History.json')
            if loss>threshold:
                anom_count += 10
                text='Anomalies Detected'
                for j in range(len(v_frames)):
                    mp.ShowVideo(cap, v_frames[j], text)
                if verbose:
                    WriteInfo(err, text, norm_count, anom_count)
            else:
                text='Normal'
                norm_count += 10
                for j in range(len(v_frames)):
                    mp.ShowVideo(cap, v_frames[j], text)
                if verbose:
                    WriteInfo(err, text, norm_count, anom_count)



def StaticServing(path, model, threshold, frames_ext, serve_type='frames', verbose=True):
    if serve_type=='frames':
        onlyfiles, _, _ = ReadFileNames(path, frames_ext)
        all_files = mp.ListCopy(onlyfiles)
        num = 10
        ten_list = np.reshape(all_files, (len(all_files)//num, num))
        img_lst = Fit_Preprocessing(path, frames_ext)
        X_test = GlobalNormalization(img_lst, save_data=False)    
    elif serve_type=='npy':
        X_test = np.load(path)
        
    X_test = mp.PrepareData(X_test)
    norm_count = 0
    anom_count = 0
    test_history = {'Serving Type':serve_type, 'Loss':[], 'Normal Frames': [], 
                    'Anomaly Frames':[], 'Total Frames':[]}
    print('\n\t------------- Now Serving will begin! -------------\n\n')
    for number, bunch in enumerate(X_test):
        #Reshaping batch to 5 dimensions
        batch = np.expand_dims(bunch,axis=0)
        pred_batch = model.predict(batch)
        #computing loss
        loss = mp.MSE(batch, pred_batch)
        err = 'Loss: {0:.5f}'.format(loss)
        test_history['Loss'].append(loss); test_history['Normal Frames'].append(norm_count)
        test_history['Anomaly Frames'].append(anom_count)
        test_history['Total Frames'].append(norm_count+anom_count)
        ToJson(test_history, name='Test History.json')
        if loss>threshold:
            anom_count += 10
            text='Anomalies Detected'
            if serve_type=='frames':
                for j in range(len(ten_list[number])):
                    v_frame = cv2.imread(ten_list[number][j])
                    cap=None
                    mp.ShowVideo(cap, v_frame, text)
            if verbose:
                WriteInfo(err, text, norm_count, anom_count)
        else:
            text='Normal'
            norm_count += 10
            if serve_type=='frames':
                for j in range(len(ten_list[number])):
                    v_frame = cv2.imread(ten_list[number][j])
                    cap=None
                    mp.ShowVideo(cap, v_frame, text)
            if verbose:
                WriteInfo(err, text, norm_count, anom_count)
    print('\n\t------------- Serving is Completed! -------------\n\n')
    return test_history

def DeploySystem(serve_type, model_path, preset_threshold=True, data_model=None, verbose=True, path=None, frames_ext=None, threshold=None, config_gpu=False):
    serving_types = ['real-time', 'video', 'frames', 'npy']
    if preset_threshold:
        if threshold is not None:
            raise TypeError('Invalid value given to argument `threshold`, its value must be None when `preset_threshold` argument is set to True.')
        if data_model=='UCSD':
            threshold=0.00026
        elif data_model=='Avenue':
            threshold=0.00040
        else:
            raise ValueError('Invalid value given to the Argument `data_model`, it can be either `UCSD` or `Avenue`!')
    else:
        if threshold is None:
            raise TypeError('None value given to argument `threshold`, it cannot be None when `preset_threshold` argument is set to False, provide a value of `float` datype or set the `preset_threshold` argument to True, to use Preset Values of Threshold.')
    if serve_type!='real-time' and serve_type != None:
        if path is None:
            raise TypeError('None value given to argument `path`, it cannot be None when value of `serve_type` is other than None.')
    if config_gpu:
        #Setting up the GPU to avoid VRAM and other conflicts.
        #For refernce visit: https://github.com/irdanish11/AnomalyEventDetection/issues/1
        mp.TF_GPUsetup()
    #loading the model
    model = get_model(model_path)
    ####################### Different Serving Techinques ######################
    
    #Serve the Anomaly Detection from the WebCam or any video device that is attached.
    if serve_type=='real-time':
        RealTimeDetection(model, threshold, serve_type, verbose=verbose)
        test_hist = None
    #Serve the Anomaly Detection from the given video.
    elif serve_type=='video':
        RealTimeDetection(model, threshold, serve_type, vid_path=path, verbose=verbose)
        test_hist = None
    #Serve the Anomaly Detection from the directory which contain frames, the Hirerachy of 
    #directories must be like this: <path>/*Directories/Here all the images
    #The path you provide must contain a further directory or directories and in those directories
    #should have the frames.
    elif serve_type=='frames':
        test_hist = StaticServing(path, model, threshold, frames_ext, serve_type, verbose=verbose)
    ##Serve the Anomaly Detection from the .npy file.
    elif serve_type=='npy':
        test_hist = StaticServing(path, model, threshold, frames_ext, serve_type, verbose=verbose)
    else:
        raise ValueError('Invalid value given to the  `serve_type` argument. Possible values: {0}'.format(serving_types))
    return test_hist



if __name__=='__main__':
    
    #model_path = 'checkpoints/Train_UCSDped2_Model.h5'
    model_path = 'checkpoints/Train_AvenueDataset_Model.h5'
    
    vid_path = './AvenueDataset/testing_videos/05.avi' #5,9
    
    frames_ext='.tif'
    frames_dir='Datasets/UCSDped2/Test'
    
    npy_file='./Test_Data/Test_UCSDped2.npy'
    
    #possible serving types
    serving_types = ['real-time', 'video', 'frames', 'npy']
    #Serving of Model
    serve_type = serving_types[1]
    test_hist = DeploySystem(serve_type, model_path, preset_threshold=True, data_model='Avenue', verbose=True, 
                             path=vid_path, frames_ext=None, threshold=None, config_gpu=False)
    