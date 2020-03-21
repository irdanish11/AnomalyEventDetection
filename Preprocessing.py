# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 21:58:29 2020

@author: danish
"""

import cv2     # for capturing videos
import os
import math
from tqdm import tqdm
import numpy as np
import glob



def Video2Frames(v_file, path='./', ext='.avi', frames_dir='train_1', extract_rate='all', frames_ext='.jpg'):
    """
    A method which extracts the frames from the given video. It can ex

    Parameters
    ----------
    v_file : str
        Name of the video file, without extension.
    path : str
        Path to the video file, if the video is in the current working directory do not specify this argument.
    ext : str, optional
        Extension of the given Video File e.g `.avi`, `.mp4`. The default is '.avi'.
    frames_dir : str, optional
        Path to the directory where frames will be saved. The default is 'train_1'.
    extract_rate : int or str, optional
        This argument specifies how many frames should be extrcated from each 1 second of video. If the value is 
        `all` it will etract all the frames in every second i.e if the frame rate of video is 25 fps it will
        extrcat all 25 frames. Other wise specify a number if you want to etract specific numbers of frames
        per each second e.g if 5 is given it will extrcat 5 frames from each 1 second. The default is `all`.
    frames_ext : str, optional
        The extension for the extracted frames/images e.h '.tif' or '.jpg'. The default is '.jpg'.

    Returns
    -------
    None.

    """
    os.makedirs(frames_dir, exist_ok=True)
    # capturing the video from the given path
    cap = cv2.VideoCapture(path+v_file+ext)   
    
    frameRate = cap.get(5) #frame rate

    #duration = int(cap.get(7)/frameRate)
    os.makedirs(frames_dir+'/'+v_file, exist_ok=True)
    count = 0
    while(cap.isOpened()):
        frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
            
        if type(extract_rate)==int:
            if extract_rate>frameRate:
                print('Frame rate of Given Video: {0} fps'.format(frameRate))
                raise ValueError('The value of `extract_rate` argument can not be greater than the Frame Rate of the video.')
            
            if (frameId % extract_rate == 0) and extract_rate>1:
                # storing the frames in a new folder named train_1
                filename = frames_dir + '/' + v_file+ '/'+"_frame{0}".format(count)+frames_ext;count+=1
                cv2.imwrite(filename, frame)
            elif extract_rate==1:
                if (frameId % math.floor(frameRate) == 0):
                    filename = frames_dir + '/' + v_file+ '/'+"_frame{0}".format(count)+frames_ext;count+=1
                    cv2.imwrite(filename, frame)
        elif type(extract_rate)==str:
            if extract_rate=='all':
                # storing the frames in a new folder named train_1
                filename = frames_dir + '/' + v_file+ '/'+  v_file + "_frame{0}".format(count)+frames_ext;count+=1
                cv2.imwrite(filename, frame)
            else:
                raise ValueError('Invalid Value for argument `extract_rate`, it can be either `all` or an integer value.')
    cap.release()    
    
    
def CannyEdges(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged   

    
def ProcessImg(img_name, read_path, write_path, canny_edge=True, canny_path=None, sigma=0.33):

    #load_image = cv2.imread(images[y], 0)
    load_image = cv2.imread(read_path, 0)
    
    resized_image = cv2.resize(load_image, (128,128))
    
    # resized_image = resized_image.astype('float32')
    # resized_image -= resized_image.mean()
    
    if canny_edge:
        if canny_path==None:
            raise ValueError('Invalid value for argument `canny_path`, the value cannot be `None`, when `canny_edge` flag is set to True. Please provide valid path to this argument.')
        edged = CannyEdges(resized_image, sigma)
        os.makedirs(canny_path, exist_ok=True)
        cv2.imwrite(os.path.join(canny_path, img_name), edged)
    
    rescaled_image = resized_image.astype('float32')
    rescaled_image /= 255.0
    
    
    #Take the global mean image
    rescaled_image -= rescaled_image.mean()
    
    os.makedirs(write_path, exist_ok=True)
    cv2.imwrite(os.path.join(write_path, img_name), rescaled_image) 
    
    return rescaled_image
 
    
def ReadFileNames(path):
    directories = [name for name in os.listdir(path) if os.path.isdir(path+'/'+name)]
    onlyfiles = []
    file_names = []
    
    for i in range (len(directories)):
        files = glob.glob(path+'/'+directories[i]+'/*.tif')
        names = []
        for file in files:
            names.append(file.split("\\")[1])
        file_names.append(names)
        onlyfiles.append(files)
    return onlyfiles, file_names, directories



if __name__=='__main__':    
    # v_file='v_ApplyEyeMakeup_g01_c01'
    # path='./'    
    # frames_ext='.tif'
    # Video2Frames(v_file, path='./', ext='.avi', frames_dir='train_1', 
    #              extract_rate='all', frames_ext='.tif')   
    
    path = 'UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train'
    onlyfiles, file_names, dirs = ReadFileNames(path)
    
    for i in tqdm(range(len(onlyfiles))):
        images = onlyfiles[i]
        count = 0
        for img in images:
            img.split('/')
            img_name = file_names[i][count]
            write_path = 'ProcessedImages/'+path+'/'+dirs[i]
            canny_path = 'CannyImages/'+path+'/'+dirs[i]
            rescaled_image = ProcessImg(img_name, read_path=img, write_path=write_path, canny_edge=True,
                                        canny_path=canny_path, sigma=0.33)
            count += 1



