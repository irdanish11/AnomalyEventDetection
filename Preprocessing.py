# -*- coding: utf-8 -*-
"""
Created on  Fri 20 14:48:49 2020

@author: danish
"""
#Import the necessary libraries
import cv2     # for capturing videos
import os
import math
from tqdm import tqdm
import numpy as np
import glob
from numba import njit


def Frame_Extractor(v_file, path='./', ext='.avi', frames_dir='train_1', extract_rate='all', frames_ext='.jpg'):
    """
    A method which extracts the frames from the guven video. It can ex

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
    
    
def Edge_Detector(image, sigma=0.33):
    """
    A method that will apply Canny Edge detection on the images.

    Parameters
    ----------
    
    image : uint8
      Image/Frame upon which Canny Edge detection will be applied
    
    sigma : float, optional
      Sigma is a real number, typically between 0 and 2. It is Standard Deviation of the
      Gaussian. Sigma plays important roles of a scale parameter for the edges: lager values
      of sigma produce coarser scale edges and small values of sigma produce finer scale edges.
      Larger values of sigma also result in greater noise suppression. Default value is 0.33.
      
    Returns
    -------
    
    edged : uint8
      Canned Edge images.

    """
    # compute the median of the single channel pixel intensities
    v = np.median(image)
	# apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
	# return the edged image
    return edged   

#@njit    
def PreProcessing(img_name, read_path, write_path, canny_edge=True, canny_path=None, sigma=0.33):
    """
    This method applies the preprocessing steps i.e Resizing, Canny Edge Detection,
    Normalization, Rescaling and Global mean.

    Parameters
    ----------
    
    img_name : string
      Name of the frame/image.
    
    read_path : string
      Path location for the images.
    
    write_path : string
      Location where to save the processed images.
    
    canny_edge : boolean, optional
      True: Applies Canny Edge Function
      False: Doesn't applies Canny Edge Function. The default is True.
    
    canny_path : string, optional
      Path where to store the edge enhanced images. The default is None.
    
    sigma : float, optional
      Explained in CannyEdges Documentation. The default is 0.33.

    Raises
    ------
    
    ValueError
      This error will occur for following reasons:
        1. Image does not exist at the given path.
        2. Datatype of the image is None.

    Returns
    -------
    
    rescaled_image : uint8
      Resturns completly processed(Resizing, Canny Edge Detection,
      Normalization, Rescaling and Global mean) images

    """
    load_image = cv2.imread(read_path, 0)
    
    resized_image = cv2.resize(load_image, (128,128))
    
    if canny_edge:
        if canny_path==None:
            raise ValueError('Invalid value for argument `canny_path`, the value cannot be `None`, when `canny_edge` flag is set to True. Please provide valid path to this argument.')
        edged = Edge_Detector(resized_image, sigma)
        os.makedirs(canny_path.split('/')[0], exist_ok=True)
        cv2.imwrite(canny_path+'_'+img_name, edged)
    
    rescaled_image = resized_image.astype('float32')
    rescaled_image /= 255.0
    
    
    #Take the global mean of the image
    rescaled_image -= rescaled_image.mean()
    
    os.makedirs(write_path.split('/')[0], exist_ok=True)
    cv2.imwrite(write_path+'_'+img_name, rescaled_image) 
    
    return rescaled_image
 
    
def ReadFileNames(path):
    """
    This method will retrieve the Folder/File names from the dataset.

    Parameters
    ----------
    path : string
      Location of the data set that needs to be preprocessed.

    Returns
    -------
    onlyfiles : list
      A list containing all the subfolder names in the dataset.
    file_names : list
      A list containing all the names of the frames/images from 'onlyfiles'.
    directories: list
      A list containing all the names of the folders containing the images.
    """
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

def load_data(orig_frames, canned_frames, seq_size = 8):
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
  X = np.array(lst)
  return X, lst

if __name__=='__main__':    
    #v_file='video'
    #path='./'    
    #frames_ext='.tif'
    #frames_dir = 'Extracted_Frames'
    #Frame_Extractor(v_file, path='./', ext='.mp4', frames_dir='Extracted_Frames', 
    #              extract_rate='all', frames_ext='.tif')   
    
    #path = frames_dir
    path = 'UCSDped1/Train'
    onlyfiles, file_names, dirs = ReadFileNames(path)
    
    for i in tqdm(range(len(onlyfiles))):
        images = onlyfiles[i]
        count = 0
        for img in images:
            img.split('/')
            img_name = file_names[i][count]
            write_path = 'ProcessedImages/'+dirs[i]
            canny_path = 'CannyImages/'+dirs[i]
            rescaled_image = PreProcessing(img_name, read_path=img, write_path=write_path, canny_edge=True,
                                           canny_path=canny_path, sigma=0.33)
            count += 1
		
