#!/usr/bin/env python
# coding: utf-8

# # Import the necessary libraries

# In[1]:


import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt 
import os
import cv2


# # Function to initialise the mean and variance images

# In[2]:


def initBackground(initImage):
    mean = mpimg.imread(initImage)
    variance = 9*np.ones(mean.shape)
    return(mean,variance)


# # Classify images into foreground and background pixels using a Chebyshev inequality based classifier

# In[3]:


def ForegroundDetection(img_file,mean,variance,lmda):
    img = cv2.imread(img_file) 
    d = img - mean
    y = variance*(lmda**2)  
    d_2 = np.square(d)
    I = d_2 - y
    mask = np.all(I>0,axis=2)  # Chebychev's Check
    rI = 255*mask.astype(int)
    rI = rI.astype(np.uint8)
    return(rI)


# # Reduce the image noise using a voting scheme

# In[4]:


def Voting(rI,eta,m,n):
    r,c = rI.shape
    cI = np.zeros((rI.shape[0],rI.shape[1]))
    for i in range(m,r-1-m):
        for j in range(n,c-1-n):
            img_patch = rI[i-m:i,j-n:j]
            y_unq, counts = np.unique(img_patch,return_counts=True)
            if len(counts) == 1 and y_unq[0] == 1:
                cI[i,j] = 255
            if len(counts)>1:
                if counts[1] > eta*m*n:
                    cI[i,j] = 255
    cI = cI.astype(np.uint8)
    return cI


# # Update the mean and variance images using a weighted average scheme

# In[5]:


def meanvarUpdate(cI,img_path,M,V,alpha):
    img = mpimg.imread(img_path)
    mean_upd = np.zeros(img.shape)
    var_upd = np.zeros(img.shape)
    d = img - M
    d_2 = np.square(d)
    for i in range(cI.shape[0]):
        for j in range(cI.shape[1]):
            if cI[i,j] == 0:
                mean_upd[i,j,:] = (1-alpha)*M[i,j,:] + alpha*img[i,j,:]
                var_upd[i,j,:] = (1-alpha)*(V[i,j,:] + alpha*d_2[i,j,:])
                var_upd[i,j,:] = np.clip(var_upd[i,j,:],a_min = 9,a_max = None)
    return(mean_upd,var_upd)


# In[11]:


def frame_capture(path='intrusion_test.webm'):
    """Captures and saves the frames from the video specified at path"""
    vid_obj = cv2.VideoCapture(path)

    count = 0

    success = 1
    
    while success:
        success, image = vid_obj.read()
        if count < 10:
            cv2.imwrite("video_frames/frame_00%d.jpg" % count, image)
        elif count < 100:
            cv2.imwrite("video_frames/frame_0%d.jpg" % count, image)
        else:
            cv2.imwrite("video_frames/frame_%d.jpg" % count, image)
        print('saving frame ' + count)
        count += 1


# In[7]:


def Background_Subtraction(img_dir,lmda,eta,m,n,alpha):
    
    img_file_name = os.listdir(img_dir)
    initImage = os.path.join(img_dir,img_file_name[0])
    mean, variance = initBackground(initImage)
    
    for i in range(1,600):
        img_path = os.path.join(img_dir,img_file_name[i])
        
        fig, ax = plt.subplots(1,3,figsize=(10,10))
        rI = ForegroundDetection(img_path,mean,variance,lmda)
        ax[0].imshow(rI,cmap="gray")
        
        cI = Voting(rI,eta,m,n)
        mean, variance = meanvarUpdate(cI,img_path,mean,variance,alpha)
        ax[1].imshow(cI,cmap="gray")
        
        img = cv2.imread(img_path)
        ax[2].imshow(img,cmap="gray")
        
        plt.show()
    return(mean,variance)


# In[7]:


mean,variance = Background_Subtraction("./Images00",0.8,0.7,7,7,0.8)


# In[ ]:




