#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 
import pickle 
import numpy as np
from sklearn import preprocessing
import scipy
import math
import time

from ipynb.fs.full.Iris_functions import computeSIFTEye 
from ipynb.fs.full.Palmprint_functions import computeSIFTPalm

from ipynb.fs.full.Matching import fetchLeftEyeDescriptorFromFile 
from ipynb.fs.full.Matching import fetchRightEyeDescriptorFromFile
from ipynb.fs.full.Matching import fetchPalmDescriptorFromFile

from ipynb.fs.full.Matching import calculateMatches


# In[2]:


def cancelleftiris(img):
    _, descriptor = computeSIFTEye(img)
    desl2 = preprocessing.normalize(descriptor,norm='l2')
    descriptornorm = preprocessing.normalize(desl2,norm='l1')
    return descriptornorm

def cancelrightiris(img):
    _, descriptor = computeSIFTEye(img)
    desl2 = preprocessing.normalize(descriptor,norm='l2')
    descriptornorm = preprocessing.normalize(desl2,norm='l1')
    return descriptornorm

def cancelpalm(img):
    _, descriptor = computeSIFTPalm(img)
    desl2 = preprocessing.normalize(descriptor,norm='l2')
    descriptornorm = preprocessing.normalize(desl2,norm='l1')
    return descriptornorm


# In[3]:


def verify(left_iris,right_iris,palm_img):
    

    descriptorleft = cancelleftiris(left_iris)
    descriptorright = cancelrightiris(right_iris)
    descriptorpalm = cancelpalm(palm_img)
    
    for j in range(25):
        filename1,descriptor_find_left = fetchLeftEyeDescriptorFromFile(j)
        matchleft = calculateMatches(descriptorleft, descriptor_find_left)
        left = len(matchleft)
        
        if(left>=10):
            print("User left eye verified.")
          
            for k in range(25):
                filename2,descriptor_find_right = fetchRightEyeDescriptorFromFile(k)
                matchright = calculateMatches(descriptorright, descriptor_find_right)
                right = len(matchright)
                
                if(right>=10):
                    print("User right eye verified.")
                    
                    for l in range(25):
                        filename3,descriptor_find_palm = fetchPalmDescriptorFromFile(l)
                        matchpalm = calculateMatches(descriptorpalm, descriptor_find_palm)
                        palm = len(matchpalm)
                        
                        if(palm>=100):  
                            print("User palm verified.")
                            print("User Validated.")
                            f=open("/Users/apple/Desktop/datasets/data/"+filename1.split('_')[0]+".txt")
                            print(f.readline())
                            f.close()
                            break
                break
            if(left<=10 | right<=10 | palm<=100):
                print("User Rejected")
        break
            
    


# In[4]:


left_iris1 = cv2.imread("/Users/apple/Desktop/datasets/iris/left/0/060_1_1.bmp",0)
right_iris1 = cv2.imread("/Users/apple/Desktop/datasets/iris/right/0/060_2_1.bmp",0)
palm_img1 = cv2.imread("/Users/apple/Desktop/datasets/palm/0/032_F_0.JPG",0)

verify(left_iris1,right_iris1,palm_img1)


# In[5]:


#-0.0020818710327148438
#-0.006985187530517578
#-16.00835108757019

