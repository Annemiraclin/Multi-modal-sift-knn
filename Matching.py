#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 
import pickle 
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import disk
from sklearn.metrics.pairwise import cosine_similarity
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from sklearn import preprocessing
import scipy
import math
import random
import os
from os import listdir
from os.path import isfile, join
import time


imageListleft = []
imageListright = []
imageListpalm = []


path = "/Users/apple/Desktop/datasets/desc/"

imageList = [f for f in listdir(path) if isfile(join(path, f))]
if ".DS_Store" in imageList: imageList.remove(".DS_Store")
        
for j,img in enumerate(imageList): 
    if "_1_" in imageList[j]: 
        imageListleft.append(img)
    elif "_2_" in imageList[j]:
        imageListright.append(img)
    elif "_F_" in imageList[j]:
        imageListpalm.append(img)


        
def fetchLeftEyeDescriptorFromFile(j): 
    start = time.time()
    filepath = "/Users/apple/Desktop/datasets/desc/"+ str(imageListleft[j].split('.')[0]) + ".txt"
    file = open(filepath,'rb') 
    descriptor = pickle.load(file) 
    file.close()
    #print(filepath)
    filename = imageListleft[j].split('.')[0]
    stop = time.time()
    print(start-stop)
    return filename,descriptor

def fetchRightEyeDescriptorFromFile(k): 
    start = time.time()
    filepath = "/Users/apple/Desktop/datasets/desc/" + str(imageListright[k].split('.')[0]) + ".txt"
    file = open(filepath,'rb') 
    descriptor = pickle.load(file) 
    file.close()
    #print(filepath)
    filename = imageListright[k].split('.')[0]
    stop = time.time()
    print(start-stop)
    return filename,descriptor

def fetchPalmDescriptorFromFile(l): 
    start = time.time()
    filepath = "/Users/apple/Desktop/datasets/desc/" + str(imageListpalm[l].split('.')[0]) + ".txt"
    file = open(filepath,'rb') 
    descriptor = pickle.load(file) 
    file.close()
    #print(filepath)
    filename = imageListpalm[l].split('.')[0]
    stop = time.time()
    print(start-stop)
    return filename,descriptor


bf = cv2.BFMatcher(crossCheck=True)
def calculateMatches(des1,des2):
        start = time.time()
        FLANNINDEXKDTREE = 1
        indexparams = dict(algorithm = FLANNINDEXKDTREE, trees = 100)
        searchparams = dict(checks=50)
        flann = cv2.FlannBasedMatcher(indexparams,searchparams)
        matches = flann.knnMatch(des1,des2,k=2)
        Results = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                Results.append([m])
        stop = time.time()
        print(start-stop)
        return Results


# In[2]:


# one = fetchLeftEyeDescriptorFromFile(0)
# two = fetchLeftEyeDescriptorFromFile(0)


#-0.0005240440368652344


# In[3]:


# fetchRightEyeDescriptorFromFile(0)

#-0.0003371238708496094


# In[4]:


# fetchPalmDescriptorFromFile(0)

#-0.005274057388305664


# In[ ]:




