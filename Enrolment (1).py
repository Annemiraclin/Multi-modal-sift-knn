#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 
import pickle 
import numpy as np
from sklearn import preprocessing
import scipy
import math
import os
from os import listdir
from os.path import isfile, join
import time

from ipynb.fs.full.Iris_functions import computeSIFTEye 
from ipynb.fs.full.Palmprint_functions import computeSIFTPalm


def enrol(number):
    start = time.time()
    if(number>=30):
        print('invalid')
    else:
        iris1path = "/Users/apple/Desktop/datasets/iris/left/"+str(number)+"/"
        iris2path = "/Users/apple/Desktop/datasets/iris/right/"+str(number)+"/"
        palmpath = "/Users/apple/Desktop/datasets/palm/"+str(number)+"/"
        
        imageListleft = []
        imageListright = []
        imageListpalm = []


        imageListleft = [f for f in listdir(iris1path) if isfile(join(iris1path, f))]
        if ".DS_Store" in imageListleft: imageListleft.remove(".DS_Store")

        imageListright = [f for f in listdir(iris2path) if isfile(join(iris2path, f))]
        if ".DS_Store" in imageListright: imageListright.remove(".DS_Store")

        imageListpalm = [f for f in listdir(palmpath) if isfile(join(palmpath, f))]
        if ".DS_Store" in imageListpalm: imageListpalm.remove(".DS_Store")


        imagesBWleft = [] 
        imagesleft = []
        for imageName in imageListleft: 
            imagePathleft = "/Users/apple/Desktop/datasets/iris/left/"+str(number)+"/"+ str(imageName) 
            imagesBWleft.append(cv2.imread(imagePathleft,0))

        imagesBWright = [] 
        imagesright = []
        for imageName in imageListright:  
            imagePathright = "/Users/apple/Desktop/datasets/iris/right/"+str(number)+"/"+ str(imageName) 
            imagesBWright.append(cv2.imread(imagePathright,0)) 

        imagesBWpalm = [] 
        imagespalm = []
        for imageName in imageListpalm:  
            imagePathpalm = "/Users/apple/Desktop/datasets/palm/"+str(number)+"/"+ str(imageName) 
            imagesBWpalm.append(cv2.imread(imagePathpalm,0)) 


        descriptorsleft = []
        for i,image in enumerate(imagesBWleft): 
            print(" Enrolling user left iris: " + imageListleft[i]) 
            des = []
            l2norm = []
            _, descriptorTemp = computeSIFTEye(image) 
            #convert descriptors to another format by using normalization
            l2norm = preprocessing.normalize(descriptorTemp,norm='l2')
            des = preprocessing.normalize(l2norm,norm='l1')
            descriptorsleft.append(des) 
            print(" Enrolling user left iris: " + imageListleft[i])


        descriptorsright = []
        for j,image in enumerate(imagesBWright): 
            print(" Enrolling user right iris: " + imageListright[j]) 
            des = []
            l2norm = []
            _, descriptorTemp = computeSIFTEye(image) 
            #convert descriptors to another format by using normalization
            l2norm = preprocessing.normalize(descriptorTemp,norm='l2')
            des = preprocessing.normalize(l2norm,norm='l1')
            descriptorsright.append(des) 
            print(" Enrolling user right iris: " + imageListright[j])


        descriptorspalm = []
        for k,image in enumerate(imagesBWpalm): 
            print(" Enrolling user palm: " + imageListpalm[k]) 
            des = []
            l2norm = []
            _, descriptorTemp = computeSIFTPalm(image) 
            #convert descriptors to another format by using normalization
            l2norm = preprocessing.normalize(descriptorTemp,norm='l2')
            des = preprocessing.normalize(l2norm,norm='l1')
            descriptorspalm.append(des) 
            print(" Enrolling user palm: " + imageListpalm[k])


        # Store the normalized descriptors for future use
        for i,left in enumerate(descriptorsleft):
            filepath = "/Users/apple/Desktop/datasets/desc/"+ str(imageListleft[i].split('.')[0]) + ".txt" 
            with open(filepath, 'wb') as fp: 
                pickle.dump(left, fp) 

        for j,right in enumerate(descriptorsright):
            filepath = "/Users/apple/Desktop/datasets/desc/" + str(imageListright[j].split('.')[0]) + ".txt" 
            with open(filepath, 'wb') as fp:
                pickle.dump(right, fp)  

        for k,palm in enumerate(descriptorspalm):
            filepath = "/Users/apple/Desktop/datasets/desc/" + str(imageListpalm[k].split('.')[0]) + ".txt" 
            with open(filepath, 'wb') as fp:
                pickle.dump(palm, fp)

    stop = time.time()
    print(start-stop)


# In[2]:


enrol(0)


# In[ ]:


#-43.60429787635803


# In[ ]:




