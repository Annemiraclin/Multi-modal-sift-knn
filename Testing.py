#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2 
import pickle 
import numpy as np


import os
from os import listdir
from os.path import isfile, join

from ipynb.fs.full.Verify import verify 
from ipynb.fs.full.Iris_functions import computeSIFTEye 
from ipynb.fs.full.Palmprint_functions import computeSIFTPalm

from ipynb.fs.full.Matching import fetchLeftEyeDescriptorFromFile 
from ipynb.fs.full.Matching import fetchRightEyeDescriptorFromFile
from ipynb.fs.full.Matching import fetchPalmDescriptorFromFile

from ipynb.fs.full.Matching import calculateMatches



def testimposter(i): 
    imposterpath = "/Users/apple/Desktop/datasets/imposter/"+str(i)+"/"
    
    imposterimagelist = []
    
    imposterimagelist = [f for f in listdir(imposterpath) if isfile(join(imposterpath, f))]
    if ".DS_Store" in imposterimagelist: imposterimagelist.remove(".DS_Store")
        
    for j in range(3):
        if "_1_" in imposterimagelist[j]:
            imposteriris1path = imposterpath+imposterimagelist[j]
        if "_2_" in imposterimagelist[j]:
            imposteriris2path = imposterpath+imposterimagelist[j]
        if ".JPG" in imposterimagelist[j]:
            imposterpalmpath = imposterpath+imposterimagelist[j]
    
    
    print(imposteriris1path)
    print(imposteriris2path)
    print(imposterpalmpath)
    
    
    imposteriris1 = cv2.imread(imposteriris1path,0) 
    imposteriris2 = cv2.imread(imposteriris2path,0)
    imposterpalm = cv2.imread(imposterpalmpath,0)

    verify(imposteriris1,imposteriris2,imposterpalm)


for k in range(5):
    testimposter(k)



def testlegitimate(i): 
    legitimatepath = "/Users/apple/Desktop/datasets/legitimate/"+str(i)+"/"
    
    legitimateimagelist = []
    
    legitimateimagelist = [f for f in listdir(legitimatepath) if isfile(join(legitimatepath, f))]
    if ".DS_Store" in legitimateimagelist: legitimateimagelist.remove(".DS_Store")
        
    for j in range(3):
        if "_1_" in legitimateimagelist[j]:
            legitimateiris1path = legitimatepath+legitimateimagelist[j]
        if "_2_" in legitimateimagelist[j]:
            legitimateiris2path = legitimatepath+legitimateimagelist[j]
        if ".JPG" in legitimateimagelist[j]:
            legitimatepalmpath = legitimatepath+legitimateimagelist[j]
    
    
    print(legitimateiris1path)
    print(legitimateiris2path)
    print(legitimatepalmpath)
    
    
    legitimateiris1 = cv2.imread(legitimateiris1path,0)

    
    legitimateiris2 = cv2.imread(legitimateiris2path,0)

    
    legitimatepalm = cv2.imread(legitimatepalmpath,0)

    
    verify(legitimateiris1,legitimateiris2,legitimatepalm)

for i in range(5):
    testlegitimate(i)

