#IMPORTS

import cv2 
import pickle 
import numpy as np
import os
from os import listdir
from os.path import isfile, join

from iVerify import verify 
from Iris_functions import computeSIFTEye 
from Palmprint_functions import computeSIFTPalm

from Matching import fetchLeftEyeDescriptorFromFile 
from Matching import fetchRightEyeDescriptorFromFile
from Matching import fetchPalmDescriptorFromFile

from Matching import calculateMatches

#MODULE FLOW
#Imposter images are verified by calling verify function of Verify module in testimposter(i)
#Legitimate images are verified by calling verify function of Verify module in testlegitimate(i)

#Function to test FAR
#INPUT - PATH OF IMPOSTER DATA
#OUTPUT - USER VALIDATED/REJECTED
def testimposter(i): 
    imposterpath = "imposter/"+str(i)+"/"
    
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


#Function to test FRR
#INPUT - PATH OF LEGITIMATE DATA
#OUTPUT - USER VALIDATED/REJECTED
def testlegitimate(i): 
    legitimatepath = "legitimate/"+str(i)+"/"
    
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

