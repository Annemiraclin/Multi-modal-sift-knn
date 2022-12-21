#IMPORTS

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


#FLOW OF MODULE
#Images are extracted from the paths using the fetchDescriptor functions
#Matching is done using Brute Force Matching with FLANN and KNN, with Lowe's ratio test.


#FUNCTIONS


imageListleft = []
imageListright = []
imageListpalm = []

#descriptor paths
path = "datasets/descriptors/"

imageList = [f for f in listdir(path) if isfile(join(path, f))]
if ".DS_Store" in imageList: imageList.remove(".DS_Store")

        
#seperating descriptors into left iris, right iris and palm descriptors for easy comparison.
for j,img in enumerate(imageList): 
    if "_1_" in imageList[j]: 
        imageListleft.append(img)
    elif "_2_" in imageList[j]:
        imageListright.append(img)
    elif "_F_" in imageList[j]:
        imageListpalm.append(img)


#getting the left iris descriptors from the system 
#code is from the Dumrewal, A. (2022), fetchDescriptors function, changed the parameters to fit the project scope
def fetchLeftEyeDescriptorFromFile(j): 
    start = time.time()
    filepath = "datasets/descriptors/"+ str(imageListleft[j].split('.')[0]) + ".txt"
    file = open(filepath,'rb') 
    descriptor = pickle.load(file) 
    file.close()
    #print(filepath)
    filename = imageListleft[j].split('.')[0]
    stop = time.time()
    print(start-stop)
    return filename,descriptor

#getting the right iris descriptors from the system
#code is from the Dumrewal, A. (2022), fetchDescriptors function, changed the parameters to fit the project scope
def fetchRightEyeDescriptorFromFile(k): 
    start = time.time()
    filepath = "datasets/descriptors/" + str(imageListright[k].split('.')[0]) + ".txt"
    file = open(filepath,'rb') 
    descriptor = pickle.load(file) 
    file.close()
    #print(filepath)
    filename = imageListright[k].split('.')[0]
    stop = time.time()
    print(start-stop)
    return filename,descriptor

#getting the palm descriptors from the system
#code is from the Dumrewal, A. (2022), fetchDescriptors function, changed the parameters to fit the project scope
def fetchPalmDescriptorFromFile(l): 
    start = time.time()
    filepath = "datasets/descriptors/" + str(imageListpalm[l].split('.')[0]) + ".txt"
    file = open(filepath,'rb') 
    descriptor = pickle.load(file) 
    file.close()
    #print(filepath)
    filename = imageListpalm[l].split('.')[0]
    stop = time.time()
    print(start-stop)
    return filename,descriptor


#brute force matching using FLANN, KNN and Lowe's ratio number.
#code is from the Dumrewal, A. (2022), brute force matching section in the ipynb file. The parameters for indexparams and searchparams have been changed to fit needs of project in the lines [84-85]
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
