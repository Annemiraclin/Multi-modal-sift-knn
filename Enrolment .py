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

from Iris_functions import computeSIFTEye 
from Palmprint_functions import computeSIFTPalm


#MODULE FLOW
#The user enters a number
#Images corresponding to the number are extracted from the given file path
#Images are converted into grayscale
#Processed and segmented the images
#Feature Extraction and conversion to cancelable template
#Storage of Descriptors for future use

#FUNCTIONS


#Enrolment function
#INPUT - USER NUMBER
#OUTPUT - ENROLLED USER
def enrol(number):
    #calculating time of computation
    start = time.time()
        #path of images
        iris1path = "iris/left/"+str(number)+"/"
        iris2path = "iris/right/"+str(number)+"/"
        palmpath = "palm/"+str(number)+"/"
        
        imageListleft = []
        imageListright = []
        imageListpalm = []


        imageListleft = [f for f in listdir(iris1path) if isfile(join(iris1path, f))]
        if ".DS_Store" in imageListleft: imageListleft.remove(".DS_Store")

        imageListright = [f for f in listdir(iris2path) if isfile(join(iris2path, f))]
        if ".DS_Store" in imageListright: imageListright.remove(".DS_Store")

        imageListpalm = [f for f in listdir(palmpath) if isfile(join(palmpath, f))]
        if ".DS_Store" in imageListpalm: imageListpalm.remove(".DS_Store")

            
        #getting grayscale images
        #Preparing list of images section from (Dumrewal, A, 2022) was used for this function.
        imagesBWleft = [] 
        for imageName in imageListleft: 
            imagePathleft = "iris/left/"+str(number)+"/"+ str(imageName) 
            imagesBWleft.append(cv2.imread(imagePathleft,0))

        imagesBWright = [] 
        for imageName in imageListright:  
            imagePathright = "iris/right/"+str(number)+"/"+ str(imageName) 
            imagesBWright.append(cv2.imread(imagePathright,0)) 

        imagesBWpalm = [] 
        for imageName in imageListpalm:  
            imagePathpalm = "palm/"+str(number)+"/"+ str(imageName) 
            imagesBWpalm.append(cv2.imread(imagePathpalm,0)) 


        #computing the keypoints and descriptors, converting them to another format
        #Generating keypoints and Descriptors section from (Dumrewal, A, 2022) was used for this function by changing parameters to fit this project.
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
        #Storing keypoints and Descriptors section from (Dumrewal, A, 2022) was used for this function.
        for i,left in enumerate(descriptorsleft):
            filepath = "desc/"+ str(imageListleft[i].split('.')[0]) + ".txt" 
            with open(filepath, 'wb') as fp: 
                pickle.dump(left, fp) 

        for j,right in enumerate(descriptorsright):
            filepath = "desc/" + str(imageListright[j].split('.')[0]) + ".txt" 
            with open(filepath, 'wb') as fp:
                pickle.dump(right, fp)  

        for k,palm in enumerate(descriptorspalm):
            filepath = "desc/" + str(imageListpalm[k].split('.')[0]) + ".txt" 
            with open(filepath, 'wb') as fp:
                pickle.dump(palm, fp)

    #stopping the computation timer
    stop = time.time()
    print(start-stop)




