
#IMPORTS

import cv2 
import pickle 
import numpy as np
from sklearn import preprocessing
import scipy
import math
import time

from Iris_functions import computeSIFTEye 
from Palmprint_functions import computeSIFTPalm

from Matching import fetchLeftEyeDescriptorFromFile 
from Matching import fetchRightEyeDescriptorFromFile
from Matching import fetchPalmDescriptorFromFile

from Matching import calculateMatches

#MODULE FLOW
#User uploads 3 images 
#The images are preprocessed and segmented
#Feature extraction, normalization is done
#The descriptors are compared with the other descriptors in the database


#Function to compute features and get cancelable template from input
#INPUT - IRIS IMAGE
#OUTPUT - CANCELABLE TEMPLATE OF IRIS
def cancelleftiris(img):
    _, descriptor = computeSIFTEye(img)
    desl2 = preprocessing.normalize(descriptor,norm='l2')
    descriptornorm = preprocessing.normalize(desl2,norm='l1')
    return descriptornorm

#Function to compute features and get cancelable template from input
#INPUT - IRIS IMAGE
#OUTPUT - CANCELABLE TEMPLATE OF IRIS
def cancelrightiris(img):
    _, descriptor = computeSIFTEye(img)
    desl2 = preprocessing.normalize(descriptor,norm='l2')
    descriptornorm = preprocessing.normalize(desl2,norm='l1')
    return descriptornorm

#Function to compute features and get cancelable template from input
#INPUT - PALM IMAGE
#OUTPUT - CANCELABLE TEMPLATE OF PALM
def cancelpalm(img):
    _, descriptor = computeSIFTPalm(img)
    desl2 = preprocessing.normalize(descriptor,norm='l2')
    descriptornorm = preprocessing.normalize(desl2,norm='l1')
    return descriptornorm




#Verification Function
#INPUT - TWO IRIS IMAGES, ONE PALM IMAGE
#OUTPUT - VERIFIED USER CAN SEE THE DATA
def verify(left_iris,right_iris,palm_img):
    

    descriptorleft = cancelleftiris(left_iris)
    descriptorright = cancelrightiris(right_iris)
    descriptorpalm = cancelpalm(palm_img)
    
    for j in range(25):
        filename1,descriptor_find_left = fetchLeftEyeDescriptorFromFile(j)
        matchleft = calculateMatches(descriptorleft, descriptor_find_left)
        left = len(matchleft)
        
        #Score for the left iris
        if(left>=10):
            print("User left eye verified.")
          
            for k in range(25):
                filename2,descriptor_find_right = fetchRightEyeDescriptorFromFile(k)
                matchright = calculateMatches(descriptorright, descriptor_find_right)
                right = len(matchright)
                
                #Score for the right iris
                if(right>=10):
                    print("User right eye verified.")
                    
                    for l in range(25):
                        filename3,descriptor_find_palm = fetchPalmDescriptorFromFile(l)
                        matchpalm = calculateMatches(descriptorpalm, descriptor_find_palm)
                        palm = len(matchpalm)
                        
                        #Score for palm
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
            
    
