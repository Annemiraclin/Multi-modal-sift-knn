
#IMPORTS

import cv2 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from skimage.morphology import black_tophat, skeletonize, convex_hull_image


#FLOW OF MODULE
#input image -> computeSIFTPalm(image) -> crop(img) -> clahe(image) 
#output of computeSIFTPalm(image) is the keypoints and descriptors

#FUNCTIONS


#CLAHE equalization of the palm image from Bhattiprolu, D.S. (2022) lines [36-67]. Changed the values of the parameters for optimised results in the lines [21-35] in this code.
#EROSION of the palm from CHADHA, S. (2022) lines [43-50] have been used in lines [39-47] in this code.
#INPUT - GRAY SCALE IMAGE OF PALM
#OUTPUT - EQUALISED AND ERODED PALM IMAGE
def clahe(image):
    #convert Gray scale image to BGR
    image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    #convert BGR to LAB 
    lab_img_palm = cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
    #splitting the LAB image to three channels
    l_palm,a_palm,b_palm = cv2.split(lab_img_palm)
    #equalizing lightness channel
    equ_palm = cv2.equalizeHist(l_palm)
    #creating clahe object
    clahe_palm = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8,8))
    #applying CLAHE on lightness channel
    clahe_img_palm = clahe_palm.apply(l_palm)
    #merging updated lightness channel with channel a and b
    updated_lab_img_palm = cv2.merge((clahe_img_palm,a_palm,b_palm))
    #converting LAB image to BGR
    CLAHE_img_palm = cv2.cvtColor(updated_lab_img_palm, cv2.COLOR_LAB2BGR)
    #converting the image to GRAY for denoising
    gray = cv2.cvtColor(CLAHE_img_palm, cv2.COLOR_BGR2GRAY)
    #denoising the palm image
    noise = cv2.fastNlMeansDenoising(gray)
    #convert back to BGR
    noise = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)
    #create kernel for morphological operation
    kernel = np.ones((7,7),np.uint8)
    #perform erosion on image
    img_crop = cv2.morphologyEx(noise, cv2.MORPH_OPEN, kernel)
    return img_crop



#Function for ROI Extraction
#INPUT - GRAYSCALE IMAGE
#OUTPUT - CROPPED ROI OF PALM
def crop(img):
    #CLAHE function is called to get equalised and eroded palm
    img = clahe(img)
    #convert BGR clahe image to GRAY
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #height and width of image is identified
    height,width = image.shape
    #binarize image
    th, im = cv2.threshold(image, 127, 255, cv2.THRESH_OTSU)
    #finding the palm center
    moment = cv2.moments(im)
    x = int(moment ["m10"] / moment["m00"])
    y = int(moment ["m01"] / moment["m00"])
    #create mask with parameters
    mask = np.zeros((height,width), np.uint8)
    #create mask image
    cv2.circle(mask,(x,y),800,(255,255,255),thickness=-1)
    #apply mask on grayscale equalised palm
    masked_data = cv2.bitwise_and(image, image, mask=mask)
    return masked_data


#Function to perform SIFT feature extraction
#INPUT - GRAYSCALE IMAGE OF PALM
#OUTPUT - SIFT FEATURES AND DESCRIPTORS
sift = cv2.SIFT_create()
def computeSIFTPalm(image): 
    #extracting the ROI palm
    palm_data = crop(image)
    return sift.detectAndCompute(palm_data, None)





