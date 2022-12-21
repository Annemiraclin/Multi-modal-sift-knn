

#IMPORTS

import cv2 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import scipy


#FUNCTIONS

#Function to find the shape of Input Image
def shapeEye(image):
    height,width = image.shape #height and width of image 
    return height,width



#Function for de-noising and smoothing image 
def gaussianEye(image):
	#median blur for de-noising image 
    median_img_eye = cv2.medianBlur(img_ref_eye,5) 
    	#gaussian filter for smoothened image
    gaussian_img_eye = scipy.ndimage.filters.gaussian_filter(median_img_eye,sigma=1.90,order=0,output=None,mode='reflect',cval=0.0,truncate=4.0) 
    return gaussian_img_eye



#Function to apply CLAHE histogram equalisation on from Bhattiprolu, D.S. (2022). Changed the values of the parameters for optimised results
def claheEye(image):
	#convert image from grayscale to BGR
    image  = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
	#convert image into LAB format based 
    lab_img_eye = cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
	#split the image channels
    l_eye,a_eye,b_eye = cv2.split(lab_img_eye)
	#equalize the lightness channel
    equ_eye = cv2.equalizeHist(l_eye)
	#creating CLAHE object
    clahe_eye = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(9,9))
	#applying CLAHE equalisation on lightness channel
    clahe_img_eye = clahe_eye.apply(l_eye)
	#merging the CLAHE image with a and b channels
    updated_lab_img_eye = cv2.merge((clahe_img_eye,a_eye,b_eye))
	#convert updated image back to BGR
    CLAHE_img_eye = cv2.cvtColor(updated_lab_img_eye, cv2.COLOR_LAB2BGR)
    return CLAHE_img_eye




#Iris Segmentation
def Eye(image):
	#getting the de-noised image
    gaussian_img = gaussianEye(image)
    cimg  = image
    #Finding pupil circles	
	#canny edge values set to 50 for getting more edges to detect smaller circles
    edges = cv2.Canny(gaussian_img,50,50)
	#hough circle transform to detect the pupil
    circles_pupil = cv2.HoughCircles(edges,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)
	#draw circle on the image from the output values
    circles_pupil = np.uint8(np.around(circles_pupil))
    for i in circles_pupil[0,:]:
        cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
	#get height and width
    height,width = shapeEye(image)
	#creating mask
    mask_pupil = np.zeros((height,width), np.uint8)
	#creating mask image
    circle_img_pupil = cv2.circle(mask_pupil,(i[0],i[1]),i[2],(255,255,255),thickness=-1)


    #Finding iris circle 
	#canny edge detection values are lower for detecting bigger circles
    edges = cv2.Canny(gaussian_img,10,10)
	#hough circle transform with higher values to detect pupil
    circles_iris = cv2.HoughCircles(edges,cv2.HOUGH_GRADIENT,1,150,param1=80,param2=70,minRadius=0,maxRadius=0)
	#draw the circle detected on the image
    circles_iris = np.uint8(np.around(circles_iris))
    for i in circles_iris[0,:]:
        cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
	#create a iris mask
    mask_iris = np.zeros((height,width), np.uint8)
	#create iris mask image
    circle_img_iris = cv2.circle(mask_iris,(i[0],i[1]),i[2],(255,255,255),thickness=-1)
	#merge iris and pupil masks
    mask_eye = cv2.subtract(mask_iris, mask_pupil)
	#retrieve the CLAHE image
    CLAHE_img_eye = claheEye(image)
	#apply mask on CLAHE image
    masked_data = cv2.bitwise_and(CLAHE_img_eye, CLAHE_img_eye, mask=mask_eye)
    return masked_data



#SIFT feature Extraction function
sift = cv2.SIFT_create()
def computeSIFTEye(image): 
	#getting the segmented iris image
    masked_data = Eye(image)
    return sift.detectAndCompute(masked_data, None)
    


