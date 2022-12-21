
#IMPORTS

import cv2 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
import scipy
import math

def clahe(image):
    image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    lab_img_palm = cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
    l_palm,a_palm,b_palm = cv2.split(lab_img_palm)
    equ_palm = cv2.equalizeHist(l_palm)
    clahe_palm = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8,8))
    clahe_img_palm = clahe_palm.apply(l_palm)
    updated_lab_img_palm = cv2.merge((clahe_img_palm,a_palm,b_palm))
    CLAHE_img_palm = cv2.cvtColor(updated_lab_img_palm, cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(CLAHE_img_palm, cv2.COLOR_BGR2GRAY)
    noise = cv2.fastNlMeansDenoising(gray)
    noise = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)
    kernel = np.ones((7,7),np.uint8)
    img_crop = cv2.morphologyEx(noise, cv2.MORPH_OPEN, kernel)
    #img_yuv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2YUV)
    #img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    #img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_crop

def crop(img):
    img = clahe(img)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height,width = image.shape
    th, im = cv2.threshold(image, 127, 255, cv2.THRESH_OTSU)
    moment = cv2.moments(im)
    x = int(moment ["m10"] / moment["m00"])
    y = int(moment ["m01"] / moment["m00"])
    mask = np.zeros((height,width), np.uint8)
    cv2.circle(mask,(x,y),800,(255,255,255),thickness=-1)
    masked_data = cv2.bitwise_and(image, image, mask=mask)
    return masked_data


sift = cv2.SIFT_create(10000)
def computeSIFTPalm(image): 
    palm_data = crop(image)
    return sift.detectAndCompute(palm_data, None)


# In[2]:


# img = cv2.imread("/Users/apple/Desktop/datasets/imposter/2/033_F_0.JPG",0)

# palm_img = clahe(img)


# keypoints,descriptors = computeSIFTPalm(img)
# draw=cv2.drawKeypoints(crop(img),keypoints,None)
# print(len(keypoints))

# crop = crop(img)

# cv2.imshow('crop',crop)
# cv2.imshow('clahe',palm_img)
# cv2.imshow('draw',draw)
# cv2.waitKey(0)
# cv2.destroyAllWindows() 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




