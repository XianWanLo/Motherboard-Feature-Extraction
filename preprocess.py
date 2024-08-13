import cv2                         
import numpy as np                 


def crop_image(input):

    mask=np.zeros_like(input)
    mask[250:850,450:1450,:]= 1
    masked = input*mask
    #cv2.imwrite('./masked1.jpg',masked)

    return masked


def extract_and_dilate(masked):

    # Extraction of Color Feature
    image_RGB =cv2.cvtColor(masked,cv2.COLOR_BGR2HSV)
    color_mask=cv2.inRange(image_RGB, np.array([100,40,40]), np.array([125,250,250]))
    #cv2.imwrite('./extraction.jpg',color_mask)

    # Feature Dilation 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 100))
    dilated = cv2.dilate(color_mask,kernel)
    #cv2.imwrite('./dilation.jpg',dilated)

    return dilated,color_mask
