import cv2                         
import numpy as np                 
from functions import find_coordinate,max_component_2_binary,draw_bounding_box

def overlapping(color_mask,img):
    
    # undrawn input images
    clean_img1 = img.copy()
    clean_img2 = img.copy()
    clean_img3 = img.copy()
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 10))
    dilated = cv2.dilate(color_mask, kernel)
    
    numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(image=dilated, connectivity=4,ltype=cv2.CV_32S)
    sizes = stats[1:,-1]
    
    # largest component
    max_index, binary_map_max = max_component_2_binary(sizes,labels)
    x0,y0,w0,h0 = find_coordinate(clean_img1,binary_map_max)
    output_img = draw_bounding_box(clean_img3,x0,y0,w0,h0)

    #second largest component
    sizes[max_index-1]=0
    second_max_index, binary_map_second_max = max_component_2_binary(sizes,labels)
    x1,y1,w1,h1 = find_coordinate(clean_img2,binary_map_second_max)
    output_img = draw_bounding_box(clean_img3,x1,y1,w1,h1)   

    return output_img