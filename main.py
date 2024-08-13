import cv2                        
import numpy as np                 
import os

from preprocess import crop_image,extract_and_dilate
from functions import find_coordinate,draw_bounding_box,max_component_2_binary
from overlap import overlapping



########################## MAIN FUNCTION ##############################


img_folder = './input_images/'

for filename in os.listdir(img_folder):

    # IMAGE READING & CLONING 
    train_img = cv2.imread(os.path.join(img_folder,filename))

    clean_img1 = train_img.copy()
    clean_img2 = train_img.copy()

    # PREPROCESS 
    masked = crop_image(train_img)
    dilated,color_mask = extract_and_dilate(masked)

    # DETECT / LOCATE OF MAX COMPONENTS
    numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(image=dilated,connectivity=4,ltype=cv2.CV_32S)
    sizes = stats[1:,-1]
    max_index,binary_map = max_component_2_binary(sizes,labels)
    x,y,w,h = find_coordinate(train_img,binary_map)

    # detect overlapping motherboard
    if h>440:
        output_img = overlapping(color_mask,clean_img1)
        cv2.imwrite('./final_results/bounded_'+filename,output_img)

    else: 
        output_img = draw_bounding_box(clean_img2,x,y,w,h)

    cv2.imwrite('./final_results/bounded_'+filename,output_img)


# -----------------------------------------