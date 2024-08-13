import cv2
import numpy as np


def find_coordinate(img,binary_map):

    _,contours, _ = cv2.findContours(binary_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contours, -1, (255, 0, 255), 2)
    
    x,y,w,h = cv2.boundingRect(contours[0])

    return x,y,w,h



def draw_bounding_box(img,x,y,w,h):

     # If the detected motherboard's height is not large enough
    if w>2*h:
        
        while w>2*h and h<350:
            h = h*1.2 
            y = y - 0.1*h

        cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (0,0,255), 4)  

    #  If the detected motherboard's width is not large enough
    elif w<1.33*h:
        
        w = w*1.4
        x = x+0.05*w
        cv2.rectangle(img, (int(x), int(y)), (int(x+w), y+h), (0, 0, 255), 4)   

    # normal board size 
    else:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 4)

    return img



def max_component_2_binary(size,label):

    component = np.zeros((label.shape))

    max_index = np.argmax(size) + 1 
    component[label == max_index] = 255
    
    binary_map = (component > 0).astype(np.uint8)

    return max_index,binary_map

