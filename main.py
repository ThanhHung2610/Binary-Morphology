import sys
import getopt
import cv2
from cv2 import MORPH_GRADIENT
import numpy as np
from morphological_operator import binary

def operator(in_file, out_file, mor_op, wait_key_time=0):
    img_origin = cv2.imread(in_file)
    cv2.imshow('original image', img_origin)

    img_gray = cv2.imread(in_file, 0)
    cv2.imshow('gray image', img_gray)

    (thresh, img) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow('binary image', img)
    
    kernel = np.ones((3, 3), np.uint8)
    img_out = None

    '''
    TODO: implement morphological operators
    '''
    if mor_op == 'dilate':
        img_dilation = cv2.dilate(img, kernel)
        cv2.imshow('OpenCV dilation image', img_dilation)

        img_dilation_manual = binary.dilate(img, kernel)
        cv2.imshow('manual dilation image', img_dilation_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_dilation_manual
    elif mor_op == 'erode':
        img_erosion = cv2.erode(img, kernel)
        cv2.imshow('OpenCV erosion image', img_erosion)

        img_erosion_manual = binary.erode(img, kernel)
        cv2.imshow('manual erosion image', img_erosion_manual)
        cv2.waitKey(wait_key_time)
        
        img_out = img_erosion_manual
    elif mor_op == 'open':
        img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN , kernel)
        cv2.imshow('OpenCV opening image', img_opening)

        img_opening_manual = binary.open(img, kernel)
        cv2.imshow('manual opening image', img_opening_manual)
        cv2.waitKey(wait_key_time)
        
        img_out = img_opening_manual
    elif mor_op == 'close':
        img_closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE , kernel)
        cv2.imshow('OpenCV closing image', img_closing)

        img_closing_manual = binary.close(img, kernel)
        cv2.imshow('manual closing image', img_closing_manual)
        cv2.waitKey(wait_key_time)
    elif mor_op == 'hitmiss':
        img_hitMiss= cv2.morphologyEx(img, cv2.MORPH_HITMISS , kernel)
        cv2.imshow('OpenCV hit or mis image', img_hitMiss)

        img_hitMiss_manual = binary.hitOrMiss(img, kernel)
        cv2.imshow('manual hit or miss image', img_hitMiss_manual)
        cv2.waitKey(wait_key_time)
        
        img_out = img_hitMiss_manual
    elif mor_op == 'thin':
        img_thin = cv2.ximgproc.thinning(img)
        cv2.imshow('OpenCV thinning image',img_thin)
        
        img_thin_manual = binary.thinning(img)
        cv2.imshow('manual thinning image',img_thin_manual)
        cv2.waitKey(wait_key_time)
        
        img_out = img_thin_manual
    elif mor_op == 'boundary_extract':
        
        img_BE_manual = binary.boundaryExtract(img)
        cv2.imshow('manual boundary-extract image',img_BE_manual)
        cv2.waitKey(wait_key_time)
        
        img_out = img_BE_manual
    elif mor_op == 'hole_fill':
        
        imgWithPInside = np.zeros(img.shape)
        # Đặt điểm p (nằm trong biên là trung tâm ảnh)
        imgWithPInside[img.shape[0]//2,img.shape[1]//2] = 255
        
        img_manual = binary.holeFilling(img, imgWithPInside)
        cv2.imshow('manual holeFilling image',img_manual)
        cv2.waitKey(wait_key_time)
        
        img_out = img_BE_manual
    elif mor_op == 'acc':
        # Tạo ảnh với 1 điểm P ở trong biên
        img_k = np.zeros(img.shape)
        # Đặt điểm p (nằm trong biên là trung tâm ảnh)
        img_k[img.shape[0]//2,img.shape[1]//2] = 255
        
        img_manual = binary.ExtractConnectedComponents(img,imgWithPInSide=img_k)
        cv2.imshow('manual ExtractConnectedComponents image',img_manual)
        cv2.waitKey(wait_key_time)
        
        img_out = img_manual    
    elif mor_op == 'convex_hull':
        img_manual = binary.ConvexHull(img)
        cv2.imshow('manual ConvexHull image',img_manual)
        cv2.waitKey(wait_key_time)
        
        img_out = img_manual      
    elif mor_op == 'thicken':
        img_manual = binary.thickening(img)
        cv2.imshow('manual thickening image',img_manual)
        cv2.waitKey(wait_key_time)
        
        img_out = img_manual       
    elif mor_op == 'skeleton':
        img_manual = binary.skeleton(img)
        cv2.imshow('manual skeletons image',img_manual)
        cv2.waitKey(wait_key_time)
        
        img_out = img_manual    
    
    if img_out is not None:
        cv2.imwrite(out_file, img_out)


def main(argv):
    input_file = ''
    output_file = ''
    mor_op = ''
    wait_key_time = 0

    description = 'main.py -i <input_file> -o <output_file> -p <mor_operator> -t <wait_key_time>'

    try:
        opts, args = getopt.getopt(argv, "hi:o:p:t:", ["in_file=", "out_file=", "mor_operator=", "wait_key_time="])
    except getopt.GetoptError:
        print(description)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print(description)
            sys.exit()
        elif opt in ("-i", "--in_file"):
            input_file = arg
        elif opt in ("-o", "--out_file"):
            output_file = arg
        elif opt in ("-p", "--mor_operator"):
            mor_op = arg
        elif opt in ("-t", "--wait_key_time"):
            wait_key_time = int(arg)

    print('Input file is ', input_file)
    print('Output file is ', output_file)
    print('Morphological operator is ', mor_op)
    print('Wait key time is ', wait_key_time)

    
    operator(input_file, output_file, mor_op, wait_key_time)
    cv2.waitKey(wait_key_time)


if __name__ == "__main__":
    main(sys.argv[1:])

    