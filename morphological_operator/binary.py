from pickletools import uint8
import numpy as np

import cv2

def erode(img, kernel):
    kernel_center = (kernel.shape[0] // 2, kernel.shape[1] // 2)
    kernel_ones_count = kernel.sum()
    img_shape = img.shape
    # Padding: cv2.copyMakeBorder(src, top, bottom, left, right, borderType, None, value)
    img = cv2.copyMakeBorder(img, kernel_center[0], kernel_center[0], kernel_center[1], kernel_center[1], cv2.BORDER_CONSTANT, None, 0)
    eroded_img = np.zeros(img.shape)
    for i in range(kernel_center[0],img_shape[0]+kernel_center[0]):
        for j in range(kernel_center[1], img_shape[1] + kernel_center[1]):
            left = i - kernel_center[0]
            right = i + kernel_center[0]
            top = j - kernel_center[1]
            bottum = j + kernel_center[1]
            if kernel_ones_count == (kernel * img[left: right + 1, top:bottum + 1]).sum() / 255:
                eroded_img[i, j] = 255
    return eroded_img[kernel_center[0]:img_shape[0]+kernel_center[0], kernel_center[1] :img_shape[1]+kernel_center[1]]


'''
TODO: implement morphological operators
'''


def dilate(img, kernel):
    kernel_center = (kernel.shape[0] // 2, kernel.shape[1] // 2)
    kernel_ones_count = kernel.sum()
    
    img_shape = img.shape
    
   # Padding: cv2.copyMakeBorder(src, top, bottom, left, right, borderType, None, value)
    img = cv2.copyMakeBorder(img, kernel_center[0], kernel_center[0], kernel_center[1], kernel_center[1], cv2.BORDER_CONSTANT, None, 0)
    
    dilated_img = img.copy()

    for i in range(kernel_center[0],img_shape[0]+kernel_center[0]):
        for j in range(kernel_center[1], img_shape[1] + kernel_center[1]):
            left = i - kernel_center[0]
            right = i + kernel_center[0]
            top = j - kernel_center[1]
            bottum = j + kernel_center[1]
            crop = img[left: right + 1, top:bottum + 1]
            # Điểm gốc có giá trị là màu trắng
            if crop[kernel_center[0],kernel_center[1]] == 255 and kernel_ones_count > (kernel * crop).sum() / 255:
                # Chuyển pixel có giá trị đen trong lân cận thành trắng
                for k in range(left,right + 1):
                    for l in range(top,bottum + 1):
                        if dilated_img[k,l] == 0 and img[k,l] == 0 and kernel[k-left,l-top] == 1:
                            dilated_img[k,l] = 255

    return dilated_img[kernel_center[0]:img_shape[0]+kernel_center[0], kernel_center[1] :img_shape[1]+kernel_center[1]]

def open(img,kernel):
    erosion_img = erode(img,kernel)
    opening_img = dilate(erosion_img,kernel)
    return opening_img

def neg(img):
    img_shape = img.shape
    neg_img = np.zeros(img_shape)
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            if img[i,j] == 0:
                neg_img[i,j] = 255
            else:
                neg_img[i,j] = 0
    return neg_img

def close(img,kernel):
    dilation_img = dilate(img,kernel)
    closing_img = erode(dilation_img,kernel)
    return closing_img

def hitOrMiss(img,kernel_hit,kernel_miss = np.zeros((3,3))):
    hit_img = erode(img,kernel_hit)
    neg_img = neg(img)
    miss_img = erode(neg_img,kernel_miss)
    HM_img = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if hit_img[i,j] == 255 and miss_img[i,j] == 255:
                HM_img[i,j] = 255
    return HM_img

# X\A
def subtract_set(X,A):
    img_shape = X.shape
    result = X.copy()
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            if X[i,j] == 255 and A[i,j] == 255:
                result[i,j] = 0
    return result


def boundaryExtract(img):
    kernel = np.ones((3, 3), np.uint8)
    eroded_img = erode(img,kernel)
    # Tính img\eroded_img
    BD_img = subtract_set(img,eroded_img)
    return BD_img
    
def intersect(img1, img2):
    inter_img = np.zeros(img1.shape)
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            if img1[i,j] == 255 and img2[i,j] == 255:
                inter_img[i,j] = img1[i,j]
    return inter_img    
    
def union(img1, img2):
    union_img = np.zeros(img1.shape)
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            if img1[i,j] == 255 or img2[i,j] == 255:
                union_img[i,j] = 255
    return union_img  
    
# Thin ảnh img với 1 kernel    
def thin(img,kernel_hit,kernel_miss):    
    HM_img = hitOrMiss(img,kernel_hit,kernel_miss)
    # X - hitOrMiss
    #thin_img = subtract_set(img,HM_img)
    
    # X intersect complement of hitOrMiss
    HM_img_neg = neg(HM_img)
    thin_img = intersect(img,HM_img_neg)
    return thin_img
    # thin_img = np.zeros(img.shape)
    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         if img[i,j] == 255 and HM_img_neg[i,j] == 255:
    #             thin_img[i,j] = 255
    # return thin_img
    
def thinning(img):
    B1_hit = np.array([[0,0,0],
                       [0,1,0],
                       [1,1,1]])
    B1_miss = np.array([[1,1,1],
                        [0,0,0],
                        [0,0,0]])
    
    B2_hit = np.array([[0,0,0],
                       [1,1,0],
                       [1,1,0]])
    B2_miss = np.array([[0,1,1],
                        [0,0,1],
                        [0,0,0]])
    
    B3_hit = np.array([[1,0,0],
                       [1,1,0],
                       [1,0,0]])
    B3_miss = np.array([[0,0,1],
                        [0,0,1],
                        [0,0,1]])
    
    B4_hit = np.array([[1,1,0],
                       [1,1,0],
                       [0,0,0]])
    B4_miss = np.array([[0,0,0],
                        [0,0,1],
                        [0,1,1]])
    
    B5_hit = np.array([[1,1,1],
                       [0,1,0],
                       [0,0,0]])
    B5_miss = np.array([[0,0,0],
                        [0,0,0],
                        [1,1,1]])
    
    B6_hit = np.array([[0,1,1],
                       [0,1,1],
                       [0,0,0]])
    B6_miss = np.array([[0,0,0],
                        [1,0,0],
                        [1,1,0]])
    
    B7_hit = np.array([[0,0,1],
                       [0,1,1],
                       [0,0,1]])
    B7_miss = np.array([[1,0,0],
                        [1,0,0],
                        [1,0,0]])
    
    B8_hit = np.array([[0,0,0],
                       [0,1,1],
                       [0,1,1]])
    B8_miss = np.array([[1,1,0],
                        [1,0,0],
                        [0,0,0]])
    
    kernel_hit = np.array([B1_hit,B2_hit,B3_hit,B4_hit,B5_hit,B6_hit,B7_hit,B8_hit])
    kernel_miss = np.array([B1_miss,B2_miss,B3_miss,B4_miss,B5_miss,B6_miss,B7_miss,B8_miss])
    # result lưu ảnh ở bước k 
    result = img.copy()
    
    # result_img kết quả sau n_kernel bước
    result_img = img.copy()
    
    n_kernel = 8
    _n_k = 0
    while(True):
        _n_k += 1
        print("thinning loop: ",_n_k)
        for i in range(n_kernel):
            # thin ảnh bước k+1
            thin_img = thin(result,kernel_hit[i],kernel_miss[i])
            result = thin_img
        # Nếu ảnh thin(X_(k+1)) = thin(X_k)
        if (thin_img==result_img).all():
            return result_img
        # Gán ảnh thin ở bước k vào result
        result_img = thin_img
        
        
    
# Lấp vùng
def holeFilling(img,imgWithPInside):
    kernel = np.array([[0,1,0],
                       [1,1,1],
                       [0,1,0]])
    neg_img = neg(img)
    result_img = imgWithPInside
    
    k = 0
    while(True):
        print(k)
        k+=1
        # Dilation
        dilated_img_k = dilate(imgWithPInside, kernel)
        # Intersection
        result_img = intersect(neg_img,dilated_img_k)
        # Kiểm tra điều kiện dừng img
        if (imgWithPInside==result_img).all():
            return union(result_img,img)
        # Gán ảnh thin ở bước k vào result
        imgWithPInside = result_img
    
# Rút thành phần liên thông
def ExtractConnectedComponents(img, imgWithPInSide):
    kernel = np.ones((3,3),dtype=np.uint8)
    
    result_img = imgWithPInSide
    k = 0
    while(True):
        print(k)
        k+=1
        # Dilation
        dilated_img_k = dilate(imgWithPInSide, kernel)
        # Intersection
        result_img = intersect(img,dilated_img_k)
        # Kiểm tra điều kiện dừng img
        if (imgWithPInSide==result_img).all():
            return result_img
        # Gán ảnh thin ở bước k vào result
        imgWithPInSide = result_img
        
# Bao lồi
def ConvexHull(img):
    B1_hit = np.array([[1,0,0],
                       [1,0,0],
                       [1,0,0]])
    B1_miss = np.array([[0,0,0],
                        [0,1,0],
                        [0,0,0]])
    
    B2_hit = np.array([[1,1,1],
                       [0,0,0],
                       [0,0,0]])
    B2_miss = np.array([[0,0,0],
                        [0,1,0],
                        [0,0,0]])
    
    B3_hit = np.array([[0,0,1],
                       [0,0,1],
                       [0,0,1]])
    B3_miss = np.array([[0,0,0],
                        [0,1,0],
                        [0,0,0]])
    
    B4_hit = np.array([[0,0,0],
                       [0,0,0],
                       [1,1,1]])
    B4_miss = np.array([[0,0,0],
                        [0,1,0],
                        [0,0,0]])
    
    kernel_hit = np.array([B1_hit,B2_hit,B3_hit,B4_hit])
    kernel_miss = np.array([B1_miss,B2_miss,B3_miss,B4_miss])
    
    
    n_kernel = 4
    _iImg = []
    for i in range(n_kernel):
        _iImg.append(img.copy())
    # Lưu ảnh khi xử lý kernel Bi tại bước k - 1
    img_i = np.array(_iImg)
    
    for j in range(n_kernel):
        print("Kernel ",j + 1,":")
        count = 0
        img_ik = img_i[j]
        # Lưu ảnh khi xử lý kernel Bi tại bước k
        result_imgi = np.zeros(img_ik.shape)
        flag = True
        while(flag):
            count += 1
            print(count)
            # Hit or Miss Bi
            HM_img_k = hitOrMiss(img_ik, kernel_hit[j],kernel_miss[j])
            # Union
            result_imgi = union(img_ik,HM_img_k)
            # Kiểm tra điều kiện dừng img
            if (img_ik==result_imgi).all():
                flag = False
            # Gán ảnh thin ở bước k vào result
            img_ik = result_imgi
        # Lưu ảnh kết quả vào lại mảng img_i
        img_i[j] = img_ik
    
    # Hợp Di
    result = img_i[0]
    for j in range(1,n_kernel):
        result = union(result,img_i[j])
    return result
        
        
# Thick ảnh img với 1 kernel    
def thick(img,kernel_hit,kernel_miss):    
    HM_img = hitOrMiss(img,kernel_hit,kernel_miss)
    thicken_img = union(img,HM_img)
    return thicken_img
    
def thickening1(img):
    ca = neg(img)
    ca = thinning(ca)
    return neg(ca)    
    
def thickening(img):
    B1_hit = np.array([[0,0,0],
                       [0,1,0],
                       [1,1,1]])
    B1_miss = np.array([[1,1,1],
                        [0,0,0],
                        [0,0,0]])
    
    B2_hit = np.array([[0,0,0],
                       [1,1,0],
                       [1,1,0]])
    B2_miss = np.array([[0,1,1],
                        [0,0,1],
                        [0,0,0]])
    
    B3_hit = np.array([[1,0,0],
                       [1,1,0],
                       [1,0,0]])
    B3_miss = np.array([[0,0,1],
                        [0,0,1],
                        [0,0,1]])
    
    B4_hit = np.array([[1,1,0],
                       [1,1,0],
                       [0,0,0]])
    B4_miss = np.array([[0,0,0],
                        [0,0,1],
                        [0,1,1]])
    
    B5_hit = np.array([[1,1,1],
                       [0,1,0],
                       [0,0,0]])
    B5_miss = np.array([[0,0,0],
                        [0,0,0],
                        [1,1,1]])
    
    B6_hit = np.array([[0,1,1],
                       [0,1,1],
                       [0,0,0]])
    B6_miss = np.array([[0,0,0],
                        [1,0,0],
                        [1,1,0]])
    
    B7_hit = np.array([[0,0,1],
                       [0,1,1],
                       [0,0,1]])
    B7_miss = np.array([[1,0,0],
                        [1,0,0],
                        [1,0,0]])
    
    B8_hit = np.array([[0,0,0],
                       [0,1,1],
                       [0,1,1]])
    B8_miss = np.array([[1,1,0],
                        [1,0,0],
                        [0,0,0]])
    
    kernel_miss = np.array([B1_hit,B2_hit,B3_hit,B4_hit,B5_hit,B6_hit,B7_hit,B8_hit])
    kernel_hit = np.array([B1_miss,B2_miss,B3_miss,B4_miss,B5_miss,B6_miss,B7_miss,B8_miss])
    # result lưu ảnh ở bước k 
    result = img.copy()
    
    # result_img kết quả sau n_kernel bước
    result_img = img.copy()
    
    n_kernel = 8
    _n_k = 0
    while(True):
        _n_k += 1
        print("thickening loop: ",_n_k)
        for i in range(n_kernel):
            # thin ảnh bước k+1
            thicken_img = thick(result,kernel_hit[i],kernel_miss[i])
            result = thicken_img
        # Nếu ảnh thin(X_(k+1)) = thin(X_k)
        if (thicken_img==result_img).all():
            return result_img
        # Gán ảnh thin ở bước k vào result
        result_img = thicken_img
     
     
def skeleton(img):
    # Set kernel
    kernel = np.ones((3,3),dtype = np.uint8)
    kernel = np.array([[0,1,0],
                       [1,1,1],
                       [0,1,0]])
    k = 0
    #
    eroded_img_k = img.copy()
    
    Opened_EroImg_k = open(eroded_img_k,kernel)
    # S_k(A)
    sket_k = subtract_set(eroded_img_k,Opened_EroImg_k)
    
    # S(A)
    result = np.zeros(img.shape)
    result = union(result, sket_k)
    
    while(True):
        k += 1
        print("K = ",k)
        # Erosion
        eroded_img_k = erode(eroded_img_k,kernel)
        # Opening
        Opened_EroImg_k = open(eroded_img_k,kernel)
        # subtract
        sket_k = subtract_set(eroded_img_k,Opened_EroImg_k)
        # Union
        result = union(result, sket_k)
        # Check if empty set
        if cv2.countNonZero(eroded_img_k) == 0:
            return result
        
        
        
       