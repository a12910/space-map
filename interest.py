import spacemap
import cv2
import numpy as np

def compute_interest_area(img, distance):
    if distance < 1:
        distance = int(distance * img.shape[0])
    img = img.copy()
    img[img > 0] = 1
    kernel = np.ones((distance * 2 + 1, distance * 2 + 1))
    img2 = spacemap.conv_2d(img, kernel)
    img2[img2 > 0] = 1
    return img2

def compute_interest_area_inter(imgI, imgJ, distance):
    inteI = compute_interest_area(imgI, distance)
    inteJ = compute_interest_area(imgJ, distance)
    inteI += inteJ
    inteI[inteI < 2] = 0
    inteI[inteI > 1] = 1
    imgI = imgI.copy()
    imgI[inteI < 1] = 0
    imgJ = imgJ.copy()
    imgJ[inteI < 1] = 0
    return imgI, imgJ, inteI
