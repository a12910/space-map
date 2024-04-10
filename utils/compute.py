import cv2
import spacemap
import numpy as np

def rotate_img(imgJ, rotate, center=None):
    """ rotate: int
        center: int, int/img 
    """
    h, w = imgJ.shape[:2]
    imgJ = np.array(imgJ, dtype=np.uint8)
    if center is not None:
        center = center[1], center[0]
    else:
        center = h // 2, w // 2
    M = cv2.getRotationMatrix2D(center, rotate, 1.0)
    rotatedI = cv2.warpAffine(imgJ, M, (w, h))
    return rotatedI

def rotate_H(imgJ, rotate, center=None):
    xyd = spacemap.XYD
    h, w = imgJ.shape[:2]
    if center is None:
        center = h // 2, w // 2
    else:
        center = center[0], center[1]
    meanX = center[0] * xyd
    meanY = center[1] * xyd
    H11 = np.array([[1, 0, -meanX], [0, 1, -meanY], [0, 0, 1]])
    
    r = rotate / 360 * np.pi * 2
    cosr = np.cos(r)
    sinr = np.sin(r)
    H12 = np.array([[cosr, -sinr, 0], [sinr, cosr, 0], [0, 0, 1]])
    H13 = np.array([[1, 0, meanX], [0, 1, meanY], [0, 0, 1]])    
    H = np.dot(H13, np.dot(H12, H11))
    return H
