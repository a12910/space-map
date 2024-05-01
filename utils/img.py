import scipy.signal as ss
import matplotlib.pyplot as plt 
import numpy as np
import spacemap
import cv2

def conv_2d(img, kernel):
    img1 = ss.convolve2d(img, kernel, mode="same")
    return img1

def convert_H_to_M(H):
    M = [[H[1, 1], H[1, 0], H[1, 2]], [H[0, 1], H[0, 0], H[0, 2]]]
    M = np.array(M)
    return M

def convert_M_to_H(M):
    H0 = [[M[1, 1], M[1, 0], M[1, 2]], [M[0, 1], M[0, 0], M[0, 2]]]
    H = np.eye(3)
    H[:2] = H0
    return H

def rotate_H(rotate, center, scale):
    center = center[1], center[0]
    M = cv2.getRotationMatrix2D(center, rotate, scale)
    H = convert_M_to_H(M)
    return H, M

def rotate_imgH(imgJ, H, scale=None):
    if scale is not None:
        H = H * scale
    return rotate_imgM(imgJ, convert_H_to_M(H))

def rotate_imgM(imgJ, M):
    h, w = imgJ.shape[:2]
    rotatedI = cv2.warpAffine(imgJ, M, (w, h))
    return rotatedI

def to_npH(H: np.array):
    H = H.copy()
    xyd = spacemap.XYD
    H[0, 2] = H[0, 2] * xyd
    H[1, 2] = H[1, 2] * xyd
    return H

def to_imgH(H: np.array):
    H = H.copy()
    xyd = spacemap.XYD
    H[0, 2] = H[0, 2] / xyd
    H[1, 2] = H[1, 2] / xyd
    return H
