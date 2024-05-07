import scipy.signal as ss
import matplotlib.pyplot as plt 
import numpy as np
import spacemap
import cv2
import torch
import torch.nn.functional as F

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

def apply_img_by_Grid(img_, grid):
    I = torch.tensor(img_).type(torch.FloatTensor)
    grid = torch.tensor(grid).type(torch.FloatTensor)
    It = torch.squeeze(F.grid_sample(I.unsqueeze(0).unsqueeze(0),grid,padding_mode='zeros',mode='bilinear', align_corners=True))
    return It.cpu().numpy()

def merge_img_grid(grid0, grid1):
    # 1. grid0 - 2. grid1
    grid0 = torch.tensor(grid0).type(torch.FloatTensor)
    grid1 = torch.tensor(grid1).type(torch.FloatTensor)
    grid01 = F.grid_sample(grid0.permute(0, 3, 1, 2), 
                           grid1, mode='bilinear', 
                           padding_mode='zeros', align_corners=True).permute(0, 2, 3, 1)
    return grid01.cpu().numpy()
