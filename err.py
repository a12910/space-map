import numpy as np
import spacemap
import pandas as pd

import cv2
def midblur(img, size):
    return cv2.medianBlur(img, size)

def err_mutinfo(imgI, imgJ):
    import sklearn.metrics as skm
    return skm.mutual_info_score(imgI.reshape(-1), imgJ.reshape(-1))

def err_dice4(imgI, imgJ, mX, mY):
    mX = int(mX)
    mY = int(mY)
    i1, j1 = imgI[:mX, :mY], imgJ[:mX, :mY]
    i2, j2 = imgI[:mX, mY:], imgJ[:mX, mY:]
    i3, j3 = imgI[mX:, :mY], imgJ[mX:, :mY]
    i4, j4 = imgI[mX:, mY:], imgJ[mX:, mY:]
    d1 = err_dice(i1, j1)
    d2 = err_dice(i2, j2)
    d3 = err_dice(i3, j3)
    d4 = err_dice(i4, j4)
    return d1, d2, d3, d4

def err_dice(imgI, imgJ):
    i1 = imgI.reshape(-1)
    i2 = imgJ.reshape(-1)
    try:
        ii = np.array([i1, i2])
        inter = ii.min(axis=0).sum()
        return (2 * inter + 0.001) / (i1.sum() + i2.sum() + 0.001)
    except Exception as e:
        print(e)
        print(i1.shape, i2.shape)
        return 0

def err_dice1(imgI, imgJ):
    i1 = imgI.reshape(-1)
    i2 = imgJ.reshape(-1)
    i1[i1 > 0] = 1
    i2[i2 > 0] = 1
    ii = np.array([i1, i2])
    inter = ii.min(axis=0).sum()
    all = ii.max(axis=0).sum()
    return (inter + 0.0001) / (all + 0.0001)
    return (2 * inter + 0.001) / (i1.sum() + i2.sum() + 0.001)

def err_ssim(imgI, imgJ):
    from skimage.metrics import structural_similarity
    imgI = norm(imgI)
    imgJ = norm(imgJ)
    score = structural_similarity(imgI, imgJ, data_range=1.0)
    return score

def err_psnr(imgI, imgJ):
    from skimage.metrics import peak_signal_noise_ratio
    score = peak_signal_noise_ratio(imgI, imgJ)
    return score

def norm(img):
    return (img - img.min()) / (img.max() - img.min())

def err_mse(img1,img2):
    height, width = img1.shape[:2]
    img1_ = norm(img1)
    img2_ = norm(img2)
    error = np.sum((img1_ - img2_) ** 2) / (height * width)
    return error

def img_clahe(img, limit=20):
    img = np.array(img, dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=(8,8))
    cl1 = clahe.apply(img)
    cl1 -= cl1.min()
    return cl1
