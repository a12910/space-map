import cv2
import spacemap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
import os

def __sift_kp(image: np.array, method='sift'):
    maxx, minn = np.max(image), np.min(image)
    gray_image = (image - minn) * 255 / (maxx - minn)
    gray_image = gray_image.astype(np.uint8)
    
    if method == 'sift':
        descriptor = cv2.SIFT_create()
    elif method == 'surf':
        descriptor = cv2.xfeatures2d.SURF_create()  # OpenCV4以上不可用
    elif method == 'brisk':
        descriptor = cv2.BRISK_create()
    elif method == 'orb':
        descriptor = cv2.ORB_create()
    elif method == 'akaze':
        descriptor = cv2.AKAZE_create()
    (kps, features) = descriptor.detectAndCompute(gray_image, None)
    return kps, features

def match_des(des1, des2, k):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=k)
    return matches

def siftFindMatchr(imgI, imgJ, method='sift'):
    kp1,des1 = __sift_kp(imgI, method)
    kp2,des2 = __sift_kp(imgJ, method)
    bf = cv2.BFMatcher()
    result = {}
    matches = bf.knnMatch(des1, des2, k=2)
    for match_ in range(6):
        matchr = 0.7 + match_ * 0.05
        result[matchr] = 0
        for match in matches:
            m, n = match[0], match[1]
            if m.distance < matchr * n.distance:
                result[matchr] += 1
    return result

def siftImageAlignment(imgI,imgJ, matchr=0.75, method='sift'):
    """ [[ptIx, ptIy, ptJx, ptJy, distance]] """
    kp1,des1 = __sift_kp(imgI, method)
    kp2,des2 = __sift_kp(imgJ, method)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    matches2 = []
    for match in matches:
        m, n = match[0], match[1]
        if m.distance > matchr * n.distance:
            continue
        pI1, pI2 = m.queryIdx, m.trainIdx
        pt1, pt2 = kp1[pI1].pt, kp2[pI2].pt
        # ptI, ptJ, dis
        matches2.append([pt1[1], pt1[0], pt2[1], pt2[0], m.distance])
    matches2.sort(key=lambda x: x[-1])
    return np.array(matches2)

def siftImageAlignment2(img1,img2, k, matchr=0.75, method='sift'):
    kp1,des1 = __sift_kp(img1, method)
    kp2,des2 = __sift_kp(img2, method)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=k)
    matches2 = []
    for match in matches:
        m, n = match[0], match[1]
        if m.distance > matchr * n.distance:
            continue
        match1 = []
        for m in match:
            pI1, pI2 = m.queryIdx, m.trainIdx
            pt1, pt2 = kp1[pI1].pt, kp2[pI2].pt
            # df和img顺序相反
            match1.append([pt1[1], pt1[0], pt2[1], pt2[0], m.distance])
        matches2.append(match1)
    return matches2

def createHFromPoints2(matches, xyd, method=cv2.RANSAC):
    matches = np.array(matches, dtype=np.float32)
    ptsI = matches[:, :2].reshape(-1, 1, 2)
    ptsJ = matches[:, 2:4].reshape(-1, 1, 2)
    ransacReprojThreshold = 4
    # ptsJ -> ptsI
    H, status =cv2.findHomography(ptsJ,ptsI,
                                  method,ransacReprojThreshold)
    if H is None:
        return None, None
    H1 = H.copy()
    matchesJ = spacemap.applyH_np(matches[:, 2:4] * xyd, H)
    matchesI = matches[:, :2] * xyd
    errX, errY = np.mean(matchesI - matchesJ, axis=0)
    H[0, 2] += errX
    H[1, 2] += errY
    return H, H1

def compute_H_from_3points(A, B):
    (ax, ay), (bx, by), (cx, cy) = A
    (ax1, ay1), (bx1, by1), (cx1, cy1) = B

    A = np.array([
        [ax, ay, 1, 0, 0, 0],
        [0, 0, 0, ax, ay, 1],
        [bx, by, 1, 0, 0, 0],
        [0, 0, 0, bx, by, 1],
        [cx, cy, 1, 0, 0, 0],
        [0, 0, 0, cx, cy, 1]
    ])
    B = np.array([ax1, ay1, bx1, by1, cx1, cy1]).reshape(6, 1) # 比手写6X1矩阵要省事
    M = np.linalg.inv(A.T @ A) @ A.T @ B # 套公式
    H = np.zeros((3, 3))
    H[:2, :3] = M.reshape(2, 3)
    H[2, 2] = 1
    return H
    