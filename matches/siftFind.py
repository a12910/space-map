import cv2
import spacemap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
import os

def __sift_kp(image: np.array, method='sift', scale=False):
    if scale:
        maxx, minn = np.max(image), np.min(image)
        gray_image = (image - minn) * 255 / (maxx - minn)
        gray_image = gray_image.astype(np.uint8)
    else:
        if image.max() > 1.1:
            gray_image = image.astype(np.uint8)
        else:
            gray_image = (image * 255).astype(np.uint8)
            gray_image[gray_image > 255] = 255
    if method == 'sift':
        descriptor = cv2.SIFT_create()
    elif method == 'surf':
        # OpenCV4以上不可用
        descriptor = cv2.xfeatures2d.SURF_create()  
    elif method == 'orb':
        descriptor = cv2.ORB_create()
    elif method == 'sift_vgg':
        descriptor = cv2.SIFT_create()
    elif method == "orb_vgg":
        descriptor = cv2.ORB_create()
        # elif method == "surf_vgg":
        # descriptor = cv2.xfeatures2d.SURF_create()
    else:
        raise Exception("Unknown method")
    
    if method in ["sift", "surf", "orb"]:
        (kps, features) = descriptor.detectAndCompute(gray_image, None)
        return kps, features
    else:
        kps = descriptor.detect(gray_image)
        kps, features = descriptor.compute(gray_image, kps)
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

def autoSetMatchr(I, J, minCount):
    matchrs = spacemap.matches.siftFindMatchr(I, J)
    for match_ in range(6):
        i = 0.7 + match_ * 0.05
        if matchrs[i] > minCount:
            return i
    return 1.0

def siftImageAlignment(imgI,imgJ, matchr=0.75, method='sift', scale=False):
    """ [[ptIx, ptIy, ptJx, ptJy, distance]] """
    kp1,des1 = __sift_kp(imgI, method, scale)
    kp2,des2 = __sift_kp(imgJ, method, scale)
    bf = cv2.BFMatcher()
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return np.array([])
    matches = bf.knnMatch(des1, des2, k=2)
    matches2 = []
    for mat in matches:
        m, n = mat[0], mat[1]
        if m.distance > matchr * n.distance:
            continue
        pI1, pI2 = m.queryIdx, m.trainIdx
        pt1, pt2 = kp1[pI1].pt, kp2[pI2].pt
        # ptI, ptJ, dis
        matches2.append([pt1[1], pt1[0], pt2[1], pt2[0], 1000/m.distance])
    # 距离从小到大
    matches2.sort(key=lambda x: x[-1], reverse=True)
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
    """ H: df, H1: img """
    matches = np.array(matches, dtype=np.float32)
    ptsI = matches[:, :2].reshape(-1, 1, 2)
    ptsJ = matches[:, 2:4].reshape(-1, 1, 2)
    ransacReprojThreshold = 4
    # ptsJ -> ptsI
    H, mask =cv2.findHomography(ptsJ,ptsI,
                                  method,ransacReprojThreshold)
    if H is None:
        return None, None
    selectMatches = matches[mask.ravel() == 1]
    H1 = H.copy()
    matchesJ = spacemap.points.applyH_np(selectMatches[:, 2:4] * xyd, H, fromImgH=False)
    matchesI = selectMatches[:, :2] * xyd
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
    