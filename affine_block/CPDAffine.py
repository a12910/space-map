
import spacemap
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from pycpd import AffineRegistration
from pycpd import RigidRegistration

def extract_points(img, num_points=2000):
    # 仅提取边缘
    img = img.copy()
    img[img > 0] = 255
    img = img.astype(np.uint8)
    _, binary = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    y, x = np.where(binary > 0)
    points = np.column_stack((x, y))
    if len(points) > num_points:
        idx = np.random.choice(len(points), num_points, replace=False)
        points = points[idx]
    points = points.astype(np.float64)
    return points

def process_img_mid(img):
    # 中值滤波
    img = cv2.medianBlur(img, 5)
    return img

def process_img_hull(img):
    img[img > 0] = 255
    img = img.astype(np.uint8)
    _, binary = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_cnt = max(contours, key=cv2.contourArea)
    hull_mask = np.zeros_like(img)
    cv2.drawContours(hull_mask, [max_cnt], -1, 255, thickness=3)
    img = hull_mask
    return img

# 使用RigidRegistration
def cpd_rigid_imgs(imgI, imgJ, mode="mid"):
    if mode == "hull":
        imgI = process_img_hull(imgI)
        imgJ = process_img_hull(imgJ)
    elif mode == "mid":
        imgI = process_img_mid(imgI)
        imgJ = process_img_mid(imgJ)
    pts_fixed = extract_points(imgI)  # 目标点集
    pts_moving = extract_points(imgJ)  # 动图点集
    reg = RigidRegistration(X=pts_fixed, Y=pts_moving, max_iterations=100, tolerance=1e-3)
    TY, (s, R, t) = reg.register()
    M = np.zeros((2, 3), dtype=np.float64)
    M[:, :2] = (s * R).T
    # M[:, :2] = R.T
    M[:, 2] = t
    H = np.eye(3, dtype=np.float64)
    H[:2, :3] = M
    return H

def extract_points_3d(img, num_points=2000):
    # 转换为浮点型防止溢出
    img_float = img.astype(np.float32)
    n = 5
    kernel = np.ones((n, n), np.float32)
    z_map = cv2.filter2D(img_float, -1, kernel)
    
    # 提取非零点
    mask = img > 0
    y, x = np.where(mask)
    z = z_map[mask] * 2
    z[z > 255] = 255
    
    points = np.column_stack((x, y, z))
    
    # 降采样
    if len(points) > num_points:
        idx = np.random.choice(len(points), num_points, replace=False)
        points = points[idx]
        
    return points.astype(np.float64)

def cpd_affine_imgs(imgI, imgJ, mode="3d"):
    if mode == "hull":
        imgI = process_img_hull(imgI)
        imgJ = process_img_hull(imgJ)
    elif mode == "mid":
        imgI = process_img_mid(imgI)
        imgJ = process_img_mid(imgJ)
    if mode == '3d':
        pts_fixed = extract_points_3d(imgI)
        pts_moving = extract_points_3d(imgJ)
    else:
        pts_fixed = extract_points(imgI) # 目标点集
        pts_moving = extract_points(imgJ) # 动图点集
    reg = AffineRegistration(**{'X': pts_fixed, 'Y': pts_moving, 'max_iterations': 100, 'tolerance': 0.001})
    out_pts, (B, t) = reg.register()
    M_cpd = np.zeros((2, 3))
    M_cpd[:, :2] = B[:2, :2].T # 旋转缩放部分
    M_cpd[:, 2] = t[:2]   # 平移部分
    # rotated_cpd = cv2.warpAffine(img_mov, M_cpd, (w, h))
    H = np.eye(3)
    H[:2, :3] = M_cpd
    return H

from multiprocessing import Pool

class CPDAffine(spacemap.AffineBlock):
    def __init__(self):
        super().__init__("CPDAffine")
    
    def compute_img(self, imgI: np.array, imgJ: np.array, finder=None):
        return cpd_affine_imgs(imgI, imgJ)
        # return cpd_rigid_imgs(imgI, imgJ)

    def compute(self, dfI: np.array, dfJ: np.array, finder=None):
        imgI = spacemap.show_img(dfI)
        imgJ = spacemap.show_img(dfJ)
        H = self.compute_img(imgI, imgJ)
        H_np = spacemap.img.to_npH(H)
        return H_np

    @staticmethod
    def compute_each(imgIJ):
        imgI, imgJ = imgIJ
        f = CPDAffine()
        H = f.compute_img(imgI, imgJ)
        return H

    @staticmethod
    def compute_all_imgs(imgs):
        datas = []
        for i in range(len(imgs) - 1):
            datas.append((imgs[i], imgs[i+1]))
        with Pool(os.cpu_count()) as p:
            result = p.map(CPDAffine.compute_each, datas)
        Ms = [np.eye(3)] + result
        Hs = [convert_M_to_H(M) for M in Ms]
        return np.array(Hs)    

    @staticmethod
    def compute_all(dfs):
        imgs = [spacemap.show_img(df) for df in dfs]
        Hs = CPDAffine.compute_all_imgs(imgs)
        return Hs

def convert_M_to_H(M):
    H0 = [[M[1, 1], M[1, 0], M[1, 2]], [M[0, 1], M[0, 0], M[0, 2]]]
    H = np.eye(3)
    H[:2] = H0
    return H