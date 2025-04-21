import spacemap
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import os
from scipy.spatial.distance import cdist
import cv2

class LastImgStore(spacemap.AffineBlock):
    def __init__(self):
        super().__init__("LastImgStore")
        self.lastImgs = []
        self.lastEdges = []
        self.N = 10
        self.finder = None
        
        self.losses = [] # loss 4
        self.lossResults = [] # H, i
        self.lastH = np.eye(3)

        self.weight = [0.2, 0.2, 0.2, 0.4] # edge_loss, sim_loss, last_sim, affine_loss
        
    def compute_img(self, imgI: np.array, imgJ: np.array, finder=None):
        self.finder = finder
        if len(self.lastImgs) == 0:
            self.lastImgs.append(imgI)   
            self.lastEdges.append(get_edge_points(imgI))
        else:
            self.lastImgs.append(imgJ)
            self.lastEdges.append(get_edge_points(imgJ))
        return np.eye(3)
    
    def get_last_imgs(self):
        if len(self.lastEdges) < self.N * 2:
            return self.lastImgs[-self.N:], self.lastEdges[-self.N:]
        elif len(self.lastEdges) < self.N * 3:
            return self.lastImgs[-self.N*2::2], self.lastEdges[-self.N*2::2]
        elif len(self.lastEdges) < self.N * 4:
            return self.lastImgs[-self.N*3::3], self.lastEdges[-self.N*3::3]
        else:
            return self.lastImgs[-self.N*4::4], self.lastEdges[-self.N*4::4]
    
    def clear(self):
        self.lastImgs = []
        self.lastEdges = []
        self.losses = []
        self.lossResults = []
        
    def add_img(self, img):
        self.lastImgs.append(img)
        self.lastEdges.append(get_edge_points(img))
        
    def clear_best(self):
        self.losses = []
        self.lossResults = []
    
    def _compute_current_edge_loss(self, img):
        if self.weight[0] == 0:
            return 0
        current_edges = get_edge_points(img)
        edge_loss = 0
        _, lastEdges = self.get_last_imgs()
        N = len(lastEdges)
        for i in range(N):
            prev_edges = lastEdges[i]
            if len(prev_edges) == 0 or len(current_edges) == 0:
                continue
            dist_forward = cdist(prev_edges, current_edges).min(axis=1).mean()
            dist_backward = cdist(current_edges, prev_edges).min(axis=0).mean()
            edge_loss += (dist_forward + dist_backward) / 2
            # print("edge_loss", dist_forward, dist_backward)
        edge_loss /= N
        return edge_loss * 0.1
    
    def _compute_sim_loss(self, img):
        if self.weight[1] == 0:
            return 0
        loss = 0
        lastImgs, _ = self.get_last_imgs()
        for i in range(len(lastImgs)):
            imgI = lastImgs[i]
            loss += self.finder.err(imgI, img)
        loss /= len(lastImgs)
        return loss
    
    def _compute_last_sim(self, imgI, imgJ):
        if self.weight[2] == 0:
            return 0
        return self.finder.err(imgI, imgJ)
    
    def _compute_loss(self, imgI, imgJ, H):
        if imgJ.max() < 0.1:
            return [1e10, 1e10, 1e10, 1e10]
        imgJ2 = spacemap.he_img.rotate_imgH(imgJ, H)
        edge_loss = self._compute_current_edge_loss(imgJ2)
        sim_loss = self._compute_sim_loss(imgJ2)
        last_sim = self._compute_last_sim(imgI, imgJ)
        affine_loss = self._compute_affine_loss(H)
        # return edge_loss * self.weight[0] + sim_loss * self.weight[1] + last_sim * self.weight[2] + affine_loss * self.weight[3]
        return [edge_loss, sim_loss, last_sim, affine_loss]
    
    def _compute_last_only(self, imgI, imgJ):
        loss = self.finder.err(imgI, imgJ)
        return loss
    
    def compute_loss(self, imgI, imgJ, H):
        loss = self._compute_loss(imgI, imgJ, H)
        return loss
    
    def _compute_affine_loss(self, H):
        if self.weight[3] == 0:
            return 0
        sx, sy, sh = decompose_affine(H)
        affine_loss = abs(sx ** 2 - 1) + abs(sy ** 2 - 1) + abs(sh)
        return affine_loss * 100
    
    def update_best(self, i, H, imgI, imgJ, loss):
        self.losses.append(loss)
        self.lossResults.append((i, H, imgI, imgJ))
        
    def best(self):
        losses = np.array(self.losses)
        for i in range(len(self.weight)):
            median = np.median(losses[:, i])
            mask = losses[:, i] > median
            losses[mask, i] = 1e10
        # minn = losses.min(axis=0)
        # maxx = losses.max(axis=0) + 0.01
        # losses = (losses - minn) / (maxx - minn)
        losses1 = np.zeros(len(losses))
        for i in range(len(self.weight)):
            losses1 += losses[:, i] * self.weight[i]
        bestIndex = losses1.argmin()
        bestResult = self.lossResults[bestIndex]
        return bestResult
    
def get_edge_points(img):
    return get_outermost_edge_points(img)
    # return get_convex_hull_edges(img)

def get_edge_points_basic(img, threshold=50):
    """提取二值化边缘点"""
    img = img / img.max() * 255
    edges = cv2.Canny(img.astype(np.uint8), threshold, 150)
    y, x = np.where(edges > 0)
    return np.column_stack((x, y)) if len(x) > 0 else np.zeros((0,2))

def get_outermost_edge_points(img, downsample_step=3):
    """ 
    仅提取最外层轮廓点（多物体场景会合并所有外层轮廓）
    """
    if img.max() < 0.1:
        return np.zeros((0,2))
    img = img / img.max() * 255
    # 二值化处理（假设输入为0-255灰度图）
    _, thresh = cv2.threshold(img.astype(np.uint8), 1, 255, cv2.THRESH_BINARY)
    
    # 仅检测最外层轮廓（RETR_EXTERNAL模式）
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 合并所有外层轮廓点并下采样
    all_points = []
    for cnt in contours:
        # 轮廓点格式转换: (N,1,2) -> (N,2)
        pts = cnt.squeeze(1)
        # 下采样
        all_points.extend(pts[::downsample_step])
    
    return np.array(all_points) if len(all_points)>0 else np.zeros((0,2))

def get_convex_hull_edges(img, downsample_step=3, merge_contours=True):
    """
    提取最外侧轮廓的凸包点
    """
    # 二值化与最外层轮廓检测
    _, thresh = cv2.threshold(img.astype(np.uint8), 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return np.zeros((0,2))
    
    # 选项1: 合并所有轮廓为一个凸包（适用于相邻物体）
    if merge_contours:
        all_points = np.vstack([cnt[::downsample_step] for cnt in contours])
        if len(all_points) < 3: 
            return all_points.squeeze()
        hull = cv2.convexHull(all_points)
        return hull.squeeze()
    
    # 选项2: 每个轮廓独立计算凸包（保留多个独立物体）
    hull_points = []
    for cnt in contours:
        cnt_sampled = cnt[::downsample_step]
        if len(cnt_sampled) < 3:
            continue
        hull = cv2.convexHull(cnt_sampled)
        hull_points.extend(hull.squeeze().tolist())
    
    return np.array(hull_points) if hull_points else np.zeros((0,2))

def decompose_affine(matrix):
    # 分解矩阵为平移(tx,ty)、旋转θ、缩放(sx,sy)、剪切sh
    tx, ty = matrix[0,2], matrix[1,2]
    a, b = matrix[0,0], matrix[0,1]
    c, d = matrix[1,0], matrix[1,1]
    
    sx = np.sqrt(a**2 + b**2)
    sy = np.sqrt(c**2 + d**2)
    theta = np.arctan2(c, a)
    sh = np.arctan2(-b, a) - theta
    return sx, sy, sh