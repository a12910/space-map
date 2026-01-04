
from spacemap import SliceImg

class AffineGlobalFix:
    def __init__(self):
        pass

    def applyH_ps(self, H, ps):
        ps = S.ps(key)
        ps2 = SliceImg.applyH_ps(ps, H)
        img2 = spacemap.show_img(ps2)
        return img2, ps2

    def applyH_img(self, H, img):
        return SliceImg.applyH_img(img, H)

    def compute(self, initHs, initPS):
        imgs = [self.applyH_img(H, img) for H, img in zip(initHs, initPS)]
        

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d

def extract_contour_radial(img, center, num_angles=360):
    """
    极坐标采样：以中心为原点，提取图像轮廓在各个角度上的半径。
    返回: (num_angles, 2) 的坐标数组 [(x0,y0), (x1,y1), ...]
    """
    mask = img > 0
    # 2. 寻找最大轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((num_angles, 2))
    
    cnt = max(contours, key=cv2.contourArea)
    
    cnt = cnt.squeeze() # (N, 2)
    if cnt.ndim == 1: return np.zeros((num_angles, 2)) # 异常处理

    # 转换为相对于中心的坐标
    dx = cnt[:, 0] - center[0]
    dy = cnt[:, 1] - center[1]
    
    angles = np.arctan2(dy, dx) # (-pi, pi)
    radii = np.sqrt(dx**2 + dy**2)
    
    # 处理角度跳变 (-pi -> pi)，为了插值需要排序
    sort_idx = np.argsort(angles)
    angles = angles[sort_idx]
    radii = radii[sort_idx]
    
    # 解决周期性边界问题 (拼接首尾)
    angles = np.concatenate(([angles[-1] - 2*np.pi], angles, [angles[0] + 2*np.pi]))
    radii = np.concatenate(([radii[-1]], radii, [radii[0]]))
    
    # 生成目标角度 (-pi 到 pi)
    target_angles = np.linspace(-np.pi, np.pi, num_angles, endpoint=False)
    
    # 插值得到每个角度的半径
    target_radii = np.interp(target_angles, angles, radii)
    
    # 转回笛卡尔坐标 (相对于中心)
    x_rel = target_radii * np.cos(target_angles)
    y_rel = target_radii * np.sin(target_angles)
    
    # 转回绝对坐标
    x_abs = x_rel + center[0]
    y_abs = y_rel + center[1]
    
    return np.column_stack((x_abs, y_abs))

def global_surface_smoothing(images, initial_matrices, smooth_sigma_z=5.0):
    """
    全局表面平滑对齐算法 (Virtual Smooth Surface Alignment)
    
    参数:
        images: 图片列表
        initial_matrices: 粗略对齐后的矩阵列表 (必须先有粗略对齐，否则无法建立对应关系)
        smooth_sigma_z: 纵向平滑的强度 (Z轴方向的高斯核大小)
                        值越大，生成的"管子"越直/越滑；值越小，越贴近原始数据。
    """
    n_imgs = len(images)
    h, w = images[0].shape[:2]
    center = np.array([w/2, h/2])
    
    num_angles = 360 # 采样分辨率
    
    # 存储所有层采样后的点: shape (N_imgs, 360, 2)
    all_layer_points_global = np.zeros((n_imgs, num_angles, 2))
    
    for i in range(n_imgs):
        # A. 提取原始图像轮廓 (未变换)
        raw_contour = extract_contour_radial(images[i], center, num_angles)
        
        # B. 应用当前的粗略矩阵，将其映射到全局空间
        # 这是为了让所有层在同一个坐标系下比较
        H = initial_matrices[i]
        
        # 变换: P_global = H * P_raw
        ones = np.ones((num_angles, 1))
        pts_homo = np.hstack([raw_contour, ones])
        pts_global = (H @ pts_homo.T).T
        
        all_layer_points_global[i] = pts_global[:, :2]

    # === 2. 纵向平滑阶段 (核心) ===
    print(f"正在进行纵向表面平滑 (Sigma={smooth_sigma_z})...")
    
    # 我们现在的 all_layer_points_global 可以看作是一个 (N, 360, 2) 的张量
    # 我们沿着 axis=0 (层数 Z) 进行高斯平滑
    # 这一步相当于把每一条贯穿 z 轴的纤维给拉直了
    smoothed_points_global = gaussian_filter1d(all_layer_points_global, sigma=smooth_sigma_z, axis=0)

    # === 3. 矩阵修正阶段 ===
    print("正在计算最终修正矩阵...")
    final_matrices = []
    
    for i in range(n_imgs):
        # 原始图像上的点 (Raw Source)
        # 注意：我们要计算的是 从 "Raw Image" 到 "Smoothed Global Surface" 的变换
        # 所以源点是 extract_contour_radial 直接出来的点，不带 H 的
        src_pts = extract_contour_radial(images[i], center, num_angles).astype(np.float32)
        
        # 目标点 (Smoothed Target)
        # 这是经过 Z 轴平滑后的理想位置
        dst_pts = smoothed_points_global[i].astype(np.float32)
        
        # 剔除无效点 (比如全黑图像导致的 (0,0))
        valid_idx = np.linalg.norm(src_pts - center, axis=1) > 1.0
        if np.sum(valid_idx) < 10:
            final_matrices.append(initial_matrices[i]) # 无法对齐，保持原样
            continue
            
        src_pts = src_pts[valid_idx]
        dst_pts = dst_pts[valid_idx]
        
        # 计算最佳变换 (Similarity: Rotation + Translation + Scale)
        # 这样每一层都会被迫移动、旋转、缩放，以贴合那个光滑的"虚拟管道"
        H_new, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
        
        if H_new is None:
            final_matrices.append(initial_matrices[i])
        else:
            final_matrices.append(H_new)
            
    return final_matrices
    