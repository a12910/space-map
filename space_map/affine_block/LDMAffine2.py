from space_map.registration.lddmm import SVFLDDMM
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import space_map

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import numpy as np
import torch

import numpy as np
def estimate_affine_matrix(psJ, psJ2):
    # 确保输入的点集是 numpy 数组
    psJ = np.asarray(psJ)
    psJ2 = np.asarray(psJ2)
    ones = np.ones((psJ.shape[0], 1))
    psJ_homogeneous = np.hstack([psJ, ones])
    # 通过最小二乘法求解线性方程，得到仿射变换矩阵
    # Reshape psJ2 为 (N*2, 1) 形状
    psJ2_reshaped = psJ2.reshape(-1, 1)
    # 求解 A * X = psJ2_reshaped，其中 A 是 psJ_homogeneous 的卡特西乘积，即 Kronecker product
    A = np.kron(psJ_homogeneous, np.eye(2))
    X, _, _, _ = np.linalg.lstsq(A, psJ2_reshaped, rcond=None)
    # 将结果 X 再转换为 2*3 的仿射变换矩阵
    affine_matrix = X.reshape(2, 3)
    return affine_matrix

class SVFAffine(space_map.AffineBlock):
    def __init__(self):
        super().__init__("SVFAffine")
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.H = np.eye(3)
        self.aff = SVFLDDMM()

    def compute(self, dfI: np.array, dfJ: np.array, finder=None):
        imgI = space_map.show_img(dfI)
        imgJ = space_map.show_img(dfJ)
        self.aff.load_img(imgI, imgJ)
        psJ2 = self.aff.apply_points2d(psJ, space_map.XYD)
        H = estimate_affine_matrix(psJ, psJ2)
        return H
    
    def compute_img(self, imgI: np.array, imgJ: np.array, finder=None):
        # self.aff.
        # imgJ2, flow = self.aff.run_high_res_loss_low_res_grid(imgJ, imgI)
        # self.H = fit_affine_from_flow(flow, imgJ.shape[0], imgJ.shape[1], 1)
        # return self.H
        raise Exception("")
        return None
