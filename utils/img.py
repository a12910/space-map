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

def scale_H(H, shape, target):
    H = H.copy()
    s = shape[0]
    t = target[0]
    H[0, 2] *= t / s
    H[1, 2] *= t / s
    return H

def get_shape():
    xyd = spacemap.XYD
    xyr = spacemap.XYRANGE
    affineShape = (int(xyr // xyd), int(xyr // xyd))
    return affineShape

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

def to_npH(H: np.array, xyd=None):
    H = H.copy()
    xyd = xyd if xyd is not None else spacemap.XYD
    H[0, 2] = H[0, 2] * xyd
    H[1, 2] = H[1, 2] * xyd
    return H

def to_imgH(H: np.array):
    H = H.copy()
    xyd = spacemap.XYD
    H[0, 2] = H[0, 2] / xyd
    H[1, 2] = H[1, 2] / xyd
    return H

def apply_transform(S: spacemap.Slice2, img: np.array, affineShape, 
                        initIndex, 
                        affineKey="cell", gridKey="final_ldm"):
    affine = S.data.loadH(initIndex, affineKey)
    grid = S.data.loadGrid(initIndex, gridKey)
    shape = img.shape
    if affineShape is None:
        affineShape = get_shape()
    affine1 = scale_H(affine, affineShape, shape)
    img1 = rotate_imgH(img, affine1)
    img2 = apply_img_by_grid(img1, grid)
    return img2

def apply_img_by_grid(img_: np.array, grid: np.array):
    import torch
    import torch.nn.functional as F
    
    # grid: N*N*2 img: N*N*C / N*N
    if len(grid.shape) == 4:
        grid = grid[0]
    grid = torch.tensor(grid).type(torch.FloatTensor).unsqueeze(0)
    if grid.shape[1:3] != img_.shape[:2]:
        upscaled_grid = F.interpolate(grid.permute(0, 3, 1, 2), 
                                      size=img_.shape[:2], mode='bilinear', align_corners=True)
        grid = upscaled_grid.permute(0, 2, 3, 1)
    I = torch.tensor(img_).type(torch.FloatTensor)
    if len(img_.shape) == 2:
        I = I.unsqueeze(0)
    else:
        I = I.permute(2, 0, 1)
    I = I.unsqueeze(0)
    It = F.grid_sample(I,grid,padding_mode='zeros',mode='bilinear', align_corners=True)
    It = It.squeeze(0)
    if len(img_.shape) == 2:
        It = It.squeeze(0)
    else:
        It = It.permute(1, 2, 0)
    It = It.cpu().numpy()
    It = It.clip(0, 1)
    # meanOld = np.mean(img_)
    # meanNew = np.mean(It)
    # It = It * meanOld / meanNew
    if img_.max() > 1.1:
        It = It.astype(np.uint8)
    return It

# def merge_affine_to_grid(affine: np.array, grid=None, shape=None):
#     import torch
#     import torch.nn.functional as F
#     if shape is None and grid is None:
#         raise Exception("need shape")
#     if shape is None:
#         shape = grid.shape
#     # affine = spacemap.img.convert_H_to_M(affine)
#     theta = torch.tensor(affine[:2, :], dtype=torch.float)
#     theta[0, 2] = theta[0, 2] / (shape[0] * 2)
#     theta[1, 2] = theta[1, 2] / (shape[1] * 2)
#     theta = theta.unsqueeze(0)
#     grid0 = F.affine_grid(theta, [1, 1, shape[0], shape[1]], align_corners=True).squeeze(0)
#     grid0 = grid0.cpu().numpy()
#     if grid is not None:
#         grid0 = merge_img_grid(grid0, grid)
#     return grid0

