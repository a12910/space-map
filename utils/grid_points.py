import torch

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from scipy.interpolate import griddata
from scipy.interpolate import Rbf
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import binary_dilation
from skimage.transform import resize

def _grid_sample_points_vectorized(points, phi, xyd=10):
    N = phi.shape[1]
    xyd = int(xyd)
    xymax = N * xyd
    size = xymax // xyd - 1
    if isinstance(points, np.ndarray):
        points = torch.tensor(points, dtype=torch.float32)
    if isinstance(phi, np.ndarray):
        phi = torch.tensor(phi, dtype=torch.float32)
    x_index = torch.clamp((points[:, 0] // xyd).long(), 0, size)
    y_index = torch.clamp((points[:, 1] // xyd).long(), 0, size)

    x_ratio = (points[:, 0] % xyd) / xyd
    y_ratio = (points[:, 1] % xyd) / xyd

    x_index_plus_one = torch.clamp(x_index + 1, 0, size)
    y_index_plus_one = torch.clamp(y_index + 1, 0, size)

    top_left = phi[x_index, y_index, :]
    top_right = phi[x_index_plus_one, y_index, :]
    bottom_left = phi[x_index, y_index_plus_one, :]
    bottom_right = phi[x_index_plus_one, y_index_plus_one, :]

    x_ratio = x_ratio[:, None]
    y_ratio = y_ratio[:, None]
    
    top_interp = (1 - x_ratio) * top_left + x_ratio * top_right
    bottom_interp = (1 - x_ratio) * bottom_left + x_ratio * bottom_right
    interpolated = (1 - y_ratio) * top_interp + y_ratio * bottom_interp

    interpolated = (interpolated + 1) / 2 * xymax
    interpolated[:, [0, 1]] = interpolated[:, [1, 0]]  # 交换 x, y 坐标
    result = (interpolated - xymax / 2) * ((size - 2) / size) + xymax / 2
    return result

def _fill_nan(nan_mask, arr0, v, edge_width):
    arr = arr0.copy()
    if not np.any(nan_mask):
        return arr
    # 创建一个结构元素，用于扩展NaN区域
    structure = np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]])
    nrows, ncols = arr.shape
    
    grid_x, grid_y = np.meshgrid(np.arange(ncols), np.arange(nrows))
    
    edge_mask = nan_mask.copy()
    for _ in range(int(edge_mask)):
        edge_mask = binary_dilation(edge_mask, structure=structure)

    no_nan_mask = 1 - nan_mask
    dilated_nan_mask = no_nan_mask.copy()
    for _ in range(int(edge_width*2)):
        dilated_nan_mask = binary_dilation(dilated_nan_mask, structure=structure)

    # nan区
    dilated_nan_mask = dilated_nan_mask * nan_mask
    # 非nan区
    edge_mask = edge_mask * (1 - nan_mask)
    # return dilated_nan_mask, edge_mask

    arr[edge_mask == 0] = v
    non_zero_indices = np.argwhere(arr != v)
    non_zero_values = arr[arr != v]
    
    # 缩小分辨率
    scale_size = 100
    small_arr = resize(arr, (scale_size, scale_size), order=0, preserve_range=True, anti_aliasing=False)
    small_nan_mask = resize(nan_mask, (scale_size, scale_size), order=0, preserve_range=True, anti_aliasing=False).astype(bool)
    
    small_grid_x, small_grid_y = np.meshgrid(np.arange(scale_size), np.arange(scale_size))

    # 使用非零值和现有插值值作为RBF的输入
    non_zero_indices_small = np.argwhere(small_arr != v)
    non_zero_values_small = small_arr[small_arr != v]

    known_x_small = np.concatenate([non_zero_indices_small[:, 1]])
    known_y_small = np.concatenate([non_zero_indices_small[:, 0]])
    known_values_small = np.concatenate([non_zero_values_small])
    
    rbf = Rbf(known_x_small, known_y_small, known_values_small, function='linear')
    small_dilated_nan_mask = resize(dilated_nan_mask, (scale_size, scale_size), order=0, preserve_range=True, anti_aliasing=False).astype(bool)
    small_arr[small_dilated_nan_mask] = rbf(small_grid_x[small_dilated_nan_mask], small_grid_y[small_dilated_nan_mask])
    
    # 将结果映射回原始分辨率
    large_arr = np.full((nrows, ncols), v)
    
    for i in range(scale_size):
        for j in range(scale_size):
            factor = no_nan_mask.shape[0] / scale_size
            original_i = int(i * factor)
            original_j = int(j * factor)
            if original_i < nrows and original_j < ncols:
                large_arr[original_i, original_j] = small_arr[i, j]
    large_arr = resize(small_arr, (nrows, ncols), order=0, preserve_range=True, anti_aliasing=False)
    arr[dilated_nan_mask != 0] = large_arr[dilated_nan_mask != 0]
    arr[nan_mask == 0] = arr0[nan_mask == 0]

    # 使用非零值和现有插值值作为RBF的输入
    # known_x = np.concatenate([non_zero_indices[:, 1]])
    # known_y = np.concatenate([non_zero_indices[:, 0]])
    # known_values = np.concatenate([non_zero_values])

    # rbf = Rbf(known_x, known_y, known_values, function='linear')
    # dilated_nan_mask_ = dilated_nan_mask != 0
    # arr[dilated_nan_mask_] = rbf(grid_x[dilated_nan_mask_], grid_y[dilated_nan_mask_])
    # arr[nan_mask == 0] = arr0[nan_mask == 0]
    
    nan_mask2 = arr == v
    if np.any(nan_mask2):
        non_zero_indices = np.argwhere(arr != v)
        non_zero_values = arr[arr != v]
        grid_x, grid_y = np.meshgrid(np.arange(ncols), np.arange(nrows))
        arr[nan_mask2] = griddata(
            non_zero_indices, 
            non_zero_values, 
            (grid_y[nan_mask2], grid_x[nan_mask2]), 
            method='nearest'
        )
    return arr

def _interpolate_2d_array(arr, v, edgeWidth=20):
    # 获取数组的形状
    nrows, ncols = arr.shape
    non_zero_indices = np.argwhere(arr != v)
    non_zero_values = arr[arr != v]
    grid_x, grid_y = np.meshgrid(np.arange(ncols), np.arange(nrows))
    # 使用griddata进行插值，使用外推处理边界
    interpolated_values = griddata(non_zero_indices, non_zero_values, (grid_y, grid_x), method='linear')

    nan_mask = np.isnan(interpolated_values)
    interpolated_values[nan_mask] = v
    interpolated_values = _fill_nan(nan_mask, interpolated_values, v, edge_width=edgeWidth)
    return interpolated_values, nan_mask

def points_gen_grid_train(ps1, ps2, N, device="cpu", epochs=1000, lr=0.1, xyd=10, err=1e-3, edgeWidthRatio=0.1, show=False):
    """ Train a grid to represent the inverse of the input grid. """
    initial_grid = torch.full((N, N, 2), -1.0, dtype=torch.float32, device=device)
    init_points = torch.tensor(ps1, dtype=torch.float32, device=device)
    target_points = torch.tensor(ps2, dtype=torch.float32, device=device)
    # Initialize the inverse transformation grid
    target_grid = torch.nn.Parameter(initial_grid.clone())

    optimizer = optim.Adam([target_grid], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.75)
    loss_fn = nn.MSELoss()

    lastErr = None
    for epoch in range(epochs):
        optimizer.zero_grad()
        transformed_points1 = _grid_sample_points_vectorized(init_points, target_grid, xyd)        
        loss = loss_fn(transformed_points1, target_points)
        loss.backward()
        optimizer.step()
        scheduler.step()
        l = loss.item()
        if lastErr is None:
            lastErr = l + 100
        if abs(l - lastErr) < err:
            break
        if l <= lastErr:
            lastErr = l
        if show:
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")
    result = target_grid.detach().cpu().numpy()
    nan_mask = np.zeros_like(result)
    edgeWidth = int(N * edgeWidthRatio)
    for i in range(2):
        result[:, :, i], nan_mask[:, :, i]  = _interpolate_2d_array(result[:, :, i], -1, edgeWidth=edgeWidth)
    return result, nan_mask

def export_grid_use_points(dfRaw, dfAlign, layerFrom, layerTo, 
                           N=1000, xyd=4, lr=0., outPath=None):
    """ 3 - 23 """
    grids = []
    inv_grids = []
    affines = []
    for index in range(layerFrom, layerTo):
        ps1 = dfRaw[dfRaw["layer"] == index][["x", "y"]].values
        ps2 = dfAlign[dfAlign["layer"] == index][["x", "y"]].values
        
        grid, _ = points_gen_grid_train(ps2, ps1, N, xyd=xyd, lr=lr)
        grids.append(grid)
        inv_grid, _ = points_gen_grid_train(ps1, ps2, N, xyd=xyd, lr=lr)
        inv_grids.append(inv_grid)
        affines.append(np.eye(3))
    pack = {
        "affine_shape": [400, 400],
        "affines": affines,
        "grids": grids,
        "inv_grids": inv_grids,
    }
    if outPath is not None:
        np.savez_compressed(outPath, **pack)
    return pack
