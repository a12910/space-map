import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import spacemap

def _initialize_identity_grid(N, device):
    """ Initialize an identity grid for a given size N. """
    # Generate a grid of coordinates
    x = torch.linspace(-1, 1, N, device=device)
    y = torch.linspace(-1, 1, N, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    identity_grid = torch.stack((grid_y, grid_x), dim=-1)  # Shape: [N, N, 2]
    return identity_grid

def _apply_transformation(grid, transformation):
    """ Apply a transformation to a grid. """
    # Reshape grid to [1, H, W, 2] and transformation to [1, 2, H, W]
    return nn.functional.grid_sample(transformation.unsqueeze(0).permute(0, 3, 1, 2),
                                     grid.unsqueeze(0),
                                     mode='bilinear',
                                     padding_mode='zeros',
                                     align_corners=True).permute(0, 2, 3, 1).squeeze(0)

def inverse_grid_train(grid, device="cpu", epochs=1000, lr=0.001):
    transformed_grid1 = torch.tensor(grid, dtype=torch.float32, device=device)
    N = grid.shape[0]
    initial_grid = _initialize_identity_grid(N, device)
    identity_grid = _initialize_identity_grid(N, device)

    # Initialize the inverse transformation grid (trying to reverse transformation1)
    inverse_grid = torch.nn.Parameter(initial_grid.clone())

    optimizer = optim.Adam([inverse_grid], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.75)
    loss_fn = nn.MSELoss()

    lastErr = 1.0
    for epoch in range(epochs):
        optimizer.zero_grad()
        recovered_grid = _apply_transformation(transformed_grid1, inverse_grid)
        # recovered_grid = _apply_transformation(inverse_grid, transformed_grid1)
        loss = loss_fn(recovered_grid, identity_grid)
        loss.backward()
        optimizer.step()
        scheduler.step()
        l = loss.item()
        if l < lastErr:
            lastErr = l
        else:
            break

    return inverse_grid.data

def grid_sample_points_vectorized(points, phi, xyd=10):
    N = phi.shape[1]
    xymax = N * xyd
    size = xymax // xyd - 1

    x_index = np.clip((points[:, 0] // xyd).astype(int), 0, size)
    y_index = np.clip((points[:, 1] // xyd).astype(int), 0, size)

    x_ratio = (points[:, 0] % xyd) / xyd
    y_ratio = (points[:, 1] % xyd) / xyd

    x_index_plus_one = np.clip(x_index + 1, 0, size)
    y_index_plus_one = np.clip(y_index + 1, 0, size)

    top_left = phi[x_index, y_index, :]
    top_right = phi[x_index_plus_one, y_index, :]
    bottom_left = phi[x_index, y_index_plus_one, :]
    bottom_right = phi[x_index_plus_one, y_index_plus_one, :]

    top_interp = (1 - x_ratio)[:, None] * top_left + x_ratio[:, None] * top_right
    bottom_interp = (1 - x_ratio)[:, None] * bottom_left + x_ratio[:, None] * bottom_right
    interpolated = (1 - y_ratio)[:, None] * top_interp + y_ratio[:, None] * bottom_interp

    interpolated = (interpolated + 1) / 2 * xymax
    interpolated[:, [0, 1]] = interpolated[:, [1, 0]]  # 交换 x, y 坐标
    result = (interpolated - xymax / 2) * ((size - 2) / size) + xymax / 2

    return result

def apply_points_by_grid(grid: np.array, ps: np.array, inv_grid=None, xyd=None):
    xyd = xyd if xyd is not None else spacemap.XYD
    if inv_grid is None:
        inv_grid = inverse_grid_train(grid)
    ps2 = grid_sample_points_vectorized(ps, inv_grid, xyd)
    return ps2, inv_grid

def applyH_np(df: np.array, H: np.array, xyd=None) -> np.array:
    df2 = df.copy()
    H = np.array(H)
    H = to_npH(H, xyd)
    df2[:, 0] = (df[:, 0] * H[0, 0] + df[:, 1] * H[0, 1]) + H[0, 2]
    df2[:, 1] = (df[:, 0] * H[1, 0] + df[:, 1] * H[1, 1]) + H[1, 2]
    return df2

def to_npH(H: np.array, xyd=None):
    H = H.copy()
    xyd = xyd if xyd is not None else spacemap.XYD
    H[0, 2] = H[0, 2] * xyd
    H[1, 2] = H[1, 2] * xyd
    return H
