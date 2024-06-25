import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import spacemap

def _initialize_identity_grid(N, device):
    """ Initialize an identity grid for a given size N.
        NxNx2
    """
    # Generate a grid of coordinates
    x = torch.linspace(-1, 1, N, device=device)
    y = torch.linspace(-1, 1, N, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    identity_grid = torch.stack((grid_y, grid_x), dim=-1)  # Shape: [N, N, 2]
    return identity_grid

def _apply_transformation(grid0, grid1):
    """ Apply a transformation to a grid. 
        img -> grid0 -> grid1 -> img
    """
    if isinstance(grid0, np.ndarray):
        grid0 = torch.tensor(grid0, dtype=torch.float32)
    if isinstance(grid1, np.ndarray):
        grid1 = torch.tensor(grid1, dtype=torch.float32)
    if len(grid0.shape) == 4:
        grid0 = grid0.squeeze(0)
    if len(grid1.shape) == 4:
        grid1 = grid1.squeeze(0)
    # Reshape grid to [1, H, W, 2] and transformation to [1, 2, H, W]
    return nn.functional.grid_sample(grid0.unsqueeze(0).permute(0, 3, 1, 2),
                                     grid1.unsqueeze(0),
                                     mode='bilinear',
                                     padding_mode='zeros',
                                     align_corners=True).permute(0, 2, 3, 1).squeeze(0)

def inverse_grid_train(grid, device="cpu", epochs=1000, lr=0.001):
    """ grid: 1xNxNx2 """
    transformed_grid1 = torch.tensor(grid, dtype=torch.float32, device=device)
    if len(transformed_grid1.shape) == 3:
        transformed_grid1 = transformed_grid1.unsqueeze(0)
    N = grid.shape[1]
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
    return inverse_grid.data.cpu().numpy()

def minus_grid_train(grid1, grid2, target, device="cpu", epochs=1000, lr=0.001):
    """ grid1 + grid2 = target """
    if grid1 is not None:
        grid1 = torch.tensor(grid1, dtype=torch.float32, device=device)
        N = grid2.shape[0]
    if grid2 is not None:
        grid2 = torch.tensor(grid2, dtype=torch.float32, device=device)
        N = grid2.shape[0]
    
    target_grid = torch.tensor(target, dtype=torch.float32, device=device)
    
    initial_grid = _initialize_identity_grid(N, device)
    initial_grid = torch.nn.Parameter(initial_grid.clone())

    optimizer = optim.Adam([initial_grid], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.75)
    loss_fn = nn.MSELoss()

    lastErr = 1.0
    for epoch in range(epochs):
        optimizer.zero_grad()
        if grid1 is not None:
            transformed_grid = _apply_transformation(grid1, initial_grid)
        else:
            transformed_grid = _apply_transformation( initial_grid, grid2)
        loss = loss_fn(transformed_grid, target_grid)
        loss.backward()
        optimizer.step()
        scheduler.step()
        l = loss.item()
        if l < lastErr:
            lastErr = l
        else:
            break
    return initial_grid.data.cpu().numpy()

def grid_sample_points_vectorized(points, phi, xyd=10):
    N = phi.shape[1]
    xyd = int(xyd)
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

    x_ratio = x_ratio[:, None]
    y_ratio = y_ratio[:, None]
    
    top_interp = (1 - x_ratio) * top_left + x_ratio * top_right
    bottom_interp = (1 - x_ratio) * bottom_left + x_ratio * bottom_right
    interpolated = (1 - y_ratio) * top_interp + y_ratio * bottom_interp

    interpolated = (interpolated + 1) / 2 * xymax
    interpolated[:, [0, 1]] = interpolated[:, [1, 0]]  # 交换 x, y 坐标
    result = (interpolated - xymax / 2) * ((size - 2) / size) + xymax / 2
    return result

def apply_points_by_grid(grid: np.array, ps: np.array, inv_grid=None, xyd=None):
    xyd = xyd if xyd is not None else int(spacemap.XYRANGE[1] // grid.shape[1])
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
