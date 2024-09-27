import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import spacemap

def applyH_np(df: np.array, H: np.array, xyd=None, fromImgH=True) -> np.array:
    df2 = df.copy()
    H = np.array(H)
    if fromImgH:
        H = to_npH(H, xyd)
    df2[:, 0] = (df[:, 0] * H[0, 0] + df[:, 1] * H[0, 1]) + H[0, 2]
    df2[:, 1] = (df[:, 0] * H[1, 0] + df[:, 1] * H[1, 1]) + H[1, 2]
    return df2

def _initialize_identity_grid(N, device):
    """ Initialize an identity grid for a given size N. """
    x = torch.linspace(-1, 1, N, device=device)
    y = torch.linspace(-1, 1, N, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    identity_grid = torch.stack((grid_y, grid_x), dim=-1)  # Shape: [N, N, 2]
    return identity_grid


def inverse_grid_train(grid, epochs=500, lr=0.2, xyd=10, err=1e-2, show=False, appendPoints=None):
    """ Train a grid to represent the inverse of the input grid. """
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    grid = torch.tensor(grid, dtype=torch.float32, device=device)
    N = grid.shape[0]
    
    initial_grid = _initialize_identity_grid(N, device)
    
    identity_points = grid_to_points(initial_grid, xyd)
    if isinstance(identity_points, np.ndarray):
        identity_points = torch.tensor(identity_points, dtype=torch.float32, device=device)

    inverse_grid1 = torch.nn.Parameter(initial_grid.clone())

    optimizer = optim.Adam([inverse_grid1], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.75)
    loss_fn = nn.MSELoss()
    if appendPoints is not None:
        appendPoints = torch.tensor(appendPoints, dtype=torch.float32, device=device)

    lastErr = None
    for epoch in range(epochs):
        optimizer.zero_grad()
        transformed_points1 = grid_sample_points_vectorized(identity_points, grid, xyd)
        transformed_points1 = grid_sample_points_vectorized(transformed_points1, inverse_grid1, xyd)
        if appendPoints is not None:
            appendPoints1 = grid_sample_points_vectorized(appendPoints, inverse_grid1, xyd)
            appendPoints2 = grid_sample_points_vectorized(appendPoints1, grid, xyd)
            loss = loss_fn(transformed_points1, identity_points) + loss_fn(appendPoints2, appendPoints)
        else:
            loss = loss_fn(transformed_points1, identity_points)
        loss.backward()
        optimizer.step()
        scheduler.step()
        l = loss.item()
        if lastErr is None:
            lastErr = l + 100
        if abs(l - lastErr) <= err:
            break
        if l <= lastErr + err:
            lastErr = l
        if epoch % 20 == 0 and show:
            print(f"Epoch {epoch}, Loss: {l}")
    return inverse_grid1.detach().cpu().numpy()

def grid_sample_points_vectorized(points, phi, xyd=10):
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

def points_to_grid(ps, xyd):
    """ ps: (NxN)x2 """
    # ps = ps.copy()
    N = int(np.sqrt(ps.shape[0]))
    ps = ps.reshape((N, N, 2))
    ps = ps / (N // 2 * xyd) - 1
    return ps

def grid_to_points(grid, xyd):
    """ grid: NxNx2 """
    # grid = grid.copy()
    N = grid.shape[0]
    grid = (grid + 1) * (N // 2 * xyd)
    return grid.reshape((-1, 2))

def apply_points_by_grid(grid: np.array, ps: np.array, inv_grid=None, xyd=None):
    xyd = xyd if xyd is not None else int(spacemap.XYRANGE // grid.shape[1])
    if len(grid.shape) == 4:
        grid = grid.squeeze(0)
    if inv_grid is None:
        inv_grid = inverse_grid_train(grid)
    ps2 = grid_sample_points_vectorized(ps, inv_grid, xyd)
    ps2 = ps2.detach().cpu().numpy()
    return ps2, inv_grid

def to_npH(H: np.array, xyd=None):
    H = H.copy()
    xyd = xyd if xyd is not None else spacemap.XYD
    H[0, 2] = H[0, 2] * xyd
    H[1, 2] = H[1, 2] * xyd
    return H

def merge_grid_img(grid0, grid1, xyd=10):
    """ img -> grid0 -> grid1 """
    device="cpu"
    grid0 = torch.tensor(grid0, dtype=torch.float32, device=device)
    N = grid0.shape[1]
    grid1 = torch.tensor(grid1, dtype=torch.float32, device=device)
    identity_grid = _initialize_identity_grid(N, device)
    identity_points = grid_to_points(identity_grid, xyd)
    target_points = identity_points 
    grid1 = grid1.reshape(N, N, 2)
    grid0 = grid0.reshape(N, N, 2)
    target_points = grid_sample_points_vectorized(target_points, grid1, xyd)
    target_points = grid_sample_points_vectorized(target_points, grid0, xyd)
    merged_points = target_points.detach().cpu().numpy()
    merged = points_to_grid(merged_points, xyd)
    return merged
