import torch
import torch.nn as nn
import torch.optim as optim
import spacemap

def _grid_sample_points_torch(points, phi, xyd=10):
    N = phi.shape[0]  # 网格尺寸
    xymax = N * xyd
    size = xymax // xyd - 1

    # 计算索引和比例
    x_index = torch.clamp((points[:, 0] // xyd).long(), 0, size)
    y_index = torch.clamp((points[:, 1] // xyd).long(), 0, size)
    x_ratio = (points[:, 0] % xyd) / xyd
    y_ratio = (points[:, 1] % xyd) / xyd

    x_index_plus_one = torch.clamp(x_index + 1, 0, size)
    y_index_plus_one = torch.clamp(y_index + 1, 0, size)

    # 提取四个角的值
    top_left = phi[x_index, y_index, :]
    top_right = phi[x_index_plus_one, y_index, :]
    bottom_left = phi[x_index, y_index_plus_one, :]
    bottom_right = phi[x_index_plus_one, y_index_plus_one, :]
    
    x_ratio = x_ratio[:, None]
    y_ratio = y_ratio[:, None]
    
    # 执行双线性插值
    top_interp = (1 - x_ratio) * top_left + x_ratio * top_right
    bottom_interp = (1 - x_ratio) * bottom_left + x_ratio * bottom_right
    interpolated = (1 - y_ratio) * top_interp + y_ratio * bottom_interp
    
    # 调整结果
    interpolated = (interpolated + 1) / 2 * xymax
    interpolated[:, [0, 1]] = interpolated[:, [1, 0]]  # 交换 x, y 坐标
    result = (interpolated - xymax / 2) * ((size - 2) / size) + xymax / 2
    return result


import numpy as np
def grid_generate(pFrom: np.array, pTo, inverse=True, 
                  maxShape=4000, phiShape=400, 
                  epoch=1000, lr=0.01, verbose=100, err=1e-5, device="cpu"):
    pFrom = torch.tensor(pFrom, device=device).float()
    pTo = torch.tensor(pTo, device=device).float()
    if inverse:
        pFrom, pTo = pTo, pFrom
    pFrom = pFrom / maxShape * 2 - 1
    pTo = pTo / maxShape * 2 - 1
    phi = nn.Parameter(torch.zeros(phiShape, phiShape, 2, device=device))
    optimizer = optim.Adam([phi], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.75)

    lastErr = None
    # 训练循环
    for epoch in range(epoch):
        # random sample points
        optimizer.zero_grad()
        transformed_points = _grid_sample_points_torch(pFrom, phi)
        loss = nn.MSELoss()(transformed_points, pTo)
        loss.backward()
        optimizer.step()
        scheduler.step()
        l = loss.item()
        if lastErr is not None and abs(lastErr - l) < err:
            print(f"Converged at epoch {epoch}, Loss: {l}")
            break
        lastErr = l
        if verbose > 0 and epoch % verbose == 0:
            print(f"Epoch {epoch}, Loss: {l}")
    result = phi.detach().numpy()
    return result

def grid_generate_test(pF, pT, phi, xyd):
    pTo1 = spacemap.points.grid_sample_points_vectorized(pF, phi, xyd)
    pTo2 = _grid_sample_points_torch(torch.tensor(pF).float(), 
                                     torch.tensor(phi).float(), xyd=10).detach().numpy()
    return np.mean(np.sqrt(np.sum((pTo1 - pTo2)**2, axis=1)))
