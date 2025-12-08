import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

def gaussian_blur(inputt, kernel_size, sigma, device):
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Create 1D Gaussian kernel
    x = torch.arange(kernel_size, device=device).float() - (kernel_size - 1) / 2
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    
    # Reshape for separable convolution
    kernel_x = kernel.view(1, 1, 1, kernel_size)
    kernel_y = kernel.view(1, 1, kernel_size, 1)
    
    pad = kernel_size // 2
    
    out = F.conv2d(inputt, kernel_x, padding=(0, pad))
    out = F.conv2d(out, kernel_y, padding=(pad, 0))
    
    return out


class LDDMM_Core(nn.Module):
    def __init__(self, input_shape, device='cpu'):
        super(LDDMM_Core, self).__init__()
        self.H, self.W = input_shape
        self.device = device
        self.grid = self._create_grid(self.H, self.W).to(device)
        self.flow = None

    def _create_grid(self, h, w):
        vectors = [torch.arange(0, s) for s in (h, w)]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids).unsqueeze(0).type(torch.FloatTensor)
        grid[:, 0, ...] = 2 * (grid[:, 0, ...] / (h - 1) - 0.5)
        grid[:, 1, ...] = 2 * (grid[:, 1, ...] / (w - 1) - 0.5)
        return grid

    def spatial_transform(self, src, flow):
        # 同样的 grid_sample 逻辑...
        new_locs = self.grid + flow
        new_locs = new_locs.permute(0, 2, 3, 1)[..., [1, 0]]
        return F.grid_sample(src, new_locs, align_corners=True, padding_mode="border")

    def integrate_scaling_squaring(self, velocity, steps=7):
        flow = velocity / (2 ** steps)
        for _ in range(steps):
            warped_flow = self.spatial_transform(flow, flow)
            flow = flow + warped_flow
        return flow

    def smooth_velocity(self, v, kernel_size, sigma):
        """
        关键修改：支持传入更大的 kernel_size 和 sigma
        """
        # 确保 kernel_size 是奇数
        if kernel_size % 2 == 0: kernel_size += 1
        
        # 增加 Padding 以防止边缘效应
        padding = kernel_size // 2
        
        # 对 X 和 Y 通道分别做强力平滑
        v_y = gaussian_blur(v[:, 0:1], kernel_size, sigma, self.device)
        v_x = gaussian_blur(v[:, 1:2], kernel_size, sigma, self.device)
        return torch.cat([v_y, v_x], dim=1)

    def compute_jacobian_determinant(self, flow):
        """
        计算雅可比行列式，用于检测局部扭曲
        J = det([[1+dy/dy, dy/dx], [dx/dy, 1+dx/dx]])
        """
        # flow shape: (1, 2, H, W) -> (y, x) normalized [-1, 1]
        # 需要转换回像素单位才能计算真实的 Jacobian，或者直接计算相对变化
        
        # 使用中心差分计算梯度
        dy = flow[:, 0, :, :]
        dx = flow[:, 1, :, :]
        
        # gradient 返回 (dy, dx)
        dy_dy, dy_dx = torch.gradient(dy, dim=(1, 2)) 
        dx_dy, dx_dx = torch.gradient(dx, dim=(1, 2))
        
        # 注意：这里的梯度是基于归一化坐标的，加 1 需要考虑 scale
        # 但为了作为正则项，我们只需要惩罚梯度的剧烈变化即可
        # 简化版 Jacobian 惩罚：惩罚 flow 的梯度过大

        jacobian_det = (1 + dy_dy) * (1 + dx_dx) - (dy_dx * dx_dy)

        # 惩罚项：惩罚任何体积的剧烈改变
        # 这种写法会极大地抑制局部压缩和局部膨胀
        loss_jac = torch.mean((jacobian_det - 1) ** 2)
        
        return loss_jac

