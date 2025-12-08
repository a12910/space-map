import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from .lddmm import LDDMM_Core

class GlobalLDDMM_Register:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device

    def run(self, moving_img, fixed_img, 
            scales=[0.25, 0.5, 1.0], 
            iterations=[200, 100, 50], 
            smooth_sigma=[20, 15, 10], 
            reg_weight=1000.0): 
        
        """
        smooth_sigma (list): 对应每个 scale 的平滑程度。数值越大，形变越"硬/全局"。
        reg_weight (float): 正则化权重。越大越不允许局部扭曲。
        """
        
        mov_t = torch.from_numpy(moving_img).float().to(self.device).unsqueeze(0).unsqueeze(0)
        fix_t = torch.from_numpy(fixed_img).float().to(self.device).unsqueeze(0).unsqueeze(0)
        H_orig, W_orig = mov_t.shape[2], mov_t.shape[3]
        
        velocity = None
        
        for idx, (scale, n_iter) in enumerate(zip(scales, iterations)):
            curr_sigma = smooth_sigma[idx]
            # Kernel size 通常设为 sigma 的 3-4 倍覆盖范围
            curr_kernel = int(curr_sigma * 4) + 1 
            
            curr_H = int(H_orig * scale)
            curr_W = int(W_orig * scale)
            
            # print(f"Scale {scale} | Sigma {curr_sigma} | Global-Mode Optimization...")
            
            mov_s = F.interpolate(mov_t, size=(curr_H, curr_W), mode='bilinear', align_corners=True)
            fix_s = F.interpolate(fix_t, size=(curr_H, curr_W), mode='bilinear', align_corners=True)
            
            if velocity is None:
                velocity = torch.zeros((1, 2, curr_H, curr_W), device=self.device, requires_grad=True)
            else:
                velocity = F.interpolate(velocity.detach(), size=(curr_H, curr_W), mode='bilinear', align_corners=True)
                velocity.requires_grad_(True)
            
            core = LDDMM_Core((curr_H, curr_W), self.device)
            optimizer = optim.Adam([velocity], lr=0.01) # 略微降低 LR 保证稳定
            
            for i in range(n_iter):
                optimizer.zero_grad()
                
                # 1. 强力平滑 (Enforce Global Smoothness)
                smooth_v = core.smooth_velocity(velocity, kernel_size=curr_kernel, sigma=curr_sigma)
                
                # 2. 积分
                flow = core.integrate_scaling_squaring(smooth_v, steps=7)
                
                # 3. 变形
                warped = core.spatial_transform(mov_s, flow)
                
                # 4. Loss 计算
                loss_sim = F.mse_loss(warped, fix_s)
                # loss_sim = F.dice_loss(warped, fix_s)
                
                # 正则项：惩罚速度场的模长 (L2 Norm)
                # reg_weight 设得很大，迫使 velocity 接近 0，除非必须动
                loss_reg = torch.mean(velocity ** 2)
                
                # 雅可比惩罚 (可选，进一步防止局部扭曲)
                loss_jac = core.compute_jacobian_determinant(flow)
                
                loss = loss_sim + reg_weight * loss_reg + 2000 * loss_jac
                
                loss.backward()
                optimizer.step()
        
        # 输出阶段：上采样到原图
        final_velocity = F.interpolate(velocity.detach(), size=(H_orig, W_orig), mode='bilinear', align_corners=True)
        final_core = LDDMM_Core((H_orig, W_orig), self.device)
        
        # 注意：最后生成结果时，也要用同样的强力平滑，否则上采样会带来高频噪声
        # 我们使用最后一层的 sigma 进行平滑
        final_sigma = smooth_sigma[-1] * (1.0/scales[-1]) # 调整 sigma 适应原分辨率
        final_kernel = int(final_sigma * 4) + 1
        
        final_smooth_v = final_core.smooth_velocity(final_velocity, kernel_size=final_kernel, sigma=final_sigma)
        final_flow = final_core.integrate_scaling_squaring(final_smooth_v, steps=7)
        final_warped = final_core.spatial_transform(mov_t, final_flow)
        self.flow = final_flow.squeeze().cpu().numpy()
        return final_warped.squeeze().cpu().numpy(), self.flow

    def run_high_res_loss_low_res_grid(self, moving_img, fixed_img, 
                                       grid_size=(8, 8), # 关键：流场分辨率（控制点的密度）
                                       iterations=200, 
                                       smooth_sigma=3,     # 在低分网格上的平滑
                                       reg_weight=1):
        """
        Input: 高清原图 (H, W)
        Optimization: 仅优化低分辨率速度场 (grid_size)
        Loss Calculation: 在高清原图上计算 MSE
        """
        H_orig, W_orig = moving_img.shape
        
        # 1. 准备高清原图 (不缩放！)
        mov_high = torch.from_numpy(moving_img).float().to(self.device).unsqueeze(0).unsqueeze(0)
        fix_high = torch.from_numpy(fixed_img).float().to(self.device).unsqueeze(0).unsqueeze(0)
        
        print(f"High-Res Guidance ({H_orig}x{W_orig}) | Low-Res Control {grid_size}...")
        
        # 2. 定义低分辨率的优化变量 (Velocity)
        # 这就是我们的"粗糙控制手柄"，自由度很低，物理上杜绝了局部聚集
        velocity_low = torch.zeros((1, 2, *grid_size), device=self.device, requires_grad=True)
        
        # 针对低分网格的核心模块
        core_low = LDDMM_Core(grid_size, self.device)
        # 针对高清原图的变换模块
        core_high = LDDMM_Core((H_orig, W_orig), self.device)
        
        optimizer = optim.Adam([velocity_low], lr=0.01) # 学习率可能需要根据 grid_size 微调
        
        for i in range(iterations):
            optimizer.zero_grad()
            
            # A. 在低分辨网格上处理流场 (平滑 & 积分)
            # 这一步计算非常快
            smooth_v_low = core_low.smooth_velocity(velocity_low, kernel_size=9, sigma=smooth_sigma)
            flow_low = core_low.integrate_scaling_squaring(smooth_v_low, steps=7)
            
            # B. 关键步骤：实时上采样流场到原图大小
            # mode='bilinear' 保证了流场在原图上是绝对平滑连续的
            flow_high = F.interpolate(flow_low, size=(H_orig, W_orig), 
                                      mode='bilinear', align_corners=True)
            
            # C. 在高清原图上应用变换
            warped_high = core_high.spatial_transform(mov_high, flow_high)
            
            # D. 计算 Loss (使用高清细节！)
            # 这样优化器就能"看到"孔洞的边缘，并试图调整粗网格来对齐它
            loss_sim = F.mse_loss(warped_high, fix_high)
            
            # 正则项只约束低分网格即可
            loss_reg = torch.mean(velocity_low ** 2)
            
            loss = loss_sim + reg_weight * loss_reg
            loss.backward()
            optimizer.step()
            
        # 3. 输出最终结果
        with torch.no_grad():
            # 重新计算一次最终结果
            smooth_v_low = core_low.smooth_velocity(velocity_low, kernel_size=9, sigma=smooth_sigma)
            flow_low = core_low.integrate_scaling_squaring(smooth_v_low, steps=7)
            flow_high = F.interpolate(flow_low, size=(H_orig, W_orig), 
                                      mode='bilinear', align_corners=True)
            warped_high = core_high.spatial_transform(mov_high, flow_high)
            
        self.flow = flow_high.squeeze().cpu().numpy()
        return warped_high.squeeze().cpu().numpy(), self.flow
    
    def map_fix_points_to_moving_space(self, ps1_points, flow_field, d, device='cpu'):
        """
        将 ps1 (Fixed) 上的点，利用 Backward Flow，找到在 ps0 (Moving) 上的对应位置。
        """
        # flow_field shape: (2, H, W) -> [dy, dx]
        ps1_points = ps1_points[:, [1, 0]]
        _, H_flow, W_flow = flow_field.shape
        
        # 转 Tensor
        pts = torch.from_numpy(ps1_points).float().to(device)
        flow = torch.from_numpy(flow_field).float().to(device).unsqueeze(0) # (1, 2, H, W)
        
        # A. 坐标归一化 (ps1 点 -> Flow 网格坐标 [-1, 1])
        # 注意：flow 的坐标系是基于 ps1i 图片的
        grid_x = 2 * (pts[:, 0] / d) / (W_flow - 1) - 1
        grid_y = 2 * (pts[:, 1] / d) / (H_flow - 1) - 1
        
        # B. 在 ps1 的点位置，采样 Flow
        sample_grid = torch.stack((grid_x, grid_y), dim=1).unsqueeze(0).unsqueeze(0)
        sampled_flow = F.grid_sample(flow, sample_grid, align_corners=True, padding_mode="border")
        
        # C. 提取位移量 (dy, dx)
        sampled_flow = sampled_flow.squeeze(0).squeeze(1).permute(1, 0) 
        flow_dy = sampled_flow[:, 0]
        flow_dx = sampled_flow[:, 1]
        
        # D. 计算 ps0 空间的新坐标
        # 公式: Pos_in_Moving = Pos_in_Fixed + Flow_Vector
        
        # 将归一化 Flow ([-1, 1]) 转回像素单位
        pixel_dx = flow_dx * (W_flow - 1) / 2
        pixel_dy = flow_dy * (H_flow - 1) / 2
        
        # 原始坐标 + 位移
        new_x = (pts[:, 0] / d + pixel_dx) * d
        new_y = (pts[:, 1] / d + pixel_dy) * d
        ps2 = torch.stack((new_x, new_y), dim=1).detach().cpu().numpy()
        ps2 = ps2[:, [1, 0]]
        return ps2

