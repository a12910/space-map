from .lddmm import LDDMM_Core
import torch
import torch.nn.functional as F
import torch.optim as optim

import cv2
import numpy as np

def extract_outer_hull(img):
    """
    提取图像的最外层轮廓，并填充为实心掩膜。
    忽略内部所有空洞。
    """
    # 1. 二值化
    # 假设 img 是 0-1 float, 转为 0-255 uint8
    img_u8 = (img * 255).astype(np.uint8)
    img_u8 = cv2.GaussianBlur(img_u8, (3, 3), 0)
    img_u8[img_u8 > 0] = 255
    
    # 阈值选取：只要不是纯黑背景都算前景
    _, binary = cv2.threshold(img_u8, 1, 255, cv2.THRESH_BINARY)
    
    # 2. 查找轮廓
    # RETR_EXTERNAL 只检测最外层轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return np.zeros_like(img)
        
    # 3. 找到最大轮廓 (假设最大的那个是组织本身，噪点忽略)
    max_cnt = max(contours, key=cv2.contourArea)
    
    # 4. 绘制实心掩膜
    hull_mask = np.zeros_like(img_u8)
    # thickness=-1 表示填充内部
    cv2.drawContours(hull_mask, [max_cnt], -1, 255, thickness=-1)
    
    return hull_mask.astype(np.float32) / 255.0

import scipy.ndimage


def generate_smooth_3d_hull(raw_slices, sigma_z=5.0):
    """
    生成一个表面完美光滑的 3D 目标体积。
    """
    N, H, W = raw_slices.shape
    hulls = []
    
    print("Extracting hulls...")
    for i in range(N):
        hulls.append(extract_outer_hull(raw_slices[i]))
        
    hulls_np = np.array(hulls)
    
    print(f"Smoothing 3D Hull (Sigma Z={sigma_z})...")
    # 关键：在 Z 轴 (axis=0) 上做强力平滑
    # 这会消除层与层之间形状的突变，让外表面像丝绸一样顺滑
    smooth_hulls = scipy.ndimage.gaussian_filter(hulls_np, sigma=(sigma_z, 1.0, 1.0))
    
    # 平滑后是浮点数，我们需要硬边缘，所以做一个阈值切断
    # 0.5 是分界线，这样边缘也是平滑过渡的
    # (如果 LDDMM 喜欢软边缘，可以保留浮点数，这里保留浮点数以提供梯度)
    
    return smooth_hulls
    return hulls_np

class HullGuidedAligner:
    def __init__(self, device='cpu'):
        self.device = device
    
    def get_forward_flow_high_res(self, velocity_low, output_size):
        """
        利用优化好的 velocity_low 计算前向流 (Forward Flow)。
        原理：Forward Flow = Integrate(-velocity)
        """
        H, W = output_size
        
        # 1. 速度场取反 (关键步骤)
        forward_velocity = -velocity_low
        
        # 2. 积分 (得到低分辨率的前向流)
        # 注意：这里需要新建一个 core，因为 integrate 依赖内部 grid
        grid_size = velocity_low.shape[2:]
        core_low = LDDMM_Core(grid_size, self.device)
        
        # 平滑并积分
        smooth_v = core_low.smooth_velocity(forward_velocity, kernel_size=9, sigma=3)
        flow_low_fwd = core_low.integrate_scaling_squaring(smooth_v, steps=7)
        
        # 3. 上采样到原图大小
        flow_high_fwd = F.interpolate(flow_low_fwd, size=(H, W), 
                                      mode='bilinear', align_corners=True)
        
        return flow_high_fwd.detach() # (1, 2, H, W)

    def run_hull_alignment(self, moving_img, fixed_hull, 
                           grid_size=(4, 4), 
                           iterations=200, 
                           reg_weight=5.0): # 极高的正则权重，强制内部刚性
        """
        Input: 
            moving_img: 原始含纹理的切片 (用于最后输出)
            fixed_hull: 平滑后的完美外壳 (作为对齐目标)
        """
        H, W = moving_img.shape
        
        # 1. 动态提取 Moving 的外壳 (我们需要对齐的是形状)
        # 注意：这里需要在 CPU 上做 opencv 操作，或者如果你有 tensor 版的也可以
        # 为简单起见，我们预先计算好
        mov_hull_np = extract_outer_hull(moving_img)
        
        # 转 Tensor
        hull_mov_t = torch.from_numpy(mov_hull_np).float().to(self.device).unsqueeze(0).unsqueeze(0)
        hull_fix_t = torch.from_numpy(fixed_hull).float().to(self.device).unsqueeze(0).unsqueeze(0)
        
        # 原始图片也转 Tensor，但只用于最后应用变形，不参与 Loss 计算！
        # img_mov_t = torch.from_numpy(moving_img).float().to(self.device).unsqueeze(0).unsqueeze(0)
        
        # 2. 定义变量
        velocity_low = torch.zeros((1, 2, *grid_size), device=self.device, requires_grad=True)
        core_low = LDDMM_Core(grid_size, self.device)
        core_high = LDDMM_Core((H, W), self.device)
        
        optimizer = optim.Adam([velocity_low], lr=0.01)
        
        for i in range(iterations):
            optimizer.zero_grad()
            
            # A. 优化流场
            smooth_v = core_low.smooth_velocity(velocity_low, kernel_size=9, sigma=3)
            flow_low = core_low.integrate_scaling_squaring(smooth_v)
            flow_high = F.interpolate(flow_low, size=(H, W), mode='bilinear', align_corners=True)
            
            # B. 变形 Moving Hull (注意：这里变的是外壳！)
            warped_hull = core_high.spatial_transform(hull_mov_t, flow_high)
            
            # C. 计算 Loss (只比较外壳的形状)
            # 这样内部血管乱七八糟根本不会影响 Loss
            loss_sim = F.mse_loss(warped_hull, hull_fix_t)
            
            # D. 正则化
            loss_reg = torch.mean(velocity_low ** 2)
            loss = loss_sim + reg_weight * loss_reg
            
            loss.backward()
            optimizer.step()
            
        # 3. 最后应用流场到 原始图片 (Texture)
        with torch.no_grad():
            smooth_v = core_low.smooth_velocity(velocity_low, kernel_size=9, sigma=3)
            flow_low = core_low.integrate_scaling_squaring(smooth_v)
            flow_high = F.interpolate(flow_low, size=(H, W), mode='bilinear', align_corners=True)
            
            # 这里应用到原始血管图上
            # warped_img = core_high.spatial_transform(img_mov_t, flow_high)
 
        flow_fwd = self.get_forward_flow_high_res(velocity_low, (H, W))[0]
        return None, flow_fwd
        # return warped_img.squeeze().cpu().numpy(), flow_fwd

    def transform_points_with_flow(self, points, forward_flow, img_shape, device='cpu', xyd=1):
        """
        使用前向流变换坐标点。
        
        Args:
            points: (N, 2) numpy array, [x, y] 格式 (列, 行)
            forward_flow: (1, 2, H, W) tensor, 来自 get_forward_flow_high_res
            img_shape: (H, W)
        """
        H, W = img_shape
        
        # 1. 准备点数据
        # 假设 points 输入是 [x, y]，转为 Tensor
        pts_t = torch.from_numpy(points).float().to(device)
        pts_t /= xyd
        
        # 2. 坐标归一化 (Pixel -> [-1, 1])
        # grid_sample 需要归一化坐标来采样流场
        grid_x = 2 * (pts_t[:, 0] / (W - 1)) - 1
        grid_y = 2 * (pts_t[:, 1] / (H - 1)) - 1
        
        # 拼装采样网格 (1, 1, N, 2)
        sample_grid = torch.stack((grid_x, grid_y), dim=1).unsqueeze(0).unsqueeze(0)
        
        # 3. 在点的位置采样前向流 (Sample Forward Flow)
        # flow shape: (1, 2, H, W) -> Channel 0: Y位移, Channel 1: X位移
        sampled_flow = F.grid_sample(forward_flow, sample_grid, align_corners=True, padding_mode="border")
        
        # 调整形状 (1, 2, 1, N) -> (N, 2) [dy, dx]
        sampled_flow = sampled_flow.squeeze(0).squeeze(1).permute(1, 0)
        
        # 4. 计算位移并应用
        # LDDMM Flow 是归一化偏移 [-1, 1]，需要转回像素单位
        # pixel_flow = norm_flow * (size - 1) / 2
        
        flow_dx = sampled_flow[:, 1] * (W - 1) / 2.0
        flow_dy = sampled_flow[:, 0] * (H - 1) / 2.0
        
        # 新坐标 = 旧坐标 + 前向流
        new_x = pts_t[:, 0] + flow_dx
        new_y = pts_t[:, 1] + flow_dy
        
        result = torch.stack((new_x, new_y), dim=1).detach().cpu().numpy()
        result *= xyd
        return result

def align_stack_perfect_shape(raw_slices):
    """
    主流程
    """
    raw_slices = np.array(raw_slices)
    N, H, W = raw_slices.shape
    
    # Step 1: 生成完美的 3D 模具 (Mold)
    print("Generating ideal 3D shape reference...")
    # sigma_z=5.0 意味着在 Z 轴 5 层范围内平滑，保证外形绝对连续
    ideal_hulls = generate_smooth_3d_hull(raw_slices, sigma_z=5.0)
    
    aligner = HullGuidedAligner(device='cpu')
    aligned_stack = []
    flows = []
    
    print("Aligning slices to ideal shape...")
    for i in range(N):
        # 让每一张原始图，去适应这个完美的模具
        # reg_weight=5.0 保证流场非常硬，内部不扭曲，只做整体搬运
        warped, flow = aligner.run_hull_alignment(
            raw_slices[i], 
            ideal_hulls[i], 
            grid_size=(8, 8), 
            reg_weight=5.0
        )
        aligned_stack.append(warped)
        flows.append(flow)
        print(f"  Slice {i} done.")
        
    return np.array(aligned_stack), np.array(flows)

