import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def compute_jacobian_map(flow):
    """
    计算流场的雅可比行列式图。
    输入 flow: numpy array (2, H, W) -> [dy, dx]
    注意：假设 flow 是归一化坐标 [-1, 1] 的位移，或者是像素位移。
    为了可视化模式，单位不影响相对趋势，但为了数值准确，建议 flow 为像素单位。
    """
    # 如果 flow 是 PyTorch tensor，转为 numpy
    if isinstance(flow, torch.Tensor):
        flow = flow.detach().cpu().numpy()
        
    dy = flow[0, :, :] # Y轴位移
    dx = flow[1, :, :] # X轴位移
    
    # 计算梯度 (使用中心差分)
    # np.gradient 返回列表 [gradient_axis_0, gradient_axis_1]
    # 即 [d/dy, d/dx]
    
    # J = det([[y_y, y_x], [x_y, x_x]])
    # 坐标变换 T(x,y) = (x + u(x,y), y + v(x,y))
    # Jacobian Matrix J = [[1 + du/dx, du/dy], [dv/dx, 1 + dv/dy]]
    # 注意：这里的 flow 是位移场。
    
    # 简单起见，我们直接计算位移场的梯度，然后加单位矩阵
    # 假设 flow 单位是像素 (如果是归一化的，数值会很小，但正负模式不变)
    
    grad_dy_y, grad_dy_x = np.gradient(dy)
    grad_dx_y, grad_dx_x = np.gradient(dx)
    
    # 加上单位矩阵 (Identity)
    # J = (1 + dy_y) * (1 + dx_x) - (dy_x * dx_y)
    jacobian_det = (1 + grad_dy_y) * (1 + grad_dx_x) - (grad_dy_x * grad_dx_y)
    
    return jacobian_det

def create_grid_image(H, W, spacing=20):
    """创建一个网格图像用于测试变形"""
    grid_img = np.zeros((H, W), dtype=np.float32)
    # 画横线
    grid_img[::spacing, :] = 1.0
    # 画竖线
    grid_img[:, ::spacing] = 1.0
    return grid_img

def warp_image_with_flow(img, flow, device='cpu'):
    """应用流场变形图片"""
    H, W = img.shape
    img_t = torch.from_numpy(img).float().to(device).unsqueeze(0).unsqueeze(0)
    
    # 确保 flow 是 Tensor
    if isinstance(flow, np.ndarray):
        flow_t = torch.from_numpy(flow).float().to(device).unsqueeze(0)
    else:
        flow_t = flow.unsqueeze(0)
        
    # 构建 Grid Sample 需要的 grid
    vectors = [torch.arange(0, s) for s in (H, W)]
    grids = torch.meshgrid(vectors, indexing='ij')
    base_grid = torch.stack(grids).float().to(device).unsqueeze(0)
    
    # 归一化
    base_grid[:, 0, ...] = 2 * (base_grid[:, 0, ...] / (H - 1) - 0.5)
    base_grid[:, 1, ...] = 2 * (base_grid[:, 1, ...] / (W - 1) - 0.5)
    
    # Flow 应该也是归一化的，直接相加 (Forward Mapping Logic usually needs warp logic)
    # 这里假设 flow 是 LDDMM 输出的 Backward Flow (用于 grid_sample)
    # grid_sample(input, grid + flow)
    
    final_grid = base_grid + flow_t
    final_grid = final_grid.permute(0, 2, 3, 1)[..., [1, 0]] # (N, H, W, 2) xy
    
    warped = F.grid_sample(img_t, final_grid, align_corners=True, padding_mode="border")
    return warped.squeeze().cpu().numpy()

def analyze_distortion(flow, title="Flow Analysis"):
    """
    主函数：可视化雅可比行列式和网格变形
    flow: (2, H, W) numpy array
    """
    _, H, W = flow.shape
    
    # 1. 计算雅可比图 (Jacobian)
    # 为了让数值更有物理意义，我们把归一化流场转回近似的像素流场
    # flow_pixel = flow * (H / 2)
    # jac = compute_jacobian_map(flow_pixel)
    # 或者直接看相对值：
    jac = compute_jacobian_map(flow * (H/2)) # 简单缩放以获得近似像素级梯度
    
    # 2. 生成网格并变形
    grid_img = create_grid_image(H, W, spacing=max(H//30, 10))
    warped_grid = warp_image_with_flow(grid_img, flow, device='cpu')
    
    # 3. 绘图
    plt.figure(figsize=(18, 6))
    
    # Subplot 1: 原始网格 vs 变形网格
    plt.subplot(1, 3, 1)
    plt.title("Grid Deformation (Visual Check)")
    plt.imshow(warped_grid, cmap='gray')
    plt.axis('off')
    
    # Subplot 2: 雅可比热力图
    plt.subplot(1, 3, 2)
    plt.title("Jacobian Map (Expansion/Compression)")
    # 使用 diverging colormap: 1.0 (白色) 是正常，红色膨胀，蓝色压缩
    im = plt.imshow(jac, cmap='coolwarm', vmin=0, vmax=2) 
    plt.colorbar(im, label="Determinant Value")
    plt.axis('off')
    
    # Subplot 3: 负雅可比区域 (折叠警告)
    plt.subplot(1, 3, 3)
    plt.title("Folding Detection (J <= 0)")
    folding_mask = jac <= 0
    plt.imshow(folding_mask, cmap='Reds', vmin=0, vmax=1)
    plt.text(10, 20, f"Folding Pixels: {np.sum(folding_mask)}", color='black', fontsize=12, backgroundcolor='white')
    plt.axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.show()
    
    # 打印统计信息
    print(f"=== Distortion Statistics ===")
    print(f"Min Jacobian: {np.min(jac):.4f} (Should be > 0)")
    print(f"Max Jacobian: {np.max(jac):.4f}")
    print(f"Mean Jacobian: {np.mean(jac):.4f} (Should be close to 1)")
    print(f"Folding Ratio: {np.sum(folding_mask) / jac.size * 100:.2f}%")

# ================= 使用示例 =================
# 假设你已经跑完了配准，拿到了 flow
# flow shape 应为 (2, H, W)

# analyze_distortion(flow, title="High Res Loss + Low Res Grid Result")