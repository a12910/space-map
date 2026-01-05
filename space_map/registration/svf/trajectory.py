
# ==========================================
# 2. 轨迹预测对齐器 (Trajectory Predictive Aligner)
# ==========================================
class TrajectoryAligner:
    def __init__(self, device='cuda'):
        self.device = device

    def compute_trend_flow(self, prev_img, curr_img):
        """
        计算两层之间的生长趋势 (Optical Flow)。
        这里使用 Farneback 稠密光流，能够捕捉血管的倾斜方向。
        """
        # 转换为 uint8 供 opencv 使用 (假设输入是 0-1 float)
        prev_u8 = (prev_img * 255).astype(np.uint8)
        curr_u8 = (curr_img * 255).astype(np.uint8)
        
        # 计算光流
        # 参数说明: pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2
        # winsize=15 保证了平滑性，不会因为噪点乱跳
        flow = cv2.calcOpticalFlowFarneback(prev_u8, curr_u8, None, 
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        return flow

    def predict_next_target(self, last_aligned_img, flow_trend):
        """
        基于上一层对齐结果和生长趋势，预测当前层应该长什么样。
        Target = Warp(Last_Aligned, Trend_Flow)
        """
        H, W = last_aligned_img.shape
        
        # 构造映射矩阵
        # Flow 是 (dx, dy), remap 需要 (x+dx, y+dy)
        grid_y, grid_x = np.mgrid[0:H, 0:W].astype(np.float32)
        
        # 这里的 flow 是从 prev -> curr 的前向流
        # 我们要把 last_aligned 推向未来
        map_x = grid_x + flow_trend[..., 0]
        map_y = grid_y + flow_trend[..., 1]
        
        # 生成预测图
        predicted_target = cv2.remap(last_aligned_img, map_x, map_y, 
                                     interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        
        return predicted_target

    def run_stable_lddmm(self, moving_img, fixed_img, 
                         grid_size=(8, 8), iterations=200, reg_weight=1):
        """
        你提供的最稳定的 LDDMM 代码 (未修改)
        """
        H_orig, W_orig = moving_img.shape
        mov_high = torch.from_numpy(moving_img).float().to(self.device).unsqueeze(0).unsqueeze(0)
        fix_high = torch.from_numpy(fixed_img).float().to(self.device).unsqueeze(0).unsqueeze(0)
        
        velocity_low = torch.zeros((1, 2, *grid_size), device=self.device, requires_grad=True)
        core_low = LDDMM_Core(grid_size, self.device)
        core_high = LDDMM_Core((H_orig, W_orig), self.device)
        optimizer = optim.Adam([velocity_low], lr=0.01)
        
        for i in range(iterations):
            optimizer.zero_grad()
            smooth_v_low = core_low.smooth_velocity(velocity_low, kernel_size=9, sigma=3)
            flow_low = core_low.integrate_scaling_squaring(smooth_v_low, steps=7)
            flow_high = F.interpolate(flow_low, size=(H_orig, W_orig), mode='bilinear', align_corners=True)
            warped_high = core_high.spatial_transform(mov_high, flow_high)
            
            loss_sim = F.mse_loss(warped_high, fix_high)
            loss_reg = torch.mean(velocity_low ** 2)
            loss = loss_sim + reg_weight * loss_reg
            loss.backward()
            optimizer.step()
            
        with torch.no_grad():
            smooth_v_low = core_low.smooth_velocity(velocity_low, kernel_size=9, sigma=3)
            flow_low = core_low.integrate_scaling_squaring(smooth_v_low, steps=7)
            flow_high = F.interpolate(flow_low, size=(H_orig, W_orig), mode='bilinear', align_corners=True)
            warped_high = core_high.spatial_transform(mov_high, flow_high)
            
        return warped_high.squeeze().cpu().numpy()

    def align_stack_predictive(self, raw_slices):
        """
        主流程: 基于趋势预测的序列对齐
        """
        N, H, W = raw_slices.shape
        aligned_stack = []
        
        # 前两张图无法预测趋势，做刚性对齐或简单 LDDMM
        print("Aligning first 2 slices as baseline...")
        aligned_stack.append(raw_slices[0])
        
        # 第2张图对齐第1张 (无趋势可用)
        s1_aligned = self.run_stable_lddmm(raw_slices[1], raw_slices[0])
        aligned_stack.append(s1_aligned)
        
        print(f"Starting Predictive Alignment (Total {N} slices)...")
        
        # 维护一个平滑的趋势流场 (Momentum)
        current_trend = np.zeros((H, W, 2), dtype=np.float32)
        alpha = 0.5 # 趋势更新率，0.5 表示 50% 历史 + 50% 新趋势
        
        for i in range(2, N):
            # 1. 计算过去两层的生长趋势 (Feature Tracking)
            # 看看 S[i-2] 是怎么变到 S[i-1] 的
            # 这代表了血管的"斜率"
            recent_flow = self.compute_trend_flow(aligned_stack[-2], aligned_stack[-1])
            
            # 2. 更新平滑趋势 (避免单层测量误差)
            # 如果是第一步，直接初始化
            if i == 2:
                current_trend = recent_flow
            else:
                current_trend = (1 - alpha) * current_trend + alpha * recent_flow
            
            # 3. 生成预测目标 (Predicted Target)
            # 假设血管继续按这个斜率生长，S[i] 应该长什么样？
            # Target = Warp(S[i-1], Trend)
            predicted_target = self.predict_next_target(aligned_stack[-1], current_trend)
            
            # 4. 对齐当前层
            # 让 Raw[i] 去对齐 Predicted_Target
            # 因为 Predicted_Target 的血管已经"斜"过来了，Raw[i] 只需要修正微小的非刚性误差
            warped = self.run_stable_lddmm(
                raw_slices[i], 
                predicted_target, 
                grid_size=(8, 8), # 保持你的稳定参数
                reg_weight=1
            )
            
            aligned_stack.append(warped)
            
            if i % 5 == 0:
                print(f"  Slice {i} aligned. (Trend magnitude: {np.mean(np.abs(current_trend)):.2f})")
                
        return np.array(aligned_stack)
        