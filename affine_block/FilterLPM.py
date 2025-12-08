import numpy as np
from scipy.spatial import KDTree
import spacemap

def filter_non_rigid_matches(ptsA, ptsB, k=5, threshold=0.1):
    """
    针对非刚性形变的误匹配剔除 (简化版局部一致性算法)
    
    Args:
        ptsA: 图A的点集 (N, 2)
        ptsB: 图B的点集 (N, 2)
        k: 检查最近邻的数量 (局部性)
        threshold: 相对距离变化的容忍度 (0.1 代表允许 10% 的局部形变误差)
        
    Returns:
        mask: 布尔数组，True 表示内点
    """
    N = len(ptsA)
    if N < k + 1:
        return np.ones(N, dtype=bool)

    # 1. 构建图A的 KD-Tree 以便快速查找局部邻居
    tree = KDTree(ptsA)
    
    # 查找每个点的 k 个最近邻 (包括它自己，所以取 k+1)
    # distances: (N, k+1), indices: (N, k+1)
    distances_A, indices = tree.query(ptsA, k=k+1)
    
    # 去掉自己 (第一列通常是自己，距离为0)
    indices = indices[:, 1:] 
    distances_A = distances_A[:, 1:]
    
    valid_counts = np.zeros(N)
    
    # 2. 检查局部一致性
    for i in range(N):
        # 获取图A中邻居的索引
        neighbor_indices = indices[i]
        
        # 计算图A中，当前点i到邻居的向量长度
        # (已经由 KDTree 返回在 distances_A 中，或者重新计算)
        dist_A = distances_A[i]
        
        # 获取图B中对应的点，并计算距离
        # 图B中的当前点
        ptB_current = ptsB[i]
        # 图B中的邻居点
        ptB_neighbors = ptsB[neighbor_indices]
        
        # 计算图B中的距离
        dist_B = np.linalg.norm(ptB_neighbors - ptB_current, axis=1)
        
        # 3. 核心判断：局部长度变化率
        # 为了防止分母为0，加一个小 epsilon
        ratio = np.abs(dist_A - dist_B) / (dist_A + 1e-6)
        
        # 如果长度变化率小于阈值，认为该邻居支持当前匹配
        support = np.sum(ratio < threshold)
        
        valid_counts[i] = support

    # 如果超过半数的邻居支持，则认为是内点
    mask = valid_counts > (k / 2)
    return mask

# --- 使用示例 ---
# 假设 input_A, input_B 是你的 N 个坐标对
# mask = filter_non_rigid_matches(input_A, input_B, k=8, threshold=0.2)
# clean_A = input_A[mask]
# clean_B = input_B[mask]

class FilterLPMImg(spacemap.AffineBlock):
    def __init__(self, k=8, threshold=0.05):
        super().__init__("FilterLPMImg")
        self.k = k
        self.threshold = threshold
        self.updateMatches = True

    def compute_img(self, imgI: np.array, imgJ: np.array, finder=None):
        return self.compute(None, None, finder)

    def compute(self, dfI: np.array, dfJ: np.array, finder=None):
        matches1 = np.array(self.matches.copy())
        inputA = matches1[:, :2]
        inputB = matches1[:, 2:4]
        mask = filter_non_rigid_matches(inputA, inputB, k=self.k, threshold=self.threshold)
        matches1 = matches1[mask]
        self.matches = matches1
        print(mask.sum(), len(matches1))
        return None
