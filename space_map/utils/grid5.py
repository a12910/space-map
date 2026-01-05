
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import tqdm

def transform_coordinates(A, B, C):
    """
    将B到C的变换矩阵应用到A中
    
    参数:
    A: M个xy坐标的数组，形状为(M, 2)
    B: N个xy坐标的数组，形状为(N, 2)
    C: N个xy坐标的数组，形状为(N, 2)
    
    返回:
    transformed_A: 变换后的A坐标，形状为(M, 2)
    """
    # 确保输入是numpy数组
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    
    # 确保B和C有相同数量的点
    assert B.shape == C.shape, "B和C必须有相同数量的点"
    
    # 为B构建KD树以便快速查找最近邻
    kdtree = cKDTree(B)
    
    # 对A中的每个点，找到B中最近的点的索引
    _, indices = kdtree.query(A, k=4)  # 使用k=4获取4个最近邻
    
    # 初始化变换后的A
    transformed_A = np.zeros_like(A)
    
    # 对A中的每个点应用变换
    for i in tqdm.tqdm(range(len(A))):
        # 获取4个最近邻点的索引
        nn_indices = indices[i]
        # 计算A[i]到B中4个最近邻的距离
        dists = np.array([np.linalg.norm(A[i] - B[j]) for j in nn_indices])
        
        # 如果距离为0（A[i]恰好是B中的点），直接使用对应的C点
        if np.min(dists) < 1e-10:
            idx = nn_indices[np.argmin(dists)]
            transformed_A[i] = C[idx]
        else:
            # 计算基于距离的权重（距离越近权重越大）
            weights = 1.0 / (dists + 1e-10)
            weights = weights / np.sum(weights)  # 归一化权重
            
            # 使用加权平均计算变换
            b_points = B[nn_indices]
            c_points = C[nn_indices]
            
            # 计算B到C的局部变换，然后应用到A[i]
            weighted_offset = np.sum(weights[:, np.newaxis] * (c_points - b_points), axis=0)
            transformed_A[i] = A[i] + weighted_offset
    
    return transformed_A

def visualize_transformation(A, B, C, transformed_A):
    """可视化原始点和变换后的点"""
    plt.figure(figsize=(14, 7))
    
    # 左图：显示A和B
    plt.subplot(1, 2, 1)
    plt.scatter(A[:, 0], A[:, 1], c='blue', s=5, label='Araw')
    plt.scatter(B[:, 0], B[:, 1], c='red', s=20, label='Bdown')
    plt.title('raw')
    plt.legend()
    
    # 右图：显示变换后的A和C
    plt.subplot(1, 2, 2)
    plt.scatter(transformed_A[:, 0], transformed_A[:, 1], c='green', s=5, label='Atransformed')
    plt.scatter(C[:, 0], C[:, 1], c='purple', s=20, label='Ctarget')
    plt.title('transformed')
    plt.legend()
    
    plt.tight_layout()
    plt.show()