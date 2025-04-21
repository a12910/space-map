import spacemap
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import os

from .imgStore import LastImgStore

class MatchEachImg(spacemap.AffineBlock):
    useGlobal = True
    lastWeight = [0.2, 0.2, 0.2, 0.4]
    lastWeightN = 10
    
    def __init__(self):
        super().__init__("MatchEachImg")
        self.updateMatches = True
        self.minMatch = 4
        self.showMatch = False
        self.lastImgs = LastImgStore()
        
    def clear(self):
        self.lastImgs.clear()
    
    def compute_img(self, imgI: np.array, imgJ: np.array, finder=None):
        """ dfI, dfJ -> H """
        matches = self.matches
        self.lastImgs.weight = MatchEachImg.lastWeight
        self.lastImgs.N = MatchEachImg.lastWeightN
        if len(self.lastImgs.lastImgs) == 0:
            self.lastImgs.add_img(imgI)
        H, index = self.compute_each_match(matches, self.minMatch, 
                                           imgI, imgJ, finder)
        self.matches = matches[:index]
        if self.showMatch:
            self.show_matches_img(self.matches, imgI, imgJ, H)
        return H
    
    def compute_each_match(self, matches, minMatch, imgI, imgJ, finder):
        spacemap.Info("SiftEach Get Matches %d" % len(matches))
        finder.clear()
        self.lastImgs.finder = finder
        self.lastImgs.clear_best()
        if minMatch == -1:
            minMatch = len(matches)
        matches = np.array(matches)
        
        spacemap.Info("Compute Each Match Start")
        datas = [[matches, i, imgI, imgJ, self.lastImgs] for i in range(minMatch, len(matches)+1)]
        
        Hs = [MatchEachImg.computeH(data) for data in datas]
        unique_indices = find_unique_matrices_optimized(Hs)
        spacemap.Info("Unique Matrices %d -> %d" % (len(Hs), len(unique_indices)))
        datas = [datas[i] + [Hs[i]] for i in unique_indices]
        
        with Pool(os.cpu_count()) as p:
            result = p.map(MatchEachImg.findBestH, datas)
        for item in result:
            if len(item) > 0:
                self.lastImgs.update_best(*item)
        spacemap.Info("Compute Each Match Finish")
        
        best = self.lastImgs.best()
        bestH = best[1]
        index = best[0]
        return bestH, index
    
    @staticmethod
    def computeH(data):
        matches, i, imgI, imgJ, lastImgs = data
        matchesi = matches[:i]
        _, H = spacemap.matches.createHFromPoints2(matchesi, 1)
        return H
    
    @staticmethod
    def findBestH(data):
        matches, i, imgI, imgJ, lastImgs, H = data
        matchesi = matches[:i]
        # _, H = spacemap.matches.createHFromPoints2(matchesi, 1)
        if H is None:
            return []
        imgJ2 = spacemap.he_img.rotate_imgH(imgJ, H)
        loss = lastImgs.compute_loss(imgI, imgJ2, H)
        return [i, H, imgI, imgJ2, loss]

MATCH_EACH_IMG = MatchEachImg()


def find_unique_matrices_optimized(matrices, err=0.001):
    matrices = [h if h is not None else np.eye(3) for h in matrices ]
    matrices = np.array(matrices)
    N = len(matrices)
    used = np.zeros(N, dtype=bool)
    unique_indices = []
    
    # 预计算矩阵范数可以加速比较
    norms = np.linalg.norm(matrices.reshape(N, -1), axis=1)
    
    for i in range(N):
        if used[i]:
            continue
            
        used[i] = True
        unique_indices.append(i)
        
        # 使用范数进行快速筛选
        norm_diff = np.abs(norms - norms[i])
        candidates = np.where((norm_diff < err) & (~used))[0]
        
        # 只对可能相似的矩阵进行详细比较
        for j in candidates:
            diff = np.abs(matrices[i] - matrices[j])
            if np.all(diff < err):
                used[j] = True
    
    return unique_indices
