
from kornia.feature import LoFTR
import torch
import spacemap
import numpy as np

class AffineAlignmentLOFTR2(spacemap.AffineAlignment):
    def __init__(self):
        super().__init__()
        
    def compute(self, imgI, imgJ, matchr):
        x, y = imgI.shape[:2]
        x = x // 2
        y = y // 2
        
        if matchr > 1:
            matchr = matchr // 4
        results = []
        
        imgI1, imgJ1 = imgI[:x, :y], imgJ[:x, :y]
        part1 = self.compute_part(imgI1, imgJ1, matchr)
        part1 += np.array([0, 0, 0, 0, 0])
        results += list(part1)
        
        imgI2, imgJ2 = imgI[x:, :y], imgJ[x:, :y]
        part2 = self.compute_part(imgI2, imgJ2, matchr)
        part2 += np.array([x, 0, x, 0, 0])
        results += list(part2)
        
        imgI3, imgJ3 = imgI[:x, y:], imgJ[:x, y:]
        part3 = self.compute_part(imgI3, imgJ3, matchr)
        part3 += np.array([0, y, 0, y, 0])
        results += list(part3)
        
        imgI4, imgJ4 = imgI[x:, :y], imgJ[x:, y:]
        part4 = self.compute_part(imgI4, imgJ4, matchr)
        part4 += np.array([x, y, x, y, 0])
        results += list(part4)
        
        return np.array(results)
    
    def compute_part(self, imgI, imgJ, matchr):
        return spacemap.loftr_compute_matches(imgI, imgJ, matchr)
