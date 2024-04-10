import spacemap
import cv2
import numpy as np
    
class AutoScale(spacemap.AffineBlock):
    def __init__(self):
        """ 用于计算可能的放大缩小比例 """
        super().__init__("AutoScale")
        
    def compute(self, dfI: np.array, dfJ: np.array, finder=None):
        imgI = spacemap.show_img3(dfI)
        imgJ = spacemap.show_img3(dfJ)
        inte1 = spacemap.compute_interest_area(imgI, 0.05)
        inte2 = spacemap.compute_interest_area(imgJ, 0.05)
        ratio = np.sum(inte1) / np.sum(inte2)
        
        H = np.eye(3)
        H[0, 0] = ratio
        H[1, 1] = ratio
        
        if np.max(dfJ) * ratio > spacemap.XYRANGE[1]:
            dis = spacemap.XYRANGE[1] - np.max(dfJ) * ratio
            H[0, 2] = -dis
            H[1, 1] = -dis
        
        return H
        