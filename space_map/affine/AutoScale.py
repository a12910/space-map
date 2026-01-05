import space_map
import cv2
import numpy as np
    
class AutoScale(space_map.AffineBlock):
    def __init__(self):
        """ 用于计算可能的放大缩小比例 """
        super().__init__("AutoScale")
        
    def compute(self, dfI: np.array, dfJ: np.array, finder=None):
        imgI = space_map.show_img(dfI)
        imgJ = space_map.show_img(dfJ)
        inte1 = space_map.compute_interest_area(imgI, 0.05)
        inte2 = space_map.compute_interest_area(imgJ, 0.05)
        ratio = np.sum(inte1) / np.sum(inte2)
        
        H = np.eye(3)
        H[0, 0] = ratio
        H[1, 1] = ratio
        
        if np.max(dfJ) * ratio > space_map.XYRANGE:
            dis = space_map.XYRANGE - np.max(dfJ) * ratio
            H[0, 2] = -dis
            H[1, 1] = -dis
        
        return H
        