import numpy as np
import cv2
from numpy.core.multiarray import array as array
import spacemap

class ManualRotateImg(spacemap.AffineBlock):
    def __init__(self, name=None, rotate=0, moveX=0, moveY=0, ratio=1.0) -> None:
        if name is None:
            name = "ManualRotateImg"
        super().__init__(name)
        self.rotate = rotate
        self.moveX = moveX
        self.moveY = moveY
        self.ratio = ratio
    
    def compute_center_img(self, dfI, dfJ):
        xyrange = spacemap.XYRANGE
        
        meanX = xyrange[1] // 2
        meanY = xyrange[3] // 2
        
        meanIX, meanIY = np.mean(dfI[:, 0]), np.mean(dfI[:, 1])
        meanJX, meanJY = np.mean(dfJ[:, 0]), np.mean(dfJ[:, 1])
        dfI_ = dfI.copy()
        dfJ_ = dfJ.copy()
        dfI_[:, 0] += meanX - meanIX
        dfI_[:, 1] += meanY - meanIY
        dfJ_[:, 0] += meanX - meanJX
        dfJ_[:, 1] += meanY - meanJY
        imgI1 = spacemap.show_img3(dfI_)
        imgJ1 = spacemap.show_img3(dfJ_)
        
        H11 = np.array([[1, 0, -meanJX], [0, 1, -meanJY], [0, 0, 1]])
        H13 = np.array([[1, 0, meanIX], [0, 1, meanIY], [0, 0, 1]])        
        return H11, H13, imgI1, imgJ1
    
    def compute_img(self, imgI: np.array, imgJ: np.array, finder=None):
        h, w = imgJ.shape[:2]
        meanIX, meanIY = w // 2, h // 2
        meanJX, meanJY = w // 2, h // 2
        r = self.rotate / 360 * np.pi * 2
        cosr = np.cos(r)
        sinr = np.sin(r)
        H11 = np.array([[1, 0, -meanJX], [0, 1, -meanJY], [0, 0, 1]])
        H12 = np.array([[cosr, -sinr, 0], [sinr, cosr, 0], [0, 0, 1]])
        H13 = np.array([[self.ratio, 0, 0], [0, self.ratio, 0], [0, 0, 1]])
        H14 = np.array([[1, 0, meanIX + self.moveX], [0, 1, meanIY + self.moveY], [0, 0, 1]])        
        H1 = np.dot(H14, np.dot(H13, np.dot(H12, H11)))
        return H1
    