import spacemap
import matplotlib.pyplot as plt
import numpy as np
import cv2

class ImgPairInit(spacemap.AffineBlock):
    def __init__(self):
        super().__init__("ImgPairInit")
        self.updateMatches = False
    
    def compute_img(self, imgI: np.array, imgJ: np.array, finder=None):
        imgI, imgJ, H = spacemap.he_img.init_he_pair(imgI, imgJ)
        return H

class ImgPairInitProcess(spacemap.AffineBlock):
    def __init__(self):
        super().__init__("ImgPairInitProcess")
        self.updateImg = True
        self.median = 9
        self.bound = 10
        
        self.filterI = None
        self.filterJ = None
    
    def compute_img(self, imgI: np.array, imgJ: np.array, finder=None):
        imgI = self.process(imgI, True)
        imgJ = self.process(imgJ, False)
        self.imgIJ = (imgI, imgJ)
        return None
    
    def compute(self, dfI: np.array, dfJ: np.array, finder=None):
        return None
    
    def process(self, imgI, II):
        imgI = imgI.copy()
        meanI = self.filterI if II else self.filterJ
        if meanI is None:
            meanI = self.bound_mean(imgI, self.bound)
            if II:
                self.filterI = meanI
            else:
                self.filterJ = meanI
        if len(imgI.shape) == 3 and imgI.shape[2] > 3:
            imgI[imgI[:, :, 3] == 0] = 0
            imgI = imgI[:, :, :3]
        imgI[imgI.mean(axis=2) > meanI*0.9] = 0
        if np.max(imgI) <= 1.0:
            imgI *= 255
        imgI = imgI.astype(np.uint8)
        if self.median is not None:
            imgI = cv2.medianBlur(imgI, self.median)
        return imgI
    
    def bound_mean(self, img: np.array, size):
        img = img.copy()
        if len(img.shape) > 2:
            img = img[:, :, :3]
            img = img.mean(axis=2)
        img1 = np.zeros((img.shape[0], 4*size))
        img1[:, :size] = img[:, :size]
        img1[:, -size:] = img[:, -size:]
        img = img.transpose(1, 0)
        img1[:, size:2*size] = img[:, :size]
        img1[:, -2*size:-size] = img[:, -size:]
        mean = img1.mean()
        return mean   
    