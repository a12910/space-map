from numpy.core.multiarray import array as array
import spacemap
import numpy as np
import matplotlib.pyplot as plt
import cv2

class MatchShow(spacemap.AffineBlock):
    def __init__(self):
        super().__init__("MatchShow")
        self.updateMatches = True
        
    def compute(self, dfI: np.array, dfJ: np.array, finder=None):
        H, _ = spacemap.matches.createHFromPoints2(self.matches, spacemap.XYD)
        self.__show_matches(self.matches, dfI, dfJ, H)
        return None
    
    def compute_img(self, imgI: np.array, imgJ: np.array, finder=None):
        H, _ = spacemap.matches.createHFromPoints2(self.matches, 1)
        self.__show_matches_img(self.matches, imgI, imgJ, H)
        return None
    
    def __show_matches_img(self, matches, imgI, imgJ, H):
        xyd = 1
        matchesJ = spacemap.applyH_np(matches[:, 2:4] * xyd, H)
        if imgI.max() > 128:
            imgI = np.array(imgI, dtype=np.float32) / 32
            imgI = np.clip(imgI, 0, 1)
            
        tag = imgI.max()
        imgI_ = imgI.copy()
        for x, y in matches[:, :2]:
            x_, y_ = int(x), int(y)
            imgI_[x_-4:x_+4, y_-1:y_+1] = tag * 3
        for x, y in matchesJ:
            x_, y_ = int(x / xyd), int(y / xyd)
            imgI_[x_-1:x_+1, y_-4:y_+4] = tag * 2
        plt.imshow(imgI_)
        plt.show()

    def __show_matches(self, matches, dfI, dfJ, H):
        xyd = spacemap.XYD
        I = spacemap.show_img3(dfI)
        
        matchesJ = spacemap.applyH_np(matches[:, 2:4] * xyd, H)
        
        tag = I.max()
        imgI_ = I.copy()
        for x, y in matches[:, :2]:
            x_, y_ = int(x), int(y)
            imgI_[x_-4:x_+4, y_-1:y_+1] = tag * 3
        for x, y in matchesJ:
            x_, y_ = int(x / xyd), int(y / xyd)
            imgI_[x_-1:x_+1, y_-4:y_+4] = tag * 2
            
        plt.imshow(imgI_)
        plt.show()
        
class ImgDiffShow(spacemap.AffineBlock):
    def __init__(self):
        super().__init__("ImgDiffShow")
        
    @staticmethod
    def process(imgI, imgJ):
        def __process(imgI):
            
            if len(imgI.shape) and imgI.shape[2] > 3:
                imgI = imgI[:, :, 3]
            if len(imgI.shape) != 3:
                imgI = np.stack([imgI, imgI, imgI], axis=2)
            if np.max(imgI) <= 1.0:
                imgI = np.array(imgI * 255)
            imgI = np.array(imgI, dtype=np.uint8)
            return imgI
        imgI = __process(imgI)
        imgJ = __process(imgJ)            
        if imgJ.shape != imgI.shape:
            imgJ = cv2.resize(imgJ, imgI.shape)
        return imgI, imgJ
            
    @staticmethod
    def show(imgI, imgJ):
        imgI, imgJ = ImgDiffShow.process(imgI, imgJ)
        diff = np.abs(imgI - imgJ)
        spacemap.show_images_form([imgI, imgJ, diff], (1, 3), ["I", "J", "Diff"], size=6)
        return imgI, imgJ
        
    def compute(self, dfI: np.array, dfJ: np.array, finder=None):
        imgI = spacemap.show_img3(dfI)
        imgJ = spacemap.show_img3(dfJ)
        return self.compute_img(imgI, imgJ, finder)
    
    def compute_img(self, imgI: np.array, imgJ: np.array, finder=None):
        ImgDiffShow.show(imgI, imgJ)
        return None
    