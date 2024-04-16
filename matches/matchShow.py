from numpy.core.multiarray import array as array
import spacemap
import numpy as np
import matplotlib.pyplot as plt

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
        