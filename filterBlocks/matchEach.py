import spacemap
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import os
import cv2

class MatchShow(spacemap.AffineBlock):
    def __init__(self):
        super().__init__("MatchShow")
        self.update_matches = True
        
    def compute(self, dfI: np.array, dfJ: np.array, finder=None):
        H, _ = spacemap.createHFromPoints2(self.matches, spacemap.XYD)
        self.__show_matches(self.matches, dfI, dfJ, H)
        return None

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

class MatchEach(spacemap.AffineBlock):
    def __init__(self):
        super().__init__("MatchEach")
        self.update_matches = True
        self.minMatch = 4
        self.showMatch = False
        
    def compute(self, dfI: np.array, dfJ: np.array, finder=None):
        """ dfI, dfJ -> H """
        matches = self.matches
        imgI = spacemap.show_img3(dfI)
        H, index = self.compute_each_match(matches, self.minMatch, dfI, dfJ, finder, imgI)
        self.matches = matches[:index]
        if self.showMatch:
            self.show_matches(self.matches, dfI, dfJ, H)
        return H
    
    def compute_each_match(self, matches, minMatch, dfI, dfJ, finder, imgI):
        xyd = spacemap.XYD
        xyr = spacemap.XYRANGE
        imgC = spacemap.IMGCONF
        spacemap.Info("SiftEach Get Matches %d" % len(matches))
        finder.clear()
        if minMatch == -1:
            minMatch = len(matches)
        matches = np.array(matches)
        
        spacemap.Info("Compute Each Match Start")
        
        datas = [(matches, i, xyd, imgI, dfJ, xyr, imgC) for i in range(minMatch, len(matches)+1)]
 
        with Pool(os.cpu_count()) as p:
            result = p.map(MatchEach.findBestH, datas)
        for item in result:
            if item is not None:
                finder.add_result(*item)
        spacemap.Info("Compute Each Match Finish")
        
        best = finder.best()
        bestH = best[1]
        index = best[0]
        return bestH, index
    
    @staticmethod
    def findBestH(data):
        matches, i, xyd, imgI, dfJ, xyr, imgC = data
        spacemap.XYD = xyd
        spacemap.XYRANGE = xyr
        spacemap.IMGCONF = imgC
        matchesi = matches[:i]
        H, _ = spacemap.createHFromPoints2(matchesi, xyd)
        if H is None:
            return None
        dfOut = spacemap.applyH_np(dfJ, H)            
        imgJ2 = spacemap.show_img3(dfOut)
        return [i, H, imgI, imgJ2]
    
