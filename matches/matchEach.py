from numpy.core.multiarray import array as array
import spacemap
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import os
import cv2


class MatchEach(spacemap.AffineBlock):
    def __init__(self):
        super().__init__("MatchEach")
        self.updateMatches = True
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
            if len(item) > 0:
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
        H, _ = spacemap.matches.createHFromPoints2(matchesi, xyd)
        if H is None:
            return []
        dfOut = spacemap.points.applyH_np(dfJ, H)            
        imgJ2 = spacemap.show_img3(dfOut)
        return [i, H, imgI, imgJ2]
