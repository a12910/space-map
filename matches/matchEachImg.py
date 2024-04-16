from numpy.core.multiarray import array as array
import spacemap
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import os

class MatchEachImg(spacemap.AffineBlock):
    def __init__(self):
        super().__init__("MatchEachImg")
        self.updateMatches = True
        self.minMatch = 4
        self.showMatch = False
    
    def compute_img(self, imgI: np.array, imgJ: np.array, finder=None):
        """ dfI, dfJ -> H """
        matches = self.matches
        H, index = self.compute_each_match(matches, self.minMatch, 
                                           imgI, imgJ, finder)
        self.matches = matches[:index]
        if self.showMatch:
            self.show_matches_img(self.matches, imgI, imgJ, H)
        return H
    
    def compute_each_match(self, matches, minMatch, imgI, imgJ, finder):
        spacemap.Info("SiftEach Get Matches %d" % len(matches))
        finder.clear()
        if minMatch == -1:
            minMatch = len(matches)
        matches = np.array(matches)
        
        spacemap.Info("Compute Each Match Start")
        
        datas = [(matches, i, imgI, imgJ) for i in range(minMatch, len(matches)+1)]
 
        with Pool(os.cpu_count()) as p:
            result = p.map(MatchEachImg.findBestH, datas)
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
        matches, i, imgI, imgJ = data
        matchesi = matches[:i]
        H, _ = spacemap.matches.createHFromPoints2(matchesi, 1)
        if H is None:
            return []
        imgJ2 = spacemap.he_img.rotate_imgH(imgJ, H)
        return [i, H, imgI, imgJ2]
