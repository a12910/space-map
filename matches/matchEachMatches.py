import spacemap
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool
import os

class MatchEachMatches(spacemap.AffineBlock):
    """ 仅选择最佳特征点 """
    def __init__(self):
        super().__init__("MatchEachMatches")
        self.updateMatches = True
        self.minMatch = 4
        self.showMatch = False
        
    def compute(self, dfI: np.array, dfJ: np.array, finder=None):
        """ dfI, dfJ -> H """
        matches = self.matches
        H, index = self.compute_each_match(matches, self.minMatch)
        self.matches = matches[:index]
        if self.showMatch:
            self.show_matches(self.matches, dfI, dfJ, H)
        return H
    
    def compute_each_match(self, matches, minMatch):
        xyd = spacemap.XYD
        spacemap.Info("SiftEachM Get Matches %d" % len(matches))
        finder = spacemap.find.default()
        
        if minMatch == -1:
            minMatch = len(matches)
        matches = np.array(matches)
        for i in tqdm(range(minMatch, len(matches)+1), desc="Compute Each Match"):
            matchesi = matches[:i]
            H, _ = spacemap.matches.createHFromPoints2(matchesi, xyd)
            matchesJ = spacemap.points.applyH_np(matchesi[:, 2:4] * xyd, H)
            matchesI = matchesi[:, :2] * xyd
            errX, errY = np.mean(matchesI - matchesJ, axis=0)
            err = errX **2 + errY **2        
            finder.add_result_blank(i, H, err)
            
        best = finder.best()
        bestH = best[1]
        index = best[0]
        return bestH, index
    