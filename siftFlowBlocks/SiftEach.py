import spacemap
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool
import os

class AffineBlockSiftEach(spacemap.AffineBlock):
    def __init__(self, matchr=0.75, minMatch=4):
        super().__init__("AffineBlockSiftEach")
        self.matchr = matchr
        self.matches = []
        self.show_match = False
        self.match_rate_filter = 0
        self.match_rate_zero = 0
        self.minMatch = minMatch
    
    def compute(self, dfI: np.array, dfJ: np.array, finder=None):
        """ dfI, dfJ -> H """
        imgI = spacemap.show_img3(dfI)
        imgJ = spacemap.show_img3(dfJ)

        matches = self.alignment.compute(imgI, imgJ, 
                                         matchr=self.matchr) 
        if self.match_rate_filter > 0:
            matches = self.compute_match_filter(matches, 
                                                self.match_rate_filter,
                                                width=imgI.shape[0])
        self.matches = matches
        H, _ = self.compute_each_match(matches, self.minMatch, dfI, dfJ, finder, imgI)
        if self.show_match:
            self.show_matches(matches, dfI, dfJ, H)
        return H
    
    def compute_match_filter(self, matches, filt, width):
        count = len(matches)
        rates = np.zeros(count)
        lengths = np.zeros(count)
        for i in range(count):
            x0, y0, x1, y1 = matches[i][:4]
            rates[i] = np.arctan((y1 - y0) / (x1 + width - x0))
            lengths[i] = abs(y1 - y0) + abs(x1 + width - x0)
        meanR = np.mean(rates)
        stdR = np.std(rates)
        if self.match_rate_zero > 0:
            meanR = 0
            stdR = self.match_rate_zero
            
        meanL = np.mean(lengths)
        stdL = np.std(lengths) / 2
        new_matches = []
        for i in range(count):
            if abs(rates[i] - meanR) < stdR * filt:
                if abs(lengths[i] - meanL) < stdL * filt:
                    new_matches.append(matches[i])
        return np.array(new_matches)       

    @staticmethod
    def compute_each_match(matches, minMatch, dfI, dfJ, finder, imgI):
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
            result = p.map(AffineBlockSiftEach.findBestH, datas)
        for item in result:
            if item is not None:
                finder.add_result(*item)
        spacemap.Info("Compute Each Match Finish")
        
        bestH = finder.bestH()
        index = finder.best()[0]
        return bestH, index
    
    @staticmethod
    def findBestH(data):
        matches, i, xyd, imgI, dfJ, xyr, imgC = data
        spacemap.XYD = xyd
        spacemap.XYRANGE = xyr
        spacemap.IMGCONF = imgC
        
        H, _ = spacemap.createHFromPoints2(matches, xyd)
        if H is None:
            return None
        matchesi = matches[:i]
        matchesJ = spacemap.applyH_np(matchesi[:, 2:4] * xyd, H)
        matchesI = matchesi[:, :2] * xyd
        errX, errY = np.mean(matchesI - matchesJ, axis=0)
        H[0, 2] += errX
        H[1, 2] += errY
        dfOut = spacemap.applyH_np(dfJ, H)            
        imgJ2 = spacemap.show_img3(dfOut)
        return [i, H, imgI, imgJ2]
    
""" 
# matchesJ2 = lddmm.applyH_np(matchesi[:, 2:4] * xyd, H)
# errX, errY = np.sum(abs(matchesI - matchesJ2), axis=0)
# print(errX, errY)
"""

"""
for i in tqdm(range(minMatch, len(matches)+1), desc="Compute Each Match"):
    H = spacemap.createHFromPoints2(matches[:i])
    if H is None:
        continue
    matchesi = matches[:i]
    matchesJ = spacemap.applyH_np(matchesi[:, 2:4] * xyd, H)
    matchesI = matchesi[:, :2] * xyd
    errX, errY = np.mean(matchesI - matchesJ, axis=0)
    H[0, 2] += errX
    H[1, 2] += errY
    
    dfOut = spacemap.applyH_np(dfJ, H)            
    imgJ2 = spacemap.show_img3(dfOut)
    finder.add_result(i, H, imgI, imgJ2)
"""        