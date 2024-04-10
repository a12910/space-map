import spacemap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class FilterLabels(spacemap.AffineBlock):
    def __init__(self, labelI: np.array, 
                 labelJ: np.array, std):
        super().__init__("FilterLabels")
        self.update_matches = True
        self.labelI = labelI
        self.labelJ = labelJ
        self.std = std
        
    def compute(self, dfI: np.array, dfJ: np.array, finder=None):
        matches = self.matches
        matches1 = self.matches_filter(matches, dfI, dfJ)
        spacemap.Info("Labels Filter Matches: %d -> %d" % (len(matches), len(matches1)))
        self.matches = matches1
        return None
    
    def matches_filter(self, matches, dfI, dfJ):
        labelI = np.array(self.labelI)
        labelJ = np.array(self.labelJ)
        mappI, mapcI = spacemap.show_img_labels(dfI, labelI)
        mappJ, mapcJ = spacemap.show_img_labels(dfJ, labelJ)
        
        def distance(x1, y1, x2, y2):
            c1 = mapcI[int(x1), int(y1)]
            if c1 == 0:
                c1 = 1
            xx = mappI[int(x1), int(y1)] / c1
            c2 = mapcJ[int(x2), int(y2)]
            if c2 == 0:
                c2 = 1
            yy = mappJ[int(x2), int(y2)] / c2
            dis = np.sum(abs(xx - yy))
            return dis
        
        matchDis = np.zeros(len(matches))
        for i in range(len(matches)):
            x1, y1, x2, y2 = matches[i][:4]
            matchDis[i] = distance(x1, y1, x2, y2)
        
        while True:
            maxIndex = np.argmax(matchDis)
            maxV = matchDis[maxIndex]
            mean = np.mean(matchDis[matchDis > 0])
            std = np.std(matchDis[matchDis > 0])
            if abs(maxV - mean) > std * self.std:
                matchDis[maxIndex] = 0
                matches1 = matches[matchDis > 0]
            else:
                break
            if len(matches1) < 8:
                break
            
        matches1 = matches[matchDis > 0]
        return matches1
    