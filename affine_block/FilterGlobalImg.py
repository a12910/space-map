from numpy.core.multiarray import array as array
import spacemap
import matplotlib.pyplot as plt
import numpy as np

class FilterGlobalImg(spacemap.AffineBlock):
    def __init__(self, count=200):
        super().__init__("FilterGlobalImg")
        self.updateMatches = True
        self.count = count
        self.dis = 2
        self.filter_maxdis = None
        
    def compute_img(self, imgI: np.array, imgJ: np.array, finder=None):
        return self.compute(None, None, finder)
        
    def compute(self, dfI: np.array, dfJ: np.array, finder=None):
        matches1 = self.matches.copy()
        if len(matches1) < self.count:
            return None
        while self.dis < spacemap.XYRANGE[1] // 40:
            matches1 = self.matches_filter(self.matches)
            if len(matches1) > self.count * 0.8 or \
                len(matches1) < self.count * 0.2:
                self.dis = int(self.dis * 1.5)
            else:
                break
        spacemap.Info("Global Filter Matches: %d -> %d" % (len(self.matches), len(matches1)))
        self.matches = matches1
        return None
    
    def matches_filter(self, matches):
        matches1 = []
        db = set()
        index = 0
        while len(matches1) < self.count and index < len(matches):
            mat = matches[index]
            index += 1
            if self.filter_maxdis is not None:
                dis = abs(mat[0] - mat[2]) + abs(mat[1] - mat[3])
                if dis > self.filter_maxdis:
                    continue
            x, y = mat[:2]
            x_, y_ = int(x // self.dis), int(y // self.dis)
            pair = (x_, y_)
            if pair in db:
                continue
            db.add(pair)
            matches1.append(mat)
        return np.array(matches1)

    