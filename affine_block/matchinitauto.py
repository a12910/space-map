import spacemap
import matplotlib.pyplot as plt
import numpy as np

class MatchInitAuto(spacemap.AffineBlock):
    def __init__(self):
        super().__init__("MatchInitAuto")
        self.updateMatches = True
        
    def compute(self, dfI: np.array, dfJ: np.array, finder=None):
        """ dfI, dfJ -> H """
        conf_ = spacemap.IMGCONF.copy()
        conf = conf_.copy()
        conf["raw"] = 1
        spacemap.IMGCONF = conf
        imgI = spacemap.show_img(dfI)
        imgJ = spacemap.show_img(dfJ)
        align1 = spacemap.matches.LOFTR()
        matches1 = align1.compute(imgI, imgJ)
        align2 = spacemap.AffineAlignment("sift_vgg")
        imgI[imgI > 0] = 1
        imgJ[imgJ > 0] = 1
        matches2 = align2.compute(imgI, imgJ)
        matches = self.merge_matches(matches1, matches2)
        self.matches = matches
        spacemap.Info("Init Matches finished: %d %d" % (len(matches1), len(matches2)))
        spacemap.IMGCONF = conf_
        return None
    
    def compute_img(self, imgI: np.array, imgJ: np.array, finder=None):
        spacemap.Info("Init Matches start")
        align1 = spacemap.matches.LOFTR()
        matches1 = align1.compute(imgI, imgJ)
        imgI = imgI.copy()
        imgJ = imgJ.copy()
        imgI[imgI > 0] = 1
        imgJ[imgJ > 0] = 1
        align2 = spacemap.AffineAlignment("sift_vgg")
        matches2 = align2.compute(imgI, imgJ)
        matches = self.merge_matches(matches1, matches2)
        self.matches = matches
        spacemap.Info("Init Matches finished: %d %d" % (len(matches1), len(matches2)))
        return None
    
    def merge_matches(self, matches1, matches2):
        matches = []
        if len(matches1) < len(matches2):
            matches1, matches2 = matches2, matches1
        for i, match1 in enumerate(matches1):
            matches.append(match1)
            if i >= len(matches2):
                continue
            matches.append(matches2[i])
        return matches
    