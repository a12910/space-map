import spacemap
import matplotlib.pyplot as plt
import numpy as np

class MatchInitImg(spacemap.AffineBlock):
    def __init__(self, matchr=0.75, method="loftr"):
        super().__init__("MatchInitImg")
        self.updateMatches = True
        self.matchr = matchr
        self.method = method
        if method == "loftr":
            self.alignment = spacemap.matches.LOFTR()
        else:
            self.alignment = spacemap.AffineAlignment(method=method)
        
    def compute(self, dfI: np.array, dfJ: np.array, finder=None):
        """ dfI, dfJ -> H """
        imgI = spacemap.show_img3(dfI)
        imgJ = spacemap.show_img3(dfJ)
        return self.compute_img(imgI, imgJ, finder)
    
    def compute_img(self, imgI: np.array, imgJ: np.array, finder=None):
        spacemap.Info("Init Matches start %s" % self.method)
        matches = self.alignment.compute(imgI, imgJ, 
                                         matchr=self.matchr) 
        self.matches = matches
        spacemap.Info("Init Matches finished: %d" % len(matches))
        return None
    