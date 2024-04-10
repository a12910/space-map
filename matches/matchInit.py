import spacemap
import matplotlib.pyplot as plt
import numpy as np

class MatchInit(spacemap.AffineBlock):
    def __init__(self, matchr=0.75):
        super().__init__("MatchInit")
        self.update_matches = True
        self.matchr = matchr
        
    def compute(self, dfI: np.array, dfJ: np.array, finder=None):
        """ dfI, dfJ -> H """
        imgI = spacemap.show_img3(dfI)
        imgJ = spacemap.show_img3(dfJ)
        spacemap.Info("Init Matches start")
        matches = self.alignment.compute(imgI, imgJ, 
                                         matchr=self.matchr) 
        self.matches = matches
        spacemap.Info("Init Matches finished: %d" % len(matches))
        return None
    