import space_map
import matplotlib.pyplot as plt
import numpy as np

class MatchInit(space_map.AffineBlock):
    def __init__(self, matchr=0.75, method="loftr"):
        super().__init__("MatchInit")
        self.updateMatches = True
        self.matchr = matchr
        if method == "loftr":
            self.alignment = space_map.matches.LOFTR()
        else:
            self.alignment = space_map.AffineAlignment(method=method)
        
    def compute(self, dfI: np.array, dfJ: np.array, finder=None):
        """ dfI, dfJ -> H """
        imgI = space_map.show_img(dfI)
        imgJ = space_map.show_img(dfJ)
        return self.compute_img(imgI, imgJ, finder)
    
    def compute_img(self, imgI: np.array, imgJ: np.array, finder=None):
        space_map.Info("Init Matches start")
        matches = self.alignment.compute(imgI, imgJ, 
                                         matchr=self.matchr) 
        self.matches = matches
        space_map.Info("Init Matches finished: %d" % len(matches))
        return None
    