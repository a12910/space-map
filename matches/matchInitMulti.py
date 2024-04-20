import spacemap
import matplotlib.pyplot as plt
import numpy as np

class MatchInitMulti(spacemap.AffineBlock):
    def __init__(self, methods=None, number=200):
        super().__init__("MatchInitMulti")
        self.updateMatches = True
        self.number = number 
        if methods is None:
            methods = ["loftr"]
        self.alignments = []
        for method in methods:
            if method == "loftr":
                self.alignments.append(spacemap.matches.LOFTR())
            else:
                self.alignments.append(spacemap.AffineAlignment(method=method))
        
    def compute(self, dfI: np.array, dfJ: np.array, finder=None):
        """ dfI, dfJ -> H """
        imgI = spacemap.show_img3(dfI)
        imgJ = spacemap.show_img3(dfJ)
        return self.compute_img(imgI, imgJ, finder)
    
    def compute_img(self, imgI: np.array, imgJ: np.array, finder=None):
        spacemap.Info("Init Matches start")
        matches = []
        for alignment in self.alignments:
            matches += alignment.compute(imgI, imgJ, 
                                         matchr=self.matchr)
        self.matches = matches
        spacemap.Info("Init Matches finished: %d" % len(matches))
        return None
    
class MatchInitAuto(MatchInitMulti):
    def __init__(self, methods=None, number=200):
        if methods is None:
            methods = ["loftr", "sift"]
        super().__init__(methods, number)
    
    def compute_img(self, imgI: np.array, imgJ: np.array, finder=None):
        spacemap.Info("Init Matches start")
        matches = []
        for alignment in self.alignments:
            matches += alignment.compute(imgI, imgJ, 
                                         matchr=self.matchr)
            if len(matches) > self.number:
                break
        self.matches = matches
        spacemap.Info("Init Matches finished: %d" % len(matches))
        return None
    