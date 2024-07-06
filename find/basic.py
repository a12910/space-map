import spacemap
import numpy as np
import cv2
        
class FinderBasic(spacemap.AffineFinder):
    def __init__(self, method):
        super().__init__("AffineFinder-" + method)
        self.method = method
        
    def err(self, imgI: np.array, imgJ: np.array):
        if len(imgI.shape) == 3:
            if imgI.shape[2] == 4:
                imgI = imgI[:, :, :3]
            imgI = np.mean(imgI[:3], axis=2)
        if len(imgJ.shape) == 3:
            if imgJ.shape[2] == 4:
                imgJ = imgJ[:, :, :3]
            imgJ = np.mean(imgJ[:3], axis=2)
        if imgI.shape != imgJ.shape:
            imgJ = cv2.resize(imgJ, imgI.shape[:2])
        
        if self.method == 'mse':
            return spacemap.err.err_mse(imgI, imgJ)
        elif self.method == 'dice':
            return -spacemap.err.err_dice(imgI, imgJ)
        elif self.method == 'dice1':
            return -spacemap.err.err_dice1(imgI, imgJ)
        elif self.method == "conv_dice":
            result, _ = spacemap.find.err_conv_edge2(imgI, imgJ, 10)
            return -result
        else:
            return spacemap.err.err_mse(imgI, imgJ)
        
class SiftCount(spacemap.AffineFinder):
    def __init__(self, matchr=0.75, method="sift"):
        super().__init__("AffineFinderSiftCount")
        self.matchr = matchr
        self.method = method
        
    def err(self, imgI, imgJ):
        matches = []
        if self.method == "sift":
            matches = spacemap.matches.siftImageAlignment(imgI, imgJ, self.matchr)
        elif self.method == "loftr":
            matches = spacemap.matches.loftr_compute_matches(imgI, imgJ, self.matchr)
        elif self.method == "loftr2":
            xyd = spacemap.XYD
            spacemap.XYD = xyd // 2
            matches = spacemap.matches.loftr_compute_matches(imgI, imgJ, self.matchr)
            spacemap.XYD = xyd
        return -len(matches)
        
def default():
    return FinderBasic("dice")
        
