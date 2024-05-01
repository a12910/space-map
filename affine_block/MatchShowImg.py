from numpy.core.multiarray import array as array
import spacemap
import numpy as np
import matplotlib.pyplot as plt
import cv2

class MatchShowImg(spacemap.AffineBlock):
    def __init__(self):
        super().__init__("MatchShowImg")
        self.updateMatches = True
        
    def compute_img(self, imgI: np.array, imgJ: np.array, finder=None):
        H, _ = spacemap.matches.createHFromPoints2(self.matches, 1)
        J_ = self.__show_matches_img(self.matches, imgI, imgJ, H)
        if finder is not None:
            err = finder.err(imgI, J_)
            spacemap.Info("Error: %f" % err)
        return None
    
    def __show_matches_img(self, matches, imgI, imgJ, H):
        matchesJ = spacemap.applyH_np(matches[:, 2:4], H)
        if imgI.max() > 128:
            imgI = np.array(imgI, dtype=np.float32) / 32
            imgI = np.clip(imgI, 0, 1)
            
        tag = imgI.max()
        imgI_ = imgI.copy()
        for x, y in matches[:, :2]:
            x_, y_ = int(x), int(y)
            imgI_[x_-4:x_+4, y_-1:y_+1] = tag * 3
        for x, y in matchesJ:
            x_, y_ = int(x), int(y)
            imgI_[x_-1:x_+1, y_-4:y_+4] = tag * 2
            
        imgJ_ = spacemap.img.rotate_imgH(imgJ, H)
        imgJ2 = imgI - imgJ_
        spacemap.show_images_form([imgI_, imgJ2], (1, 2), ["KeyPoints", "Diff"], size=10)
        return imgJ_
