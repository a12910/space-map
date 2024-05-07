import spacemap
import numpy as np
import cv2

class ImgDiffShow(spacemap.AffineBlock):
    def __init__(self, size=6, channel=False):
        super().__init__("ImgDiffShow")
        self.size = size
        self.channel = channel
            
    @staticmethod
    def show(imgI, imgJ, channel=False):
        if channel:
            spacemap.show_compare_channel(imgI, imgJ)
        else:
            spacemap.show_compare_img(imgI, imgJ)
        return imgI, imgJ
        
    def compute(self, dfI: np.array, dfJ: np.array, finder=None):
        imgI = spacemap.show_img3(dfI)
        imgJ = spacemap.show_img3(dfJ)
        self.compute_img(imgI, imgJ, finder=finder)
        return None
    
    def compute_img(self, imgI: np.array, imgJ: np.array, finder=None):
        ImgDiffShow.show(imgI, imgJ, channel=self.channel)
        if finder is not None:
            err = finder.err(imgI, imgJ)
            spacemap.Info("Error: %f" % err)
        return None
    