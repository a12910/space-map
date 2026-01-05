import space_map
import numpy as np
import cv2

class ImgDiffShow(space_map.AffineBlock):
    def __init__(self, size=6, channel=False):
        super().__init__("ImgDiffShow")
        self.size = size
        self.channel = channel
            
    @staticmethod
    def show(imgI, imgJ, channel=False):
        if channel:
            space_map.show_compare_channel(imgI, imgJ)
        else:
            space_map.show_compare_img(imgI, imgJ)
        return imgI, imgJ
        
    def compute(self, dfI: np.array, dfJ: np.array, finder=None):
        imgI = space_map.show_img(dfI)
        imgJ = space_map.show_img(dfJ)
        self.compute_img(imgI, imgJ, finder=finder)
        return None
    
    def compute_img(self, imgI: np.array, imgJ: np.array, finder=None):
        ImgDiffShow.show(imgI, imgJ, channel=self.channel)
        if finder is not None:
            err = finder.err(imgI, imgJ)
            space_map.Info("Error: %f" % err)
        return None
    