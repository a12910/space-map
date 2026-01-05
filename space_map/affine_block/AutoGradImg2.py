import cv2
import space_map
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from space_map import he_img

class AutoGradImg2(space_map.AffineBlock):
    def __init__(self, method="Powell", useH=True):
        super().__init__("BestGradImg2")
        self.method = method
        self.finder = None
        self.useH = useH

    def compute(self, dfI: np.array, dfJ: np.array, finder=None):
        conf = space_map.IMGCONF_CMP or space_map.IMGCONF
        imgI = space_map.show_img(dfI, conf)
        imgJ = space_map.show_img(dfJ, conf)
        return self.compute_img(imgI, imgJ, finder)
        
    def compute_img(self, imgI: np.array, imgJ: np.array, finder=None):
        self.finder = finder
        # imgI = spacemap.show_img(imgI, {"hull": 0})
        # imgJ = spacemap.show_img(imgJ, {"hull": 0})
        initial_params = [1, 0, 0, 0, 1, 0]
        result = minimize(self._err, initial_params, args=(imgI, imgJ), method=self.method)
        H = np.array([[result.x[0], result.x[1], result.x[2]], [result.x[3], result.x[4], result.x[5]], [0, 0, 1]])
        return H
    
    def _err(self, params, imgI, imgJ):
        transform_matrix = np.array([[params[0], params[1], params[2]], [params[3], params[4], params[5]], [0, 0, 1]])
        imgJ_ = space_map.img.rotate_imgH(imgJ, transform_matrix)
        return self.finder.err(imgI, imgJ_)
