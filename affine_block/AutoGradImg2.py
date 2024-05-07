import cv2
from numpy.core.multiarray import array as array
import spacemap
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from spacemap import he_img

class AutoGradImg2(spacemap.AffineBlock):
    def __init__(self, method="Powell", useH=True):
        super().__init__("BestGradImg2")
        self.method = method
        self.finder = None
        self.useH = useH
        
    def compute_img(self, imgI: np.array, imgJ: np.array, finder=None):
        self.finder = finder
        initial_params = [1, 0, 0, 0, 1, 0]
        result = minimize(self._err, initial_params, args=(imgI, imgJ), method=self.method)
        H = np.array([[result.x[0], result.x[1], result.x[2]], [result.x[3], result.x[4], result.x[5]], [0, 0, 1]])
        return H
    
    def _err(self, params, imgI, imgJ):
        transform_matrix = np.array([[params[0], params[1], params[2]], [params[3], params[4], params[5]], [0, 0, 1]])
        imgJ_ = spacemap.img.rotate_imgH(imgJ, transform_matrix)
        return self.finder.err(imgI, imgJ_)
