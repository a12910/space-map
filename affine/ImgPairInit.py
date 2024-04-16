import spacemap
import matplotlib.pyplot as plt
import numpy as np

class ImgPairInit(spacemap.AffineBlock):
    def __init__(self):
        super().__init__("ImgPairInit")
        self.updateMatches = False
    
    def compute_img(self, imgI: np.array, imgJ: np.array, finder=None):
        imgI, imgJ, H = spacemap.he_img.init_he_pair(imgI, imgJ)
        return H
            