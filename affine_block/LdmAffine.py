import cv2
import numpy as np
import spacemap
import numpy as np
import matplotlib.pyplot as plt

class LDMAffine(spacemap.AffineBlock):
    def __init__(self):
        super().__init__("LDMAffine")
        self.updateMatches = False
        self.H = np.eye(3)
        self.err = 0.1
        self.finder = spacemap.find.default()
        
    def compute(self, dfI: np.array, dfJ: np.array, finder=None):
        imgI = spacemap.show_img3(dfI)
        imgJ = spacemap.show_img3(dfJ)
        A = self.compute_img(imgI, imgJ, finder)
        A_np = spacemap.points.to_npH(A)
        return A_np
    
    def compute_img(self, imgI: np.array, imgJ: np.array, finder=None):
        ldm = spacemap.registration.LDDMMRegistration()
        ldm.load_img(imgI, imgJ)
        ldm.err = self.err
        ldm.verbose = 1000
        A = ldm.run_affine()
        return A
    