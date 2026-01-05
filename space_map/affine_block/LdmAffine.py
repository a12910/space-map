import cv2
import numpy as np
import space_map
import numpy as np
import matplotlib.pyplot as plt

class LDMAffine(space_map.AffineBlock):
    def __init__(self):
        super().__init__("LDMAffine")
        self.updateMatches = False
        self.H = np.eye(3)
        self.err = 0.1
        self.finder = space_map.find.default()
        
    def compute(self, dfI: np.array, dfJ: np.array, finder=None):
        imgI = space_map.show_img(dfI)
        imgJ = space_map.show_img(dfJ)
        A = self.compute_img(imgI, imgJ, finder)
        A_np = space_map.points.to_npH(A)
        return A_np
    
    def compute_img(self, imgI: np.array, imgJ: np.array, finder=None):
        ldm = space_map.registration.LDDMMRegistration()
        ldm.load_img(imgI, imgJ)
        ldm.err = self.err
        ldm.gpu = space_map.DEVICE
        ldm.verbose = 1000
        A = ldm.run_affine()
        return A
    