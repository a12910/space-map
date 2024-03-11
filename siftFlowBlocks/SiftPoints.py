import spacemap
import numpy as np

class AffineBlockSiftPoint(spacemap.AffineBlock):
    def __init__(self, matches):
        super().__init__("AffineBlockSiftPoint")
        self.matches = matches
    
    def compute(self, dfI: np.array, dfJ: np.array, finder=None):
        m = self.matches
        A = m[:3][:2]
        B = m[:3][2:4]
        H = spacemap.compute_H_from_3points(A, B)
        return H
    
    def compute_align(self, dfI, dfJ, indexs=None):
        m = self.matches
        if indexs is None:
            A = m[:3][:2]
            B = m[:3][2:4]
        else:
            i1, i2, i3 = indexs
            A = np.array([m[i1][:2], m[i2][:2], m[i3][:2]])
            B = np.array([m[i1][2:4], m[i2][2:4], m[i3][2:4]])
        H = spacemap.compute_H_from_3points(A, B)
        dfJ2 = spacemap.applyH_np(dfJ, H)
        spacemap.show_align_np(dfI, dfJ2, "TARGET_I", "NEW_J")
        