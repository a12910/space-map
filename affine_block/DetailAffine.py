import cv2
import numpy as np
from numpy.core.multiarray import array as array
import spacemap
import numpy as np
import matplotlib.pyplot as plt

class DetailAffine(spacemap.AffineBlock):
    def __init__(self):
        super().__init__("DetailAffine")
        self.updateMatches = False
        self.H = np.eye(3)
        self.block = 4
        
    def compute(self, dfI: np.array, dfJ: np.array, finder=None):
        imgI = spacemap.show_img(dfI)
        imgJ = spacemap.show_img(dfJ)
        A = self.compute_img(imgI, imgJ, finder)
        A_np = spacemap.points.to_npH(A)
        return A_np
    
    def compute_img(self, imgI: np.array, imgJ: np.array, finder=None):
        return None
        allmatches = []
        width = imgI.shape[1]
        step =  width // self.block
        self.alignment = spacemap.matches.LOFTR()
        # self.alignment = spacemap.AffineAlignment("sift")
        for i in range(0, width, step):
            for j in range(0, width, step):
                imgI1 = imgI[i:i+step, j:j+step]
                imgJ1 = imgJ[i:i+step, j:j+step]
                if np.mean(imgI1) < 0.1 or np.mean(imgJ1) < 0.1:
                    continue
                matches = self.alignment.compute(imgI1, imgJ1, matchr=0.75)
                if len(matches) < 5:
                    continue
                matches[:, 0] += i
                matches[:, 1] += j
                matches[:, 2] += i
                matches[:, 3] += j
                allmatches += list(matches[:, :4])
                img2 = np.concatenate((imgI, imgJ), axis=1)
                for m in matches:
                    plt.plot([m[1], m[3]+imgI.shape[1]], [m[0], m[2]], 'r-', linewidth=1)
                plt.imshow(img2)
                plt.show()
                
                
                
        spacemap.Info("DetailAffine Get Matches %d block %d" % (len(allmatches), self.block))
        if len(matches) < 5:
            return np.eye(3)
        _, H = spacemap.matches.createHFromPoints2(allmatches, 1)
        if H is None:
            return np.eye(3)
        print(H)
        return H
    