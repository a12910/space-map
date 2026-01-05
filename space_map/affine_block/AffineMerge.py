
import space_map
import numpy as np

class AffineFix:
    def __init__(self, slices, affineKey):
        self.affineKey = affineKey
        self.slices = slices
        self.err = space_map.find.default()

    def fix(self):
        errs = []
        for i, s in enumerate(self.slices[:-1]):
            S1 = self.slices[i]
            S2 = self.slices[i+1]
            H21 = S2.data.loadH(S1.index, self.affineKey)
            img1 = S1.get_img(S1.rawKey, mchannel=True, scale=False, fixHe=False)
            img2 = S2.get_img(S2.rawKey, mchannel=True, scale=False, fixHe=False)
            img2_ = SliceImg.applyH_img(img2, H21)
            err = self.err.err(img1, img2_)
            errs.append(err)
        
        # find 
        errs = np.array(errs)
        mean = np.mean(errs)
        std = np.std(errs)
        for i in range(len(self.slices)-1):
            if errs[i] > mean + std:
                S2.data.saveH(S1.index, self.affineKey, np.eye(3))


