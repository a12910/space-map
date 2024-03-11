import spacemap
import numpy as np
import matplotlib.pyplot as plt

class AffineFinderMultiDice(spacemap.AffineFinder):
    def __init__(self, clas=10):
        super().__init__("MultiDice")
        self.clas = clas
        
    def err_all(self, imgI, imgJ):
        ranges = self.compute_range(imgI, imgJ)
        dices = []
        for i in range(self.clas):
            e = self.err_part(imgI, imgJ, ranges[i], ranges[i+1])
            dices.append(e)
        dices2 = []
        for i in dices:
            if i is not None:
                dices2.append(i)
        if len(dices2) == 0:
            return 0
        result = sum(dices2) / len(dices2)
        return dices, result
        
    def err(self, imgI, imgJ):
        dices, result = self.err_all(imgI, imgJ)
        return -result
    
    def compute_range(self, imgI, imgJ):
        maxx = max(np.max(imgI), np.max(imgJ)) + 1
        minn = min(np.min(imgI), np.min(imgJ)) - 1
        step = (maxx - minn) / self.clas
        ranges = [minn + i * step for i in range(self.clas + 3)]
        return ranges
    
    def err_part(self, imgI, imgJ, minn, maxx):
        imgI_ = self.filter_part(imgI, minn, maxx)
        imgJ_ = self.filter_part(imgJ, minn, maxx)
        if np.sum(imgI_) < 1 and np.sum(imgJ_) < 1:
            return None
        return spacemap.err_dice(imgI_, imgJ_)
    
    def filter_part(self, img, minn, maxx):
        img = img.copy()
        img[img < minn] = 0
        img[img > maxx] = 0
        img[img > 0] = 1
        return img
    
    @staticmethod
    def show(imgI, imgJ, clas):
        err = AffineFinderMultiDice(clas)
        ranges = err.compute_range(clas, imgI, imgJ)
        Is = []
        Js = []
        for i in range(clas):
            imgI_, imgJ_ = err.err_part(imgI, imgJ, ranges[i], ranges[i+1])
            if imgI_ is not None:
                Is.append(imgI_)
                Js.append(imgJ_)
        for i in range(len(Is)):
            imgg = np.zeros((imgI.shape[0],
                             imgI.shape[1], 3))
            imgg[:, :, 0] = Is[i]
            imgg[:, :, 1] = Js[i]
            plt.imshow(imgg)
            plt.show()
            