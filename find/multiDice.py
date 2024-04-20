import spacemap
import numpy as np
import matplotlib.pyplot as plt

class MultiDice(spacemap.AffineFinder):
    def __init__(self, clas=10):
        super().__init__("MultiDice")
        self.clas = clas
        
    def err_all(self, imgI, imgJ):
        ranges = self.compute_range(imgI, imgJ)
        dices = []
        summ = imgI.sum() + imgJ.sum()
        for i in range(self.clas):
            e, pixs = self.err_part(imgI, imgJ, ranges[i], ranges[i+1])
            if e is not None:
                e = e * pixs / summ
                dices.append(e)
        if len(dices) == 0:
            return [], 0
        result = sum(dices)
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
            return None, None
        return [spacemap.err.err_dice(imgI_, imgJ_), np.sum(imgI_) + np.sum(imgJ_)]
    
    def filter_part(self, img, minn, maxx):
        img = img.copy()
        img[img < minn] = 0
        img[img > maxx] = 0
        img[img > 0] = 1
        return img
    
    @staticmethod
    def show(imgI, imgJ, clas):
        err = MultiDice(clas)
        ranges = err.compute_range(imgI, imgJ)
        Is = []
        Js = []
        for i in range(clas):
            minn, maxx = ranges[i], ranges[i+1]
            imgI_ = err.filter_part(imgI, minn, maxx)
            imgJ_ = err.filter_part(imgJ, minn, maxx)
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
            
    