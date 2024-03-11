import spacemap
import numpy as np

class AffineFinderMultiLabel(spacemap.AffineFinder):
    def __init__(self, labelI, labelJ, clas=10):
        super().__init__("MultiLabel")
        self.clas = clas
        self.labelI = labelI
        self.labelJ = labelJ
        
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
        ranges = [minn + i * step for i in range(self.clas + 1)]
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

from sklearn.cluster import KMeans
def kmeans(data, clas):
    km = KMeans(n_clusters=clas)
    km.fit(data)
    labels = km.labels_
    return labels
    