import spacemap
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class AffineFinderCellLabelDice(spacemap.AffineFinder):
    def __init__(self, labelI: np.array, labelJ: np.array):
        super().__init__("CellLabelDice")
        self.labelI = labelI
        self.labelJ = labelJ
        self.clas = max(np.max(labelI), np.max(labelJ)) + 1
        
    @staticmethod
    def convert_label(labels: list, conf=None):
        if conf is None:
            conf = {}
        maxx = len(conf.values()) 
        result = np.zeros(len(labels))
        for i in range(len(labels)):
            key = labels[i]
            if key in conf:
                result[i] = conf[key]
            else:
                result[i] = maxx
                conf[key] = maxx
                maxx += 1
        return result, conf
    
    def compute(self, dfI, dfJ, show=True):
        result = 0
        summ = dfI.shape[0]
        for i in range(self.clas):
            dfI_ = dfI[self.labelI == i].copy()
            dfJ_ = dfJ[self.labelI == i].copy()
            imgI_ = spacemap.show_img3(dfI_)
            imgJ_ = spacemap.show_img3(dfJ_)
            e = spacemap.err_dice1(imgI_, imgJ_)
            e = e * dfI_.shape[0] / summ
            result += e
        return result
    
    def show(self, dfI, dfJ):
        result = 0
        summ = dfI.shape[0]
        imgIs = []
        imgJs = []
        for i in range(self.clas):
            dfI_ = dfI[self.labelI == i].copy()
            dfJ_ = dfJ[self.labelI == i].copy()
            imgI_ = spacemap.show_img3(dfI_)
            imgJ_ = spacemap.show_img3(dfJ_)
            imgIs.append(imgI_)
            imgJs.append(imgJ_)
            e = spacemap.err_dice1(imgI_, imgJ_)
            e = e * dfI_.shape[0] / summ
            result += e
        imgIs = np.concatenate(imgIs, axis=0)
        imgJs = np.concatenate(imgJs, axis=0)
        return result, imgIs, imgJs
    