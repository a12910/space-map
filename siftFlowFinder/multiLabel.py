import spacemap
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class AffineFinderMultiLabelDice(spacemap.AffineFinder):
    def __init__(self, labelI, labelJ, clas=10):
        super().__init__("MultiLabelDice")
        self.clas = clas
        self.labelI = labelI
        self.labelJ = labelJ
        self.clasI = np.zeros(labelI.shape[0])
        self.clasJ = np.zeros(labelJ.shape[0])
        self.init_model()
        
    def init_model(self):
        kmeans = KMeans(n_clusters=self.clas)
        kmeans.fit(self.labelI)
        self.clasI = kmeans.labels_
        self.clasJ = kmeans.predict(self.labelJ)
    
    def compute(self, dfI, dfJ, show=True):
        result = 0
        summ = dfI.shape[0]
        for i in range(self.clas):
            dfI_ = dfI[self.clasI == i].copy()
            dfJ_ = dfJ[self.clasJ == i].copy()
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
            dfI_ = dfI[self.clasI == i].copy()
            dfJ_ = dfJ[self.clasJ == i].copy()
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
    