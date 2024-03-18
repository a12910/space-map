import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacemap


class AffineAlignment:
    def __init__(self):
        pass
    
    def compute(self, imgI, imgJ, matchr):
        return spacemap.siftImageAlignment(imgI, imgJ, matchr=matchr)

class AffineData:
    def __init__(self, name):
        self.name = name
    
    def update(self, dfI, dfJ):
        return dfI, dfJ

class AffineBlock:
    def __init__(self, name):
        self.name = name
        self.matches = []
        self.compute_raw = False
        self.update_matches = False
        self.alignment = AffineAlignment()
    
    def compute(self, dfI: np.array, dfJ: np.array, finder=None):
        """ dfI, dfJ -> H """
        return np.eye(3,3)
    
    @staticmethod
    def show_matches(matches, dfI, dfJ, H):
        xyd = spacemap.XYD
        I = spacemap.show_img3(dfI)
        J = spacemap.show_img3(dfJ)
        plt.figure(figsize=(10,10))
        plt.imshow(np.concatenate((I, J), axis=1))
        for m in matches:
            plt.plot([m[1], m[3]+I.shape[1]], [m[0], m[2]], 'r-', linewidth=1)
        plt.show()
        if H is not None:
            imgI_ = I.copy()
            for x, y in matches[:, :2]:
                x_, y_ = int(x), int(y)
                imgI_[x_-4:x_+4, y_-1:y_+1] = 20
            matchesJ = spacemap.applyH_np(matches[:, 2:4] * xyd, H)
            for x, y in matchesJ:
                x_, y_ = int(x / xyd), int(y / xyd)
                imgI_[x_-1:x_+1, y_-4:y_+4] = 10
                
            plt.imshow(imgI_)
            plt.show()
        

class AffineFlowMgr:
    def __init__(self, name, 
                 dfI: np.array, 
                 dfJ: np.array, finder=None):
        self.name = name
        self.__dfI = dfI
        self.__dfJ = dfJ
        self.dfI = dfI
        self.dfJ = dfJ
        self.matches = []
        
        self.initImgC = spacemap.IMGCONF
        self.AffineH: [(np.array, float, str)] = []
        self.affineFinder = finder
        
    def find_matchr(self, minCount=75):
        I = spacemap.show_img3(self.__dfI)
        J = spacemap.show_img3(self.__dfJ)
        matchrs = spacemap.siftFindMatchr(I, J)
        for match_ in range(6):
            i = 0.7 + match_ * 0.05
            if matchrs[i] > minCount:
                return i
        return 1.0
         
    def run_flow(self, flow: AffineBlock, showErr=True):
        self.affineFinder.clear()
        if flow.update_matches:
            flow.matches = np.array(self.matches)
        self.affineFinder.clear()
        H = flow.compute(self.dfI, self.dfJ, self.affineFinder)
        if flow.update_matches:
            self.matches = np.array(flow.matches)
        if H is not None:
            self.dfJ = spacemap.applyH_np(self.dfJ, H)
            err = self.current_err(show=showErr)
            self.AffineH.append((H, err, flow.name))
        return self.dfI, self.dfJ
    
    def current_err(self, show=True):
        return self.affineFinder.compute(self.dfI, self.dfJ, show=show)
    
    def update_data(self, flow: AffineData):
        self.dfI, self.dfJ = flow.update(self.dfI, self.dfJ)

    def resultH(self):
        finalH = np.eye(3, 3)
        for i in range(len(self.AffineH)):
            H = self.AffineH[i][0]
            finalH = np.dot(H, finalH) 
            finalH[2, :2] = 0
            finalH[2, 2] = 1
        return finalH
    
    def best(self, show=True):
        finalH = np.eye(3, 3)
        minErr = 1000
        minH = None
        bestName = None
        for i in range(len(self.AffineH)):
            H, err, name = self.AffineH[i]
            finalH = np.dot(H, finalH) 
            finalH[2, :2] = 0
            finalH[2, 2] = 1
            if minErr > err:
                minErr = err
                minH = finalH.copy()
                bestName = name
        if show:
            spacemap.Info("Find BestH Flow: %s err=%.4f" % (str(bestName), minErr))
        return minH, minErr, bestName
    
    def bestH(self):
        return self.best()[0]
        
    def clear(self):
        self.AffineH = []
        
