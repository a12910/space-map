import spacemap
import numpy as np


class AffineFlowMgrBase:
    def __init__(self, name, finder=None) -> None:
        self.name = name
        if finder is None:
            finder = spacemap.find.FinderBasic("dice")
        self.affineFinder: spacemap.AffineFinder = finder
        self.AffineH: [(np.array, float, str)] = []
        self.matches = []
        self.matchr_std = 1.5
        self.matchr = 200
        self.center = True
        self.glob_count = 100
        self.initImgC = spacemap.IMGCONF
        
    def addH(self, H: np.array, err: float, name: str):
        self.AffineH.append((H, err, name))
        
    def resultH(self):
        finalH = np.eye(3, 3)
        for i in range(len(self.AffineH)):
            H = self.AffineH[i][0]
            finalH = np.dot(H, finalH) 
            finalH[2, :2] = 0
            finalH[2, 2] = 1
        return finalH
    
    def resultH_np(self):
        return self.resultH()
    
    def resultH_img(self):
        return self.resultH()
    
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

class AffineFlowMgr(AffineFlowMgrBase):
    def __init__(self, name, 
                 dfI: np.array, 
                 dfJ: np.array, finder=None):
        super().__init__(name, finder)
        self.__dfI = dfI
        self.__dfJ = dfJ
        self.dfI = dfI
        self.dfJ = dfJ
        
    def find_matchr(self, minCount=75):
        I = spacemap.show_img3(self.__dfI)
        J = spacemap.show_img3(self.__dfJ)
        return spacemap.matches.autoSetMatchr(I, J, minCount)
         
    def run_flow(self, flow: spacemap.AffineBlock, showErr=True):
        self.affineFinder.clear()
        if flow.updateMatches:
            flow.matches = np.array(self.matches)
        self.affineFinder.clear()
        H = flow.compute(self.dfI, self.dfJ, self.affineFinder)
        if flow.updateMatches:
            self.matches = np.array(flow.matches)
        if H is not None:
            self.dfJ = spacemap.applyH_np(self.dfJ, H)
            err = self.current_err(show=showErr)
            self.AffineH.append((H, err, flow.name))
        return self.dfI, self.dfJ
    
    def current_err(self, show=True):
        return self.affineFinder.compute(self.dfI, self.dfJ, show=show)
    