import spacemap
import numpy as np


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
        if finder is None:
            finder = spacemap.find.FinderBasic("dice")
        self.affineFinder = finder
        
        self.matchr_std = 1.5
        self.matchr = 200
        self.center = True
        self.glob_count = 100
        
    def find_matchr(self, minCount=75):
        I = spacemap.show_img3(self.__dfI)
        J = spacemap.show_img3(self.__dfJ)
        matchrs = spacemap.matches.siftFindMatchr(I, J)
        for match_ in range(6):
            i = 0.7 + match_ * 0.05
            if matchrs[i] > minCount:
                return i
        return 1.0
         
    def run_flow(self, flow: spacemap.AffineBlock, showErr=True):
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
        
