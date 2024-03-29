import spacemap
import numpy as np

class AffineFinder:
    def __init__(self, name):
        self.name = name
        self.DB = {}
        # index, H, err
        self.minItem = None
        
    def err(self, imgI, imgJ):
        raise Exception("Not Implemented")
        return np.sum(abs(imgI - imgJ))
    
    def err_np(self, npI, npJ):
        imgI = spacemap.show_img3(npI)
        imgJ = spacemap.show_img3(npJ)
        return self.err(imgI, imgJ)
        
    def add_result(self, index, H: np.array, imgI: np.array, imgJ: np.array):
        err = self.err(imgI, imgJ)
        self.DB[index] = [index, H, err]
        if self.minItem is None or self.minItem[2] > err:
            self.minItem = self.DB[index]
            
    def computeI(self, imgI, imgJ, show=True):
        e = self.err(imgI, imgJ)
        if show:
            spacemap.Info("Compute %s Error: %.4f" % (self.name, e))
        return e
            
    def compute(self, dfI, dfJ, show=True):
        imgI = spacemap.show_img3(dfI)
        imgJ = spacemap.show_img3(dfJ)
        return self.computeI(imgI, imgJ, show)
    
    def add_result_blank(self, index, H: np.array, err):
        self.DB[index] = [index, H, err]
        if self.minItem is None or self.minItem[2] > err:
            self.minItem = self.DB[index]
            
    def best(self):
        return self.minItem
    
    def bestH(self):
        return self.best()[1]
    
    def clear(self):
        self.DB = {}
        self.minItem = None
        
    def copy(self):
        return AffineFinder(self.name)
        
class AffineFinderBasic(AffineFinder):
    def __init__(self, method):
        super().__init__("AffineFinder-" + method)
        self.method = method
        
    def err(self, imgI, imgJ):
        if self.method == 'mse':
            return spacemap.err_mse(imgI, imgJ)
        elif self.method == 'dice':
            return -spacemap.err_dice(imgI, imgJ)
        elif self.method == 'dice1':
            return -spacemap.err_dice1(imgI, imgJ)
        elif self.method == "conv_dice":
            result, _ = spacemap.err_conv_edge2(imgI, imgJ, 10)
            return -result
        else:
            return spacemap.err_mse(imgI, imgJ)
        
class AffineFinderSiftCount(AffineFinder):
    def __init__(self, matchr=0.75, method="sift"):
        super().__init__("AffineFinderSiftCount")
        self.matchr = matchr
        self.method = method
        
    def err(self, imgI, imgJ):
        matches = []
        if self.method == "sift":
            matches = spacemap.siftImageAlignment(imgI, imgJ, self.matchr)
        elif self.method == "loftr":
            matches = spacemap.loftr_compute_matches(imgI, imgJ, self.matchr)
        elif self.method == "loftr2":
            xyd = spacemap.XYD
            spacemap.XYD = xyd // 2
            matches = spacemap.loftr_compute_matches(imgI, imgJ, self.matchr)
            spacemap.XYD = xyd
        return -len(matches)
        
class AffineFinderDice4(AffineFinder):
    def __init__(self):
        super().__init__("AffineFinderDice4")
        self.centerX = int(spacemap.XYRANGE[1] // spacemap.XYD // 2)
        self.centerY = int(spacemap.XYRANGE[3] // spacemap.XYD // 2)
        self.minErr = np.zeros(4)
        self.minErrIndex = np.zeros(4)
        self.allErr = np.zeros(4)
        
    def add_result(self, index, H: np.array, 
                   imgI: np.array, imgJ: np.array):
        err4 = self.err(imgI, imgJ)
        for i in range(4):
            if self.minErr[i] > err4[i]:
                self.minErrIndex[i] = index
            self.allErr[i] += err4[i]
        self.DB[index] = [index, H, err4]
        
    def clear(self):
        self.minErr = np.zeros(4)
        self.minErrIndex = np.zeros(4)
        self.allErr = np.zeros(4)
        
    def best(self):
        minPart = np.argmax(self.allErr)
        minIndex = self.minErrIndex[minPart]
        return self.DB[minIndex]
        
    def err(self, imgI, imgJ):
        mX = int(imgI.shape[0] // 2)
        mY = int(imgI.shape[1] // 2)
        i1, j1 = imgI[:mX, :mY], imgJ[:mX, :mY]
        i2, j2 = imgI[:mX, mY:], imgJ[:mX, mY:]
        i3, j3 = imgI[mX:, :mY], imgJ[mX:, :mY]
        i4, j4 = imgI[mX:, mY:], imgJ[mX:, mY:]
        d1 = spacemap.err_dice(i1, j1)
        d2 = spacemap.err_dice(i2, j2)
        d3 = spacemap.err_dice(i3, j3)
        d4 = spacemap.err_dice(i4, j4)
        return -d1, -d2, -d3, -d4
  
  # 30-31 31-32 33-34 