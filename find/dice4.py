
import spacemap
import numpy as np

class Dice4(spacemap.AffineFinder):
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
  
  