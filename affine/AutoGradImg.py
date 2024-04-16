import cv2
from numpy.core.multiarray import array as array
import spacemap
import numpy as np
import matplotlib.pyplot as plt
from spacemap import he_img

class AutoGradImg(spacemap.AffineBlock):
    def __init__(self, sort=None):
        super().__init__("BestGradImg")
        self.updateMatches = False
        self.showGrad = False
        self.H = np.eye(3)
        self.finder = spacemap.find.default()
        self.sort = ["X", "Y", "R", "S"] if sort is None else sort
        
        self.initDis = 0.3
        self.initSkip = 4
        self.multiDis = 0.5
        self.multiSkip = 0.5
        
    def compute_img(self, imgI: np.array, imgJ: np.array, finder=None):
        dis = self.initDis
        skip = self.initSkip
        lastErr = 0
        iter = 0
        if finder is not None:
            self.finder = finder
        
        imgJ_ = imgJ.copy() # last step
        
        while True:
            iter += 1
            for s in self.sort:
                if s == "X":
                    xH, err, c = self.find_bestX(imgI, imgJ_, dis, skip)
                    if xH is not None: 
                        self.H = np.dot(xH, self.H)
                        imgJ_ = he_img.rotate_imgH(imgJ, self.H)
                elif s == "Y":
                    yH, err, c = self.find_bestY(imgI, imgJ_, dis, skip) 
                    if yH is not None:  
                        self.H = np.dot(yH, self.H)
                        imgJ_ = he_img.rotate_imgH(imgJ, self.H)
                elif s == "R":
                    rH, err, c = self.find_bestRotate(imgI, imgJ_, dis, skip)  
                    if rH is not None: 
                        self.H = np.dot(rH, self.H)
                        imgJ_ = he_img.rotate_imgH(imgJ, self.H)
                elif s == "S":
                    sH, err, c = self.find_bestScale(imgI, imgJ_, dis, skip)
                    if sH is not None:
                        self.H = np.dot(sH, self.H)
                        imgJ_ = he_img.rotate_imgH(imgJ, self.H)
    
            spacemap.Info("Iter: %d Grad Find: err=%.5f" % (iter, err))
            dis = dis * self.multiDis
            skip = skip * self.multiDis
            
            if abs(lastErr - err) < 1e-5:
                break
            lastErr = err 
            if self.showGrad:
                imgJ2 = imgJ_[:imgI.shape[0], :imgI.shape[1]]
                plt.imshow(abs(imgI - imgJ2))
                plt.colorbar()
                plt.show()
            
        return self.H
    
    def find_best(self, imgI: np.array, imgJ: np.array, imgFunc, dis, skip=1):
        self.finder.clear()
        v = -dis
        count = 0
        while v < dis:
            imgJ_ = np.zeros_like(imgJ) 
            imgJ_ = imgFunc(imgJ, v, imgJ_)
            self.finder.add_result(v, None, imgI, imgJ_)
            v += skip
            count += 1
        best = self.finder.best()
        v = best[0]
        err = best[2]
        return v, err, count
    
    def find_bestX(self, imgI: np.array, imgJ: np.array, dis, skip=1):
        def moveX(imgJ, v, imgJ_):
            if v < 0:
                imgJ_[0:v, :] = imgJ[-v:, :]
            elif v == 0:
                imgJ_ = imgJ.copy()
            else:
                imgJ_[v:, :] = imgJ[:-v, :]
            return imgJ_
        dis = int(dis * imgI.shape[0])
        if skip < 1:
            skip = 1
        if (dis / skip) < 3:
            return None, 1, 0
        skip = int(skip)
        
        v, err, count = self.find_best(imgI, imgJ, moveX, dis, skip)
        H = np.eye(3)
        H[0, 2] = v
        return H, err, count
    
    def find_bestY(self, imgI: np.array, imgJ: np.array, dis, skip=1):
        def moveY(imgJ, v, imgJ_):
            imgJ_ = np.zeros_like(imgJ)
            if v < 0:
                imgJ_[:, 0:v] = imgJ[:, -v:]
            elif v == 0:
                imgJ_ = imgJ.copy()
            else:
                imgJ_[:, v:] = imgJ[:, :-v]
            return imgJ_
        
        dis = int(dis * imgI.shape[0])
        if dis < 3:
            return None, 1, 0
    
        if skip < 1:
            skip = 1
        skip = int(skip)
        
        v, err, count = self.find_best(imgI, imgJ, moveY, dis, skip)
        H = np.eye(3)
        H[1, 2] = v
        return H, err, count
    
    def find_bestRotate(self, imgI: np.array, imgJ: np.array, dis, skip=1):
        midX, midY = he_img.img_center(imgJ)
        dis = int(dis * 360 * 2)
        if dis > 360:
            dis = 360
        if dis < 3:
            return None, 1, 0
        if skip > 1:
            skip = 1
        
        def rotate(imgJ, v, imgJ_):
            imgJ_, _ = he_img.rotate_img(imgJ, v, (midX, midY), 1.0)
            return imgJ_        
        # print("Search Rotate dis=%.1f skip=%.1f" % (dis, skip))
        v, err, count = self.find_best(imgI, imgJ, rotate, dis, skip)
        H, _ = he_img.rotate_H(v, (midX, midY), 1.0)
        return H, err, count
    
    def find_bestScale(self, imgI: np.array, imgJ: np.array, dis, skip=1):
        midX, midY = he_img.img_center(imgJ)
        dis = dis * 0.5
        skip = skip * 0.01
        if dis / skip < 3:
            return None, 1, 0
        def scale(imgJ, v, imgJ_):
            imgJ_, _ = he_img.rotate_img(imgJ, 0, (midX, midY), v+1)
            return imgJ_
        # print("Search Scale dis=%.3f skip=%.3f" % (dis, skip))
        v, err, count = self.find_best(imgI, imgJ, scale, dis, skip)
        H, _ = he_img.rotate_H(0, (midX, midY), v+1)
        return H, err, count
    