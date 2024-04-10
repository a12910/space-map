import cv2
import spacemap
import numpy as np

class AutoGrad(spacemap.AffineBlock):
    def __init__(self) -> None:
        super().__init__("BestGrad")
        self.update_matches = False
        self.showGrad = False
        
    def compute(self, dfI: np.array, dfJ: np.array, finder=None):
        imgC = {"raw": 1, "gauss": 0}
        imgI = spacemap.show_img3(dfI, imgConf=imgC)
        imgJ = spacemap.show_img3(dfJ, imgConf=imgC)
        H = np.eye(3)
        
        dis = 0.6
        skip = 16
        lastErr = 0
        
        while True:
            xH, err = self.find_bestX(imgI, imgJ, finder, dis, skip)
            if xH is not None: 
                H = np.dot(xH, H)
                ps = spacemap.applyH_np(dfJ, H)
                imgJ = spacemap.show_img3(ps, imgConf=imgC)
            
            yH, err = self.find_bestY(imgI, imgJ, finder, dis, skip) 
            if yH is not None:  
                H = np.dot(yH, H)
                ps = spacemap.applyH_np(dfJ, H)
                imgJ = spacemap.show_img3(ps, imgConf=imgC)
            
            rH, err = self.find_bestRotate(imgI, imgJ, finder, dis, skip)  
            if rH is not None: 
                H = np.dot(rH, H)
                ps = spacemap.applyH_np(dfJ, H)
                imgJ = spacemap.show_img3(ps, imgConf=imgC)
            
            spacemap.Info("Grad Find: err=%.3f" % (err))
            dis = dis * 0.5
            skip = skip * 0.5
            
            if abs(lastErr - err) < 0.001:
                break
            lastErr = err 
            if self.showGrad:
                spacemap.show_align_np(dfI, ps, "TARGET", "NEW")
            
        return H
    
    def find_best(self, imgI: np.array, imgJ: np.array, imgFunc, finder: spacemap.AffineFinder, dis, skip=1):
        finder.clear()
        v = -dis
        while v < dis:
            imgJ_ = np.zeros_like(imgJ)
            imgJ_ = imgFunc(imgJ, v, imgJ_)
            finder.add_result(v, None, imgI, imgJ_)
            v += skip
        best = finder.best()
        v = best[0]
        err = best[2]
        return v, err
    
    def find_bestX(self, imgI: np.array, imgJ: np.array,
                   finder: spacemap.AffineFinder, dis, skip=1):
        def moveX(imgJ, v, imgJ_):
            if v < 0:
                imgJ_[0:v, :] = imgJ[-v:, :]
            elif v == 0:
                imgJ_ = imgJ
            else:
                imgJ_[v:, :] = imgJ[:-v, :]
            return imgJ_
        dis = int(dis * imgI.shape[0])
        if skip < 1:
            skip = 1
        if (dis / skip) < 3:
            return None, 1
        skip = int(skip)
        
        v, err = self.find_best(imgI, imgJ, moveX, finder, dis, skip)
        H = np.eye(3)
        H[0, 2] = v * spacemap.XYD
        return H, err
    
    def find_bestY(self, imgI: np.array, imgJ: np.array, 
                   finder: spacemap.AffineFinder, dis, skip=1):
        def moveY(imgJ, v, imgJ_):
            imgJ_ = np.zeros_like(imgJ)
            if v < 0:
                imgJ_[:, 0:v] = imgJ[:, -v:]
            elif v == 0:
                imgJ_ = imgJ
            else:
                imgJ_[:, v:] = imgJ[:, :-v]
            return imgJ_
        
        dis = int(dis * imgI.shape[0])
        if dis < 3:
            return None, 1
    
        if skip < 1:
            skip = 1
        skip = int(skip)
        
        v, err = self.find_best(imgI, imgJ, moveY, finder, dis, skip)
        H = np.eye(3)
        H[1, 2] = v * spacemap.XYD
        return H, err
    
    def find_bestRotate(self, imgI: np.array, imgJ: np.array, finder: spacemap.AffineFinder, dis, skip=1):
        
        Xs = np.sum(imgJ, axis=1)
        Ys = np.sum(imgJ, axis=0)
        dd = np.array(range(imgJ.shape[0]))
        midX = int(np.sum(Xs * dd) / np.sum(Xs))
        midY = int(np.sum(Ys * dd) / np.sum(Ys))
        
        dis = int(dis * 360 * 2)
        if dis > 360:
            dis = 360
        if dis < 3:
            return None, 1
        
        def rotate(imgJ, v, imgJ_):
            imgJ_ = spacemap.compute.rotate_img(imgJ, v, (midX, midY))
            return imgJ_        
        
        v, err = self.find_best(imgI, imgJ, rotate, finder, dis, skip)
        H = spacemap.compute.rotate_H(imgJ, v, (midX, midY))
        return H, err
    