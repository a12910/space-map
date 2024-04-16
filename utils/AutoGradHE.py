import cv2
import spacemap
import numpy as np
import matplotlib.pyplot as plt
from spacemap import he_img

class AutoGradHE:
    def __init__(self):
        self.showGrad = False
        self.finder_DB = {}
        self.finder_minItem = None  # index, img, err
        self.H = np.eye(3)
    
    def finder_clear(self):
        self.finder_DB = {}
        self.finder_minItem = None
        
    def finder_add_result(self, index, imgI: np.array, imgJ: np.array):
        err = he_img.finder_err(imgI, imgJ)
        self.finder_DB[index] = [index, imgJ.copy(), err]
        if self.finder_minItem is None or self.finder_minItem[2] > err:
            self.finder_minItem = self.finder_DB[index]
            
    def init_img(self, imgI, imgJ):
        if len(imgJ.shape) == 3:
            imgJ = cv2.cvtColor(imgJ, cv2.COLOR_BGR2GRAY)
        if len(imgI.shape) == 3:
            imgI = cv2.cvtColor(imgI, cv2.COLOR_BGR2GRAY)
        
        imgI, imgJ = he_img.img_norm(imgI, imgJ)
        imgJ1 = he_img.fill_square(imgJ)
        
        scale = he_img.auto_scale(imgI, imgJ1)
        length2 = int(imgJ1.shape[0] * scale)
        
        imgJ2 = cv2.resize(imgJ1, (length2, length2))
        midJ = he_img.img_center(imgJ2, True)
        sIx, sIy = imgI.shape
        
        imgJ3 = np.zeros((length2*2, length2*2))
        
        imgJ3 = he_img.cut_img(imgJ2, midJ[0]-sIx//2, midJ[1]-sIy//2, 
                        midJ[0]+sIx//2, midJ[1]+sIy//2)
        
        H = he_img.rotate_H(0, (0, 0), scale)[0]
        H2 = np.eye(3)
        H2[0, 2] = -(midJ[1] - sIy // 2)
        H2[1, 2] = -(midJ[0] - sIx // 2)
        H = np.dot(H2, H)
        self.H = H
        
        return imgI, imgJ3
        
    def compute(self, imgI: np.array, imgJ: np.array):
        dis = 0.3
        skip = 4
        lastErr = 0
        iter = 0
        
        imgI, imgJ = self.init_img(imgI, imgJ)
        imgJ_ = imgJ.copy() # last step
        
        while True:
            iter += 1
            xH, err = self.find_bestX(imgI, imgJ_, dis, skip)
            if xH is not None: 
                self.H = np.dot(xH, self.H)
                imgJ_ = he_img.rotate_imgH(imgJ, xH)
            
            yH, err = self.find_bestY(imgI, imgJ_, dis, skip) 
            if yH is not None:  
                self.H = np.dot(yH, self.H)
                imgJ_ = he_img.rotate_imgH(imgJ, yH)
            
            rH, err = self.find_bestRotate(imgI, imgJ_, dis, skip)  
            if rH is not None: 
                self.H = np.dot(rH, self.H)
                imgJ_ = he_img.rotate_imgH(imgJ, rH)
                
            sH, err = self.find_bestScale(imgI, imgJ_, dis, skip)
            if sH is not None:
                self.H = np.dot(sH, self.H)
                imgJ_ = he_img.rotate_imgH(imgJ, sH)
            
            print("Iter: %d Grad Find: err=%.5f" % (iter, err))
            dis = dis * 0.5
            skip = skip * 0.5
            
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
        self.finder_clear()
        v = -dis
        while v < dis:
            imgJ_ = np.zeros_like(imgJ) 
            imgJ_ = imgFunc(imgJ, v, imgJ_)
            self.finder_add_result(v, imgI, imgJ_)
            v += skip
        best = self.finder_minItem
        v = best[0]
        err = best[2]
        return v, err
    
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
            return None, 1
        skip = int(skip)
        
        v, err = self.find_best(imgI, imgJ, moveX, dis, skip)
        H = np.eye(3)
        H[0, 2] = v
        return H, err
    
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
            return None, 1
    
        if skip < 1:
            skip = 1
        skip = int(skip)
        
        v, err = self.find_best(imgI, imgJ, moveY, dis, skip)
        H = np.eye(3)
        H[1, 2] = v
        return H, err
    
    def find_bestRotate(self, imgI: np.array, imgJ: np.array, dis, skip=1):
        midX, midY = he_img.img_center(imgJ)
        dis = int(dis * 360 * 2)
        if dis > 360:
            dis = 360
        if dis < 3:
            return None, 1
        if skip > 1:
            skip = 1
        
        def rotate(imgJ, v, imgJ_):
            imgJ_, _ = he_img.rotate_img(imgJ, v, (midX, midY), 1.0)
            return imgJ_        
        print("Search Rotate dis=%.1f skip=%.1f" % (dis, skip))
        v, err = self.find_best(imgI, imgJ, rotate, dis, skip)
        H, _ = he_img.rotate_H(v, (midX, midY), 1.0)
        return H, err
    
    def find_bestScale(self, imgI: np.array, imgJ: np.array, dis, skip=1):
        midX, midY = he_img.img_center(imgJ)
        dis = dis * 0.5
        skip = skip * 0.01
        if dis / skip < 3:
            return None, 1
        def scale(imgJ, v, imgJ_):
            imgJ_, _ = he_img.rotate_img(imgJ, 0, (midX, midY), v+1)
            return imgJ_
        print("Search Scale dis=%.3f skip=%.3f" % (dis, skip))
        v, err = self.find_best(imgI, imgJ, scale, dis, skip)
        H, _ = he_img.rotate_H(0, (midX, midY), v+1)
        return H, err
    