import numpy as np
import cv2
import spacemap

class AffineBlockRotate(spacemap.AffineBlock):
    def __init__(self, name=None, rotate=0, moveX=0, moveY=0) -> None:
        if name is None:
            name = "Rotate"
        super().__init__(name)
        self.rotate = rotate
        self.moveX = moveX
        self.moveY = moveY
        
    def compute_center(self, dfI, dfJ):
        xyrange = spacemap.XYRANGE
        
        meanX = xyrange[1] // 2
        meanY = xyrange[3] // 2
        
        meanIX, meanIY = np.mean(dfI[:, 0]), np.mean(dfI[:, 1])
        meanJX, meanJY = np.mean(dfJ[:, 0]), np.mean(dfJ[:, 1])
        dfI_ = dfI.copy()
        dfJ_ = dfJ.copy()
        dfI_[:, 0] += meanX - meanIX
        dfI_[:, 1] += meanY - meanIY
        dfJ_[:, 0] += meanX - meanJX
        dfJ_[:, 1] += meanY - meanJY
        imgI1 = spacemap.show_img3(dfI_)
        imgJ1 = spacemap.show_img3(dfJ_)
        
        H11 = np.array([[1, 0, -meanJX], [0, 1, -meanJY], [0, 0, 1]])
        H13 = np.array([[1, 0, meanIX], [0, 1, meanIY], [0, 0, 1]])        
        return H11, H13, imgI1, imgJ1
    
    def compute_center_img(self, dfI, dfJ):
        xyrange = spacemap.XYRANGE
        
        meanX = xyrange[1] // 2
        meanY = xyrange[3] // 2
        
        meanIX, meanIY = np.mean(dfI[:, 0]), np.mean(dfI[:, 1])
        meanJX, meanJY = np.mean(dfJ[:, 0]), np.mean(dfJ[:, 1])
        dfI_ = dfI.copy()
        dfJ_ = dfJ.copy()
        dfI_[:, 0] += meanX - meanIX
        dfI_[:, 1] += meanY - meanIY
        dfJ_[:, 0] += meanX - meanJX
        dfJ_[:, 1] += meanY - meanJY
        imgI1 = spacemap.show_img3(dfI_)
        imgJ1 = spacemap.show_img3(dfJ_)
        
        H11 = np.array([[1, 0, -meanJX], [0, 1, -meanJY], [0, 0, 1]])
        H13 = np.array([[1, 0, meanIX], [0, 1, meanIY], [0, 0, 1]])        
        return H11, H13, imgI1, imgJ1
        
    def compute(self, dfI: np.array, dfJ: np.array, finder=None):
        meanIX, meanIY = np.mean(dfI[:, 0]), np.mean(dfI[:, 1])
        meanJX, meanJY = np.mean(dfJ[:, 0]), np.mean(dfJ[:, 1])
        r = self.rotate / 360 * np.pi * 2
        cosr = np.cos(r)
        sinr = np.sin(r)
        H11 = np.array([[1, 0, -meanJX], [0, 1, -meanJY], [0, 0, 1]])
        H12 = np.array([[cosr, -sinr, 0], [sinr, cosr, 0], [0, 0, 1]])
        H13 = np.array([[1, 0, meanIX + self.moveX], [0, 1, meanIY + self.moveY], [0, 0, 1]])        
        H1 = np.dot(H13, np.dot(H12, H11))
        return H1
    
    @staticmethod
    def rotate_img(imgJ, rotate):
        h, w = imgJ.shape[:2]
        center = h // 2, w // 2
        M = cv2.getRotationMatrix2D(center, rotate, 1.0)
        rotatedI = cv2.warpAffine(imgJ, M, (w, h))
        return rotatedI
    
class AffineBlockBestRotate(AffineBlockRotate):
    def __init__(self, step1=5, step2=0.2) -> None:
        super().__init__("BestRotate")
        self.step1 = step1
        self.step2 = step2
        
    def compute(self, dfI: np.array, dfJ: np.array, finder=None):
        H11, H13, imgI1, imgJ1 = self.compute_center(dfI, dfJ)
        rotate = self.find_rotate(imgI1, imgJ1, finder)
        spacemap.Info("Best Rotate Find: %s" % str(rotate))
        self.rotate = rotate
        r = rotate / 360 * np.pi * 2
        cosr = np.cos(r)
        sinr = np.sin(r)
        H12 = np.array([[cosr, -sinr, 0], [sinr, cosr, 0], [0, 0, 1]])
        H1 = np.dot(H13, np.dot(H12, H11))
        return H1
        
    def find_rotate(self, imgI, imgJ, finder: spacemap.AffineFinder):
        w, h = imgI.shape[:2]
        imgI_ = np.zeros((w * 2, h * 2))
        imgJ_ = np.zeros((w * 2, h * 2))
        imgI_[int(w // 2): int(w // 2 + w), int(h // 2): int(h // 2 + h)] = imgI
        imgJ_[int(w // 2): int(w // 2 + w), int(h // 2): int(h // 2 + h)] = imgJ
        for rotate in range(0, 360, self.step1):
            imgJ1 = AffineBlockRotate.rotate_img(imgJ_, rotate)
            finder.add_result(rotate, None, imgI_, imgJ1)
        min_rotate = finder.best()[0]
        finder.clear()
        for rotate_ in range(int((min_rotate - self.step1) // self.step2), int((min_rotate + self.step1)//self.step2) + 1):
            rotate = rotate_ * self.step2
            imgJ1 = AffineBlockRotate.rotate_img(imgJ_, rotate)
            finder.add_result(rotate, None, imgI_, imgJ1)
        return finder.best()[0] # rotate
    
