import spacemap
import numpy as np


class AffineFlowMgrImg(spacemap.AffineFlowMgrBase):
    def __init__(self, name, 
                 imgI: np.array, imgJ: np.array, finder=None):
        super().__init__(name, finder)
        self.name = name
        self.__imgI = imgI
        self.__imgJ = imgJ
        self.imgI = imgI.copy()
        self.imgJ = imgJ.copy()
        
    def find_matchr(self, minCount=75):
        return spacemap.matches.autoSetMatchr(self.__imgI, self.__imgJ, minCount)

    def run_flow(self, flow: spacemap.AffineBlock, showErr=True):
        self.affineFinder.clear()
        if flow.updateMatches:
            flow.matches = np.array(self.matches)
        if flow.updateImg:
            flow.imgIJ = (self.imgI, self.imgJ)
        self.affineFinder.clear()
        H = flow.compute_img(self.imgI, self.imgJ, self.affineFinder)
        if flow.updateMatches:
            self.matches = np.array(flow.matches)
        if flow.updateImg:
            self.imgI, self.imgJ = flow.imgIJ
        if H is not None:
            # H_ = self.resultH()
            # H_ = spacemap.he_img.multiply_HH([H_, H])
            # self.imgJ = spacemap.he_img.rotate_imgH(self.__imgJ, H_)
            self.imgJ = spacemap.he_img.rotate_imgH(self.imgJ, H)
            err = self.current_err(show=showErr)
            self.AffineH.append((H, err, flow.name))
        return self.imgI, self.imgJ
    
    def resultH_np(self):
        H = self.resultH()
        xyd = spacemap.XYD
        H[0, 2] *= xyd
        H[1, 2] *= xyd
        return H
    
    def current_err(self, show=True):
        return self.affineFinder.computeI(self.imgI, self.imgJ, show=show)
    
    @staticmethod
    def export_w(genH: np.array, rawImg: np.array, sampleShape: np.array):
        H = genH
        he01 = np.flip(rawImg, axis=1)
        size = max(rawImg.shape)
        if len(he01.shape) == 2:
            he01_ = np.zeros((size, size))
        else:
            he01_ = np.zeros((size, size, he01.shape[2]))
        he01_[:he01.shape[0], :he01.shape[1]] = he01

        scale2 = size / max(sampleShape)
        
        H0 = np.eye(3)
        H0[0, 0] = 1 / scale2
        H0[1, 1] = 1 / scale2
        
        H2 = np.eye(3)
        H2[0, 0] = scale2
        H2[1, 1] = scale2
        
        H = np.dot(H2, np.dot(H, H0))
        
        he02 = spacemap.he_img.rotate_imgH(he01_, H)
        he02 = he02.astype(np.uint8)
        return he02, H

    def export(genH: np.array, rawImg: np.array, 
                sampleImg: np.array):
        sampleShape = sampleImg.shape[:2]
        return AffineFlowMgrImg.export_w(genH, rawImg, sampleShape)
    
    def generateH(self, initH: np.array, mgrH: np.array, rawImg: np.array):
        M0 = spacemap.he_img.convert_H_to_M(initH)
        MM = spacemap.he_img.convert_H_to_M(mgrH)
        M0H = np.eye(3)
        M0H[:2] = M0
        MMH = np.eye(3)
        MMH[:2] = MM

        finalM = np.matmul(MMH, M0H)
        H = spacemap.he_img.convert_M_to_H(finalM[:2])
        
        scale = max(rawImg.shape[:2]) / max(self.imgI.shape[:2])
        H *= scale
        H[2, 2] = 1
        return H
        