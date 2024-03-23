import spacemap
import numpy as np
import cv2

def err_conv_edge(imgI, imgJ, kernel, maxx=0.7):
    err = spacemap.err_dice
    xx, yy = imgI.shape
    errI = np.zeros((xx, yy))
    count = 0
    
    edgeI = np.array(imgI[:, :, np.newaxis], dtype=np.uint8)
    edgeI[edgeI > 0] = 128
    edgeI = cv2.medianBlur(edgeI, 9)
    edge = cv2.Canny(edgeI, 50, 150)
    for x in range(kernel, xx-kernel):
        for y in range(kernel, yy-kernel):
            partI = imgI[x-kernel:x+kernel+1, y-kernel:y+kernel+1]
            partJ = imgJ[x-kernel:x+kernel+1, y-kernel:y+kernel+1]
            edgeI = edge[x-kernel:x+kernel+1, y-kernel:y+kernel+1]
            # if np.sum(edgeI) < 3:
            #     continue
            if np.sum(partI[0]) < 5 and np.sum(partJ[0]) < 5:
                continue
            exp = np.mean(edgeI) / 128
            e = exp
            # e = 1
            count += e
            e2 = err(partI, partJ)
            if e2 > maxx:
                e2 = maxx
            e2 = e2 / maxx
            errI[x, y] = e2 * e
            # errI[x, y] = exp
    return np.sum(errI) / count, errI

def err_conv_create_edge_weight(imgI, mid, kernel):
    if kernel < 1:
        kernel = int(imgI.shape[0] * kernel)
    
    kernel = int(kernel * 2)
    meanKernel = np.ones((kernel, kernel)) / (kernel**2)
    edgeI = np.array(imgI[:, :, np.newaxis], dtype=np.uint8)
    edgeI[edgeI > 0] = 128
    edgeI = cv2.medianBlur(edgeI, mid)
    edge = cv2.Canny(edgeI, 50, 150)
    edgeMean = cv2.filter2D(edge, -1, meanKernel) / 128
    return edgeMean
        
def err_conv_edge2(imgI, imgJ, kernel, maxx=0.7, mid=9, edge=None):
    if kernel < 1:
        kernel = int(imgI.shape[0] * kernel)
    
    kernel = int(kernel * 2)
    meanKernel = np.ones((kernel, kernel)) / (kernel**2)
    sumKernel = np.ones((kernel, kernel))
        
    if edge is None or edge.shape != imgI.shape:
        edgeI = np.array(imgI[:, :, np.newaxis], dtype=np.uint8)
        edgeI[edgeI > 0] = 128
        edgeI = cv2.medianBlur(edgeI, mid)
        edge = cv2.Canny(edgeI, 50, 150)
        edgeMean = cv2.filter2D(edge, -1, meanKernel) / 128
    else:
        edgeMean = edge
        
    diceIJ = np.array([imgI, imgJ]).min(axis=0)
    
    sumDiceIJ = cv2.filter2D(diceIJ, -1, sumKernel)
    blank = sumDiceIJ < 5
    sumI = cv2.filter2D(imgI, -1, sumKernel)
    sumJ = cv2.filter2D(imgJ, -1, sumKernel)
    sumIJ = sumI + sumJ
    sumIJ[sumIJ < 0.01] = 0.01
    
    diceV = (sumDiceIJ / sumIJ) * 2
    diceV[np.isnan(diceV)] = 0
    diceV[np.isinf(diceV)] = 0
    diceV[diceV > maxx] = maxx
    diceV = diceV / maxx
    diceV[blank] = 0
    edgeMean[blank] = 0
    diceV = diceV * edgeMean
    
    result = np.sum(diceV) / np.sum(edgeMean)
    return result, diceV

def err_edge_dice(imgI, imgJ, kernel, maxx=0.7):
    edgeI = np.array(imgI[:, :, np.newaxis], dtype=np.uint8)
    edgeI[edgeI > 0] = 128
    edgeI = cv2.medianBlur(edgeI, 9)
    edge = cv2.Canny(edgeI, 50, 150)
    edge[edge > 0] = 1
    ckernel = np.ones((kernel * 2 + 1, kernel * 2 + 1), np.uint8)
    # 对edge进行二维卷积平均
    edge = cv2.filter2D(edge, -1, ckernel)
    edge = np.array(edge, dtype=np.float32)
    # edge = edge / np.sum(ckernel)
    
    ii = imgI.copy().reshape(-1)
    ij = imgJ.copy().reshape(-1)
    ii[ii > 0] = 1
    ij[ij > 0] = 1
    ee = edge.reshape(-1)
    
    ii1 = ii * ee
    ii1[ii1 > maxx] = maxx
    ii1 = ii1 / maxx
    
    ij1 = ij * ee
    ij1[ij1 > maxx] = maxx
    ij1 = ij1 / maxx

    iii = np.array([ii1, ij1])
    inter = iii.min(axis=0).sum()
    return (2 * inter + 0.001) / (ii.sum() + ij.sum() + 0.001), edge

class AffineFinderEdgeDice(spacemap.AffineFinder):
    def __init__(self, mid=9, kernel=10):
        super().__init__("AffineFinderEdgeDice")
        self.mid = mid
        self.kernel = kernel
        self.edge = None
        
    def err(self, imgI, imgJ):
        if self.edge is None:
            self.edge = err_conv_create_edge_weight(imgI, self.mid, self.kernel)
        e, _ = err_conv_edge2(imgI, imgJ, self.kernel, edge=self.edge)
        return -e
    
    def clear(self):
        self.edge = None
        return super().clear()