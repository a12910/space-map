import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacemap

class AffineAlignment:
    def __init__(self):
        pass
    
    def compute(self, imgI, imgJ, matchr):
        return spacemap.matches.siftImageAlignment(imgI, imgJ, matchr=matchr)

class AffineBlock:
    def __init__(self, name):
        self.name = name
        self.matches = []
        self.computeRaw = False
        self.updateMatches = False
        self.alignment = AffineAlignment()
    
    def compute(self, dfI: np.array, dfJ: np.array, finder=None):
        """ dfI, dfJ -> H """
        raise Exception("Not Implemented")
    
    def compute_img(self, imgI: np.array, imgJ: np.array, finder=None):
        """ imgI, imgJ -> H """
        raise Exception("Not Implemented")
    
    @staticmethod
    def show_matches_img(matches, I, J, H):
        xyd = spacemap.XYD
        plt.figure(figsize=(10,10))
        plt.imshow(np.concatenate((I, J), axis=1))
        for m in matches:
            plt.plot([m[1], m[3]+I.shape[1]], [m[0], m[2]], 'r-', linewidth=1)
        plt.show()
        if H is not None:
            imgI_ = I.copy()
            for x, y in matches[:, :2]:
                x_, y_ = int(x), int(y)
                imgI_[x_-4:x_+4, y_-1:y_+1] = 20
            matchesJ = spacemap.applyH_np(matches[:, 2:4] * xyd, H)
            for x, y in matchesJ:
                x_, y_ = int(x / xyd), int(y / xyd)
                imgI_[x_-1:x_+1, y_-4:y_+4] = 10
            plt.imshow(imgI_)
            plt.show()
    
    @staticmethod
    def show_matches(matches, dfI, dfJ, H):
        I = spacemap.show_img3(dfI)
        J = spacemap.show_img3(dfJ)
        AffineBlock.show_matches_img(matches, I, J, H)

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
        
# def manual_flow(dfI, dfJ):
#     mgr = AffineFlowMgr("ManualFlow", dfI, dfJ)
#     if mgr.center:
#         rotate = spacemap.AffineBlockBestRotate()
#         mgr.run_flow(rotate)
#     matches = spacemap.MatchInit(matchr=mgr.matchr)
#     matches.alignment = spacemap.AffineAlignmentLOFTR()
#     mgr.run_flow(matches)
#     if mgr.center:
#         graph = spacemap.MatchFilterGraph(std=mgr.matchr_std)
#         # graph.show_graph_match = True
#         mgr.run_flow(graph)
#         mgr.run_flow(spacemap.MatchShow())
#     glob = spacemap.MatchFilterGlobal(count=mgr.glob_count)
#     mgr.run_flow(glob)
#     mgr.run_flow(spacemap.MatchShow())
    
#     each = spacemap.MatchEach()
#     mgr.run_flow(each)
#     mgr.run_flow(spacemap.MatchShow())
#     # H = self.resultH()
#     H = mgr.bestH()
#     return H
