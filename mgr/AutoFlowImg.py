import numpy as np
import spacemap
import pandas as pd

class AutoFlowImg(spacemap.AffineFlowMgrImg):
    def __init__(self, imgI: np.array, imgJ: np.array, finder=None):
        super().__init__("AutoFlowImg", imgI, imgJ, finder)
        
    def run(self):
        # if self.center:
        #     rotate = spacemap.affine.AutoGradImg()
        #     rotate.showGrad = True
        #     self.run_flow(rotate)
        matches = spacemap.matches.MatchInit(matchr=self.matchr, method="loftr")
        # matches.alignment = spacemap.matches.LOFTR()

        self.run_flow(matches)
        if self.center:
            graph = spacemap.affine.FilterGraph(std=self.matchr_std)
            # graph.show_graph_match = True
            self.run_flow(graph)
            self.run_flow(spacemap.matches.MatchShow())
        # glob = spacemap.affine.FilterGlobal(count=self.glob_count)
        # self.run_flow(glob)
        # show = spacemap.matches.MatchShow()
        
        self.run_flow(spacemap.matches.MatchShow())
        
        each = spacemap.matches.MatchEachImg()
        self.run_flow(each)
        self.run_flow(spacemap.matches.MatchShow())
        # H = self.resultH()
        H = self.bestH()
        return H