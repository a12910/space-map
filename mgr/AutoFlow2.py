import numpy as np
import spacemap
import pandas as pd

class SpaceMapAutoFlow2(spacemap.AffineFlowMgr):
    def __init__(self, dfI: np.array, dfJ: np.array, center=True):
        finder = spacemap.find.default()
        super().__init__("SpaceMapAutoFlow", dfI, dfJ, finder)        
        self.center = center

        
    def run(self):
        if self.center:
            rotate = spacemap.affine.BestRotate()
            self.run_flow(rotate)
        matches = spacemap.matches.MatchInit(matchr=self.matchr)
        xyd = spacemap.XYD
        spacemap.XYD = int(xyd // 2)
        matches.alignment = spacemap.matches.LOFTR2()
        self.run_flow(matches)
        self.matches = self.matches / 2
        spacemap.XYD = xyd
        if self.center:
            graph = spacemap.affine.FilterGraph(std=self.matchr_std)
            # graph.show_graph_match = True
            self.run_flow(graph)
            self.run_flow(spacemap.matches.MatchShow())

        glob = spacemap.affine.FilterGlobal(count=self.glob_count)
        self.run_flow(glob)
        self.run_flow(spacemap.matches.MatchShow())
        
        each = spacemap.matches.MatchEach()
        self.run_flow(each)
        self.run_flow(spacemap.matches.MatchShow())
        # H = self.resultH()
        H = self.bestH()
        return H

class SpaceMapAutoFlowLabel(spacemap.AffineFlowMgr):
    def __init__(self, dfI: np.array, dfJ: np.array, 
                 lI: pd.DataFrame=None, lJ: pd.DataFrame=None, center=True):
        finder = spacemap.find.default()
        super().__init__("SpaceMapAutoFlow", dfI, dfJ, finder)
        self.matchr = 200
        self.center = center
        self.lI = lI
        self.lJ = lJ
        
    def run(self):
        if self.center:
            rotate = spacemap.affine.BestRotate(step2=1)
            self.run_flow(rotate)
        matches = spacemap.matches.MatchInit(matchr=200)
        matches.alignment = spacemap.matches.LOFTR()
        self.run_flow(matches)
        graph = spacemap.affine.FilterGraph(std=2)
        # graph.show_graph_match = True
        self.run_flow(graph)
        self.run_flow(spacemap.matches.MatchShow())
        
        if self.lI is not None:
            labels = spacemap.affine.FilterLabels(self.lI, self.lJ, 2)
            self.run_flow(labels)
            self.run_flow(spacemap.matches.MatchShow())
            
        dis = spacemap.XYRANGE[1] // 20
        glob = spacemap.matches.FilterGlobal(count=200, dis=int(dis))
        self.run_flow(glob)
        self.run_flow(spacemap.matches.MatchShow())
        
        each = spacemap.matches.MatchEach()
        self.run_flow(each)
        self.run_flow(spacemap.matches.MatchShow())
        # H = self.resultH()
        H = self.bestH()
        return H
    