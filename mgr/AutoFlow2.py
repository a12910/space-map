import numpy as np
import spacemap
import pandas as pd

class SpaceMapAutoFlow2(spacemap.AffineFlowMgr):
    def __init__(self, dfI: np.array, dfJ: np.array, center=True):
        finder = spacemap.AffineFinderBasic("dice")
        super().__init__("SpaceMapAutoFlow", dfI, dfJ, finder)
        self.matchr = 200
        self.center = center
        
    def run(self):
        if self.center:
            rotate = spacemap.AffineBlockBestRotate()
            self.run_flow(rotate)
        matches = spacemap.MatchInit(matchr=200)
        xyd = spacemap.XYD
        spacemap.XYD = int(xyd // 2)
        matches.alignment = spacemap.AffineAlignmentLOFTR2()
        self.run_flow(matches)
        self.matches = self.matches / 2
        spacemap.XYD = xyd
        if self.center:
            graph = spacemap.MatchFilterGraph(std=1.5)
            # graph.show_graph_match = True
            self.run_flow(graph)
            self.run_flow(spacemap.MatchShow())
        if len(self.matches) > 200:
            dis = spacemap.XYRANGE[1] // 40
            glob = spacemap.MatchFilterGlobal(count=200, dis=int(dis))
            self.run_flow(glob)
            self.run_flow(spacemap.MatchShow())
        
        each = spacemap.MatchEach()
        self.run_flow(each)
        self.run_flow(spacemap.MatchShow())
        # H = self.resultH()
        H = self.bestH()
        return H

class SpaceMapAutoFlowLabel(spacemap.AffineFlowMgr):
    def __init__(self, dfI: np.array, dfJ: np.array, 
                 lI: pd.DataFrame=None, lJ: pd.DataFrame=None, center=True):
        finder = spacemap.AffineFinderBasic("dice")
        super().__init__("SpaceMapAutoFlow", dfI, dfJ, finder)
        self.matchr = 200
        self.center = center
        self.lI = lI
        self.lJ = lJ
        
    def run(self):
        if self.center:
            rotate = spacemap.AffineBlockBestRotate(step2=1)
            self.run_flow(rotate)
        matches = spacemap.MatchInit(matchr=200)
        matches.alignment = spacemap.AffineAlignmentLOFTR()
        self.run_flow(matches)
        graph = spacemap.MatchFilterGraph(std=2)
        # graph.show_graph_match = True
        self.run_flow(graph)
        self.run_flow(spacemap.MatchShow())
        
        if self.lI is not None:
            labels = spacemap.MatchFilterLabels(self.lI, self.lJ, 2)
            self.run_flow(labels)
            self.run_flow(spacemap.MatchShow())
            
        dis = spacemap.XYRANGE[1] // 20
        glob = spacemap.MatchFilterGlobal(count=200, dis=int(dis))
        self.run_flow(glob)
        self.run_flow(spacemap.MatchShow())
        
        each = spacemap.MatchEach()
        self.run_flow(each)
        self.run_flow(spacemap.MatchShow())
        # H = self.resultH()
        H = self.bestH()
        return H
    