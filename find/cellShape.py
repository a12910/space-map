import spacemap
import numpy as np
import pandas as pd
import tqdm
from .nearBound import NearBoundGenerate
from .nearBoundCellData import NearBoundCellData

class CellShapeGenerate(NearBoundGenerate):
    def __init__(self, db: spacemap.TransformDB, folder: str, 
                 cellDB: pd.DataFrame, cFrom: int, maxShape) -> None:
        """ cellDB: cell_id,x,y,layer """
        super().__init__(db, folder, cellDB, cFrom, maxShape)
    
    def _parse_edge(self, edgesDF: pd.DataFrame):
        edges_group = edgesDF.groupby("cell_id")
        edges = {}
        for cid, group in edges_group:
            if "x" not in group.columns:
                group["x"] = group["vertex_x"]
                group["y"] = group["vertex_y"]
            g = np.array(group[["x", "y", "x_align", "y_align"]].values, dtype=np.int32)
            if g.shape[0] < 3: continue
            if np.sum(g[:, 2:].max(axis=0) - g[:, 2:].min(axis=0)) > 200: continue
            edges[cid] = g
        return edges
    
    def transform_bound(self, boundPath):
        for i in range(self.count):
            _ = self.get_data(i, "all")
            spacemap.Info("Transform Bound: %d" % i)
            boundDF = pd.read_csv(boundPath % (self.cFrom + i))
            if not "x" in boundDF.columns:
                boundDF["x"] = boundDF["vertex_x"]
                boundDF["y"] = boundDF["vertex_y"]
            xy = boundDF[["x", "y"]].values
            xy2 = self.db.apply_point(xy, i, self.maxShape)
            boundDF["x_align"] = xy2[:, 0]
            boundDF["y_align"] = xy2[:, 1]
            spacemap.Info("Parse Bound: %d" % i)
            edgeDF = self._parse_edge(boundDF)
            spacemap.Info("Save Bound: %d" % i)
            for key, group in edgeDF.items():
                data = self.get_data(i, key)
                rawXY  = group[:, :2]
                alignXY = group[:, 2:]
                data.save("bound_raw", rawXY, "np")
                data.save("bound_align", alignXY, "np")
        self.auto_close()
        
    def compute_err(self):
        for i in range(self.count-1):
            spacemap.Info("Compute Err: %d" % i)
            rawCells = self.get_cell_ids(i)
            count = len(rawCells)
            result = np.zeros(count)
            for celli in range(count):
                data = self.get_data(i, rawCells[celli])
                e = self.compute_shape_err(data)
                result[celli] = e
                data.save("err", e, "num")
            path = self.dataFolder + "/err_%d.npy" % i
            np.save(path, result)
            spacemap.Info("Save Err: %d %f" % (i, result.mean()))
        self.auto_close()
    
    def compute_shape_err(self, data: NearBoundCellData):
        group1 = data.load("bound_raw", "np")
        group2 = data.load("bound_align", "np")
        if group1 is None or group2 is None:
            raise Exception()
            return 0
        minx1, miny1 = group1.min(axis=0)
        minx2, miny2 = group2.min(axis=0)
        maxx1, maxy1 = group1.max(axis=0)
        maxx2, maxy2 = group2.max(axis=0)
        sizex1, sizey1 = maxx1 - minx1, maxy1 - miny1
        sizex2, sizey2 = maxx2 - minx2, maxy2 - miny2
        ratio1 = max(sizex1, sizey1) / min(sizex1, sizey1)
        ratio2 = max(sizex2, sizey2) / min(sizex2, sizey2)
        area1 = sizex1 * sizey1
        area2 = sizex2 * sizey2
        value = min(area1, area2) / max(area1, area2)
        value *= min(ratio1, ratio2) / max(ratio1, ratio2)
        return value