import spacemap
import pandas as pd
import numpy as np
import tqdm
import cv2

class NearBoundGenerate:
    def __init__(self, db: spacemap.TransformDB, folder: str, 
                 cellDB: pd.DataFrame, cFrom: int, maxShape) -> None:
        """ cellDB: cell_id,x,y,layer """
        self.db = db
        self.count = db.count+1
        self.folder = folder
        self.cellDB = cellDB
        self.cFrom = cFrom
        self.maxShape = maxShape
        self.dataFolder = folder + "/data"
        self.dbs = {}
        spacemap.mkdir(folder)
        spacemap.mkdir(self.dataFolder)
        
    def get_data(self, index, key):
        if not index in self.dbs:
            self.dbs[index] = spacemap.find.SimpleKVDB(self.dataFolder + "/%d.db" % index)
            spacemap.Info("Init DB: %d" % (index))
        db = self.dbs[index]
        key = "%d_%s" % (index, key)
        return spacemap.find.NearBoundCellData(db, key)
    
    def get_cell_ids(self, index):
        df = self.cellDB[self.cellDB["layer"] == (index + self.cFrom)]
        return df["cell_id"].tolist()
    
    def _parse_edge(self, edgesDF: pd.DataFrame):
        edges_group = edgesDF.groupby("cell_id")
        edges = {}
        for cid, group in edges_group:
            g = np.array(group[["x", "y"]].values, dtype=np.int32)
            if g.shape[0] < 3: continue
            if np.sum(g.max(axis=0) - g.min(axis=0)) > 200: continue
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
            boundDF["x"] = xy2[:, 0]
            boundDF["y"] = xy2[:, 1]
            spacemap.Info("Parse Bound: %d" % i)
            edgeDF = self._parse_edge(boundDF)
            spacemap.Info("Save Bound: %d" % i)
            for key, group in edgeDF.items():
                data = self.get_data(i, key)
                data.save("bound", group, "np")
                
    def transform_raw(self):
        for i in range(self.count):
            spacemap.Info("Transform Raw: %d" % i) 
            df1 = self.cellDB[self.cellDB["layer"] == (i+self.cFrom)].copy()
            xy = df1[["x", "y"]].values
            xy2 = self.db.apply_point(xy, i, self.maxShape)
            df1["x"] = xy2[:, 0]
            df1["y"] = xy2[:, 1]
            df1.to_csv(self.folder + "/raw_%d.csv.gz" % i)
    
    def compute_nearst(self):
        boundFolder = self.folder + "/bound"
        spacemap.mkdir(boundFolder)
        ps = []
        for i in range(self.count):
            pxy = pd.read_csv(boundFolder + "/raw_%d.csv.gz" % i)[["x", "y"]].values
            ps.append(pxy)
        
        for i in range(self.count-1):
            spacemap.Info("Compute Nearst: %d" % i)
            ps0 = ps[i]
            ps1 = ps[i+1]
            nearst = spacemap.find.NearPointErr(10)
            spacemap.Info("Init Nearst: %d" % i)
            nearst.init_db(ps0)
            spacemap.Info("Search Nearst: %d" % i)
            result = nearst.search(ps1, maxDis=7)
            cellids = self.get_cell_ids(i)
            cellids2 = self.get_cell_ids(i+1)
            spacemap.Info("Save Nearst: %d" % i)
            for j in range(len(cellids)):
                data = self.get_data(i, cellids[j])
                cell2Index = result[j][0]
                data.save("nearst_index", [cell2Index, cellids2[int(cell2Index)]], "lis")
    
    def compute_bound_err(self, data1, 
                        data2, xyd=2):
        group1 = data1.load("bound", "np")
        group2 = data2.load("bound", "np")
        # link two groups
        groups = np.concatenate([group1, group2], axis=0)
        minXY = np.min(groups, axis=0)
        maxXY = np.max(groups, axis=0)
        group1 = group1 - minXY
        group2 = group2 - minXY
        img1 = np.zeros((maxXY[0]-minXY[0], maxXY[1]-minXY[1], 3), dtype=np.uint8)
        img2 = np.zeros_like(img1)
        group1 = group1.astype(np.int32)
        group2 = group2.astype(np.int32)
        img1 = cv2.fillPoly(img1, [group1], (128, 128, 128))
        img2 = cv2.fillPoly(img2, [group2], (128, 128, 128))
        e = spacemap.find.default()
        return -e.err(img1, img2)
    
    def compute_err(self):
        for i in range(self.count-1):
            spacemap.Info("Compute Err: %d" % i)
            rawCells = self.get_cell_ids(i)
            count = len(rawCells)
            result = np.zeros(count)
            
            for celli in tqdm.trange(count):
                rawCell = self.get_data(i, rawCells[celli])
                nearstIndex, nearCellID = rawCell.load("nearst_index", "lis")
                nearCell = self.get_data(i+1, nearCellID)
                e = self.compute_bound_err(rawCell, nearCell)
                result[celli] = e
                rawCell.save("err", e, "num")
            path = self.dataFolder + "/err_%d.npy" % i
            np.save(path, result)
            print(i, result.mean())
        