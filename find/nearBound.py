import spacemap
import pandas as pd
import numpy as np
import tqdm
import cv2
from multiprocessing import Pool
import os



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
            self.dbs[index] = spacemap.find.CacheKVDB(self.dataFolder + "/%d" % index)
        db = self.dbs[index]
        key = "%d_%s" % (index, key)
        return spacemap.find.NearBoundCellData(db, key)
        
    def connect(self):
        for i in range(self.count):
            _ = self.get_data(i, "all")
            
    def auto_close(self):
        for db in self.dbs.values():
            db.close()
        self.dbs = {}
        
    def load_transform_raw(self, transformRaw):
        for i in range(self.count):
            spacemap.Info("Transform Raw Load: %d" % i) 
            df2 = transformRaw[transformRaw["layer"] == (i+self.cFrom)].copy()
            xy2 = df2[["x", "y"]].values
            df1 = self.cellDB[self.cellDB["layer"] == (i+self.cFrom)].copy()
            xy = df1[["x", "y"]].values
            cells = self.get_cell_ids(i)
            spacemap.Info("Save Raw: %d" % i)
            for celli, cell in enumerate(cells):
                data = self.get_data(i, cell)
                data.save("new_xy", xy2[celli], "np")
                data.save("raw_xy", xy[celli], "np")
            df2.to_csv(self.dataFolder + "/raw_%d.csv.gz" % i)
        self.auto_close()
    
    def load_transform_bound(self, transformBoundPath):
        for i in range(self.count):
            _ = self.get_data(i, "all")
            spacemap.Info("Transform Bound Load: %d" % i)
            boundDF = pd.read_csv(transformBoundPath % (self.cFrom + i))
            spacemap.Info("Parse Bound: %d" % i)
            edgeDF = self._parse_edge(boundDF)
            spacemap.Info("Save Bound: %d" % i)
            for key, group in edgeDF.items():
                data = self.get_data(i, key)
                data.save("bound", group, "np")
        self.auto_close()
    
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
        self.auto_close()
                
    def transform_raw(self):
        for i in range(self.count):
            spacemap.Info("Transform Raw: %d" % i) 
            df1 = self.cellDB[self.cellDB["layer"] == (i+self.cFrom)].copy()
            xy = df1[["x", "y"]].values
            xy2 = self.db.apply_point(xy, i, self.maxShape)
            df1["x"] = xy2[:, 0]
            df1["y"] = xy2[:, 1]
            cells = self.get_cell_ids(i)
            spacemap.Info("Save Raw: %d" % i)
            for celli, cell in enumerate(cells):
                data = self.get_data(i, cell)
                data.save("new_xy", xy2[celli], "np")
                data.save("raw_xy", xy[celli], "np")
            df1.to_csv(self.dataFolder + "/raw_%d.csv.gz" % i)
        self.auto_close()
    
    def compute_nearst(self):
        ps = []
        for i in range(self.count):
            pxy = pd.read_csv(self.dataFolder + "/raw_%d.csv.gz" % i)[["x", "y"]].values
            ps.append(pxy)
        
        for i in range(self.count-1):
            spacemap.Info("Compute Nearst: %d" % i)
            ps0 = ps[i]
            ps1 = ps[i+1]
            nearst = spacemap.find.NearPointErr(10)
            spacemap.Info("Init Nearst: %d" % i)
            nearst.init_db(ps1)
            spacemap.Info("Search Nearst: %d" % i)
            result = nearst.search(ps0, maxDis=7)
            
            cellids = self.get_cell_ids(i)
            cellids2 = self.get_cell_ids(i+1)
            spacemap.Info("Save Nearst: %d" % i)
            for j in range(len(cellids)):
                data = self.get_data(i, cellids[j])
                cell2Index = result[j][0]
                data.save("nearst_index", 
                          [cell2Index, cellids2[int(cell2Index)]], 
                          "lis")
        self.auto_close()
    
    def compute_bound_err(self, data1, 
                        data2):
        group1 = data1.load("bound", "np")
        group2 = data2.load("bound", "np")
        if group1 is None or group2 is None:
            return 0, None
        group1 = group1
        group2 = group2
        
        minx1, miny1 = group1.min(axis=0)
        minx2, miny2 = group2.min(axis=0)
        maxx1, maxy1 = group1.max(axis=0)
        maxx2, maxy2 = group2.max(axis=0)
        if not (minx1 < maxx2 and maxx1 > minx2 and miny1 < maxy2 and maxy1 > miny2):
            return 0, None
        
        groups = np.concatenate([group1, group2], axis=0)
        minXY = np.min(groups, axis=0)
        maxXY = np.max(groups, axis=0)
        shape = max(maxXY - minXY) + 5
        xyd = int(50 / shape)
        if xyd > 1:
            group1 = group1 * xyd
            group2 = group2 * xyd
            minXY = minXY * xyd
            shape = shape * xyd
        
        group1 = group1 - minXY
        group2 = group2 - minXY
        
        img1 = np.zeros((shape, shape, 3), dtype=np.uint8)
        img2 = np.zeros_like(img1)
        group1 = group1.astype(np.int32)
        group2 = group2.astype(np.int32)
    
        img1 = cv2.fillPoly(img1, [group1], (128, 128, 128))
        img2 = cv2.fillPoly(img2, [group2], (128, 128, 128))
        img12 = np.zeros_like(img1)
        img12[:, :, 0] = img1[:, :, 0]
        img12[:, :, 1] = img2[:, :, 1]
        e = spacemap.find.default()
        return -e.err(img1, img2), img12
    
    def compute_err(self):
        for i in range(self.count-1):
            spacemap.Info("Compute Err: %d" % i)
            rawCells = self.get_cell_ids(i)
            count = len(rawCells)
            result = np.zeros(count)
            
            for celli in range(count):
                rawCell = self.get_data(i, rawCells[celli])
                nearstIndex, nearCellID = rawCell.load("nearst_index", "lis")
                nearCell = self.get_data(i+1, nearCellID)
                e, _ = self.compute_bound_err(rawCell, nearCell)
                result[celli] = e
                rawCell.save("err", e, "num")
            path = self.dataFolder + "/err_%d.npy" % i
            np.save(path, result)
            spacemap.Info("Save Err: %d %f" % (i, result.mean()))
        self.auto_close()
        
    def export_err(self):
        err_raw = []
        err_mean = []
        for i in range(self.count-1):
            path = self.dataFolder + "/err_%d.npy" % i
            err = np.load(path)
            err_raw.append(err.tolist())
            err_mean.append(err.mean())
        return err_raw, np.array(err_mean)

    def draw_err(self, lis, size=1000):
        step = 1 / size
        img = np.zeros((size, size))
        if isinstance(lis[0], list):
            lis1 = []
            for l in lis:
                lis1 += l
            lis = lis1
        count = np.zeros(size)
        for i in lis:
            i = int(i / step)
            count[i] += 1
        all = 0
        allCount = len(lis)
        for i in range(size):
            v = count[size - i - 1]
            all += v
            v2 = int(all / allCount * size)
            img[:v2, i] = 1
        return img, np.mean(img)
            