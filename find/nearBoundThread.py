import spacemap
import pandas as pd
import numpy as np
import tqdm
import cv2
from multiprocessing import Pool
from .nearBound import NearBoundGenerate
import os

class NearBoundGenerateThread(NearBoundGenerate):
    def __init__(self, db: spacemap.TransformDB, folder: str, 
                 cellDB: pd.DataFrame, cFrom: int, maxShape) -> None:
        super().__init__(db, folder, cellDB, cFrom, maxShape)
    
    def get_thread_datas(self, count):
        return [[self, i] for i in range(count)]
    
    @staticmethod
    def _transform_bound(data):
        self, i, boundPath = data
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
        spacemap.Info("Transform Bound Finish: %d" % i)
        return ""
        
    @staticmethod
    def _transform_raw(data):
        spacemap.Info("Transform Raw: %d" % data[1]) 
        df1 = data[0].cellDB[data[0].cellDB["layer"] == (data[1]+data[0].cFrom)].copy()
        xy = df1[["x", "y"]].values
        xy2 = data[0].db.apply_point(xy, data[1], data[0].maxShape)
        df1["x"] = xy2[:, 0]
        df1["y"] = xy2[:, 1]
        cells = data[0].get_cell_ids(data[1])
        spacemap.Info("Save Raw: %d" % data[1])
        for celli, cell in enumerate(cells):
            data1 = data[0].get_data(data[1], cell)
            data1.save("new_xy", xy2[celli], "np")
            data1.save("raw_xy", xy[celli], "np")
        df1.to_csv(data[0].dataFolder + "/raw_%d.csv.gz" % data[1])
        data[0].auto_close()
        spacemap.Info("Transform Raw Finish: %d" % data[1])
        return ""

    def transform_bound(self, boundPath):
        spacemap.Info("Transform Bound")
        datas = [[self, i, boundPath] for i in range(self.count)]
        with Pool(os.cpu_count()) as p:
            p.map(NearBoundGenerateThread._transform_bound, datas)
                
    def transform_raw(self):
        spacemap.Info("Transform Raw")
        datas = self.get_thread_datas(self.count)
        with Pool(os.cpu_count()) as p:
            p.map(NearBoundGenerateThread._transform_raw, datas)
    
    @staticmethod
    def _compute_err(data):
        index = data[1]
        spacemap.Info("Compute Err: %d" % index)
        rawCells = data[0].get_cell_ids(index)
        count = len(rawCells)
        result = np.zeros(count)
        for celli in range(count):
            rawCell = data[0].get_data(index, rawCells[celli])
            nearstIndex, nearCellID = rawCell.load("nearst_index", "lis")
            nearCell = data[0].get_data(index+1, nearCellID)
            e, _ = data[0].compute_bound_err(rawCell, nearCell)
            result[celli] = e
            rawCell.save("err", e, "num")
        path = data[0].dataFolder + "/err_%d.npy" % index
        np.save(path, result)
        spacemap.Info("Save Err: %d %f" % (index, result.mean()))
        data[0].auto_close()
        spacemap.Info("Compute Err Finish: %d" % index)
        return ""
    
    def compute_err(self):
        datas = self.get_thread_datas(self.count-1)
        with Pool(os.cpu_count()) as p:
            p.map(NearBoundGenerateThread._compute_err, datas)
    