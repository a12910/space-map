import pandas as pd
import numpy as np
import spacemap
import tqdm
import cv2
from PIL import Image
import os

class ModelGenerate:
    def __init__(self, baseFolder, cellDB, start, end):
        self.baseFolder = baseFolder
        self.start = start
        self.celllDB = cellDB # 对齐后的DB
        self.end = end
        self.gridShape = (400, 400)
        self.gridXYD = 10
        
        self.edgeCache = {}
        self.edgeKeyX = "vertex_x"
        self.edgeKeyY = "vertex_y"
        
        self.genValue = 128
        self.genCellSize = 10
        self.genRatio = 0.8
        self.genShape = (40000, 40000)
        self.genXYD = 10
        
        self.genStart = start
        self.genEnd = end
        self.genOutFolder = baseFolder + "/data"
        self.genSaveThub = False
        self.genPrefix = "d"
        
    def prepare_grid(self, dfRaw, dfAlign, force=False):
        folder = self.baseFolder + "/grid"
        spacemap.mkdir(folder)
        
        for i in range(self.start, self.end+1):
            path = "%s/grid_%d.npy" % (folder, i)
            if os.path.exists(path) and not force:
                continue
            spacemap.Info("Prepare grid %d" % i)
            raw = dfRaw[dfRaw["layer"] == i][["x", "y"]].values
            align = dfAlign[dfAlign["layer"] == i][["x", "y"]].values
            grid = spacemap.GridGenerate(self.grid_shape, 
                                         self.grid_xyd, 1)
            grid.init_db(raw, align)
            grid.generate()
            grid.fix()
            
            np.save(path, grid)
        spacemap.Info("Prepare grid done")

    def prepare_edge(self, edgeFolder, force=False):
        folder = self.baseFolder + "/edge"
        spacemap.mkdir(folder)
        
        for index in range(self.start, self.end+1):
            path = "%s/edge/align_edge_%d.csv" % (folder, index)
            if os.path.exists(path) and not force:
                continue
            spacemap.Info("Prepare edge %d" % index)
            df = pd.read_csv("%s/%d.csv.gz" % (edgeFolder, index))
            if index == self.start:
                df["x"] = df[self.edgeKeyX]
                df["y"] = df[self.edgeKeyY]
            else:
                gridPath = "%s/grid/grid_%.2d.npy" % (self.baseFolder, index)
                grid = np.load(gridPath)
                points = np.array(df[[self.edgeKeyX, self.edgeKeyY]].values)
                gen = spacemap.GridGenerate(self.gridShape, self.gridXYD)
                gen.grid = grid
                points2 = gen.grid_sample_points(points)
                df["x"] = points2[:, 0]
                df["y"] = points2[:, 1]
            df.to_csv(path, index=False)
        spacemap.Info("Prepare edge done")
        
    @staticmethod
    def parse_edge(edgesDF: pd.DataFrame):
        edges_group = edgesDF.groupby("cell_id")
        edges = {}
        for cid, group in edges_group:
            g = np.array(group[(group["x"] > 200) & (group["y"] > 200) & (group["x"] < 3800) & (group["y"] < 3800)][["x", "y"]].values)
            if g.shape[0] < 3:
                continue
            if np.sum(g.max(axis=0) - g.min(axis=0)) > 200:
                continue
            edges[cid] = g
        return edges
    
    def __get_layer_db(self, layer: int):
        if layer in self.edgeCache:
            return self.edgeCache[layer]
        path = "%s/edge/align_edge_%d.csv" % (self.baseFolder, layer)
        spacemap.Info("load layer %d" % layer)
        db = self.parse_edge(pd.read_csv(path))
        self.edgeCache[layer] = db
        return db
    
    def generate_cells(self, cellMapping: dict):
        db = self.celllDB.copy()
        db["channel"] = ""
        other = None
        for key, value in cellMapping.items():  
            prefix = "%s_%d" % (self.genPrefix, value)    
            if key == "ALL":
                spacemap.Info("Generate CellType-ALL")
                self.generate(prefix=prefix, cDB=db)
            elif key != "OTHER":
                db.loc[db["cell_type"] == key, "channel"] = str(value)
                spacemap.Info("Generate CellType-%s" % key)
                cDB = db[db["cell_type"] == key].copy()
                self.generate(prefix=prefix, cDB=cDB)
            else:
                other = value
        if other is not None:
            spacemap.Info("Generate CellType-OTHER")
            prefix = "%s_%d" % (self.genPrefix, other)    
            cDB = db[db["channel"] == ""].copy()
            self.generate(prefix="%s_OTHER" % self.genPrefix, cDB=cDB)
        
    def generate(self, prefix=None, cDB=None):
        if prefix is None:
            prefix = self.genPrefix
        outFolder = self.genOutFolder
        spacemap.mkdir(outFolder)
        if self.genSaveThub:
            thubFolder = outFolder + "/thub"
            spacemap.mkdir(thubFolder)
            
        if cDB is None:
            cDB = self.celllDB
        
        for layer in tqdm.trange(self.genStart, self.genEnd):
            spacemap.Info("Generate layer %d" % layer)
            edgeDB = self.__get_layer_db(layer)
            
            img = np.zeros((*self.genShape, 3), dtype=np.uint8)
            
            cellsDF = cDB[cDB["layer"] == layer].copy()
            cells = cellsDF[["cell_id", "x", "y"]].values.tolist()
            areas = []
            for cell in cells:
                cell_id = cell[0]
                if cell_id not in edgeDB:
                    continue
                edges = edgeDB[cell_id]
                
                center = np.array(cell[1:])
                edges = np.array(edges)
                edgeDis = np.sum(abs(edges - center), axis=1)
                if self.genCellSize is not None:
                    bigDis = edgeDis[edgeDis > self.genCellSize]
                    bigEdge = edges[edgeDis > self.genCellSize].copy()
                    bigEdge = center + (bigEdge - center) * self.genCellSize / bigDis[:, np.newaxis]
                    edges[edgeDis > self.genCellSize] = bigEdge
                
                if self.genRatio is not None:
                    edges = edges * self.genRatio + center * (1 - self.genRatio)
                edges = edges * self.genXYD
                edges = np.array(edges, dtype=np.int32)
                areas.append(edges)
            spacemap.Info("Generate layer filter %s %d %d->%d" % (prefix, layer, len(cells), len(areas)))
            
            img = cv2.fillPoly(img, areas, (self.genValue, self.genValue, self.genValue))
            image = Image.fromarray(img[:, :, 0])
            out_path = outFolder + "/d%s_%d.tiff" % (prefix, layer)
            image.save(out_path)
            spacemap.Info("Generate layer done %s %d" % (prefix, layer))
            if self.genSaveThub:
                thub_path = thubFolder + "/d%s_%d.jpg" % (prefix, layer)
                image.thumbnail((200, 200))
                image.save(thub_path)
            