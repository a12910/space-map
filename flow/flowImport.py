import numpy as np
import spacemap
import json, os
import pandas as pd

class FlowImport:
    def __init__(self, basePath: str) -> None:
        self.slices = []
        self.basePath = basePath
        spacemap.init_path(basePath)
        self.auto_init()
        
    def init_xys(self, xys: list[np.array], ids=None) -> None:
        size = 0
        for i, s in enumerate(xys):
            sX, sY = s[:, 0], s[:, 1]
            sizeS = max(np.max(sX) - np.min(sX), np.max(sY) - np.min(sY))
            size = max(size, sizeS)
        xys2 = []
        targetSize = (int(size * 1.2 / 1000)) * 1000
        for i, s in enumerate(xys):
            # mid = np.mean(s, axis=0)
            mid = np.max(s, axis=0) / 2 + np.min(s, axis=0) / 2
            xys2.append(targetSize // 2 + s - mid)
        spacemap.XYRANGE = targetSize
        xyd = int(targetSize / 400 / 5) * 5 + 5
        spacemap.XYD = xyd
        spacemap.Info(f"XYRANGE: {spacemap.XYRANGE}, XYD: {spacemap.XYD}")
        if ids is None:
            ids = [str(i) for i in range(len(xys))]
        
        self.slices = []
        for i, s in enumerate(xys2):
            slice = spacemap.Slice2(ids[i], self.basePath, i==0)
            df = pd.DataFrame(s, columns=["x", "y"])
            slice.init_df(df)
            slice.save_config()
            self.slices.append(slice)
        conf = {
            "XYD": spacemap.XYD,
            "XYRANGE": spacemap.XYRANGE,
            "ids": ids
        }
        with open(self.basePath + "/conf.json", "w") as f:
            json.dump(conf, f)
        return self.slices
    
        
    def auto_init(self):
        path = self.basePath + "/conf.json"
        if os.path.exists(path):
            conf = json.load(open(path))
            spacemap.XYRANGE = conf["XYRANGE"]
            spacemap.XYD = conf["XYD"]
            spacemap.Info(f"Auto Init: XYRANGE: {spacemap.XYRANGE}, XYD: {spacemap.XYD}")
            ids = conf["ids"]   
            self.slices = []
            for i, idd in enumerate(ids):
                slice = spacemap.Slice2(idd, self.basePath, i==0)
                self.slices.append(slice)
            return self.slices
        return []
    
    def init_from_codex(self, csvPath):
        df = pd.read_csv(csvPath)
        groups = df.groupby("array")
        pack = {}
        for g, dff in groups:
            xy = dff[["x", "y"]].values
            pack[g] = xy
        
        keys = list(pack.keys())
        keys.sort()
        keys2 = [(k, int(k[1:])) for k in keys]
        keys2.sort(key=lambda x: x[1])
        # print(keys2)
        xys = [pack[k[0]] for k in keys2]
        keys = [k[0] for k in keys2]
        # conf = {}
        # for i in range(len(xys)):
        #     conf[i] = keys[i]
        # with open(self.basePath + "/codex.json", "w") as f:
        #     json.dump(conf, f)
        spacemap.Info(f"Init from codex: {keys}")
        return self.init_xys(xys, keys)