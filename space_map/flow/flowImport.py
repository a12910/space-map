from __future__ import annotations
import numpy as np
import space_map
import json, os
import pandas as pd

class FlowImport:
    def __init__(self, basePath: str) -> None:
        self.slices = []
        self.basePath = basePath
        space_map.init_path(basePath)
        self.ratio = 1.4
        self.auto_init()
        
    def init_xys(self, xys: list[np.array], ids=None) -> None:
        size = 0
        for i, s in enumerate(xys):
            sX, sY = s[:, 0], s[:, 1]
            sizeS = max(np.max(sX) - np.min(sX), np.max(sY) - np.min(sY))
            size = max(size, sizeS)
        xys2 = []
        targetSize = (int(size * self.ratio / 1000)) * 1000
        for i, s in enumerate(xys):
            # mid = np.mean(s, axis=0)
            mid = np.max(s, axis=0) / 2 + np.min(s, axis=0) / 2
            xys2.append(targetSize // 2 + s - mid)
        space_map.XYRANGE = targetSize
        xyd = int(targetSize / 400 / 5) * 5 + 5
        space_map.XYD = xyd
        space_map.Info(f"XYRANGE: {space_map.XYRANGE}, XYD: {space_map.XYD}")
        if ids is None:
            ids = [str(i) for i in range(len(xys))]
        
        self.slices = []
        dfs = []
        for i, s in enumerate(xys2):
            idd = ids[i]
            space_map.Info(f"Init Slice {idd} {s.shape}")
            slice = space_map.Slice(ids[i], self.basePath, i==0)
            df = pd.DataFrame(s, columns=["x", "y"])
            df["layer"] = i
            slice.init_df(df)
            dfs.append(df)
            slice.save_config()
            self.slices.append(slice)
        conf = {
            "XYD": space_map.XYD,
            "XYRANGE": space_map.XYRANGE,
            "ids": ids
        }
        dfs = pd.concat(dfs, axis=0, ignore_index=True)
        with open(self.basePath + "/conf.json", "w") as f:
            json.dump(conf, f)
        dfs.to_csv(self.basePath + "/raw/cells.csv.gz", index=False)
        return self.slices
    
        
    def auto_init(self):
        path = self.basePath + "/conf.json"
        if os.path.exists(path):
            conf = json.load(open(path))
            space_map.XYRANGE = conf["XYRANGE"]
            space_map.XYD = conf["XYD"]
            space_map.Info(f"Auto Init: XYRANGE: {space_map.XYRANGE}, XYD: {space_map.XYD}")
            ids = conf["ids"]   
            self.slices = []
            for i, idd in enumerate(ids):
                slice = space_map.Slice(idd, self.basePath, i==0)
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
        space_map.Info(f"Init from codex: {keys}")
        return self.init_xys(xys, keys)