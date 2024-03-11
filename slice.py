import spacemap
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np

class Slice:
    rawKey = "raw"
    align1Key = "align1"
    align2Key = "align2"
    finalKey = "final"
    enhanceKey = "enhance"
    def __init__(self, initdf, index, projectf=None):
        self.df = initdf
        self.dfs = {}
        self.index = str(index)
        self.projectf = spacemap.BASE if projectf is None else projectf
        initdf2 = initdf[["x", "y"]].copy()
        initdf2["x"] += spacemap.APPEND[0]
        initdf2["y"] += spacemap.APPEND[1]
        self.dfs[Slice.rawKey] = initdf2
        self.save_df()
    
    def save_df(self):
        spacemap.Info("Slice Save DF: %s" % self.index)
        for key in self.dfs:
            path = "%s/outputs/%s_%s.csv" % (self.projectf, self.index, key)
            df = self.dfs[key]
            df.to_csv(path, index=False)
                
    def save_value(self, df: pd.DataFrame, keys=None, dfKey=None):
        keys = ["x", "y"] if keys is None else keys
        dfKey = Slice.align1Key if dfKey is None else dfKey
        spacemap.Info("LDDMM Save %s %s" % (self.index, dfKey))
        df1 = df[keys].copy()
        df1.rename(columns={
            keys[0]: "x",
            keys[1]: "y"
        }, inplace=True)
        self.dfs[dfKey] = df1
        path = "%s/outputs/%s_%s.csv" % (self.projectf, self.index, dfKey)
        df1.to_csv(path, index=False)
            
    def save_value_points(self, points, dfKey):
        df = pd.DataFrame(data=points, columns=["x", "y"])
        dfKey = Slice.finalKey if dfKey is None else dfKey
        self.save_value(df, dfKey=dfKey)
        
    def saveH(self, H, indexTo):
        path = "%s/outputs/alignH_%s_%s.npy" % (self.projectf, str(self.index), str(indexTo))
        np.save(path, H)
    
    def loadH(self, indexTo):
        path = "%s/outputs/alignH_%s_%s.npy" % (self.projectf, str(self.index), str(indexTo))
        if os.path.exists(path):
            H = np.load(path)
            H[2, :2] = 0
            H[2, 2] = 1
            return H
        return None
    
    def loadGrid(self, indexTo):
        path = "%s/outputs/grid_%s_%s.npy" % (self.projectf, str(self.index), str(indexTo))
        if os.path.exists(path):
            grid = np.load(path)
            return grid
        return None
    
    def saveGrid(self, grid, indexTo):
        path = "%s/outputs/grid_%s_%s.npy" % (self.projectf, str(self.index), str(indexTo))
        np.save(path, grid)
        
    def save_labels_df(self, df, labels):
        data =np.array(df[labels].values)
        self.save_labels(data)
        
    def save_labels(self, data: np.array):
        path = "%s/outputs/labels_%s.npy" % (self.projectf, str(self.index))
        np.save(path, data)
        
    def load_labels(self):
        path = "%s/outputs/labels_%s.npy" % (self.projectf, str(self.index))
        if os.path.exists(path):
            return np.load(path)
        return None
    
    def get_df(self, dfKey, keys=None):
        keys = ["x", "y"] if keys is None else keys
        if isinstance(dfKey, int):
            if dfKey == 0:
                dfKey = Slice.rawKey
            elif dfKey == 1:
                dfKey = Slice.align1Key
            else:
                dfKey = Slice.finalKey
        if dfKey not in self.dfs:
            path = "%s/outputs/%s_%s.csv" % (self.projectf, self.index, dfKey)
            if os.path.exists(path):
                df = pd.read_csv(path)
                self.dfs[dfKey] = df
            else:
                dfKey = Slice.rawKey
        df = self.dfs[dfKey].copy()
        df.rename(columns={
            "x": keys[0],
            "y": keys[1]
        }, inplace=True)
        return df
    
    def create_img2(self, dfk: str):
        points = self.to_points(dfk)
        img = spacemap.show_img3(points)
        img = np.array(img, dtype=int)
        return img
            
    def create_img(self, force: bool, tag: str, dfk: str):
        imgConf = spacemap.IMGCONF
        enhance = imgConf.get("kernel", 0)
        mid = imgConf.get("mid", 0)
        raw = imgConf.get("raw", 0)
        
        wid = spacemap.XYRANGE[1] // spacemap.XYD
        tagg = "%s_%s_w%se%sm%sr%s.png" % (tag, self.index, str(wid), str(enhance), str(mid), str(raw))
        path = self.projectf + "/imgs/" + tagg
        if os.path.exists(path) and not force:
            img = plt.imread(path)
            if len(img.shape) > 2:
                img = img[:, :, 1]
        else:
            spacemap.Info("Slice Create Img: %s-%s df-%s %s" % (self.index, tag, dfk, tagg))
            points = self.to_points(dfk)
            img = spacemap.show_img3(points)
            plt.imsave(path, img)
        img = np.array(img, dtype=int)
        return img
    
    def to_points(self, dfk):
        df = self.get_df(dfk)
        return np.array(df[["x", "y"]].values)
    
    def ldm_path(self, npy=False):
        if not npy:
            return self.projectf + "/outputs/ldm_%s.json" % self.index
        else:
            return self.projectf + "/outputs/ldm_%s" % self.index
        
    def applyH(self, fromDF, H, toDF):
        points = self.to_points(fromDF)
        points2 = spacemap.applyH_np(points, H)
        self.save_value_points(points2, toDF)
        
    
    @staticmethod
    def show_align(sI, sJ, keyI=None, keyJ=None):
        slice_show_align(sI, sJ, keyI, keyJ)
        
def slice_show_align(sI: Slice, sJ: Slice, 
                     keyI=None, keyJ=None):
    
    keyI = Slice.finalKey if keyI is None else keyI
    keyJ = Slice.rawKey if keyJ is None else keyJ
    spacemap.Info("Slice Align: %s-%s %s-%s" % (sI.index, keyI, sJ.index, keyJ))
    dfI = sI.get_df(keyI)
    dfJ = sJ.get_df(keyJ)
    spacemap.show_xy([dfI, dfJ], 
                    ["Target_" + str(sI.index), "New_" + str(sJ.index)], 
                    keyx="x", 
                    keyy="y", s=0.5)
    