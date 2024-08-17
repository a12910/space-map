import spacemap
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np

class SliceData:
    def __init__(self, index, projectf=None):
        self.index = str(index)
        self.projectf = spacemap.BASE if projectf is None else projectf
        
    def __path(self, prefix, ext, tag=None, indexTo=None):
        path = "%s/outputs/%s" % (self.projectf, prefix)
        if tag is not None:
            path = path + "_" + str(tag)
        path = path + "_" + self.index
        if indexTo is not None:
            path = path + "_" + str(indexTo)
        path = path + "." + ext
        return path        
        
    def saveH(self, H, indexTo, tag=None):
        path = self.__path("alignH", "npy", tag=tag, indexTo=indexTo)
        np.save(path, H)
    
    def loadH(self, indexTo, tag=None):
        path = self.__path("alignH", "npy", tag=tag, indexTo=indexTo)
        if os.path.exists(path):
            H = np.load(path)
            H[2, :2] = 0
            H[2, 2] = 1
            return H
        return None
    
    def loadGrid(self, indexTo, tag=None):
        path = self.__path("grid", "npy", tag=tag, indexTo=indexTo)
        if os.path.exists(path):
            grid = np.load(path)
            return grid
        return None
    
    def saveGrid(self, grid, indexTo, tag=None):
        path = self.__path("grid", "npy", tag=tag, indexTo=indexTo)
        np.save(path, grid)
        
    def save_labels_df(self, df: pd.DataFrame, labels=None):
        if labels is None:
            labels = df.columns
        data =np.array(df[labels].values)
        self.save_labels(data)
        
    def save_labels(self, data: np.array):
        path = self.__path("labels", "npy")
        np.save(path, data)
        
    def load_labels(self):
        path = self.__path("labels", "npy")
        if os.path.exists(path):
            return np.load(path)
        return None

    def ldm_path(self, npy=False):
        if not npy:
            return self.projectf + "/outputs/ldm_%s.json" % self.index
        else:
            return self.projectf + "/outputs/ldm_%s" % self.index
        
