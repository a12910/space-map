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
    
    def __init__(self, index, projectf=None, first=False, dfMode=True):
        self.dfs = {}
        self.imgs = {}
        self.index = str(index)
        self.projectf = spacemap.BASE if projectf is None else projectf
        self.dfMode = dfMode
        self.data = spacemap.SliceData(index, self.projectf)
        self.first = first
        self.heImg = False
        
    def init_df(self, initdf):
        initdf2 = initdf[["x", "y"]].copy()
        initdf2["x"] += spacemap.APPEND[0]
        initdf2["y"] += spacemap.APPEND[1]
        self.dfs[Slice.rawKey] = initdf2
        self.save_value_df(initdf2, dfKey=Slice.rawKey)
        self.dfMode = True

    def init_img(self, img):
        self.save_value_img(img, Slice.rawKey)
        self.dfMode = False
        
    def save_value_img(self, img, key: str):
        spacemap.Info("Slice Save-IMG %s %s" % (self.index, key))
        self.imgs[key] = img
        path = "%s/imgs/%s_%s.png" % (self.projectf, self.index, key)
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)
        spacemap.imsave(path, img)
                
    def save_value_df(self, df: pd.DataFrame, keys=None, dfKey=None):
        keys = ["x", "y"] if keys is None else keys
        dfKey = Slice.align1Key if dfKey is None else dfKey
        spacemap.Info("Slice Save-DF %s %s" % (self.index, dfKey))
        df1 = df[keys].copy()
        df1.rename(columns={
            keys[0]: "x",
            keys[1]: "y"
        }, inplace=True)
        self.dfs[dfKey] = df1
        path = "%s/outputs/%s_%s.csv.gz" % (self.projectf, self.index, dfKey)
        df1.to_csv(path, index=False)
            
    def save_value_points(self, points, dfKey):
        df = pd.DataFrame(data=points, columns=["x", "y"])
        self.save_value_df(df, dfKey=dfKey)
        
    def ps(self, dfk):
        return self.to_points(dfk)
    
    def to_points(self, dfk):
        return np.array(self.get_df(dfk)[["x", "y"]].values)
    
    def get_df(self, dfKey, keys=None):
        keys = ["x", "y"] if keys is None else keys
        if dfKey not in self.dfs:
            path = "%s/outputs/%s_%s.csv" % (self.projectf, self.index, dfKey)
            path2 = path + ".gz"
            if os.path.exists(path):
                df = pd.read_csv(path)
                self.dfs[dfKey] = df
            elif os.path.exists(path2):
                df = pd.read_csv(path2)
                self.dfs[dfKey] = df
            else:
                spacemap.Info("Slice Load %s %s->raw" % (self.index, dfKey))
                return self.get_df(Slice.rawKey, keys=keys)
        df = self.dfs[dfKey].copy()
        df.rename(columns={
            "x": keys[0],
            "y": keys[1]
        }, inplace=True)
        return df
    
    def get_img_raw(self, dfKey):
        return self.get_img(dfKey, mchannel=True, scale=False)
    
    def get_img(self, dfKey, mchannel=False, scale=True, he=None):
        xyd = spacemap.XYD
        xyr = spacemap.XYRANGE
        shape = (int(xyr[1]/xyd), int(xyr[3]/xyd))

        path = "%s/imgs/%s_%s.png" % (self.projectf, self.index, dfKey)
        if os.path.exists(path):
            img = plt.imread(path)
            self.imgs[dfKey] = img
        elif dfKey == Slice.rawKey:
            raise Exception("Raw Image not found")
        else:
            spacemap.Info("Slice Load %s %s->raw" % (self.index, dfKey))
            return self.get_img(Slice.rawKey, mchannel=mchannel, he=he, scale=scale)
        img = self.imgs[dfKey]
        if len(img.shape) == 3 and not mchannel:
            img = img[:, :, :3]
            img = img.mean(axis=2)
        elif len(img.shape) == 2 and mchannel:
            img = np.stack([img, img, img], axis=2)
        if scale:
            img = cv2.resize(img, shape)
        if (he == True) or (he is None and self.heImg):
            _, img = spacemap.he_img.split_he_background_otsu(img)
        return img        
    
    def create_img(self, dfk: str, useDF=None, mchannel=False, he=False):
        if (useDF is None and self.dfMode) or useDF == True:
            points = self.ps(dfk)
            img = spacemap.show_img3(points)
            if mchannel:
                img = np.stack([img, img, img], axis=2)
            return img
        elif (useDF is None and not self.dfMode) or useDF == False:
            img = self.get_img(dfk, mchannel=mchannel, he=he)
            return img
        else:
            raise Exception("Invalid useDF")
        
    def applyH(self, fromDF, H, toDF, forIMG=False):
        if forIMG is None or forIMG == False:
            points = self.to_points(fromDF)
            points2 = spacemap.applyH_np(points, H)
            self.save_value_points(points2, toDF)
        if forIMG is None or forIMG == True:
            img = self.get_img(fromDF, mchannel=True, scale=False, he=False)
            shape = img.shape[0]
            ishape = spacemap.XYRANGE[1] / spacemap.XYD
            ratio = shape / ishape
            H = H.copy()
            H[0, 2] *= ratio
            H[1, 2] *= ratio
            img = spacemap.he_img.rotate_imgH(img, H)
            self.save_value_img(img, toDF)
    
    @staticmethod
    def show_align(sI, sJ, keyI=None, keyJ=None, forIMG=False, imgHE=False):
        slice_show_align(sI, sJ, keyI, keyJ, forIMG, imgHE=imgHE)
        
def slice_show_align(sI: Slice, sJ: Slice, 
                     keyI=None, keyJ=None, forIMG=False, imgHE=False):
    
    keyI = Slice.finalKey if keyI is None else keyI
    keyJ = Slice.rawKey if keyJ is None else keyJ
    spacemap.Info("Slice Align: %s-%s %s-%s" % (sI.index, keyI, sJ.index, keyJ))
    if forIMG:
        imgI = sI.get_img(keyI, he=imgHE)
        imgJ = sJ.get_img(keyJ, he=imgHE)
        spacemap.show_compare_channel(imgI, imgJ, titleI=sI.index, titleJ=sJ.index)
    else:
        dfI = sI.get_df(keyI)
        dfJ = sJ.get_df(keyJ)
        spacemap.show_xy([dfI, dfJ], 
                        ["Target_" + str(sI.index), "New_" + str(sJ.index)], 
                        keyx="x", 
                        keyy="y", s=0.2, alpha=0.3)
    