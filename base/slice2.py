import spacemap
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2, json
import numpy as np
from . import SliceData

class SliceImg:
    DF = "DF"
    Img = "IMG"
    def __init__(self, index, imgKey, dfMode=False, 
                 heMode=False, projectf=None):
        self.index = str(index)
        self.projectf = spacemap.BASE if projectf is None else projectf
        self.imgKey = imgKey
        self.dfMode = dfMode
        self.heMode = heMode
        
    def to_config(self):
        return {
            "index": self.index,
            "key": self.imgKey, 
            "dfMode": "TRUE" if self.dfMode else "FALSE", 
            "heMode": "TRUE" if self.heMode else "FALSE"
        }
    
    def save_img(self, img, key):
        spacemap.Info("SliceImg Save %s-%s:%s" % (self.index, self.imgKey, key))
        path = "%s/imgs/%s_%s_%s.png" % (self.projectf, self.index, self.imgKey, key)
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)
        if len(img.shape) == 3 and img.shape[2] == 4:
            mask = img[:, :, 3]
            img = img[:, :, :3]
            img[mask == 0] = 0
        spacemap.imsave(path, img)
        
    def ps(self, dfk):
        return self.get_points(dfk)
    
    def get_points(self, key):
        path = "%s/outputs/%s_%s_%s.csv.gz" % (self.projectf, self.index, self.imgKey, key)
        if os.path.exists(path):
            df = pd.read_csv(path)
            return df[["x", "y"]].values
        elif key == Slice2.rawKey:
            raise Exception("no Data")
        else:
            spacemap.Info("Slice Load %s %s->raw" % (self.index, key))
            return self.get_points(Slice2.rawKey)
    
    def get_img(self, key, mchannel=False, scale=True, fixHe=False):
        if self.dfMode:
            points = self.get_points(key)
            img = spacemap.show_img3(points)
        else:
            path = "%s/imgs/%s_%s_%s.png" % (self.projectf, self.index, self.imgKey, key)
            if not os.path.exists(path):
                if key == Slice2.rawKey:
                    raise Exception("no Data")
                spacemap.Info("Slice Load %s %s->raw" % (self.index, key))
                return self.get_img(Slice2.rawKey, mchannel=mchannel, 
                                    scale=scale, fixHe=fixHe)
            img = cv2.imread(path)
        if len(img.shape) == 3 and not mchannel:
            img = img[:, :, :3]
            img = img.mean(axis=2)
        elif len(img.shape) == 2 and mchannel:
            img = np.stack([img, img, img], axis=2)
        if scale:
            xyd = spacemap.XYD
            xyr = spacemap.XYRANGE
            shape = (int(xyr[1]/xyd), int(xyr[3]/xyd))
            img = cv2.resize(img, shape)
        if fixHe and self.heMode:
            _, img = spacemap.he_img.split_he_background_otsu(img)
        if spacemap.IMGCONF.get("raw", 0) == 0:
            img[img > 0] = 1.0
        return img
        
    def save_points(self, points, key):
        spacemap.Info("SliceImg Save Points %s-%s:%s" % (self.index, self.imgKey, key))
        path = "%s/outputs/%s_%s_%s.csv.gz" % (self.projectf, self.index, self.imgKey, key)
        df = pd.DataFrame(data=points, columns=["x", "y"])
        df.to_csv(path, index=False)
        
    def applyH(self, fromKey, H, toKey):
        if H is None:
            if self.dfMode:
                points = self.get_points(fromKey)
                self.save_points(points, toKey)
            else:
                img = self.get_img(fromKey, mchannel=True, 
                                   scale=False, fixHe=False)
                self.save_img(img, toKey)
        else:
            if self.dfMode:
                points = self.get_points(fromKey)
                H_np = spacemap.img.to_npH(H)
                points2 = spacemap.applyH_np(points, H_np)
                self.save_points(points2, toKey)
            else:
                img = self.get_img(fromKey, mchannel=True, scale=False, fixHe=False)
                shape = img.shape[0]
                ishape = spacemap.XYRANGE / spacemap.XYD
                ratio = shape / ishape
                H = H.copy()
                H[0, 2] *= ratio
                H[1, 2] *= ratio
                img = spacemap.he_img.rotate_imgH(img, H)
                self.save_img(img, toKey)
            
    def apply_grid(self, fromKey, toKey, grid, inv_grid=None):
        if grid is None:
            self.applyH(fromKey, None, toKey)
        else:
            if self.dfMode:
                points = self.get_points(fromKey)
                points2, _ = spacemap.points.apply_points_by_grid(grid, points, inv_grid)
                self.save_points(points2, toKey)
            else:
                img = self.get_img(fromKey, mchannel=True, scale=False, fixHe=False)
                img1 = spacemap.img.apply_img_by_grid(img, grid)
                self.save_img(img1, toKey)
        
    @staticmethod
    def load_config(path, projectf) -> dict[str: 'SliceImg']:
        if os.path.exists(path):
            with open(path, "r") as f:
                text = f.read()
                packs = json.loads(text)
            imgs = {}
            for pack in packs.values():
                img = SliceImg(index=pack["index"], 
                               imgKey=pack["key"], 
                               dfMode=pack["dfMode"] == "TRUE", 
                               heMode=pack["heMode"] == "TRUE", 
                               projectf=projectf)
                imgs[img.imgKey] = img
            return imgs
        else:
            return {}

    @staticmethod
    def save_config(imgs, path):
        packs = {}
        for key, img in imgs.items():
            packs[key] = img.to_config()
        with open(path, "w") as f:
            p = json.dumps(packs)
            f.write(p)

class Slice2:
    rawKey = "raw"
    align1Key = "align1"
    align2Key = "align2"
    finalKey = "final"
    enhanceKey = "enhance"
    
    def __init__(self, index, projectf=None, first=False):
        self.imgs: dict[str: SliceImg] = {}
        self.index = str(index)
        self.projectf = spacemap.BASE if projectf is None else projectf
        self.data = SliceData(index, self.projectf)
        self.first = first
        confPath = "%s/outputs/%s_conf.json" % (self.projectf, self.index)
        self.imgs = SliceImg.load_config(confPath, self.projectf)
        
    def add_img(self, imgKey, heMode):
        img = SliceImg(self.index, imgKey, False, heMode, self.projectf)
        self.imgs[imgKey] = img
        self.save_config()
        
    def init_df(self, initdf):
        if initdf.shape[1] == 0:
            raise Exception("Empty DataFrame")
        img = SliceImg(self.index, SliceImg.DF, 
                       dfMode=True, heMode=False, 
                       projectf=self.projectf)
        initdf2 = initdf[["x", "y"]].values
        initdf2[:, 0] += spacemap.APPEND[0]
        initdf2[:, 1] += spacemap.APPEND[1]
        img.save_points(initdf2, Slice2.rawKey)
        self.imgs[SliceImg.DF] = img

    def init_img(self, img, he=False, imgKey=None):
        if imgKey is None:
            imgKey = SliceImg.Img
        s = SliceImg(self.index, imgKey,
                       dfMode=False, heMode=he, 
                       projectf=self.projectf)
        s.save_img(img, Slice2.rawKey)
        self.imgs[imgKey] = s
        
    def save_config(self):
        confPath = "%s/outputs/%s_conf.json" % (self.projectf, self.index)
        SliceImg.save_config(self.imgs, confPath)
        
    def save_value_img(self, img, imgKey: str, key: str, he=False):
        if imgKey not in self.imgs:
            slice = SliceImg(self.index, imgKey, 
                             dfMode=False, heMode=he, 
                             projectf=self.projectf)
            self.imgs[imgKey] = slice
        slice = self.imgs[imgKey]
        slice.save_img(img, key)
                   
    def save_value_points(self, points, dfKey):
        if SliceImg.DF not in self.imgs:
            slice = SliceImg(self.index, SliceImg.DF,
                                dfMode=True, heMode=False, 
                                projectf=self.projectf)
            self.imgs[SliceImg.DF] = slice
        slice = self.imgs[SliceImg.DF]
        slice.save_points(points, dfKey)
    
    def get_img_raw(self, dfKey):
        return self.get_img(dfKey, mchannel=True, scale=False, fixHe=False)
    
    def get_img(self, imgKey, dfKey, mchannel=False, scale=True, fixHe=False):
        if imgKey not in self.imgs:
            raise Exception("No Image")
        return self.imgs[imgKey].get_img(dfKey, mchannel=mchannel, 
                                         scale=scale, fixHe=fixHe)
        
    def create_img(self, imgKey, dfKey, 
                   mchannel=False, scale=True, fixHe=False):
        return self.get_img(imgKey, dfKey, 
                            mchannel=mchannel, scale=scale, fixHe=fixHe)
        
    def applyH(self, fromDF, H, toDF):
        for img in self.imgs.values():
            img.applyH(fromDF, H, toDF)
            
    def apply_grid(self, fromDF, toDF, grid, inv_grid=None):
        if grid is None and inv_grid is not None:
            if SliceImg.DF in self.imgs.keys() and len(self.imgs.keys()) > 1:
                grid = spacemap.points.inverse_grid_train(inv_grid)
        for img in self.imgs.values():
            img.apply_grid(fromDF, toDF, grid, inv_grid)
    
    @staticmethod
    def show_align(sI, sJ, keyI=None, keyJ=None, imgKey=SliceImg.Img):
        keyI = Slice2.finalKey if keyI is None else keyI
        keyJ = Slice2.rawKey if keyJ is None else keyJ
        if imgKey not in sI.imgs or imgKey not in sJ.imgs:
            raise Exception("No Image %s" % imgKey)
        spacemap.Info("Slice Align: %s-%s %s-%s" % (sI.index, keyI, sJ.index, keyJ))
        if imgKey != SliceImg.DF:
            imgI = sI.get_img(imgKey, keyI, mchannel=False, scale=True, fixHe=True)
            imgJ = sJ.get_img(imgKey, keyJ, mchannel=False, scale=True, fixHe=True)
            spacemap.show_compare_channel(imgI, imgJ, titleI=sI.index, titleJ=sJ.index)
        else:
            psI = sI.imgs[imgKey].get_points(keyI)
            psJ = sJ.imgs[imgKey].get_points(keyJ)
            dfI = pd.DataFrame(data=psI, columns=["x", "y"])
            dfJ = pd.DataFrame(data=psJ, columns=["x", "y"])
            spacemap.show_xy([dfI, dfJ], 
                            ["Target_" + str(sI.index), "New_" + str(sJ.index)], 
                            keyx="x", 
                            keyy="y", s=0.1, alpha=0.2)
        