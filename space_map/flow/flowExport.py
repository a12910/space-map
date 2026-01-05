import space_map
import numpy as np
import os
import tifffile as tiff
import cv2

class FlowExport:
    def __init__(self, slices: list[space_map.Slice]):
        self.slices = slices
        self.ldmKey = "final_ldm"
        self.affineKey = "cell"
        self.initGridSlice = slices[0]
        self.initAffineSlice = slices[0]

    def export_affine_grid(self, affineShape=None, gridKey1="final_ldm", gridKey2="img", save=None):
        if affineShape is None:
            affineShape = space_map.img.get_shape()
        pack = {}
        pack["affine_shape"] = affineShape

        affines = []
        grids = []
        initS = self.initAffineSlice.index
        initI = 0
        for i, s in enumerate(self.slices):
            if s.index == self.initAffineSlice:
                initI = i
                affines.append(None)
                continue
            affine = s.data.loadH(initS, self.affineKey)
            affines.append(affine)
        i1 = (initI+1) % (len(self.slices) - 1)
        affines[initI] = np.zeros_like(affines[i1])
        
        for i, s in enumerate(self.slices):
            if s.index == self.initGridSlice:
                initI = i
                grids.append(None)
                continue
            grid = s.data.loadGrid(initS, gridKey1)
            if grid is None:
                grid = s.data.loadGrid(initS, gridKey2)
            grids.append(grid[0])
        i1 = (initI+1) % (len(self.slices) - 1)
        grids[initI] = np.zeros_like(grids[i1])
        
        a = np.array(affines)
        g = np.array(grids)
        pack["affines"] = a
        pack["grids"] = g
        if save:
            np.savez_compressed(save, **pack)
        return pack

    def write_tiffs(self, dfKey, key, prefix):
        imgs_ = self.export_imgs(dfKey, key, mchannel=False, he=False, scale=False)
        self.write_tiffs_imgs(imgs_, prefix)
    
    def export_imgs(self, dfKey, key, mchannel=True, he=False, scale=False):
        imgs = []
        for s in self.slices:
            img = s.get_img(dfKey, key, mchannel=mchannel, scale=scale, fixHe=he)
            imgs.append(img)
        return np.array(imgs)

    def write_tiffs_imgs(self, imgs, prefix):
        imgs_ = imgs
        imgs_ *= 32
        imgs_[imgs_ > 255] = 255
        imgs_ = imgs_.astype(np.uint8)
        dirr = os.path.dirname(prefix)
        if not os.path.exists(dirr):
            os.makedirs(dirr)
        for i in range(imgs_.shape[0]):
            tiff.imwrite("%s_%d.tiff" % (prefix, i), imgs_[i])
    
    def import_imgs(self, key, imgs):
        for i, img in enumerate(imgs):
            if isinstance(img, str):
                img = cv2.imread(img)
            self.slices[i].save_value_img(img, key)
        