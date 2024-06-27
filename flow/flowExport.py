import spacemap
import numpy as np
import cv2

class FlowExport:
    def __init__(self, slices: list[spacemap.Slice]):
        self.slices = slices
        self.ldmKey = "final_ldm"
        self.affineKey = "cell"
        self.initSlice = slices[0]

    def export_affine_grid(self, affineShape=None, gridKey1="final_ldm", gridKey2="img", save=None):
        if affineShape is None:
            affineShape = spacemap.img.get_shape()
        pack = {}
        pack["affine_shape"] = affineShape

        affines = []
        grids = []
        initS = self.initSlice.index
        initI = 0
        for i, s in enumerate(self.slices):
            print(s.index)
            if s.index == self.initSlice:
                initI = i
                affines.append(None)
                grids.append(None)
                continue
            affine = s.data.loadH(initS, self.affineKey)
            grid = s.data.loadGrid(initS, gridKey1)
            if grid is None:
                grid = s.data.loadGrid(initS, gridKey2)
            affines.append(affine)
            grids.append(grid[0])
        i1 = (initI+1) % (len(self.slices) - 1)
        affines[initI] = np.zeros_like(affines[i1])
        grids[initI] = np.zeros_like(grids[i1])
        
        a = np.array(affines)
        g = np.array(grids)
        pack["affines"] = a
        pack["grids"] = g
        if save:
            np.savez_compressed(save, **pack)
        return pack
    
    def export_imgs(self, key, mchannel=True, he=False, scale=False):
        imgs = []
        for s in self.slices:
            img = s.get_img(key, mchannel=mchannel, scale=scale, he=he)
            imgs.append(img)
        return np.array(imgs)
    
    def import_imgs(self, key, imgs):
        for i, img in enumerate(imgs):
            if isinstance(img, str):
                img = cv2.imread(img)
            self.slices[i].save_value_img(img, key)
        