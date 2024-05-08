import spacemap
import numpy as np
import cv2

class FlowExport:
    def __init__(self, slices: list[spacemap.Slice]):
        self.slices = slices

    def export_affine_grid(self, affineShape=None):
        if affineShape is None:
            affineShape = spacemap.img.get_shape()
        pack = {}
        pack["affine_shape"] = affineShape
        
        affines = []
        grids = []
        initS = self.slices[0].index
        for s in self.slices[1:]:
            affine = s.data.loadH(initS, "cell")
            grid = s.data.loadGrid(initS, "final_ldm")
            affines.append(affine)
            grids.append(grid[0])
        a = np.array(affines)
        g = np.array(grids)
        pack["affines"] = a
        pack["grids"] = g
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
        