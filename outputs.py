import spacemap
import numpy as np

class Transform:
    def __init__(self, grid=None, affine=None, shape=None) -> None:
        self.grid = grid
        self.affine = affine
        self.shape = shape
    
    def pack(self, path=None):
        pack = {}
        if self.grid is not None:
            pack["grid"] = self.grid
        if self.affine is not None:
            pack["affine"] = self.affine
        if path is not None:
            np.savez(path, **pack)
        return pack
    
    def load(self, path=None, pack=None):
        if pack is not None:
            self.grid = pack.get("grid", None)
            self.affine = pack.get("affine", None)
        if path is not None:
            data = np.load(path)
            self.grid = data.get("grid", None)
            self.affine = data.get("affine", None)

class TransformDB:
    def __init__(self, path, affineShape) -> None:
        self.transforms = []
        pack = np.load(path)
        self.affine_shape = pack.get("affine_shape", affineShape) 
        self._affine = pack["affines"]
        self._grid = pack["grids"]
        self.count = len(self._affine)
        
    def apply_imgs(self, imgs):
        """ imgs: count+1 """
        result = []
        result.append(imgs[0])
        for index in range(1, len(imgs)):
            affine = self._affine[index-1]
            grid = self._grid[index-1]
            img = imgs[index]
            shape = img.shape
            affine1 = spacemap.img.scale_H(affine, self.affine_shape, shape)
            img1 = spacemap.img.rotate_imgH(img, affine1)
            img2 = spacemap.img.apply_img_by_grid(img1, grid)
            result.append(img2)
        return result
    
    def apply_points(self, imgs):
        """ imgs: count+1 """
        result = []
        for index in range(1, len(imgs)):
            affine = self._affine[index-1]
            grid = self._grid[index-1]
            img = imgs[index]
            shape = img.shape
            affine1 = spacemap.img.scale_H(affine, self.affine_shape, shape)
            img1 = spacemap.img.rotate_imgH(img, affine1)
            img2 = spacemap.img.apply_img_by_Grid(img1, grid)
            result.append(img2)
        return result
    