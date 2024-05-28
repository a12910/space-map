import spacemap
import numpy as np
import tqdm

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
        self._affine = pack.get("affines", None)
        self._grid = pack.get("grids", None)
        self._inverse_grids = pack.get("inverse_grids", None)
        if self._affine is not None:
            self.count = len(self._affine)
        elif self._grid is not None:
            self.count = len(self._grid)
        else:
            self.count = len(self._inverse_grids)
        spacemap.Info("TransformDB: %d" % self.count)
        
    def apply_img(self, img, index, useGrid=True):
        if index == 0:
            return img
        if self._affine is not None:
            affine = self._affine[index-1]
            
            shape = img.shape
            affine1 = spacemap.img.scale_H(affine, self.affine_shape, shape)
            img1 = spacemap.img.rotate_imgH(img, affine1)
        else:
            img1 = img
        if useGrid and self._grid is not None:
            grid = self._grid[index-1]
            img2 = spacemap.img.apply_img_by_grid(img1, grid)
            return img2
        return img1
    
    def apply_point(self, p, index, maxShape=None, useGrid=True):
        if index == 0:
            return p
        if maxShape is None:
            maxShape = spacemap.XYRANGE[1]
        
        if self._affine is not None:
            affine = self._affine[index-1]
            xyd = maxShape / self.affine_shape[0]
            p = spacemap.points.applyH_np(p, affine, xyd=xyd)
        if not useGrid:
            return p
        
        if self._grid is not None:
            grid = self._grid[index-1]
            xyd = maxShape / grid.shape[0]
            p, _ = spacemap.points.apply_points_by_grid(grid, p, xyd=xyd)
            return p
        elif self._inverse_grids is not None:
            grid = self._inverse_grids[index-1]
            xyd = maxShape / grid.shape[0]
            p, _ = spacemap.points.apply_points_by_grid(grid, p, grid, xyd=xyd)
            return p
        return p
    
    def apply_imgs(self, imgs):
        """ imgs: count+1 """
        result = []
        result.append(imgs[0])
        for index in tqdm.trange(1, len(imgs)):
            img = imgs[index]
            img2 = self.apply_img(img, index)
            result.append(img2)
        return result
    
    def apply_points(self, ps, maxShape=None):
        """ ps: count+1 """
        result = []
        result.append(ps[0])
        for index in tqdm.trange(1, len(ps)):
            p = ps[index]
            p2 = self.apply_point(p, index, maxShape=maxShape)
            result.append(p2)
        return result
    