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
        pack = np.load(path)
        self.affine_shape = pack.get("affine_shape", affineShape) 
        self._affine = pack.get("affines", None)
        self._grid = pack.get("grids", None)
        self.dfGrid = pack.get("dfGrid", None) != None
        if self._affine is not None:
            self.count = len(self._affine)
        elif self._grid is not None:
            self.count = len(self._grid)
        else:
            raise Exception("TransformDB: No transforms")
        spacemap.Info("TransformDB: %d" % self.count)
        self.ignoreInit = True
        if self.count == 19:
            self.ignoreInit = True
        elif self.count == 20:
            self.ignoreInit = False
        self.useGrid = True
            
    def __iszero(self, data):
        return np.sum(np.abs(data)) < 1
        
    def apply_img(self, img, index):
        if index == 0 and self.ignoreInit:
            return img
        if self.ignoreInit:
            index -= 1
        uint = img.max() > 1.0
        if self._affine is not None:
            affine = self._affine[index]
            if self.__iszero(affine):
                img = img
            else:
                shape = img.shape
                affine1 = spacemap.img.scale_H(affine, self.affine_shape, shape)
                img = spacemap.img.rotate_imgH(img, affine1)
        else:
            img = img
        if self.useGrid and self._grid is not None:
            grid = self._grid[index]
            if self.__iszero(grid):
                img = img
                return img
            else:
                if self.dfGrid:
                    grid = spacemap.points.inverse_grid_train(grid)
                img = spacemap.img.apply_img_by_grid(img, grid)
        if uint:
            img = img * 255
            img = img.astype(np.uint8)
        return img
    
    def apply_point(self, p, index, maxShape=None):
        if index == 0 and self.ignoreInit:
            return p
        if self.ignoreInit:
            index -= 1
        if maxShape is None:
            maxShape = spacemap.XYRANGE[1]
        
        if self._affine is not None:
            affine = self._affine[index-1]
            if self.__iszero(affine):
                pass
            else:
                xyd = maxShape / self.affine_shape[0]
                p = spacemap.points.applyH_np(p, affine, xyd=xyd)
        if not self.useGrid:
            return p
        if self._grid is not None:
            grid = self._grid[index-1]
            if not self.dfGrid:
                grid = spacemap.points.inverse_grid_train(grid)
            xyd = maxShape / grid.shape[0]
            p, _ = spacemap.points.apply_points_by_grid(grid, p, inv_grid=grid, xyd=xyd)
            return p
        return p
    
    def apply_imgs(self, imgs):
        """ imgs: count+1 """
        result = []
        # result.append(imgs[0])
        for index in tqdm.trange(len(imgs)):
            img = imgs[index]
            img2 = self.apply_img(img, index)
            result.append(img2)
        return result
    
    def apply_points(self, ps, maxShape=None):
        """ ps: count+1 """
        result = []
        # result.append(ps[0])
        for index in tqdm.trange(len(ps)):
            p = ps[index]
            p2 = self.apply_point(p, index, maxShape=maxShape)
            result.append(p2)
        return result
    